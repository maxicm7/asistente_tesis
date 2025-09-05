# ==============================================================================
# 1. IMPORTACIONES Y CONFIGURACIÓN INICIAL
# ==============================================================================
import streamlit as st
import os
import google.generativeai as genai

# --- Importaciones de LangChain ---
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain import hub # Para el agente ReAct

# ##############################################################################
# Se utiliza st.secrets para leer las claves de API de forma segura desde
# la configuración de Streamlit Cloud. Este método es el recomendado.
# ##############################################################################
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

    # Configurar las variables de entorno para que LangChain las detecte automáticamente
    # Aunque ya las tenemos en variables, muchas librerías buscan directamente en os.environ
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

except KeyError as e:
    st.error(f"Error: No se encontró el secreto '{e.args[0]}'.")
    st.error("Por favor, asegúrate de haber configurado todas las claves (GOOGLE_API_KEY, HUGGINGFACEHUB_API_TOKEN, TAVILY_API_KEY) en los 'Secrets' de tu app en Streamlit Cloud.")
    st.stop() # Detiene la ejecución si falta alguna clave

# ==============================================================================
# 2. DEFINICIÓN DE HERRAMIENTAS (HABILIDADES DEL AGENTE)
# ==============================================================================

@tool
def web_search(query: str) -> str:
    """Busca en la web información actualizada, incluyendo fuentes de datos para investigación académica."""
    try:
        # La clase buscará la API key en las variables de entorno si no se pasa explícitamente
        search = TavilySearchAPIWrapper()
        return search.run(query)
    except Exception as e:
        return f"Error en la búsqueda web: {e}"

@tool
def summarize_paper(pdf_path: str) -> str:
    """Carga y resume un artículo de investigación en formato PDF. Extrae la metodología, resultados y conclusiones clave."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        
        full_text = " ".join([page.page_content for page in pages])
        
        # Usar un LLM específico y rápido para la tarea de resumen
        summarizer_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
        
        prompt_template = f"""
        Basado en el siguiente texto de un paper académico, por favor, crea un resumen conciso y estructurado (aproximadamente 300 palabras).
        El resumen debe enfocarse en:
        1. **Problema y Objetivos:** ¿Qué pregunta busca responder el paper?
        2. **Metodología:** ¿Qué métodos usa (ej. modelo de panel, DSGE, SVAR)? ¿Cuál es la fuente de datos?
        3. **Hallazgos Clave:** ¿Cuáles son los resultados más importantes y significativos?
        4. **Conclusiones:** ¿Cuál es la implicación principal del estudio?

        Texto del Paper:
        {full_text[:25000]} 
        """
        
        summary = summarizer_llm.invoke(prompt_template).content
        return summary
    except Exception as e:
        return f"Error al procesar el PDF: {e}"

tools = [web_search, summarize_paper]

# ==============================================================================
# 3. CONFIGURACIÓN DEL AGENTE Y LA MEMORIA
# ==============================================================================

# Prompt para el agente Tool Calling (Gemini)
tool_calling_prompt = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente de investigación de doctorado de clase mundial.
    Tu misión es ayudar al usuario a avanzar en su tesis sobre la transición energética, precios al carbono y modelado económico.
    - Usa tus herramientas cuando sea necesario para buscar información o analizar documentos.
    - Cuando te pidan código para modelos (Panel Data, SVAR, DSGE, Streamlit), genera el código Python directamente.
    - Siempre responde de forma rigurosa, clara y académica."""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Prompt para el agente ReAct (Hugging Face)
react_prompt = hub.pull("hwchase17/react")

# Inicializar la memoria para el historial del chat
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ==============================================================================
# 4. INTERFAZ DE USUARIO CON STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Asistente de Tesis IA", layout="wide")
st.title("🤖 Asistente de Tesis IA para Transición Energética")

# La verificación de claves ahora está al principio, así que podemos quitarla de aquí.

# --- Barra Lateral para Controles ---
with st.sidebar:
    st.header("Configuración")
    model_choice = st.selectbox(
        "Elige tu modelo:",
        ("Google Gemini-1.5-Pro", "Mistral-7B (via Hugging Face)")
    )
    temperature = st.slider(
        "Temperatura (creatividad):", 
        min_value=0.0, max_value=1.0, value=0.4, step=0.1
    )
    
    uploaded_file = st.file_uploader("Sube un paper (PDF)", type="pdf")
    
    if uploaded_file:
        temp_dir = "temp_pdf"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Guardamos el archivo temporalmente
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.uploaded_file_path = temp_file_path
        st.success(f"Archivo '{uploaded_file.name}' cargado y listo para analizar.")

# --- Lógica de Selección de Modelo y Construcción de Agente ---
# ##############################################################################
# ### CAMBIO 2: CORRECCIÓN MENOR EN LA LÓGICA DEL MODELO ###
# Se ajusta la condición para que coincida exactamente con el texto del selectbox.
# ##############################################################################
if model_choice == "Google Gemini-1.5-Pro":
    st.info("Nota: Se usará el modelo 'gemini-1.5-flash' para optimizar la velocidad y la cuota gratuita.")
    # SOLUCIÓN 1: Usamos gemini-1.5-flash-latest, es más rápido y tiene una cuota más generosa.
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=temperature, convert_system_message_to_human=True)
    agent = create_tool_calling_agent(llm, tools, tool_calling_prompt)

else: # "Mistral-7B (via Hugging Face)"
    st.info("Nota: Se usará el modelo 'Mixtral-8x7B' ya que Mistral-7B no está en la capa gratuita de la API.")
    # SOLUCIÓN 2: Usamos un modelo más potente que SÍ está disponible en la capa gratuita.
    # El modelo Mixtral es una excelente alternativa.
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=temperature)
    agent = create_react_agent(llm, tools, react_prompt)


agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=st.session_state.memory, 
    verbose=True, 
    handle_parsing_errors=True # Muy útil para agentes ReAct
)

# --- Lógica del Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hola, soy tu asistente de investigación. ¿Cómo puedo ayudarte hoy con tu tesis?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Pregunta sobre papers, datos, modelos..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        input_for_agent = {"input": user_prompt}
        
        # Añadir contexto del archivo subido si existe
        if 'uploaded_file_path' in st.session_state and st.session_state.uploaded_file_path:
            input_for_agent["input"] += (
                f"\n\n[Contexto Adicional] El usuario ha subido el archivo ubicado en: "
                f"'{st.session_state.uploaded_file_path}'. "
                f"Si la pregunta se refiere a 'el paper' o 'el documento', usa la herramienta `summarize_paper` con esa ruta."
            )

        with st.spinner("Procesando tu solicitud..."):
            response = agent_executor.invoke(input_for_agent)
            st.markdown(response["output"])
        
    st.session_state.messages.append({"role": "assistant", "content": response["output"]})

    # Limpiar el archivo después de su uso para la siguiente interacción
    if 'uploaded_file_path' in st.session_state and st.session_state.uploaded_file_path:
        if os.path.exists(st.session_state.uploaded_file_path):
            os.remove(st.session_state.uploaded_file_path)
        del st.session_state.uploaded_file_path
