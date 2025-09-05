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

# ==============================================================================
# 2. CARGA DE SECRETS Y VARIABLES DE ENTORNO
# ==============================================================================
# <--- CAMBIO 1: Carga de secretos usando el método nativo y correcto de Streamlit.
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

    # Configurar las variables de entorno para que LangChain las detecte automáticamente
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

except KeyError as e:
    st.error(f"Error: No se encontró el secreto '{e.args[0]}'.")
    st.error("Por favor, asegúrate de haber configurado todas las claves (GOOGLE_API_KEY, HUGGINGFACEHUB_API_TOKEN, TAVILY_API_KEY) en los 'Secrets' de tu app en Streamlit Cloud.")
    st.stop()

# ==============================================================================
# 3. DEFINICIÓN DE HERRAMIENTAS (HABILIDADES DEL AGENTE)
# ==============================================================================

@tool
def web_search(query: str) -> str:
    """Busca en la web información actualizada, incluyendo fuentes de datos para investigación académica."""
    try:
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
        
        summarizer_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
        
        prompt_template = f"""
        Basado en el siguiente texto de un paper académico, por favor, crea un resumen conciso y estructurado (aproximadamente 300 palabras).
        El resumen debe enfocarse en:
        1. **Problema y Objetivos:** ¿Qué pregunta busca responder el paper?
        2. **Metodología:** ¿Qué métodos usa? ¿Cuál es la fuente de datos?
        3. **Hallazgos Clave:** ¿Cuáles son los resultados más importantes?
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
# 4. CONFIGURACIÓN DEL AGENTE Y LA MEMORIA
# ==============================================================================

tool_calling_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente de investigación de doctorado de clase mundial. Tu misión es ayudar al usuario a avanzar en su tesis. Usa tus herramientas cuando sea necesario. Responde de forma rigurosa, clara y académica."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

react_prompt = hub.pull("hwchase17/react")

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ==============================================================================
# 5. INTERFAZ DE USUARIO Y LÓGICA PRINCIPAL
# ==============================================================================

st.set_page_config(page_title="Asistente de Tesis IA", layout="wide")
st.title("🤖 Asistente de Tesis IA para Transición Energética")

with st.sidebar:
    st.header("Configuración")
    
    # <--- CAMBIO 2: Actualizamos las etiquetas para que reflejen los modelos reales que usaremos.
    model_choice = st.selectbox(
        "Elige tu modelo:",
        ("Google Gemini-1.5-Flash", "Hugging Face (Flan-T5)")
    )
    temperature = st.slider("Temperatura (creatividad):", 0.0, 1.0, 0.4, 0.1)
    
    uploaded_file = st.file_uploader("Sube un paper (PDF)", type="pdf")
    
    if uploaded_file:
        temp_dir = "temp_pdf"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.uploaded_file_path = temp_file_path
        st.success(f"Archivo '{uploaded_file.name}' cargado.")

# --- Lógica de Selección de Modelo y Construcción de Agente ---
# <--- CAMBIO 3: Usamos modelos fiables y estables para las capas gratuitas de las APIs.
if model_choice == "Google Gemini-1.5-Flash":
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=temperature,
        convert_system_message_to_human=True
    )
    agent = create_tool_calling_agent(llm, tools, tool_calling_prompt)

else: # "Hugging Face (Flan-T5)"
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-xxl",
        temperature=0.1, # Los agentes ReAct funcionan mejor con baja creatividad.
        max_new_tokens=1024
    )
    agent = create_react_agent(llm, tools, react_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=st.session_state.memory,
    verbose=True,
    handle_parsing_errors="Check your output and try again." # Mejora la robustez del agente
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
        
        if 'uploaded_file_path' in st.session_state and st.session_state.uploaded_file_path:
            input_for_agent["input"] += (
                f"\n\n[Contexto Adicional] El usuario ha subido el archivo ubicado en: '{st.session_state.uploaded_file_path}'. "
                f"Si la pregunta se refiere a 'el paper' o 'el documento', usa la herramienta `summarize_paper` con esa ruta."
            )

        with st.spinner("Procesando tu solicitud..."):
            response = agent_executor.invoke(input_for_agent)
            st.markdown(response["output"])
        
    st.session_state.messages.append({"role": "assistant", "content": response["output"]})

    if 'uploaded_file_path' in st.session_state and st.session_state.uploaded_file_path:
        if os.path.exists(st.session_state.uploaded_file_path):
            os.remove(st.session_state.uploaded_file_path)
        del st.session_state.uploaded_file_path
