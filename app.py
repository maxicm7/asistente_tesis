# ==============================================================================
# 1. IMPORTACIONES Y CONFIGURACIÓN INICIAL
# ==============================================================================
import streamlit as st
import os
from dotenv import load_dotenv
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

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración y validación de API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Configuración explícita de la API de Google (mejor práctica)
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ==============================================================================
# 2. DEFINICIÓN DE HERRAMIENTAS (HABILIDADES DEL AGENTE)
# ==============================================================================

@tool
def web_search(query: str) -> str:
    """Busca en la web información actualizada, incluyendo fuentes de datos para investigación académica."""
    try:
        search = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)
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

# Verificar si todas las claves están presentes antes de continuar
if not all([GOOGLE_API_KEY, HUGGINGFACEHUB_API_TOKEN, TAVILY_API_KEY]):
    st.error("Por favor, asegúrate de que todas las claves de API (Google, Hugging Face, Tavily) están en tu archivo .env")
    st.stop()

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
        
        # Usar un nombre de archivo único para evitar colisiones
        temp_file_path = os.path.join(temp_dir, f"uploaded_{uploaded_file.id}.pdf")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.uploaded_file_path = temp_file_path
        st.success(f"Archivo '{uploaded_file.name}' cargado y listo para analizar.")

# --- Lógica de Selección de Modelo y Construcción de Agente ---
if model_choice == "Google Gemini-1.5-flash":
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=temperature)
    agent = create_tool_calling_agent(llm, tools, tool_calling_prompt)
else:
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature=temperature)
    agent = create_react_agent(llm, tools, react_prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=st.session_state.memory, 
    verbose=True, 
    handle_parsing_errors=True
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
    if 'uploaded_file_path' in st.session_state:
        if os.path.exists(st.session_state.uploaded_file_path):
            os.remove(st.session_state.uploaded_file_path)
        del st.session_state.uploaded_file_path
