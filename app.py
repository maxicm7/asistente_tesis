# ==============================================================================
# 1. IMPORTACIONES Y CONFIGURACIÓN INICIAL
# ==============================================================================
import streamlit as st
import os
import google.generativeai as genai
import asyncio

# --- Importaciones de LangChain ---
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# ==============================================================================
# 2. CARGA DE SECRETS Y VARIABLES DE ENTORNO
# ==============================================================================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
except KeyError as e:
    st.error(f"Error: No se encontró el secreto '{e.args[0]}'. Revisa la configuración de tu app.")
    st.stop()

# ==============================================================================
# 3. DEFINICIÓN DE HERRAMIENTAS
# ==============================================================================
@tool
def web_search(query: str) -> str:
    """Busca en la web información actualizada, papers o documentación de código."""
    try:
        return TavilySearchAPIWrapper().run(query)
    except Exception as e:
        return f"Error en la búsqueda web: {e}"

@tool
def summarize_paper(pdf_path: str) -> str:
    """Carga y resume un artículo de investigación en formato PDF."""
    try:
        loader = PyPDFLoader(pdf_path)
        full_text = " ".join([page.page_content for page in loader.load_and_split()])
        # Usamos el modelo estable también para resumir
        summarizer_llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
        prompt_template = f"Resume este texto de un paper (aprox. 300 palabras) enfocándote en: Problema, Metodología, Hallazgos y Conclusiones.\n\nTexto:\n{full_text[:25000]}"
        return summarizer_llm.invoke(prompt_template).content
    except Exception as e:
        return f"Error al procesar el PDF: {e}"

tools = [web_search, summarize_paper]

# ==============================================================================
# 4. CONFIGURACIÓN DEL AGENTE Y LA MEMORIA
# ==============================================================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente de doctorado experto tanto en investigación econométrica como en codificación. Tu misión es ayudar al usuario con su tesis. Usa tus herramientas. Genera código directamente cuando se te pida."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(key="chat_history", return_messages=True)

# ==============================================================================
# 5. INTERFAZ DE USUARIO Y LÓGICA PRINCIPAL
# ==============================================================================
st.set_page_config(page_title="Asistente de Tesis IA", layout="wide")
st.title("🤖 Asistente de Tesis IA (Estable)")
st.info("Este asistente utiliza el modelo `gemini-pro` de Google, garantizado para funcionar sin necesidad de aprobaciones.")

with st.sidebar:
    st.header("Configuración")
    temperature = st.slider("Temperatura (creatividad):", 0.0, 1.0, 0.4, 0.1)
    
    uploaded_file = st.file_uploader("Sube un paper (PDF) para analizar", type="pdf")
    if uploaded_file:
        temp_dir = "temp_pdf"
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f: f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_file_path = temp_file_path
        st.success(f"Archivo '{uploaded_file.name}' cargado.")

# --- Lógica de Creación del Agente (simplificada) ---
# Usamos el modelo 'gemini-pro', que es universalmente accesible.
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature, convert_system_message_to_human=True)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)

# --- Lógica del Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hola, soy tu asistente. Pídeme que busque papers, resuma documentos o genere código."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

async def get_agent_response(executor, input_dict):
    # La llamada asíncrona es más robusta en el entorno de Streamlit
    return await executor.ainvoke(input_dict)

if user_prompt := st.chat_input("Busca, resume, o pide código..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"): st.markdown(user_prompt)

    with st.chat_message("assistant"):
        input_for_agent = {"input": user_prompt}
        if 'uploaded_file_path' in st.session_state and st.session_state.uploaded_file_path:
            input_for_agent["input"] += f"\n\n[Contexto] El usuario ha subido el archivo: '{st.session_state.uploaded_file_path}'."

        with st.spinner("Procesando..."):
            try:
                # Usamos asyncio.run para ejecutar nuestra función asíncrona
                response = asyncio.run(get_agent_response(agent_executor, input_for_agent))
                output = response["output"]
            except Exception as e:
                st.error(f"Ha ocurrido un error al contactar con la API de Google. Por favor, inténtalo de nuevo.\n\nDetalles: {e}")
                output = "No pude procesar tu solicitud."

        st.markdown(output)
    st.session_state.messages.append({"role": "assistant", "content": output})

    if 'uploaded_file_path' in st.session_state and st.session_state.uploaded_file_path:
        if os.path.exists(st.session_state.uploaded_file_path): os.remove(st.session_state.uploaded_file_path)
        del st.session_state.uploaded_file_path
