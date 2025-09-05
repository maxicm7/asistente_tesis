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
    """Busca en la web información actualizada."""
    try:
        search = TavilySearchAPIWrapper()
        return search.run(query)
    except Exception as e:
        return f"Error en la búsqueda web: {e}"

@tool
def summarize_paper(pdf_path: str) -> str:
    """Carga y resume un artículo de investigación en formato PDF."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        full_text = " ".join([page.page_content for page in pages])
        summarizer_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
        prompt_template = f"Basado en este texto de un paper, crea un resumen conciso (aprox. 300 palabras) enfocándote en: Problema, Metodología, Hallazgos y Conclusiones.\n\nTexto:\n{full_text[:25000]}"
        summary = summarizer_llm.invoke(prompt_template).content
        return summary
    except Exception as e:
        return f"Error al procesar el PDF: {e}"

tools = [web_search, summarize_paper]

# ==============================================================================
# 4. CONFIGURACIÓN DEL AGENTE Y LA MEMORIA
# ==============================================================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente de investigación de doctorado. Tu misión es ayudar al usuario a avanzar en su tesis. Usa tus herramientas. Responde de forma rigurosa y académica."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ==============================================================================
# 5. INTERFAZ DE USUARIO Y LÓGICA PRINCIPAL
# ==============================================================================
st.set_page_config(page_title="Asistente de Tesis IA", layout="wide")
st.title("🤖 Asistente de Tesis IA")

with st.sidebar:
    st.header("Configuración")
    model_choice = st.selectbox(
        "Elige tu modelo:",
        ("Gemini 1.5 Flash (Rápido)", "Gemini 1.5 Pro (Potente)")
    )
    temperature = st.slider("Temperatura (creatividad):", 0.0, 1.0, 0.4, 0.1)
    
    uploaded_file = st.file_uploader("Sube un paper (PDF)", type="pdf")
    if uploaded_file:
        temp_dir = "temp_pdf"
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f: f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_file_path = temp_file_path
        st.success(f"Archivo '{uploaded_file.name}' cargado.")

# --- Lógica de Selección de Modelo ---
model_name = "gemini-1.5-flash-latest" if model_choice == "Gemini 1.5 Flash (Rápido)" else "gemini-1.5-pro-latest"
llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, convert_system_message_to_human=True)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)

# --- Lógica del Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hola, soy tu asistente. ¿Cómo te ayudo?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

async def get_agent_response(agent_executor, input_dict):
    return await agent_executor.ainvoke(input_dict)

if user_prompt := st.chat_input("Pregunta sobre papers, datos, modelos..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"): st.markdown(user_prompt)

    with st.chat_message("assistant"):
        input_for_agent = {"input": user_prompt}
        if 'uploaded_file_path' in st.session_state and st.session_state.uploaded_file_path:
            input_for_agent["input"] += f"\n\n[Contexto] El usuario ha subido el archivo: '{st.session_state.uploaded_file_path}'."

        with st.spinner("Procesando..."):
            response = asyncio.run(get_agent_response(agent_executor, input_for_agent))
            st.markdown(response["output"])
        
    st.session_state.messages.append({"role": "assistant", "content": response["output"]})

    if 'uploaded_file_path' in st.session_state and st.session_state.uploaded_file_path:
        if os.path.exists(st.session_state.uploaded_file_path): os.remove(st.session_state.uploaded_file_path)
        del st.session_state.uploaded_file_path
