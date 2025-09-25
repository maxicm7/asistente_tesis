# ==============================================================================
# 1. IMPORTACIONES Y CONFIGURACIÓN INICIAL
# ==============================================================================
import streamlit as st
import os
import google.generativeai as genai
import nest_asyncio
import asyncio

# Aplicar parche para compatibilidad asíncrona
nest_asyncio.apply()

# --- Importaciones de LangChain ---
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# ==============================================================================
# 2. CARGA DE SECRETS Y VARIABLES DE ENTORNO
# ==============================================================================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
except KeyError as e:
    st.error(f"Error: No se encontró el secreto '{e.args[0]}'. Revisa la configuración de tu app.")
    st.stop()

# ==============================================================================
# 3. INTERFAZ PRINCIPAL Y PESTAÑAS
# ==============================================================================
st.set_page_config(page_title="Asistente de Tesis Avanzado", layout="wide")
st.title("🤖 Asistente de Tesis Avanzado")

tab1, tab2, tab3 = st.tabs([
    "Investigación (Gemini)", 
    "Código Confiable (Gemma)", 
    "Código Experimental (CodeQwen)"
])

# ==============================================================================
# PESTAÑA 1: ASISTENTE DE INVESTIGACIÓN (GEMINI CON HERRAMIENTAS)
# ==============================================================================
with tab1:
    st.header("Asistente de Investigación")
    st.write("Usa este asistente para buscar información, analizar documentos PDF y responder preguntas teóricas.")

    @tool
    def web_search(query: str) -> str:
        """Busca en la web información actualizada."""
        try: return TavilySearchAPIWrapper().run(query)
        except Exception as e: return f"Error en la búsqueda web: {e}"

    @tool
    def summarize_paper(pdf_path: str) -> str:
        """Carga y resume un artículo de investigación en formato PDF."""
        try:
            loader = PyPDFLoader(pdf_path)
            full_text = " ".join([page.page_content for page in loader.load_and_split()])
            summarizer_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
            prompt_template = f"Resume este texto de un paper (aprox. 300 palabras) enfocándote en: Problema, Metodología, Hallazgos y Conclusiones.\n\nTexto:\n{full_text[:25000]}"
            return summarizer_llm.invoke(prompt_template).content
        except Exception as e: return f"Error al procesar el PDF: {e}"

    research_tools = [web_search, summarize_paper]
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente de investigación de doctorado. Usa tus herramientas. Sé riguroso y académico."),
        ("placeholder", "{chat_history}"), ("human", "{input}"), ("placeholder", "{agent_scratchpad}"),
    ])
    if 'research_memory' not in st.session_state: st.session_state.research_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    research_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5, convert_system_message_to_human=True)
    research_agent = create_tool_calling_agent(research_llm, research_tools, research_prompt)
    research_agent_executor = AgentExecutor(agent=research_agent, tools=research_tools, memory=st.session_state.research_memory, verbose=True)

    if "research_messages" not in st.session_state: st.session_state.research_messages = [{"role": "assistant", "content": "Hola, ¿en qué tema de tu investigación necesitas ayuda?"}]
    for msg in st.session_state.research_messages: st.chat_message(msg["role"]).markdown(msg["content"])

    with st.sidebar:
        st.header("Herramientas de Investigación")
        uploaded_file = st.file_uploader("Sube un paper (PDF) para analizar", type="pdf")
        if uploaded_file:
            temp_dir = "temp_pdf"
            if not os.path.exists(temp_dir): os.makedirs(temp_dir)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f: f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_file_path = temp_file_path
            st.success(f"Archivo '{uploaded_file.name}' listo para analizar en la pestaña de Investigación.")

    async def get_research_response(executor, input_dict): return await executor.ainvoke(input_dict)

    if research_prompt_input := st.chat_input("Busca un paper, resume un documento..."):
        st.session_state.research_messages.append({"role": "user", "content": research_prompt_input})
        st.chat_message("user").markdown(research_prompt_input)
        with st.chat_message("assistant"):
            input_for_agent = {"input": research_prompt_input}
            if 'uploaded_file_path' in st.session_state and st.session_state.uploaded_file_path:
                input_for_agent["input"] += f"\n\n[Contexto] El usuario ha subido el archivo: '{st.session_state.uploaded_file_path}'."
            with st.spinner("Investigando..."):
                response = asyncio.run(get_research_response(research_agent_executor, input_for_agent))
                st.markdown(response["output"])
            st.session_state.research_messages.append({"role": "assistant", "content": response["output"]})
            if 'uploaded_file_path' in st.session_state:
                if os.path.exists(st.session_state.uploaded_file_path): os.remove(st.session_state.uploaded_file_path)
                del st.session_state.uploaded_file_path

# ==============================================================================
# PESTAÑA 2: ASISTENTE DE CODIFICACIÓN CONFIABLE (GEMMA)
# ==============================================================================
with tab2:
    st.header("Asistente de Codificación Confiable (Gemma)")
    st.write("Este asistente usa `google/gemma-7b-it`, un modelo abierto y potente que siempre está disponible.")
    try:
        gemma_llm = HuggingFaceEndpoint(repo_id="google/gemma-7b-it", temperature=0.1, max_new_tokens=2048)
    except Exception as e:
        st.error(f"No se pudo inicializar el modelo Gemma: {e}")
        gemma_llm = None
    if "gemma_messages" not in st.session_state: st.session_state.gemma_messages = [{"role": "assistant", "content": "¿En qué código necesitas ayuda?"}]
    for msg in st.session_state.gemma_messages: st.chat_message(msg["role"]).markdown(msg["content"])
    if gemma_prompt_input := st.chat_input("Genera código Python para un modelo SVAR..."):
        st.session_state.gemma_messages.append({"role": "user", "content": gemma_prompt_input})
        st.chat_message("user").markdown(gemma_prompt_input)
        with st.chat_message("assistant"):
            if gemma_llm:
                with st.spinner("Gemma está programando..."):
                    response_text = gemma_llm.invoke(gemma_prompt_input)
                    st.markdown(response_text)
                st.session_state.gemma_messages.append({"role": "assistant", "content": response_text})
            else: st.error("El asistente Gemma no está disponible.")

# ==============================================================================
# PESTAÑA 3: ASISTENTE DE CODIFICACIÓN EXPERIMENTAL (CODEQWEN)
# ==============================================================================
with tab3:
    st.header("Asistente de Codificación Experimental (CodeQwen)")
    st.info("Ahora que tu cuenta está aprobada, este modelo debería funcionar. Si falla, es posible que el modelo específico no esté en la capa gratuita en este momento.")
    try:
        qwen_llm = HuggingFaceEndpoint(repo_id="Qwen/CodeQwen1.5-7B-Chat", temperature=0.1, max_new_tokens=2048)
    except Exception as e:
        st.error(f"No se pudo inicializar el modelo CodeQwen. Error: {e}")
        qwen_llm = None
    if "qwen_messages" not in st.session_state: st.session_state.qwen_messages = [{"role": "assistant", "content": "¿En qué código necesitas ayuda?"}]
    for msg in st.session_state.qwen_messages: st.chat_message(msg["role"]).markdown(msg["content"])
    if qwen_prompt_input := st.chat_input("Genera código Python para un modelo DSGE..."):
        st.session_state.qwen_messages.append({"role": "user", "content": qwen_prompt_input})
        st.chat_message("user").markdown(qwen_prompt_input)
        with st.chat_message("assistant"):
            if qwen_llm:
                with st.spinner("CodeQwen está programando..."):
                    response_text = qwen_llm.invoke(qwen_prompt_input)
                    st.markdown(response_text)
                st.session_state.qwen_messages.append({"role": "assistant", "content": response_text})
            else: st.error("El asistente CodeQwen no está disponible en este momento. Por favor, usa la pestaña de Gemma.")
