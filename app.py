import streamlit as st
import io
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- 1. Definición del Rol y Configuración Inicial ---
RAG_PROMPT_TEMPLATE = """
[INICIO DE LA DEFINICIÓN DEL ROL]
**Nombre del Rol:** Investigador Doctoral IA (IDA)
**Personalidad:** Eres un asistente de investigación post-doctoral; preciso, metódico y objetivo. Tu propósito es responder preguntas basándote ESTRICTAMENTE en el contexto proporcionado.
**Tarea:** Analiza el siguiente contexto extraído de documentos académicos y responde la pregunta del usuario. Si la respuesta no se encuentra en el contexto, indica claramente: "La información no se encuentra en los documentos proporcionados." No inventes información. Cita tus respuestas basándote en los metadatos de los documentos.

[CONTEXTO]
{context}

[PREGUNTA]
{input}

[RESPUESTA PRECISA Y BASADA EN EL CONTEXTO]
"""

# --- Funciones de la Lógica RAG ---

def extract_documents_from_pdfs(pdf_files):
    documents = []
    for pdf_file in pdf_files:
        try:
            pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_file.getvalue()))
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    documents.append(Document(page_content=text, metadata={"source": pdf_file.name, "page": i + 1}))
        except Exception as e:
            st.error(f"Error leyendo el archivo {pdf_file.name}: {e}")
    return documents

@st.cache_resource
def get_embedding_model():
    return SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_db_and_retriever(_pdf_docs, _embedding_model):
    with st.spinner("Procesando documentos: dividiendo, vectorizando y almacenando..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(_pdf_docs)

        if not chunks:
            st.warning("No se pudo extraer texto de los documentos o los documentos están vacíos.")
            return None
        
        try:
            vector_db = FAISS.from_documents(chunks, _embedding_model)
            return vector_db.as_retriever()
        except Exception as e:
            st.error(f"Ocurrió un error inesperado al crear la base de datos vectorial. Detalle: {e}")
            return None

# --- 2. Interfaz de Streamlit ---
st.set_page_config(layout="wide", page_title="Asistente de Tesis con RAG")
st.title("🎓 Asistente de Tesis Doctoral (con RAG)")
st.markdown("Chatea con tu bibliografía. Sube tus papers en PDF y haz preguntas sobre su contenido.")

with st.sidebar:
    st.header("Configuración")
    api_key_value = st.secrets.get("HF_API_KEY", "")
    hf_api_key_input = st.text_input(
        "Hugging Face API Key", 
        type="password", 
        value=api_key_value,
        help="Necesaria para el modelo de lenguaje (LLM)."
    )
    st.sidebar.subheader("Parámetros del Modelo LLM")
    model_reasoning = st.sidebar.selectbox(
        "Selección de Modelo LLM",
        ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-8B-Instruct", "google/gemma-7b-it"]
    )
    temp_slider = st.sidebar.slider(
        "Temperatura", min_value=0.01, max_value=1.0, value=0.1, step=0.01,
        help="Valores bajos = respuestas más factuales. Se recomienda < 0.5 para RAG."
    )
    
    st.header("Base de Conocimiento")
    uploaded_files = st.file_uploader(
        "Sube tus archivos PDF aquí", 
        type="pdf", 
        accept_multiple_files=True
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 3. Lógica Principal de la App ---
retriever = None
if uploaded_files:
    docs = extract_documents_from_pdfs(uploaded_files)
    if docs:
        embedding_model = get_embedding_model()
        retriever = create_vector_db_and_retriever(docs, embedding_model)

if retriever:
    st.success(f"¡Base de conocimiento creada con {len(uploaded_files)} documento(s)! Lista para recibir preguntas.")
    
    llm = None
    if not hf_api_key_input or not hf_api_key_input.startswith("hf_"):
        st.warning("Por favor, introduce una Hugging Face API Key válida en la barra lateral para poder chatear.")
    else:
        try:
            # <-- EL CAMBIO ESTÁ AQUÍ -->
            llm = HuggingFaceHub(
                repo_id=model_reasoning,
                task="text-generation",
                huggingfacehub_api_token=hf_api_key_input,
                model_kwargs={
                    "temperature": temp_slider, 
                    "max_new_tokens": 1024,
                    "top_p": 0.95,
                    "repetition_penalty": 1.03,
                }
            )
        except Exception as e:
            st.error(f"No se pudo inicializar el modelo LLM. Revisa la API Key y la selección de modelo. Error: {e}")

    if llm:
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_query := st.chat_input("Haz una pregunta sobre tus documentos..."):
            st.chat_message("user").markdown(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            with st.spinner("Buscando en los documentos y generando respuesta..."):
                try:
                    response = rag_chain.invoke({"input": user_query})
                    answer = response["answer"]
                    
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        with st.expander("Ver fuentes consultadas"):
                            for doc in response["context"]:
                                st.write(f"**Fuente:** {doc.metadata.get('source', 'N/A')}, **Página:** {doc.metadata.get('page', 'N/A')}")
                                st.caption(doc.page_content)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Ocurrió un error al generar la respuesta: {e}")
                    st.exception(e)

else:
    st.info("Por favor, sube uno o más archivos PDF para comenzar.")
