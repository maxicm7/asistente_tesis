# Reemplaza todo tu bloque de importaciones de LangChain con este
import streamlit as st
import io
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.llms import HuggingFaceEndpoint  # <-- CORREGIDO
from langchain_huggingface.embeddings import HuggingFaceInferenceAPIEmbeddings # <-- CORREGIDO
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- 1. Definición del Rol y Configuración Inicial ---
# Este prompt ahora es un TEMPLATE que recibirá el contexto recuperado
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

# Función para extraer texto de los PDFs subidos. Ahora devuelve objetos Document de LangChain.
def extract_documents_from_pdfs(pdf_files):
    documents = []
    for pdf_file in pdf_files:
        try:
            pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_file.getvalue()))
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    # Creamos un objeto Document por cada página para mantener la referencia
                    documents.append(Document(page_content=text, metadata={"source": pdf_file.name, "page": i + 1}))
        except Exception as e:
            st.error(f"Error leyendo el archivo {pdf_file.name}: {e}")
    return documents

# Función para crear la base de datos vectorial (Retriever)
# @st.cache_resource es clave para la eficiencia. No se re-ejecuta si los inputs no cambian.
@st.cache_resource
def create_vector_db_and_retriever(_pdf_docs, api_key):
    if not api_key or not api_key.startswith("hf_"):
        st.error("Se necesita una Hugging Face API Key válida para crear los embeddings.")
        return None

    with st.spinner("Procesando documentos: dividiendo, vectorizando y almacenando..."):
        # 1. Dividir los documentos en fragmentos (chunks)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(_pdf_docs)

        if not chunks:
            st.warning("No se pudo extraer texto de los documentos o los documentos están vacíos.")
            return None

        # 2. Crear los embeddings (vectores) para cada fragmento
        try:
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=api_key,
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            # 3. Crear la base de datos vectorial con FAISS y almacenar los chunks
            vector_db = FAISS.from_documents(chunks, embeddings)

            # 4. Crear el retriever, que es la interfaz para buscar en la base de datos
            retriever = vector_db.as_retriever()
            return retriever
        except Exception as e:
            st.error(f"Error al crear la base de datos vectorial. ¿La API Key tiene permisos? Detalle: {e}")
            return None


# --- 2. Interfaz de Streamlit ---
st.set_page_config(layout="wide", page_title="Asistente de Tesis con RAG")
st.title("🎓 Asistente de Tesis Doctoral (con RAG)")
st.markdown("Chatea con tu bibliografía. Sube tus papers en PDF y haz preguntas sobre su contenido.")

# --- Configuración en la barra lateral ---
with st.sidebar:
    st.header("Configuración")
    api_key_value = st.secrets.get("HF_API_KEY", "")
    hf_api_key_input = st.text_input(
        "Hugging Face API Key", 
        type="password", 
        value=api_key_value,
        help="Necesaria para los embeddings y el modelo de lenguaje."
    )
    st.sidebar.subheader("Parámetros del Modelo")
    model_reasoning = st.sidebar.selectbox(
        "Selección de Modelo LLM",
        ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-8B-Instruct"]
    )
    temp_slider = st.sidebar.slider(
        "Temperatura", min_value=0.0, max_value=1.0, value=0.1, step=0.1,
        help="Valores bajos = respuestas más factuales y basadas en el texto. Se recomienda < 0.5 para RAG."
    )
    
    st.header("Base de Conocimiento")
    uploaded_files = st.file_uploader(
        "Sube tus archivos PDF aquí", 
        type="pdf", 
        accept_multiple_files=True
    )

# Inicializar el estado de la sesión
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 3. Lógica Principal de la App ---

# Solo procesamos si hay archivos subidos
if uploaded_files:
    # Extraer los documentos de los PDFs
    docs = extract_documents_from_pdfs(uploaded_files)
    if docs:
        # Crear el retriever (la base de conocimiento)
        # La función está cacheada, así que solo se ejecuta si los archivos cambian
        st.session_state.retriever = create_vector_db_and_retriever(docs, hf_api_key_input)

if st.session_state.retriever:
    st.success(f"¡Base de conocimiento creada con {len(uploaded_files)} documento(s)! Lista para recibir preguntas.")

    # Inicializar el LLM
    try:
        llm = HuggingFaceEndpoint(
            repo_id=model_reasoning,
            max_length=2048,
            temperature=temp_slider,
            huggingfacehub_api_token=hf_api_key_input
        )
    except Exception as e:
        st.error(f"No se pudo inicializar el modelo LLM. Revisa la API Key y la selección de modelo. Error: {e}")
        llm = None

    if llm:
        # Crear la cadena de RAG con LangChain
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)

        # Mostrar historial del chat
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input del usuario
        if user_query := st.chat_input("Haz una pregunta sobre tus documentos..."):
            st.chat_message("user").markdown(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            with st.spinner("Buscando en los documentos y generando respuesta..."):
                try:
                    response = rag_chain.invoke({"input": user_query})
                    answer = response["answer"]
                    
                    # Añadir la respuesta del asistente al historial y mostrarla
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        # Mostrar las fuentes utilizadas para la respuesta
                        with st.expander("Ver fuentes consultadas"):
                            for doc in response["context"]:
                                st.write(f"**Fuente:** {doc.metadata.get('source', 'N/A')}, **Página:** {doc.metadata.get('page', 'N/A')}")
                                st.caption(doc.page_content)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Ocurrió un error al generar la respuesta: {e}")
else:
    st.info("Por favor, sube uno o más archivos PDF para comenzar.")
