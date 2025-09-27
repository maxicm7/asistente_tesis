import streamlit as st
from huggingface_hub import InferenceClient
import pypdf # Para leer los PDFs
import io

# --- 1. Definición del Rol y Configuración Inicial ---
MASTER_PROMPT = """
[INICIO DE LA DEFINICIÓN DEL ROL]
**Nombre del Rol:** Investigador Doctoral IA (IDA)
**Objetivo Principal:** Asistir en la investigación y redacción de una tesis doctoral.
**Personalidad:** Eres un asistente de investigación post-doctoral; preciso, metódico y objetivo.
**Áreas de Especialización:**
1. **Analista de Literatura Académica:** Resume papers identificando pregunta de investigación, metodología, resultados, contribución y limitaciones.
2. **Razonador Económico-Matemático:** Explica conceptos, desarrolla derivaciones matemáticas paso a paso e interpreta modelos.
**Instrucciones de Interacción:** Identifica la tarea, aplica el formato de salida correcto y prioriza la integridad académica.
[FIN DE LA DEFINICIÓN DEL ROL]
"""

# --- Función para Extraer Texto de un PDF ---
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_file.getvalue()))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error al leer el archivo PDF: {e}")
        return None

# --- Función para llamar a la API de Hugging Face ---
# <-- ESTA ES LA VERSIÓN CORREGIDA Y MÁS DETALLADA QUE PREFERÍAS -->
def get_hf_response(api_key, model, prompt):
    if not api_key or not api_key.startswith("hf_"):
        st.error("Por favor, introduce una Hugging Face API Key válida en la barra lateral.")
        return None
    try:
        client = InferenceClient(token=api_key)
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages, model=model, max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al contactar la API de Hugging Face. Detalles:")
        st.info(
            "Esto puede ocurrir por varias razones:\n"
            "1. No tienes acceso a este modelo (visita su página en HF para solicitarlo).\n"
            "2. El modelo está tardando en cargar en los servidores de Hugging Face. Por favor, espera un minuto y vuelve a intentarlo.\n"
            "3. La API Key es incorrecta o no tiene los permisos necesarios."
        )
        # Imprime el error completo en la consola del servidor para depuración
        print(f"Detalle del error: {e}") 
        return None


# --- 2. Interfaz de Streamlit ---
st.set_page_config(layout="wide", page_title="Asistente de Tesis Doctoral IA")
st.title("🎓 Asistente de Tesis Doctoral IA")

# --- Configuración en la barra lateral ---
with st.sidebar:
    st.header("Configuración")
    api_key_value = st.secrets.get("HF_API_KEY", "")
    hf_api_key_input = st.text_input(
        "Hugging Face API Key", 
        type="password", 
        value=api_key_value
    )
    st.sidebar.subheader("Selección de Modelo")
    model_reasoning = st.sidebar.selectbox(
        "Modelo para Resumen y Razonamiento",
        ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-8B-Instruct"]
    )

# --- 3. Lógica Principal con Dos Pestañas ---
tab1, tab2 = st.tabs(["📄 Resumir Paper", "🧠 Razonamiento Económico/Matemático"])

# --- Pestaña 1: Resumir Paper ---
with tab1:
    st.header("Analista de Literatura Académica")
    st.markdown("Pega el texto del paper en el área de abajo **o** sube el archivo PDF.")
    
    uploaded_file = st.file_uploader("Sube un archivo PDF:", type="pdf")
    paper_text = st.text_area("Pega aquí el texto:", height=200)

    if st.button("Generar Resumen", key="summarize"):
        text_to_summarize = ""
        if uploaded_file is not None:
            with st.spinner("Extrayendo texto del PDF..."):
                text_to_summarize = extract_text_from_pdf(uploaded_file)
                if text_to_summarize:
                    st.info(f"PDF procesado. Se extrajeron {len(text_to_summarize)} caracteres.")
        elif paper_text.strip():
            text_to_summarize = paper_text
        
        if text_to_summarize:
            with st.spinner("Analizando y generando el resumen..."):
                final_prompt = f"{MASTER_PROMPT}\n\n**Tarea Actual:** Resumir el siguiente texto académico de manera detallada, identificando la pregunta de investigación, metodología, resultados clave, y contribución.\n\n**Texto a Analizar:**\n```\n{text_to_summarize}\n```\n\n**Análisis Detallado:**"
                summary = get_hf_response(hf_api_key_input, model_reasoning, final_prompt)
                if summary:
                    st.markdown(summary)
        else:
            st.warning("Por favor, sube un archivo PDF o pega texto en el área designada.")


# --- Pestaña 2: Razonamiento ---
with tab2:
    st.header("Razonador Económico-Matemático")
    question_text = st.text_area("Escribe tu pregunta o el problema a resolver:", height=200)
    if st.button("Obtener Razonamiento", key="reason"):
        if question_text:
            with st.spinner("Procesando..."):
                final_prompt = f"{MASTER_PROMPT}\n\n**Tarea Actual:** Responder a una pregunta de economía/matemáticas.\n\n**Pregunta:**\n```\n{question_text}\n```\n\n**Respuesta Detallada:**"
                reasoning = get_hf_response(hf_api_key_input, model_reasoning, final_prompt)
                if reasoning:
                    st.markdown(reasoning)
        else:
            st.warning("Por favor, introduce una pregunta.")
