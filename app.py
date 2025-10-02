# --- app.py (Versión Original sin RAG) ---

import streamlit as st
from huggingface_hub import InferenceClient
import pypdf
import io

# --- 1. Definición del Rol y Configuración Inicial ---
MASTER_PROMPT = """
[INICIO DE LA DEFINICIÓN DEL ROL]
**Nombre del Rol:** Investigador Doctoral IA (IDA)
**Objetivo Principal:** Asistir en la investigación y redacción de una tesis doctoral.
**Personalidad:** Eres un asistente de investigación post-doctoral; preciso, metódico y objetivo.
[FIN DE LA DEFINICIÓN DEL ROL]
"""

# --- Función para Extraer Texto de un PDF ---
def extract_text_from_pdf(pdf_file):
    try:
        # Usamos io.BytesIO para leer el archivo en memoria
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_file.getvalue()))
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    except Exception as e:
        st.error(f"Error al leer el archivo PDF: {e}")
        return None

# --- Función para llamar a la API de Hugging Face ---
def get_hf_response(api_key, model, prompt, temperature):
    if not api_key or not api_key.startswith("hf_"):
        st.error("Por favor, introduce una Hugging Face API Key válida en la barra lateral.")
        return None
    try:
        client = InferenceClient(token=api_key)
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages, 
            model=model, 
            max_tokens=4096, # Límite de tokens para la respuesta
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Ocurrió un error al contactar la API de Hugging Face.")
        st.info(
            "Posibles causas:\n"
            "1. No tienes acceso a este modelo (visita su página en HF para solicitarlo).\n"
            "2. El modelo está tardando en cargar en los servidores de HF. Espera un minuto y vuelve a intentarlo.\n"
            "3. La API Key es incorrecta o no tiene los permisos necesarios."
        )
        st.exception(e) # Muestra el detalle completo del error para depuración
        return None


# --- 2. Interfaz de Streamlit ---
st.set_page_config(layout="wide", page_title="Asistente de Tesis Doctoral IA")
st.title("🎓 Asistente de Tesis Doctoral IA")

# --- Configuración en la barra lateral ---
with st.sidebar:
    st.header("Configuración")
    # Es mejor usar st.secrets para la API key en producción
    api_key_value = st.secrets.get("HF_API_KEY", "")
    hf_api_key_input = st.text_input(
        "Hugging Face API Key", 
        type="password", 
        value=api_key_value
    )
    st.sidebar.subheader("Parámetros del Modelo")
    model_reasoning = st.sidebar.selectbox(
        "Selección de Modelo",
        ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-8B-Instruct", "google/gemma-7b-it"]
    )
    temp_slider = st.sidebar.slider(
        "Temperatura",
        min_value=0.1,
        max_value=1.0,
        value=0.6,
        step=0.1,
        help="Valores bajos = respuestas más predecibles. Valores altos = respuestas más creativas."
    )


# --- 3. Lógica Principal con Dos Pestañas ---
tab1, tab2 = st.tabs(["📄 Resumir Paper", "🧠 Razonamiento General"])

# --- Pestaña 1: Resumir Paper ---
with tab1:
    st.header("Analista de Literatura Académica")
    st.markdown("Sube el archivo PDF **o** pega el texto del paper en el área de abajo.")
    
    uploaded_file = st.file_uploader("Sube un archivo PDF:", type="pdf")
    paper_text = st.text_area("Pega aquí el texto:", height=200, key="paper_text_area")

    if st.button("Generar Resumen", key="summarize_button"):
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
                final_prompt = f"{MASTER_PROMPT}\n\n**Tarea:** Resumir el siguiente texto académico de manera detallada y estructurada.\n\n**Texto:**\n{text_to_summarize}"
                summary = get_hf_response(hf_api_key_input, model_reasoning, final_prompt, temp_slider)
                if summary:
                    st.markdown("### Resumen Generado")
                    st.write(summary)
        else:
            st.warning("Por favor, sube un archivo PDF o pega texto para resumir.")


# --- Pestaña 2: Razonamiento ---
with tab2:
    st.header("Asistente de Razonamiento")
    question_text = st.text_area("Escribe tu pregunta o el problema a resolver:", height=200, key="question_text_area")
    if st.button("Obtener Razonamiento", key="reason_button"):
        if question_text.strip():
            with st.spinner("Procesando tu pregunta..."):
                final_prompt = f"{MASTER_PROMPT}\n\n**Tarea:** Responder la siguiente pregunta de investigación de forma precisa y objetiva.\n\n**Pregunta:**\n{question_text}"
                reasoning = get_hf_response(hf_api_key_input, model_reasoning, final_prompt, temp_slider)
                if reasoning:
                    st.markdown("### Respuesta del Asistente")
                    st.write(reasoning)
        else:
            st.warning("Por favor, introduce una pregunta.")
