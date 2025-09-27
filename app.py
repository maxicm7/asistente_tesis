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
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_file.getvalue()))
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    except Exception as e:
        st.error(f"Error al leer el archivo PDF: {e}")
        return None

# --- Función para llamar a la API de Hugging Face ---
# <-- AÑADIMOS EL PARÁMETRO 'temperature' -->
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
            max_tokens=4096,
            temperature=temperature # <-- AQUÍ USAMOS EL PARÁMETRO -->
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
    st.sidebar.subheader("Parámetros del Modelo")
    model_reasoning = st.sidebar.selectbox(
        "Selección de Modelo",
        ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-8B-Instruct"]
    )
    # <-- NUEVO: Slider para controlar la temperatura -->
    temp_slider = st.sidebar.slider(
        "Temperatura",
        min_value=0.1,
        max_value=1.0,
        value=0.6, # Un buen valor por defecto
        step=0.1,
        help="Valores bajos = respuestas más predecibles y factuales. Valores altos = respuestas más creativas."
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
                final_prompt = f"{MASTER_PROMPT}\n\n**Tarea:** Resumir el siguiente texto académico de manera detallada.\n\n**Texto:**\n{text_to_summarize}"
                # <-- PASAMOS EL VALOR DEL SLIDER A LA FUNCIÓN -->
                summary = get_hf_response(hf_api_key_input, model_reasoning, final_prompt, temp_slider)
                if summary:
                    st.markdown(summary)
        else:
            st.warning("Por favor, sube un archivo PDF o pega texto.")


# --- Pestaña 2: Razonamiento ---
with tab2:
    st.header("Razonador Económico-Matemático")
    question_text = st.text_area("Escribe tu pregunta o el problema a resolver:", height=200)
    if st.button("Obtener Razonamiento", key="reason"):
        if question_text:
            with st.spinner("Procesando..."):
                final_prompt = f"{MASTER_PROMPT}\n\n**Tarea:** Responder la siguiente pregunta.\n\n**Pregunta:**\n{question_text}"
                # <-- PASAMOS EL VALOR DEL SLIDER A LA FUNCIÓN -->
                reasoning = get_hf_response(hf_api_key_input, model_reasoning, final_prompt, temp_slider)
                if reasoning:
                    st.markdown(reasoning)
        else:
            st.warning("Por favor, introduce una pregunta.")
