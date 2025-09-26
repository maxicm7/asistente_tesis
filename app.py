import streamlit as st
from huggingface_hub import InferenceClient

# --- Configuración de la página ---
st.set_page_config(layout="wide", page_title="Asistente de Tesis Doctoral IA")

# --- Barra Lateral de Configuración ---
with st.sidebar:
    st.header("Configuración")

    # Input para la API Key
    hf_api_key_input = st.text_input(
        "Hugging Face API Key", 
        type="password", 
        value=st.secrets.get("HF_API_KEY", "")
    )

    st.subheader("Selección de Modelos")
    
    # ----> AQUÍ ESTÁ LA CLAVE <----
    # Dropdown 1: Para tareas de lenguaje natural (Resumen, Razonamiento)
    model_reasoning = st.selectbox(
        "Modelo para Resumen y Razonamiento",
        # Opciones potentes para razonamiento
        ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-8B-Instruct"],
        help="Elige un modelo generalista fuerte para analizar texto y responder preguntas."
    )
    
    # Dropdown 2: Para tareas de CÓDIGO.
    model_coding = st.selectbox(
        "Modelo para Código (CODEQwen)",
        # Opciones especializadas en CÓDIGO. Qwen es excelente.
        ["Qwen/CodeQwen1.5-7B-Chat", "codellama/CodeLlama-34b-Instruct-hf"],
        help="Elige un modelo especializado en programación para obtener los mejores resultados."
    )

# --- Función para llamar a la API de Hugging Face ---
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
        print(f"Detalle del error: {e}")
        return None

# --- Estructura de Pestañas ---
tab1, tab2= st.tabs(["📄 Resumir Paper", "🧠 Razonamiento Económico/Matemático"])

# Pestaña 1 y 2 usan el modelo de RAZONAMIENTO
with tab1:
    st.header("Analista de Literatura Académica")
    paper_text = st.text_area("Pega aquí el abstract:", height=250, key="paper_text")
    if st.button("Generar Resumen", key="summarize"):
        if paper_text:
            final_prompt = f"Resume el siguiente texto académico:\n\n{paper_text}"
            # ----> USA EL MODELO DE RAZONAMIENTO
            summary = get_hf_response(hf_api_key_input, model_reasoning, final_prompt)
            if summary: st.markdown(summary)

with tab2:
    st.header("Razonador Económico-Matemático")
    question_text = st.text_area("Escribe tu pregunta:", height=200, key="question_text")
    if st.button("Obtener Razonamiento", key="reason"):
        if question_text:
            final_prompt = f"Responde la siguiente pregunta:\n\n{question_text}"
            # ----> USA EL MODELO DE RAZONAMIENTO
            reasoning = get_hf_response(hf_api_key_input, model_reasoning, final_prompt)
            if reasoning: st.markdown(reasoning)


