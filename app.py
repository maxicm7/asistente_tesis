import streamlit as st
from huggingface_hub import InferenceClient

# --- Configuración de la página ---
# El estado inicial de la barra lateral se puede configurar aquí
st.set_page_config(
    layout="wide", 
    page_title="Asistente de Tesis Doctoral IA",
    initial_sidebar_state="expanded" # 'auto', 'expanded', or 'collapsed'
)

# --- Definición del Rol (Master Prompt) ---
MASTER_PROMPT = """
[INICIO DE LA DEFINICIÓN DEL ROL]
**Nombre del Rol:** Investigador Doctoral IA (IDA)
**Objetivo Principal:** Asistir en la investigación y redacción de una tesis doctoral en el campo de la Economía Aplicada. Tu función es ser un colaborador riguroso, analítico y eficiente.
[FIN DE LA DEFINICIÓN DEL ROL]
"""

# --- Interfaz de la Barra Lateral (Sidebar) ---
with st.sidebar:
    st.header("Configuración")

    # Intenta obtener la API key desde los secretos de Streamlit, si no, deja el campo vacío.
    try:
        HF_API_KEY_FROM_SECRETS = st.secrets["HF_API_KEY"]
    except (FileNotFoundError, KeyError):
        HF_API_KEY_FROM_SECRETS = ""

    hf_api_key_input = st.text_input(
        "Hugging Face API Key", 
        type="password", 
        value=HF_API_KEY_FROM_SECRETS,
        help="Introduce tu API Key de Hugging Face. Es necesaria para usar los modelos."
    )

    st.subheader("Selección de Modelos")
    model_reasoning = st.selectbox(
        "Modelo para Resumen y Razonamiento",
        ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-8B-Instruct"],
        key="model_reasoning_select"
    )
    model_coding = st.selectbox(
        "Modelo para Código (CODEQwen)",
        ["Qwen/CodeQwen1.5-7B-Chat", "codellama/CodeLlama-34b-Instruct-hf"],
        key="model_coding_select"
    )


# --- Lógica de la API de Hugging Face ---
def get_hf_response(api_key, model, prompt):
    """
    Función para llamar a la API de Inferencia de Hugging Face.
    Incluye el manejo de errores que se muestra en la captura de pantalla.
    """
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
        )
        return response.choices[0].message.content
    
    # Esta sección es crucial para replicar el error que se muestra en la imagen
    except Exception as e:
        st.error("Error al contactar la API de Hugging Face. Detalles:")
        st.info(
            "Esto puede ocurrir por varias razones:\n"
            "1. No tienes acceso a este modelo (visita su página en HF para solicitarlo).\n"
            "2. El modelo está tardando en cargar en los servidores de Hugging Face. Por favor, espera un minuto y vuelve a intentarlo.\n"
            "3. La API Key es incorrecta o no tiene los permisos necesarios."
        )
        # Opcional: imprimir el error real en la consola del servidor para depuración
        print(f"Detalle completo del error: {e}") 
        return None


# --- Estructura Principal de la Aplicación ---
# No se necesita el título principal aquí, ya que cada pestaña tiene su propio encabezado.

# Definición de las pestañas
tab1, tab2, tab3 = st.tabs(["📄 Resumir Paper", "🧠 Razonamiento Económico/Matemático", "💻 Generar Código"])

# Pestaña 1: Resumir Paper
with tab1:
    st.header("Analista de Literatura Académica")
    paper_text = st.text_area("Pega aquí el abstract o el texto completo del paper:", height=250, key="paper_text")
    if st.button("Generar Resumen", key="summarize"):
        if paper_text:
            with st.spinner("Analizando..."):
                final_prompt = f"{MASTER_PROMPT}\n**Tarea:** Resumir el siguiente texto académico.\n**Texto:**\n{paper_text}"
                summary = get_hf_response(hf_api_key_input, model_reasoning, final_prompt)
                if summary: 
                    st.markdown(summary)
        else:
            st.warning("Por favor, pega el texto de un paper.")

# Pestaña 2: Razonamiento Económico/Matemático
with tab2:
    st.header("Razonador Económico-Matemático")
    question_text = st.text_area("Escribe tu pregunta o el problema a resolver:", height=200, key="question_text")
    if st.button("Obtener Razonamiento", key="reason"):
        if question_text:
            with st.spinner("Procesando..."):
                final_prompt = f"{MASTER_PROMPT}\n**Tarea:** Responder la siguiente pregunta.\n**Pregunta:**\n{question_text}"
                reasoning = get_hf_response(hf_api_key_input, model_reasoning, final_prompt)
                if reasoning: 
                    st.markdown(reasoning)
        else:
            st.warning("Por favor, introduce una pregunta.")

# Pestaña 3: Generar Código (la que se muestra en la imagen)
with tab3:
    st.header("Generador de Código con CODEQwen")
    # A
    #     JUSTE AQUÍ: El valor del `text_area` ahora coincide exactamente con la nueva imagen.
    code_description = st.text_area(
        "Describe la tarea de programación que necesitas:", 
        height=100, 
        key="code_desc", 
        value="Me podrías ayudar con un código de python para un modelo de dsge con Quantecon"
    )
    if st.button("Generar Código", key="code"):
        if code_description:
            with st.spinner("Escribiendo código..."):
                final_prompt = f"{MASTER_PROMPT}\n**Tarea:** Generar código Python.\n**Descripción:**\n{code_description}"
                code = get_hf_response(hf_api_key_input, model_coding, final_prompt)
                if code:
                    st.code(code, language='python')
        else:
            st.warning("Por favor, describe la tarea de programación.")
