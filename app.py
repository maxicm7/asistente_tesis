import streamlit as st
from huggingface_hub import InferenceClient

# --- 1. Definición del Rol y Configuración Inicial ---
# Se define un prompt maestro para darle contexto y personalidad al asistente de IA.
MASTER_PROMPT = """
[INICIO DE LA DEFINICIÓN DEL ROL]
**Nombre del Rol:** Investigador Doctoral IA (IDA)
**Objetivo Principal:** Asistir en la investigación y redacción de una tesis doctoral en el campo de la Economía Aplicada. Tu función es ser un colaborador riguroso, analítico y eficiente.
**Personalidad:** Eres un asistente de investigación post-doctoral; preciso, metódico y objetivo.
**Áreas de Especialización:**
1. **Analista de Literatura Académica:** Resume papers identificando pregunta de investigación, metodología, resultados, contribución y limitaciones.
2. **Razonador Económico-Matemático:** Explica conceptos, desarrolla derivaciones matemáticas paso a paso e interpreta modelos.
3. **Generador de Código (Especialista en CODEQwen):** Genera código Python funcional, comentado y con sus dependencias para tareas de análisis de datos.
**Instrucciones de Interacción:** Identifica la tarea, aplica el formato de salida correcto y prioriza la integridad académica.
[FIN DE LA DEFINICIÓN DEL ROL]
"""

# --- 2. Interfaz de Streamlit ---
st.set_page_config(layout="wide", page_title="Asistente de Tesis Doctoral IA")

# Barra lateral para configuración
st.sidebar.header("Configuración")

# Input para la API Key de Hugging Face, con manejo de secretos de Streamlit
try:
    HF_API_KEY_FROM_SECRETS = st.secrets["HF_API_KEY"]
except (FileNotFoundError, KeyError):
    HF_API_KEY_FROM_SECRETS = ""

hf_api_key_input = st.sidebar.text_input(
    "Hugging Face API Key", 
    type="password", 
    value=HF_API_KEY_FROM_SECRETS,
    help="Pega tu clave aquí si corres la app localmente. En la nube, se configura vía st.secrets."
)

st.sidebar.subheader("Selección de Modelos")
model_reasoning = st.sidebar.selectbox(
    "Modelo para Resumen y Razonamiento",
    ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-8B-Instruct", "microsoft/Phi-3-mini-4k-instruct", "google/gemma-7b-it"],
    key="model_reasoning_select"
)
model_coding = st.sidebar.selectbox(
    "Modelo para Código (CODEQwen)",
    ["Qwen/CodeQwen1.5-7B-Chat", "codellama/CodeLlama-34b-Instruct-hf", "bigcode/starcoder2-15b"],
    key="model_coding_select"
)

# --- 3. Lógica Principal de la App ---
# Definición de las pestañas de la interfaz principal
tab1, tab2, tab3 = st.tabs(["📄 Resumir Paper", "🧠 Razonamiento Económico/Matemático", "💻 Generar Código"])


def get_hf_response(api_key, model, prompt):
    """
    Función central para contactar la API de Hugging Face.
    Maneja la autenticación, la llamada a la API y la gestión de errores.
    """
    if not api_key or not api_key.startswith("hf_"):
        st.error("Por favor, introduce una Hugging Face API Key válida.")
        st.stop()

    try:
        # Crea el cliente de inferencia con el token proporcionado
        client = InferenceClient(token=api_key)
        messages = [{"role": "user", "content": prompt}]
        
        # Realiza la llamada a la API usando el método chat_completion
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=4096,
        )
        return response.choices[0].message.content
        
    except Exception as e:
        # Esta sección se activa si la llamada a la API falla, replicando el error de la imagen
        st.error(f"Error al contactar la API de Hugging Face. Detalles:")
        st.info("Esto puede ocurrir por varias razones:\n"
                "1. No tienes acceso a este modelo (visita su página en HF para solicitarlo).\n"
                "2. El modelo está tardando en cargar en los servidores de Hugging Face. Por favor, espera un minuto y vuelve a intentarlo.\n"
                "3. La API Key es incorrecta o no tiene los permisos necesarios.")
        print(f"Detalle del error: {e}") # Imprime el error real en la consola para depuración
        return None

# Contenido de la Pestaña 1: Resumir Paper
with tab1:
    st.header("Analista de Literatura Académica")
    paper_text = st.text_area("Pega aquí el abstract o el texto completo del paper:", height=300, key="paper_text")
    if st.button("Generar Resumen", key="summarize"):
        if paper_text:
            with st.spinner("Analizando..."):
                final_prompt = f"{MASTER_PROMPT}\n\n**Tarea Actual:** Resumir el siguiente paper académico.\n\n**Texto a Analizar:**\n```\n{paper_text}\n```\n\n**Análisis Detallado:**"
                summary = get_hf_response(hf_api_key_input, model_reasoning, final_prompt)
                if summary: st.markdown(summary)
        else: st.warning("Por favor, pega el texto de un paper.")

# Contenido de la Pestaña 2: Razonamiento
with tab2:
    st.header("Razonador Económico-Matemático")
    question_text = st.text_area("Escribe tu pregunta o el problema a resolver:", height=200, key="question_text")
    if st.button("Obtener Razonamiento", key="reason"):
        if question_text:
            with st.spinner("Procesando..."):
                final_prompt = f"{MASTER_PROMPT}\n\n**Tarea Actual:** Responder a una pregunta de economía/matemáticas.\n\n**Pregunta:**\n```\n{question_text}\n```\n\n**Respuesta Detallada:**"
                reasoning = get_hf_response(hf_api_key_input, model_reasoning, final_prompt)
                if reasoning: st.markdown(reasoning)
        else: st.warning("Por favor, introduce una pregunta.")

# Contenido de la Pestaña 3: Generar Código (la que se muestra en la imagen)
with tab3:
    st.header("Generador de Código con CODEQwen")
    code_description = st.text_area("Describe la tarea de programación que necesitas:", height=100, key="code_desc", value="Me podrías ayudar con un código de para un modelo de dsge con Quantecon")
    if st.button("Generar Código", key="code"):
        if code_description:
            with st.spinner("Escribiendo código..."):
                final_prompt = f"{MASTER_PROMPT}\n\n**Tarea Actual:** Generar código Python basado en la siguiente descripción.\n\n**Descripción de la Tarea:**\n```\n{code_description}\n```\n\n**Código Generado:**"
                code = get_hf_response(hf_api_key_input, model_coding, final_prompt)
                if code:
                    # Limpia el output para mostrar solo el bloque de código
                    try:
                        code_block = code.split("```python")[1].split("```")[0]
                        st.code(code_block.strip(), language='python')
                    except IndexError:
                         # Si el formato no es el esperado, muestra la respuesta completa
                        st.code(code, language='python')
        else: st.warning("Por favor, describe la tarea de programación.")
