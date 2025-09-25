import streamlit as st
from huggingface_hub import InferenceClient
import os

# --- 1. Definición del Rol y Configuración Inicial ---

# Tu clave de API de Hugging Face
# Es mejor usar st.secrets para producción, pero para desarrollo local esto funciona.
# O puedes pedirla en la UI.
HF_API_KEY = os.environ.get("HF_API_KEY") # O usa st.text_input("Hugging Face API Key", type="password")

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

st.title("🎓 Asistente de Tesis Doctoral IA")
st.markdown("Una herramienta para potenciar tu investigación doctoral usando IA.")

# --- Configuración en la barra lateral ---
st.sidebar.header("Configuración")
hf_api_key_input = st.sidebar.text_input("Hugging Face API Key", type="password", help="Pega tu clave de API de Hugging Face aquí.")

# Selección de modelos
st.sidebar.subheader("Selección de Modelos")
model_reasoning = st.sidebar.selectbox(
    "Modelo para Resumen y Razonamiento",
    ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Llama-2-70b-chat-hf", "google/gemma-7b-it"],
    help="Modelos grandes son mejores para entender textos complejos."
)
model_coding = st.sidebar.selectbox(
    "Modelo para Código (CODEQwen)",
    ["Qwen/CodeQwen1.5-7B-Chat", "codellama/CodeLlama-34b-Instruct-hf"],
    help="Modelos especializados en código."
)


# --- 3. Funcionalidad de la App en Pestañas ---

tab1, tab2, tab3 = st.tabs(["📄 Resumir Paper", "🧠 Razonamiento Económico/Matemático", "💻 Generar Código"])

# Función para llamar a la API de Hugging Face
def get_hf_response(api_key, model, prompt):
    if not api_key:
        st.error("Por favor, introduce tu Hugging Face API Key en la barra lateral.")
        return None
    try:
        client = InferenceClient(token=api_key)
        response = client.text_generation(prompt=prompt, model=model, max_new_tokens=2048)
        return response
    except Exception as e:
        st.error(f"Error al contactar la API de Hugging Face: {e}")
        return None

# Pestaña 1: Resumir Paper
with tab1:
    st.header("Analista de Literatura Académica")
    paper_text = st.text_area("Pega aquí el abstract o el texto completo del paper:", height=300)
    if st.button("Generar Resumen", key="summarize"):
        if paper_text:
            with st.spinner("Analizando el texto y generando resumen..."):
                final_prompt = f"{MASTER_PROMPT}\n\n**Tarea Actual:** Realizar un análisis de literatura académica sobre el siguiente texto. Sigue estrictamente el formato de salida para 'Analista de Literatura Académica'.\n\n**Texto a Analizar:**\n```\n{paper_text}\n```\n\n**Análisis Detallado:**"
                summary = get_hf_response(hf_api_key_input, model_reasoning, final_prompt)
                if summary:
                    st.markdown(summary)
        else:
            st.warning("Por favor, pega el texto de un paper.")

# Pestaña 2: Razonamiento
with tab2:
    st.header("Razonador Económico-Matemático")
    question_text = st.text_area("Escribe tu pregunta o el problema a resolver:", height=200)
    if st.button("Obtener Razonamiento", key="reason"):
        if question_text:
            with st.spinner("Procesando la solicitud..."):
                final_prompt = f"{MASTER_PROMPT}\n\n**Tarea Actual:** Responder a la siguiente pregunta de razonamiento económico/matemático. Proporciona una explicación clara, lógica y, si es necesario, paso a paso.\n\n**Pregunta:**\n```\n{question_text}\n```\n\n**Respuesta Detallada:**"
                reasoning = get_hf_response(hf_api_key_input, model_reasoning, final_prompt)
                if reasoning:
                    st.markdown(reasoning)
        else:
            st.warning("Por favor, introduce una pregunta.")

# Pestaña 3: Generar Código
with tab3:
    st.header("Generador de Código con CODEQwen")
    code_description = st.text_area("Describe la tarea de programación que necesitas:", height=200)
    if st.button("Generar Código", key="code"):
        if code_description:
            with st.spinner("Escribiendo el código..."):
                final_prompt = f"{MASTER_PROMPT}\n\n**Tarea Actual:** Generar código según la siguiente descripción. Utiliza la especialización 'Generador de Código (Especialista en CODEQwen)' para producir un código claro, comentado y con sus dependencias.\n\n**Descripción de la Tarea:**\n```\n{code_description}\n```\n\n**Código Generado:**"
                code = get_hf_response(hf_api_key_input, model_coding, final_prompt)
                if code:
                    # Los modelos de código a menudo devuelven el código dentro de bloques ```python ... ```
                    # Podemos intentar extraerlo o simplemente mostrarlo todo.
                    st.code(code, language='python')
        else:
            st.warning("Por favor, describe la tarea de programación.")
