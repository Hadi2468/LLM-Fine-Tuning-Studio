import streamlit as st

from src.models.model_registry import get_all_models
from src.configs.training_config import TrainingConfig
from src.core.orchestrator import TrainingOrchestrator
from src.core.training_worker import TrainingWorker


# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="LLM Fine-Tuning Studio",
    layout="wide"
)

st.title("🧠 LLM Fine-Tuning Studio")


# ---------------------------
# Sidebar - Config
# ---------------------------
st.sidebar.header("⚙️ Training Configuration")

models = list(get_all_models().keys())

model_name = st.sidebar.selectbox("Model", models)

method = st.sidebar.selectbox("Method", ["lora", "qlora"])

backend = st.sidebar.selectbox("Backend", ["hf", "unsloth"])

dataset_path = st.sidebar.text_input(
    "Dataset Path",
    "datasets/sample.json"
)

batch_size = st.sidebar.slider("Batch Size", 1, 16, 4)

max_seq_len = st.sidebar.selectbox(
    "Max Sequence Length",
    [512, 1024, 2048, 4096],
    index=2
)

available_vram = st.sidebar.number_input(
    "Available VRAM (GB)",
    min_value=4,
    max_value=200,
    value=24
)


# ---------------------------
# Build config
# ---------------------------
def build_config():
    return TrainingConfig(
        model_name=model_name,
        method=method,
        backend=backend,
        dataset_path=dataset_path,
        batch_size=batch_size,
        max_seq_length=max_seq_len,
        available_vram=available_vram,
    )


# ---------------------------
# Session state (worker)
# ---------------------------
if "worker" not in st.session_state:
    st.session_state.worker = None


# ---------------------------
# Start training
# ---------------------------
st.markdown("---")

if st.button("🚀 Start Fine-Tuning", use_container_width=True):

    config = build_config()

    st.subheader("🔧 Training Configuration")
    st.json(config.__dict__)

    try:
        # Create orchestrator
        orchestrator = TrainingOrchestrator(config)

        # Validate hardware first
        st.info("Checking GPU compatibility...")
        orchestrator.validate_hardware()
        st.success("GPU check passed!")

        # Start background worker
        worker = TrainingWorker(orchestrator)
        worker.start()

        st.session_state.worker = worker

        st.success("🚀 Training started in background!")

    except Exception as e:
        st.error(f"Failed to start training: {str(e)}")


# ---------------------------
# Live status panel
# ---------------------------
st.markdown("---")
st.subheader("📡 Training Status")

worker = st.session_state.worker

if worker is None:
    st.info("No training job started yet.")
else:
    status, error = worker.status()

    if status == "running":
        st.warning("🟡 Training is running in background...")
        st.write("You can safely change config or start another session.")

    elif status == "failed":
        st.error(f"❌ Training failed: {error}")

    else:
        st.success("✅ Training completed or idle.")