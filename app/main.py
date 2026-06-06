import streamlit as st
import re

from src.models.model_registry import get_all_models
from src.configs.training_config import TrainingConfig
from src.core.orchestrator import TrainingOrchestrator
from src.core.training_worker import TrainingWorker


# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="LLM Fine-Tuning Studio",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Color palette based on #40C7BB (teal/mint) ──
   Primary:   #40C7BB  (teal)
   Dark bg:   #0b1a19  (very dark teal-black)
   Panel bg:  #0f2422  (dark teal)
   Border:    #1a3a37  (muted teal border)
   Secondary: #2ee8c0  (bright mint accent)
   Tertiary:  #a8f0e8  (pale mint)
   Text:      #d4f5f1  (light teal-white)
   Muted:     #4a7a75  (muted teal)
*/

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0b1a19;
    color: #d4f5f1;
}

.stApp {
    background: #0b1a19;
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 2rem 2.5rem !important;
    max-width: 100% !important;
}

/* ── Title ── */
.studio-title {
    font-family: 'Space Mono', monospace;
    font-size: 50vw;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #40C7BB, #2ee8c0, #a8f0e8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    line-height: 1.15;
}

.studio-subtitle {
    font-size: 0.85rem;
    color: #4a7a75;
    font-family: 'Space Mono', monospace;
    margin-top: 0.3rem;
    letter-spacing: 1.5px;
}

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1a3a37, #40C7BB 40%, #2ee8c0 60%, #1a3a37, transparent);
    margin: 1.2rem 0 1.8rem 0;
    border: none;
}

/* ── Panel headers ── */
.panel-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #40C7BB;
    padding: 0.4rem 0.8rem;
    border-left: 2px solid #40C7BB;
    margin-bottom: 1.2rem;
    background: rgba(64, 199, 187, 0.06);
}

.panel-header-right {
    border-left: 2px solid #2ee8c0;
    color: #2ee8c0;
    background: rgba(46, 232, 192, 0.06);
}

.panel-header-center {
    border-left: 2px solid #a8f0e8;
    color: #a8f0e8;
    background: rgba(168, 240, 232, 0.06);
    font-size: 0.8rem;
}

/* ── Input styling ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background-color: #0f2422 !important;
    border: 1px solid #1a3a37 !important;
    border-radius: 6px !important;
    color: #d4f5f1 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s ease;
}

.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus {
    border-color: #40C7BB !important;
    box-shadow: 0 0 0 2px rgba(64, 199, 187, 0.12) !important;
}

/* ── Labels ── */
.stSelectbox label, .stNumberInput label, .stTextInput label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #4a7a75 !important;
    letter-spacing: 0.3px !important;
    text-transform: uppercase !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #40C7BB 0%, #2ee8c0 60%, #a8f0e8 100%) !important;
    color: #0b1a19 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(64, 199, 187, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(64, 199, 187, 0.5) !important;
    filter: brightness(1.1) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Alerts ── */
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}

/* ── JSON display ── */
.stJson {
    background: #0f2422 !important;
    border: 1px solid #1a3a37 !important;
    border-radius: 8px !important;
}

/* ── Status card ── */
.status-card {
    background: #0f2422;
    border: 1px solid #1a3a37;
    border-radius: 10px;
    padding: 1.5rem;
    margin-top: 1rem;
}

.status-idle {
    border-left: 3px solid #2a4a47;
}

.status-running {
    border-left: 3px solid #f0c040;
    animation: pulse-border 2s infinite;
}

.status-done {
    border-left: 3px solid #40C7BB;
}

.status-failed {
    border-left: 3px solid #f87171;
}

@keyframes pulse-border {
    0%, 100% { border-left-color: #f0c040; }
    50% { border-left-color: #d4a820; }
}

/* ── Metric-like pill ── */
.config-pill {
    display: inline-block;
    background: rgba(64, 199, 187, 0.08);
    border: 1px solid rgba(64, 199, 187, 0.25);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #40C7BB;
    margin: 0.2rem;
}

/* ── Column separators ── */
.col-separator {
    border-left: 1px solid #1a3a37;
    height: 100%;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0b1a19; }
::-webkit-scrollbar-thumb { background: #1a3a37; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #40C7BB; }
</style>
""", unsafe_allow_html=True)


# ---------------------------
# Header
# ---------------------------
st.markdown('<p class="studio-title"> ✴️ LLM Fine-Tuning Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="studio-subtitle">// train · evaluate · deploy</p>', unsafe_allow_html=True)
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ---------------------------
# Layout columns
# ---------------------------
left_col, _, middle_col, _, right_col = st.columns([2, 0.1, 3, 0.1, 2])


# ---------------------------
# Natural sort helper
# ---------------------------
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


# ---------------------------
# Left panel – Model Config
# ---------------------------
with left_col:
    st.markdown('<div class="panel-header">⚙️ Model Configuration</div>', unsafe_allow_html=True)

    models = sorted(get_all_models().keys(), key=natural_sort_key)
    model_name = st.selectbox("Model", models)

    method_options = {"LoRA": "lora", "QLoRA": "qlora"}
    method = method_options[st.selectbox("Method", method_options.keys())]

    backend_options = {"Hugging Face": "hf", "Unsloth": "unsloth"}
    backend = backend_options[st.selectbox("Backend", backend_options.keys())]

    available_vram = st.number_input("Available VRAM (GB)", min_value=4, max_value=200, value=16, step=1)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel-header" style="border-left-color:#2ee8c0;color:#2ee8c0;background:rgba(46,232,192,0.06)">📂 Data & Output</div>', unsafe_allow_html=True)

    dataset_path = st.text_input("Dataset Path", value="datasets/sample.json")
    output_dir = st.text_input("Output Directory", value="./outputs")


# ---------------------------
# Right panel – Training Config
# ---------------------------
with right_col:
    st.markdown('<div class="panel-header panel-header-right">⚙️ Training Configuration</div>', unsafe_allow_html=True)

    learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-1, value=2e-4, format="%.6f")
    epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=1, step=1)
    batch_size = st.number_input("Batch Size", min_value=1, max_value=1000, value=1, step=1)
    accumulate_grad_batches = st.number_input("Grad Accumulation Steps", min_value=1, max_value=1000, value=16, step=1)
    logging_steps = st.number_input("Logging Steps", min_value=1, max_value=100, value=5, step=1)
    max_seq_len = st.selectbox(
        "Max Sequence Length",
        [64, 128, 256, 512, 1024, 2048, 4096],
        index=2
    )

    


# ---------------------------
# Middle panel – Control & Status
# ---------------------------
with middle_col:
    st.markdown('<div class="panel-header panel-header-center">🧠 Training Console</div>', unsafe_allow_html=True)

    # Build config helper
    def build_config():
        return TrainingConfig(
            model_name=model_name,
            method=method,
            backend=backend,
            dataset_path=dataset_path,
            output_dir=output_dir,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=accumulate_grad_batches,
            max_seq_length=max_seq_len,
            logging_steps=logging_steps,
            available_vram=available_vram,
        )

    # Session state
    if "worker" not in st.session_state:
        st.session_state.worker = None

    # Config summary pills
    st.markdown(f"""
        <div style="margin-bottom:1rem; line-height:2.2;">
            <span class="config-pill">{model_name.split('/')[-1]}</span>
            <span class="config-pill">{method.upper()}</span>
            <span class="config-pill">{backend.upper()}</span>
            <span class="config-pill">lr={learning_rate:.0e}</span>
            <span class="config-pill">ep={epochs}</span>
            <span class="config-pill">bs={batch_size}</span>
            <span class="config-pill">seq={max_seq_len}</span>
            <span class="config-pill">vram={available_vram}GB</span>
        </div>
    """, unsafe_allow_html=True)

    # Start button
    if st.button("⚡ START FINE-TUNING", use_container_width=True):
        config = build_config()

        with st.expander("📋 Full Configuration", expanded=False):
            st.json(config.__dict__)

        try:
            orchestrator = TrainingOrchestrator(config)

            with st.spinner("Checking GPU compatibility..."):
                orchestrator.validate_hardware()
            st.success("✅ GPU check passed!")

            worker = TrainingWorker(orchestrator)
            worker.start()
            st.session_state.worker = worker
            st.success("🚀 Training started in background!")

        except Exception as e:
            st.error(f"❌ Failed to start: {str(e)}")

    # Status panel
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel-header panel-header-center">🚀 Live Status</div>', unsafe_allow_html=True)

    worker = st.session_state.worker

    if worker is None:
        st.markdown("""
            <div class="status-card status-idle">
                <span style="color:#475569; font-family:'Space Mono',monospace; font-size:0.8rem;">
                    ○ &nbsp; NO ACTIVE JOB
                </span><br>
                <span style="color:#2a4a47; font-size:0.78rem; margin-top:0.3rem; display:block;">
                    Configure your model and training parameters, then hit Start.
                </span>
            </div>
        """, unsafe_allow_html=True)
    else:
        status, error = worker.status()

        if status == "running":
            st.markdown("""
                <div class="status-card status-running">
                    <span style="color:#f0c040; font-family:'Space Mono',monospace; font-size:0.8rem;">
                        ◉ &nbsp; TRAINING IN PROGRESS
                    </span><br>
                    <span style="color:#94a3b8; font-size:0.78rem; margin-top:0.3rem; display:block;">
                        Running in background. You can safely adjust settings or open a new session.
                    </span>
                </div>
            """, unsafe_allow_html=True)

        elif status == "failed":
            st.markdown(f"""
                <div class="status-card status-failed">
                    <span style="color:#f87171; font-family:'Space Mono',monospace; font-size:0.8rem;">
                        ✕ &nbsp; TRAINING FAILED
                    </span><br>
                    <span style="color:#94a3b8; font-size:0.78rem; margin-top:0.3rem; display:block;">
                        {error}
                    </span>
                </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
                <div class="status-card status-done">
                    <span style="color:#40C7BB; font-family:'Space Mono',monospace; font-size:0.8rem;">
                        ✓ &nbsp; TRAINING COMPLETE
                    </span><br>
                    <span style="color:#94a3b8; font-size:0.78rem; margin-top:0.3rem; display:block;">
                        Model has been saved to the output directory.
                    </span>
                </div>
            """, unsafe_allow_html=True)
