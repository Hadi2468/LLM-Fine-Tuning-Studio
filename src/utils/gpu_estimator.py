"""
Simple GPU estimator for LLM fine-tuning (LoRA / QLoRA aware).
"""

def estimate_gpu_requirements(
    model_name: str,
    method: str,
    batch_size: int,
    seq_len: int,
):
    """
    Returns estimated VRAM (GB) required for training.
    """

    # Base VRAM estimates (rough but practical)
    base_vram_map = {
        "Llama-3.2-1B": 6,
        "Llama-3.2-3B": 10,
        "Qwen2.5-1.5B": 6,
        "Qwen2.5-3B": 10,
        "Mistral-7B": 24,
        "Qwen2.5-7B": 24,
        "Llama-3.1-8B": 24,
        "Qwen2.5-14B": 40,
        "Qwen2.5-32B": 80,
        "Qwen2.5-72B": 140,
        "Llama-3.1-70B": 140,
        "Llama-3.3-70B": 140,
    }

    base_vram = base_vram_map.get(model_name, 24)

    # Method adjustment
    method = method.lower()

    if method == "lora":
        base_vram *= 0.85
    elif method == "qlora":
        base_vram *= 0.5

    # Batch size scaling
    base_vram *= (1 + (batch_size - 1) * 0.08)

    # Sequence length scaling (relative to 2048 baseline)
    base_vram *= (seq_len / 2048)

    return round(base_vram, 2)