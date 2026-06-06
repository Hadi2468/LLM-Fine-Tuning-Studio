"""
Central registry of supported LLM models.
"""

MODEL_REGISTRY = {
    # Small Models (1B-4B)
    "Llama-3.2-1B": {
        "path": "meta-llama/Llama-3.2-1B",
        "group": "Small",
        "size": "1B",
        "recommended_vram": 8,
    },
    "Llama-3.2-3B": {
        "path": "meta-llama/Llama-3.2-3B",
        "group": "Small",
        "size": "3B",
        "recommended_vram": 12,
    },
    "Qwen2.5-1.5B": {
        "path": "Qwen/Qwen2.5-1.5B-Instruct",
        "group": "Small",
        "size": "1.5B",
        "recommended_vram": 8,
    },
    "Qwen2.5-3B": {
        "path": "Qwen/Qwen2.5-3B-Instruct",
        "group": "Small",
        "size": "3B",
        "recommended_vram": 12,
    },

    # Medium Models
    "Llama-3.1-8B": {
        "path": "meta-llama/Meta-Llama-3.1-8B",
        "group": "Medium",
        "size": "8B",
        "recommended_vram": 24,
    },
    "Mistral-7B": {
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
        "group": "Medium",
        "size": "7B",
        "recommended_vram": 24,
    },
    "Qwen2.5-7B": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "group": "Medium",
        "size": "7B",
        "recommended_vram": 24,
    },
    "Qwen2.5-14B": {
        "path": "Qwen/Qwen2.5-14B-Instruct",
        "group": "Medium",
        "size": "14B",
        "recommended_vram": 40,
    },

    # Large Models
    "Qwen2.5-32B": {
        "path": "Qwen/Qwen2.5-32B-Instruct",
        "group": "Large",
        "size": "32B",
        "recommended_vram": 80,
    },
    "Qwen2.5-72B": {
        "path": "Qwen/Qwen2.5-72B-Instruct",
        "group": "Large",
        "size": "72B",
        "recommended_vram": 80,
    },
    "Llama-3.1-70B": {
        "path": "meta-llama/Meta-Llama-3.1-70B",
        "group": "Large",
        "size": "70B",
        "recommended_vram": 80,
    },
    "Llama-3.3-70B": {
        "path": "meta-llama/Llama-3.3-70B-Instruct",
        "group": "Large",
        "size": "70B",
        "recommended_vram": 80,
    },
}

def get_all_models():
    return {k: v["path"] for k, v in MODEL_REGISTRY.items()}


def get_model_path(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}")
    return MODEL_REGISTRY[model_name]["path"]


def get_model_group(model_name: str):
    return MODEL_REGISTRY.get(model_name, {}).get("group", "Unknown")


def get_model_metadata(model_name: str):
    return MODEL_REGISTRY.get(model_name, {})

def resolve_model_name(model_name: str):
    """
    Case-insensitive model lookup
    """
    for k in MODEL_REGISTRY.keys():
        if k.lower() == model_name.lower():
            return MODEL_REGISTRY[k]["path"]

    raise ValueError(f"Model not found: {model_name}")
