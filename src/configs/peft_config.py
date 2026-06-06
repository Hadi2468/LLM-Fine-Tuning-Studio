from dataclasses import dataclass


PEFT_METHODS = [
    "None",
    "LoRA",
    "QLoRA"
]

BACKENDS = [
    "Hugging Face",
    "Unsloth"
]


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )


def get_peft_config(training_config):
    """
    Converts TrainingConfig → PEFT config
    """

    if training_config.method.lower() == "lora":
        return LoRAConfig()

    if training_config.method.lower() == "qlora":
        # same config for now (you can extend later)
        return LoRAConfig()

    return None