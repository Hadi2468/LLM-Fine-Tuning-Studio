# Configuration for training parameters and settings

from dataclasses import dataclass
from enum import Enum


class FinetuneMethod(str, Enum):
    LORA = "lora"
    QLORA = "qlora"


class Backend(str, Enum):
    HF = "hf"
    UNSLOTH = "unsloth"


@dataclass
class TrainingConfig:
    model_name: str = ""
    method: FinetuneMethod = FinetuneMethod.LORA
    backend: Backend = Backend.HF
    dataset_path: str = ""
    output_dir: str = "./outputs"
    warmup_ratio: float = 0.03
    learning_rate: float = 2e-4
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 2048
    logging_steps: int = 5
    available_vram: int = 16
    save_steps: int = 100