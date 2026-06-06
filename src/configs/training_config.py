# Configuration for training parameters and settings


from dataclasses import dataclass


@dataclass
class TrainingConfig:

    output_dir: str = "./outputs"

    learning_rate: float = 2e-4

    epochs: int = 10

    batch_size: int = 1

    gradient_accumulation_steps: int = 16

    warmup_ratio: float = 0.03

    max_seq_length: int = 2048

    logging_steps: int = 5

    save_steps: int = 100

    weight_decay: float = 0.01