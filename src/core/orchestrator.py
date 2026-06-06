"""
Central orchestrator for LLM fine-tuning pipeline.
Wires together registry, configs, GPU estimation, and trainers.
"""

from src.models.model_registry import (
    get_model_path,
    get_model_group,
    get_model_metadata,
)

from src.configs.training_config import TrainingConfig
from src.configs.peft_config import get_peft_config

from src.utils.gpu_estimator import estimate_gpu_requirements

from src.trainers.lora_hf_trainer import LoRAHFTrainer
# from src.trainers.qlora_hf_trainer import QLoRAHFTrainer
# from src.trainers.lora_unsloth_trainer import LoRAUnslothTrainer
# from src.trainers.qlora_unsloth_trainer import QLoRAUnslothTrainer


# ----------------------------
# Trainer Factory
# ----------------------------
def get_trainer(method: str, backend: str):
    """
    Select correct trainer based on method + backend.
    """

    if method == "lora" and backend == "hf":
        return LoRAHFTrainer()

    # if method == "qlora" and backend == "hf":
    #     return QLoRAHFTrainer()

    # if method == "lora" and backend == "unsloth":
    #     return LoRAUnslothTrainer()

    # if method == "qlora" and backend == "unsloth":
    #     return QLoRAUnslothTrainer()

    raise ValueError(
        f"Only LoRA + HF is supported in current test mode. "
        f"Got method={method}, backend={backend}"
    )


# ----------------------------
# Main Orchestrator
# ----------------------------
class TrainingOrchestrator:
    """
    High-level controller for fine-tuning pipeline.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Resolve model
        self.model_path = get_model_path(config.model_name)
        self.model_group = get_model_group(config.model_name)
        self.model_meta = get_model_metadata(config.model_name)

        # PEFT config
        self.peft_config = get_peft_config(config)

    # ------------------------
    # Step 1: Validate GPU
    # ------------------------
    def validate_hardware(self):
        """
        Check if GPU can handle training.
        """

        required_vram = estimate_gpu_requirements(
            model_name=self.config.model_name,
            method=self.config.method,
            batch_size=self.config.batch_size,
            seq_len=self.config.max_seq_length,
        )

        if required_vram > self.config.available_vram:
            raise RuntimeError(
                f"Insufficient VRAM. Required: {required_vram}GB, Available: {self.config.available_vram}GB"
            )

        print(f"[OK] GPU check passed ({required_vram}GB required)")

    # ------------------------
    # Step 2: Build Trainer
    # ------------------------
    def build_trainer(self):
        """
        Instantiate correct trainer.
        """

        return get_trainer(
            method=self.config.method,
            backend=self.config.backend,
        )

    # ------------------------
    # Step 3: Run training
    # ------------------------
    def run(self):
        """
        Full training pipeline execution.
        """

        print("\n===== LLM FINE-TUNING ORCHESTRATOR =====")

        print(f"Model: {self.config.model_name}")
        print(f"Path: {self.model_path}")
        print(f"Group: {self.model_group}")
        print(f"Method: {self.config.method}")
        print(f"Backend: {self.config.backend}")

        # 1. Validate GPU
        self.validate_hardware()

        # 2. Build trainer
        trainer = self.build_trainer()

        # 3. Train
        print("\n[INFO] Starting training...\n")

        trainer.train(
            model_path=self.model_path,
            dataset_path=self.config.dataset_path,
            peft_config=self.peft_config,
            training_args=self.config.to_dict(),
        )

        print("\n[SUCCESS] Training completed!")