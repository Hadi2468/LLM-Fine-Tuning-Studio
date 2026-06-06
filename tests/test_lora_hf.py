from src.trainers.lora_hf_trainer import LoRAHFTrainer
from src.configs.training_config import TrainingConfig
from src.configs.peft_config import LoRAConfig
from src.dataset.loader import load_dataset


config = TrainingConfig(
    model_name="meta-llama/Llama-3.2-1B",
    dataset_path="datasets/sample.json",
    epochs=1,
    batch_size=1,
    gradient_accumulation_steps=1,
    max_seq_length=128,
    learning_rate=2e-4,
)

dataset = load_dataset(config.dataset_path)

lora_config = LoRAConfig()

trainer = LoRAHFTrainer(
    config,
    dataset,
    lora_config
)

trainer.load_model()
trainer.apply_lora()

print("\n✅ Trainer initialized successfully")

trainer.prepare_dataset()
trainer.apply_lora()
trainer.train()
trainer.save_model()

print("\n✅ Model saved successfully")
