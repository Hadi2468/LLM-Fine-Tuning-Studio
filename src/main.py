# Main entry point for the LLM Fine-Tuning Studio application

trainer = LoRAHFTrainer(
    model_name=model_name,
    dataset=dataset,
    training_config=config,
    lora_config=lora_config
)

trainer.load_model()
trainer.prepare_dataset()
trainer.apply_lora()
trainer.train()
trainer.save_model()