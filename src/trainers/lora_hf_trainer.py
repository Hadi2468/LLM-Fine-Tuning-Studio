"""
LoRA Fine-Tuning Trainer using Hugging Face Trainer API
"""

from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from src.models.model_registry import resolve_model_name

from .base_trainer import BaseTrainer


class LoRAHFTrainer(BaseTrainer):
    """
    Hugging Face + LoRA trainer.
    """

    def __init__(self, config, dataset, lora_config=None):

        self.config = config
        self.dataset = dataset
        self.lora_config = lora_config

        self.model_name = resolve_model_name(config["model"]["name"])
        self.training_config = config

        self.model = None
        self.tokenizer = None

        self.peft_model = None
        self.tokenized_dataset = None

        self.trainer = None

    ####################################################################
    # Load Model
    ####################################################################

    def load_model(self):
        print("📦 Selected model:", self.model_name)
        print(f"\nLoading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        return self.model

    ####################################################################
    # Dataset Formatting
    ####################################################################

    @staticmethod
    def format_instruction(example):
        if "text" in example:
            return {"text": example["text"]}

        return {
            "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
        }

    ####################################################################
    # Tokenization
    ####################################################################

    def tokenize_function(self, example):

        tokens = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.training_config.max_seq_length,
            padding="max_length",
        )

        tokens["labels"] = tokens["input_ids"].copy()

        return tokens

    ####################################################################
    # Dataset Preparation
    ####################################################################

    def prepare_dataset(self):

        print("\nFormatting dataset...")

        formatted_dataset = self.dataset.map(
            self.format_instruction
        )

        print("Tokenizing dataset...")

        self.tokenized_dataset = formatted_dataset.map(
            self.tokenize_function,
            remove_columns=formatted_dataset.column_names,
        )

        return self.tokenized_dataset

    ####################################################################
    # Apply LoRA
    ####################################################################

    def apply_lora(self):

        print("\nApplying LoRA configuration...")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            target_modules=list(
                self.lora_config.target_modules
            ),
        )

        self.peft_model = get_peft_model(
            self.model,
            peft_config
        )

        self.peft_model.print_trainable_parameters()

        return self.peft_model

    ####################################################################
    # Train
    ####################################################################

    def train(self):

        print("\nStarting training...")

        bf16_supported = (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )

        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            per_device_train_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            num_train_epochs=self.training_config.epochs,
            warmup_ratio=self.training_config.warmup_ratio,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            bf16=bf16_supported,
            fp16=not bf16_supported,
            report_to="none",
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            max_grad_norm=0.3,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        self.trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            data_collator=data_collator,
        )

        self.trainer.train()

    ####################################################################
    # Save Model
    ####################################################################

    def save_model(
        self,
        output_dir: str = None,
    ):

        if output_dir is None:

            safe_name = self.model_name.split("/")[-1]

            output_dir = (
                Path("outputs")
                / f"{safe_name}_lora"
            )

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(f"\nSaving model to: {output_dir}")

        self.peft_model.save_pretrained(
            str(output_dir)
        )

        self.tokenizer.save_pretrained(
            str(output_dir)
        )

    ####################################################################
    # Generate
    ####################################################################

    def generate(
        self,
        question: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ):

        self.peft_model.eval()

        prompt = (
            f"### Instruction:\n"
            f"{question}\n\n"
            f"### Response:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.peft_model.device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():

            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return response