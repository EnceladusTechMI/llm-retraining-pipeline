# scripts/finetune.py

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def main():
    # Settings
    #model_name = "mistralai/Mistral-7B-v0.1"
    output_dir = "models/lora_falcon_rw"
    #model_name = "tiiuae/falcon-7b-instruct"  # More optimized for lighter machines
    model_name = "tiiuae/falcon-rw-1b"



    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"},
    torch_dtype=torch.float32
    )

    # Prepare model for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query_key_value"],  # Fine-tuning only specific parts
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # Load dummy dataset
    dataset = load_dataset("Abirate/english_quotes", split="train[:1%]")  # Tiny dataset to keep it fast

    def tokenize_function(examples):
        return tokenizer(examples["quote"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x["input_ids"]})


    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy="no",
        num_train_epochs=1,
        save_steps=10,
        logging_steps=5,
        save_total_limit=1,
        fp16=False,
        learning_rate=2e-4,
        optim="adamw_torch",
        report_to="none",
        no_cuda=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    # Start fine-tuning
    trainer.train()

    # Save final LoRA adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Fine-tuning complete. Model saved to {output_dir}")

if __name__ == "__main__":
    main()
