"""
train_lora.py — LoRA fine-tuning script.

This script fine-tunes the base TinyLlama model on our generated QA dataset 
using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.
"""
import os
import sys

# Ensure the project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

from lora_training.config import (
    MODEL_NAME, DATASET_PATH, OUTPUT_DIR, MAX_LENGTH,
    BATCH_SIZE, GRADIENT_ACCUMULATION, LEARNING_RATE, NUM_EPOCHS,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES
)
from lora_training.dataset import load_dataset


def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize the 'text' field in the dataset to prepare it for
    training with a causal language model.
    """
    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        return tokens

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    return tokenized_dataset


def train_model():
    """
    Main execution loop for LoRA fine-tuning.
    """
    print("Loading TinyLlama model")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model in float16 so it fits well in 8GB RAM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.config.use_cache = False

    print("Loading dataset")
    dataset = load_dataset(DATASET_PATH)

    # Make sure we got data
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    print("Applying LoRA adapters")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    
    print(f"dataset size: {len(dataset)}")
    # Print the number of trainable parameters
    model.print_trainable_parameters()

    # Create the training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none"  # avoid telemetry
    )

    # We use DataCollatorForLanguageModeling to predict the next token (causal language modeling)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Training started")
    trainer.train()

    print("Training complete")
    print(f"Adapter weights saved to:\n{OUTPUT_DIR}/")
    
    # Save our final adapter
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train_model()
