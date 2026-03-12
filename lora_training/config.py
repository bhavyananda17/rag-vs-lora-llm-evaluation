"""
config.py — Configuration parameters for LoRA fine-tuning.
"""

import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model & Data paths
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "qa_dataset.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "lora_adapter")

# Tokenizer / Text length
MAX_LENGTH = 512

# Training parameters
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

# LoRA parameters
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]
