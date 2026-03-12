"""
evaluate_models.py — Evaluation framework comparing Base LLM, RAG, and LoRA approaches.

This script runs a fixed set of evaluation questions against three distinct pipelines
to compare the quality, grounding, and domain knowledge of each method.
Results are displayed on the console and saved to a CSV file for further analysis.
"""

import sys
import os
import json
import logging
import warnings
import pandas as pd
import requests
import torch

# Ensure the project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Suppress verbose Hugging Face warnings for a cleaner console UI
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import RAG pipeline
from rag_pipeline.retrieval import initialize_rag_pipeline, rag_query

# ── Configuration ─────────────────────────────────────────────────────
EVAL_QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation_questions.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "results.csv")
LORA_ADAPTER_PATH = os.path.join(PROJECT_ROOT, "lora_adapter")
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OLLAMA_MODEL = "tinyllama"
OLLAMA_URL = "http://localhost:11434/api/generate"


# ── Functions ─────────────────────────────────────────────────────────

def load_questions() -> list:
    """Load the evaluation questions from the JSON file."""
    if not os.path.exists(EVAL_QUESTIONS_PATH):
        raise FileNotFoundError(f"Questions not found at {EVAL_QUESTIONS_PATH}")
    
    with open(EVAL_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} evaluation questions.")
    return questions


def run_base_model(question: str) -> str:
    """Inference for the Base LLM using the local Ollama instance."""
    prompt = f"Question:\n{question}\n\nAnswer:\n"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=None)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"[Error: {str(e)}]"


def run_rag_model(question: str, chunks, embedding_model, faiss_index) -> str:
    """Inference for the RAG system using the existing pipeline."""
    # We suppress standard stdout temporarily to keep the evaluation loop clean
    # Alternatively, we just extract the answer
    result_dict = rag_query(
        query=question,
        chunks=chunks,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        ollama_model=OLLAMA_MODEL
    )
    return result_dict["answer"].strip()


def run_lora_model(question: str, model, tokenizer) -> str:
    """Inference for the LoRA fine-tuned model using HuggingFace PEFT."""
    # Format matches the training dataset instruction template
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.1,      # low temperature for factual recall
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip the original prompt out to get just the generated response
    answer = output_text.replace(prompt, "").strip()
    return answer


def save_results(results: list):
    """Save the evaluation results to an evaluation/results.csv file."""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print(f"\nSaved evaluation results to {RESULTS_PATH}")


# ── Main Evaluation Execution ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Model Evaluation: Base LLM vs RAG vs LoRA")
    print("=" * 60)

    # 1. Load Questions
    questions = load_questions()
    
    # 2. Setup RAG components
    chunks, embedding_model, faiss_index = initialize_rag_pipeline()
    
    # 3. Setup LoRA Model
    print("\nLoading LoRA adapter and base model...")
    if not os.path.exists(LORA_ADAPTER_PATH):
        print(f"\nWARNING: LoRA adapter not found at {LORA_ADAPTER_PATH}.")
        print("Please ensure lora_training/train_lora.py has completed successfully.")
        sys.exit(1)
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    base_model.config.use_cache = False
    
    lora_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    lora_model.eval()  # Set model to evaluation mode
    print("LoRA model loaded successfully.\n")

    # 4. Evaluation Loop
    results = []
    
    for i, question in enumerate(questions, start=1):
        print("=" * 60)
        print(f"QUESTION [{i}/{len(questions)}]:\n{question}")
        print("=" * 60)
        
        # Base Model
        base_answer = run_base_model(question)
        print("\n[BASE]")
        print(base_answer)
        
        # RAG System
        print("\n[RAG] (Retrieving context...)")
        rag_answer = run_rag_model(question, chunks, embedding_model, faiss_index)
        print("\n[RAG ANSWER]")
        print(rag_answer)
        
        # LoRA Model
        print("\n[LORA] (Generating...)")
        lora_answer = run_lora_model(question, lora_model, tokenizer)
        print("\n[LORA ANSWER]")
        print(lora_answer)
        
        print("\n" + "-" * 60)
        
        # Store results
        results.append({
            "question": question,
            "base_answer": base_answer,
            "rag_answer": rag_answer,
            "lora_answer": lora_answer
        })

    # 5. Save Results
    save_results(results)
    print("=" * 60)
    print("  Evaluation Complete  ")
    print("=" * 60)


if __name__ == "__main__":
    main()
