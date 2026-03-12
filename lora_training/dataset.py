"""
dataset.py — QA Dataset loading and formatting for LoRA fine-tuning.
"""
import json
from datasets import Dataset

def format_prompts(example):
    """
    Format single QA pair using an instruction style template.
    Creates a new 'text' field that will be used for CausalLM training.
    """
    instruction = example.get("instruction", "")
    output = example.get("output", "")
    
    text = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )
    example["text"] = text
    return example

def load_dataset(dataset_path: str) -> Dataset:
    """
    Load the JSON dataset from the given path, 
    convert it into a HuggingFace Dataset, 
    and format the prompts.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    dataset = Dataset.from_list(data)
    
    # Map the formatting function over the dataset
    formatted_dataset = dataset.map(format_prompts)
    
    return formatted_dataset
