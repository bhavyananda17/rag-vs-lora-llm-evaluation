"""
generate_qa_dataset.py — Automatic QA dataset generator for LoRA fine-tuning.

This script reads domain-knowledge documents, chunks them, and uses the
local Ollama TinyLlama model to generate instruction / output QA pairs
for each chunk.  The resulting dataset is saved as JSON and can be
directly consumed by the LoRA training pipeline.

Workflow:
    documents → chunking (300 chars) → LLM QA generation → qa_dataset.json
"""

import sys
import os
import json
import re
import time
from typing import Dict, List, Tuple

import requests

# ── Ensure the project root is on sys.path ────────────────────────────
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag_pipeline.chunking import load_documents, chunk_text  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "tinyllama"
CHUNK_SIZE: int = 300
OVERLAP: int = 50
QA_PER_CHUNK: int = 2
DOCUMENTS_DIR: str = os.path.join(PROJECT_ROOT, "data", "documents")
OUTPUT_PATH: str = os.path.join(PROJECT_ROOT, "data", "qa_dataset.json")

# ── Prompt template ───────────────────────────────────────────────────
QA_PROMPT_TEMPLATE: str = """You are generating a dataset for training a question answering AI.

Based only on the text below, generate {n} question-answer pairs.

Text:
{chunk}

Return the result as valid JSON in this format:

[
  {{
    "instruction": "question here",
    "output": "answer here"
  }},
  {{
    "instruction": "question here",
    "output": "answer here"
  }}
]

Requirements:
- Questions must be answerable using only the provided text.
- Answers must stay faithful to the text.
- Do not invent information.
- Return ONLY the JSON array, no extra text."""


# ── Document loading and chunking ─────────────────────────────────────

def chunk_documents(
    directory: str = DOCUMENTS_DIR,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> List[str]:
    """Load documents and split them into chunks for QA generation.

    Uses a smaller chunk size (300 chars) than the RAG pipeline to
    produce more focused, single-topic chunks that yield higher
    quality question-answer pairs.

    Args:
        directory:  Path to the documents folder.
        chunk_size: Maximum characters per chunk.
        overlap:    Overlapping characters between consecutive chunks.

    Returns:
        A flat list of all chunks across every document.
    """
    documents = load_documents(directory)
    print(f"Loaded {len(documents)} documents")

    all_chunks: List[str] = []
    for _filename, content in documents:
        doc_chunks = chunk_text(content, chunk_size, overlap)
        all_chunks.extend(doc_chunks)

    print(f"Generated {len(all_chunks)} chunks (chunk_size={chunk_size}, overlap={overlap})")
    return all_chunks


# ── Ollama QA generation ──────────────────────────────────────────────

def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Send a prompt to the local Ollama server and return the response.

    Args:
        prompt: The full prompt string.
        model:  Name of the Ollama model.

    Returns:
        The model's generated text as a plain string.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload: Dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=None)
    except requests.ConnectionError:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Make sure the Ollama server is running."
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama returned status {response.status_code}: {response.text}"
        )

    return response.json().get("response", "")


def parse_qa_response(raw_response: str) -> List[Dict[str, str]]:
    """Parse the LLM response into a list of QA pair dictionaries.

    Attempts multiple strategies to extract valid JSON from the
    model's output, handling common formatting issues from small
    models like TinyLlama.

    Args:
        raw_response: The raw text returned by the LLM.

    Returns:
        A list of dicts with ``instruction`` and ``output`` keys.
        Returns an empty list if parsing fails entirely.
    """
    # Strategy 1: Try direct JSON parse
    try:
        pairs = json.loads(raw_response.strip())
        if isinstance(pairs, list):
            return [p for p in pairs if "instruction" in p and "output" in p]
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON array from surrounding text
    match = re.search(r'\[.*\]', raw_response, re.DOTALL)
    if match:
        try:
            pairs = json.loads(match.group())
            if isinstance(pairs, list):
                return [p for p in pairs if "instruction" in p and "output" in p]
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try to find individual JSON objects
    objects = re.findall(r'\{[^{}]+\}', raw_response)
    pairs = []
    for obj_str in objects:
        try:
            obj = json.loads(obj_str)
            if "instruction" in obj and "output" in obj:
                pairs.append(obj)
        except json.JSONDecodeError:
            continue

    return pairs


def generate_qa_pairs(
    chunks: List[str],
    qa_per_chunk: int = QA_PER_CHUNK,
) -> List[Dict[str, str]]:
    """Generate QA pairs from all chunks using the local Ollama model.

    Iterates over each chunk, sends a QA-generation prompt to the LLM,
    parses the JSON response, and collects all valid pairs.

    Args:
        chunks:       List of text chunks to generate QA from.
        qa_per_chunk: Number of QA pairs to request per chunk.

    Returns:
        A flat list of all generated QA pair dictionaries.
    """
    all_pairs: List[Dict[str, str]] = []
    failed_chunks: int = 0

    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        print(f"  [{idx}/{total}] Generating QA pairs...", end=" ", flush=True)

        prompt = QA_PROMPT_TEMPLATE.format(n=qa_per_chunk, chunk=chunk)

        start_time = time.time()
        try:
            raw = call_ollama(prompt)
            pairs = parse_qa_response(raw)
        except (ConnectionError, RuntimeError) as exc:
            print(f"ERROR: {exc}")
            failed_chunks += 1
            continue

        elapsed = time.time() - start_time

        if pairs:
            all_pairs.extend(pairs)
            print(f"got {len(pairs)} pairs ({elapsed:.1f}s)")
        else:
            failed_chunks += 1
            print(f"parse failed ({elapsed:.1f}s)")

    print(f"\nGeneration complete: {len(all_pairs)} QA pairs "
          f"from {total} chunks ({failed_chunks} chunks failed)")
    return all_pairs


# ── Dataset saving ────────────────────────────────────────────────────

def save_dataset(
    qa_pairs: List[Dict[str, str]],
    output_path: str = OUTPUT_PATH,
) -> None:
    """Save the QA dataset to a JSON file.

    Args:
        qa_pairs:    List of QA pair dictionaries.
        output_path: Destination file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(qa_pairs, fh, indent=2, ensure_ascii=False)

    print(f"Saved dataset to {output_path}")


# ── CLI entry-point ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  QA Dataset Generation for LoRA Fine-Tuning")
    print("=" * 60)
    print()

    # 1. Chunk documents
    chunks = chunk_documents()

    if not chunks:
        print("No chunks generated — check data/documents/ for .txt files.")
        sys.exit(1)

    # 2. Generate QA pairs
    print(f"\nGenerating {QA_PER_CHUNK} QA pairs per chunk "
          f"using Ollama ({OLLAMA_MODEL})...\n")
    qa_pairs = generate_qa_pairs(chunks)

    if not qa_pairs:
        print("No QA pairs were generated. Check Ollama connection.")
        sys.exit(1)

    # 3. Save dataset
    print()
    save_dataset(qa_pairs)

    # 4. Summary
    print()
    print("=" * 60)
    print(f"  Summary")
    print("=" * 60)
    print(f"  Documents loaded : {len(load_documents(DOCUMENTS_DIR))}")
    print(f"  Chunks generated : {len(chunks)}")
    print(f"  QA pairs created : {len(qa_pairs)}")
    print(f"  Output file      : {OUTPUT_PATH}")
    print("=" * 60)
