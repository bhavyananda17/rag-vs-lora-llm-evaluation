"""
retrieval.py — Full RAG pipeline with semantic retrieval and LLM generation.

This module ties together the chunking, embedding, and FAISS retrieval
stages, builds a context-augmented prompt from the top retrieved chunks,
and sends it to a locally running Ollama model for answer generation.

Workflow:
    query → FAISS retrieval → context assembly → prompt → Ollama LLM → answer
"""

import sys
import os
import json
from typing import Dict, List, Optional, Tuple

import requests

# ── Ensure the project root is on sys.path ────────────────────────────
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag_pipeline.chunking import chunk_documents          # noqa: E402
from rag_pipeline.embeddings import (                      # noqa: E402
    build_faiss_index,
    create_embeddings,
    load_model,
    search_index,
)

# ── Configuration ─────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "tinyllama"
TOP_K: int = 5


# ── Prompt construction ──────────────────────────────────────────────

def format_context(context_chunks: List[str]) -> str:
    """Format retrieved chunks as a numbered context block.

    Each chunk is prefixed with a bracketed index for clear
    attribution in the LLM prompt.

    Args:
        context_chunks: Ordered list of retrieved text passages.

    Returns:
        A single string with numbered, separated chunks.
    """
    context = ""
    for i, chunk in enumerate(context_chunks):
        context += f"[{i + 1}]\n{chunk}\n\n"
    return context.rstrip()


def build_prompt(query: str, context_chunks: List[str]) -> str:
    """Build a strict, grounding-focused prompt for the LLM.

    The prompt uses an instruction-based template that explicitly
    forbids the model from using outside knowledge or guessing.
    This significantly reduces hallucination with smaller models
    like TinyLlama.

    Args:
        query:          The user's natural-language question.
        context_chunks: Retrieved text passages that are relevant to
                        the query.

    Returns:
        A fully formatted prompt string ready to send to the LLM.
    """
    context_block = format_context(context_chunks)

    prompt = (
        "You are a strict question answering assistant.\n\n"
        "Answer the question ONLY using the information in the "
        "provided context.\n\n"
        "Rules:\n"
        "- Do not use outside knowledge\n"
        "- Do not guess\n"
        "- If the answer cannot be found in the context, reply "
        "exactly with: "
        "\"I don't know based on the provided documents.\"\n\n"
        f"Context:\n\n{context_block}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:\n"
    )
    return prompt


# ── Ollama integration ────────────────────────────────────────────────

def query_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    """Send a prompt to the Ollama REST API and return the generated text.

    Uses the ``/api/generate`` endpoint with streaming disabled for
    simplicity.

    Args:
        prompt:   The full prompt string (including context).
        model:    Name of the Ollama model to use (must be pulled locally).
        base_url: Base URL of the running Ollama server.

    Returns:
        The model's generated response as a plain string.

    Raises:
        ConnectionError: If the Ollama server is unreachable.
        RuntimeError:    If the API returns a non-200 status.
    """
    url = f"{base_url}/api/generate"
    payload: Dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=None)
    except requests.ConnectionError:
        raise ConnectionError(
            f"Cannot reach Ollama at {base_url}. "
            "Make sure the Ollama server is running (ollama serve)."
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama returned status {response.status_code}: "
            f"{response.text}"
        )

    result: Dict = response.json()
    return result.get("response", "")


# ── Full RAG pipeline ────────────────────────────────────────────────

def initialize_rag_pipeline() -> Tuple:
    """Set up all components needed for the RAG pipeline.

    This function runs the complete preparation workflow:
      1. Chunk the source documents.
      2. Load the SentenceTransformer embedding model.
      3. Embed all chunks.
      4. Build the FAISS index.

    Returns:
        A tuple of ``(chunks, embedding_model, faiss_index)`` ready for
        repeated queries.
    """
    print("=" * 60)
    print("  RAG Pipeline Initialization")
    print("=" * 60)

    # 1. Chunk documents
    chunks = chunk_documents()
    if not chunks:
        print("No chunks generated — check data/documents/ for .txt files.")
        sys.exit(1)

    # 2. Load embedding model
    embedding_model = load_model()

    # 3. Create embeddings
    embeddings = create_embeddings(chunks, embedding_model)

    # 4. Build FAISS index
    faiss_index = build_faiss_index(embeddings)

    print("=" * 60)
    print("  Pipeline ready — awaiting queries")
    print("=" * 60)

    return chunks, embedding_model, faiss_index


def rag_query(
    query: str,
    chunks: List[str],
    embedding_model,
    faiss_index,
    top_k: int = TOP_K,
    ollama_model: str = OLLAMA_MODEL,
) -> Dict[str, object]:
    """Execute a single end-to-end RAG query.

    Steps:
      1. Retrieve the *top_k* most relevant chunks from the FAISS index.
      2. Assemble a strict, grounding-focused prompt with numbered context.
      3. Send the prompt to Ollama for generation.

    Args:
        query:           The user's question in natural language.
        chunks:          All text chunks (aligned with the FAISS index).
        embedding_model: The SentenceTransformer model for query embedding.
        faiss_index:     The populated FAISS index.
        top_k:           Number of context chunks to retrieve.
        ollama_model:    Ollama model name.

    Returns:
        A dictionary with keys:
            - ``question``         : the original question
            - ``retrieved_chunks`` : list of retrieved chunk strings
            - ``answer``           : the LLM-generated answer
    """
    # ── Step 1: Retrieve ──────────────────────────────────────────────
    results: List[Tuple[str, float]] = search_index(
        query, faiss_index, chunks, embedding_model, top_k=top_k,
    )
    retrieved_chunks = [chunk for chunk, _dist in results]

    print(f"\nRetrieved {len(retrieved_chunks)} relevant chunks:\n")
    for rank, (chunk, dist) in enumerate(results, start=1):
        preview = chunk[:60].replace("\n", " ")
        print(f"  {rank}. [dist={dist:.4f}] {preview}...")

    # ── Step 2: Build prompt ──────────────────────────────────────────
    prompt = build_prompt(query, retrieved_chunks)

    # ── Step 3: Generate answer via Ollama ────────────────────────────
    print(f"\nSending prompt to Ollama ({ollama_model})...")
    answer = query_ollama(prompt, model=ollama_model)

    return {
        "question": query,
        "retrieved_chunks": retrieved_chunks,
        "answer": answer,
    }


# ── CLI entry-point ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Initialize the full pipeline
    chunks, embedding_model, faiss_index = initialize_rag_pipeline()

    # Test query — minimal, focused on benchmarking
    test_queries = [
        "What is the attention mechanism in transformers?",
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"  QUERY: {query}")
        print("=" * 60)

        try:
            result = rag_query(
                query=query,
                chunks=chunks,
                embedding_model=embedding_model,
                faiss_index=faiss_index,
            )
            print(f"\n{'─' * 60}")
            print("  ANSWER")
            print(f"{'─' * 60}")
            print(result["answer"])

        except ConnectionError as exc:
            print(f"\n⚠  {exc}")
            print("Skipping remaining queries.")
            break

        except RuntimeError as exc:
            print(f"\n⚠  Ollama error: {exc}")
            print("Skipping remaining queries.")
            break

    print("\n" + "=" * 60)
    print("  RAG pipeline demo complete")
    print("=" * 60)
