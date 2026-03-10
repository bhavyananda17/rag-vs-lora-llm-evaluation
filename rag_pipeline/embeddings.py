"""
embeddings.py — Embedding generator and FAISS vector index for the RAG pipeline.

This module takes chunked text produced by rag_pipeline.chunking, generates
dense vector embeddings using a SentenceTransformer model, builds a FAISS
index over those embeddings, and exposes a semantic search function for
query-time retrieval.

Workflow:
    documents → chunking → embeddings → FAISS index → semantic retrieval
"""

import sys
import os
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Ensure the project root is on sys.path so rag_pipeline is importable ──
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag_pipeline.chunking import chunk_documents  # noqa: E402

# ── Default model identifier ─────────────────────────────────────────
MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load and return a SentenceTransformer embedding model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A SentenceTransformer model instance ready for encoding.
    """
    model = SentenceTransformer(model_name)
    return model


def create_embeddings(
    chunks: List[str],
    model: SentenceTransformer | None = None,
) -> np.ndarray:
    """Generate dense vector embeddings for a list of text chunks.

    Args:
        chunks: List of text strings to embed.
        model:  A pre-loaded SentenceTransformer model.  If *None*, the
                default model is loaded automatically.

    Returns:
        A NumPy array of shape ``(len(chunks), embedding_dim)`` with
        float32 embeddings.
    """
    if model is None:
        model = load_model()

    embeddings: np.ndarray = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)

    print(f"Created embeddings for {len(chunks)} chunks "
          f"(dimension: {embeddings.shape[1]})")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS flat L2 index from a matrix of embeddings.

    Uses an exact (brute-force) L2 distance index, which is ideal for
    moderate-sized corpora where recall matters more than speed.

    Args:
        embeddings: Float32 array of shape ``(n, dim)``.

    Returns:
        A populated FAISS index containing all embedding vectors.
    """
    dimension: int = embeddings.shape[1]
    index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors "
          f"(dimension: {dimension})")
    return index


def search_index(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: List[str],
    model: SentenceTransformer,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Retrieve the *top_k* most similar chunks for a natural-language query.

    The query is embedded with the same model used to build the index,
    then an L2 nearest-neighbour search is performed against the FAISS
    index.

    Args:
        query:  The user's search query in plain text.
        index:  A populated FAISS index.
        chunks: The original text chunks (aligned with the index rows).
        model:  The SentenceTransformer model used for embedding.
        top_k:  Number of results to return.

    Returns:
        A list of ``(chunk_text, distance)`` tuples sorted by relevance
        (smallest distance first).
    """
    query_embedding: np.ndarray = model.encode(
        [query],
        convert_to_numpy=True,
    ).astype(np.float32)

    distances, indices = index.search(query_embedding, top_k)

    results: List[Tuple[str, float]] = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        results.append((chunks[idx], float(dist)))

    return results


# ── CLI entry-point ───────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Chunk documents
    chunks = chunk_documents()

    if not chunks:
        print("No chunks generated — check data/documents/ for .txt files.")
        sys.exit(1)

    # 2. Load model & create embeddings
    model = load_model()
    embeddings = create_embeddings(chunks, model)

    # 3. Build FAISS index
    index = build_faiss_index(embeddings)

    # 4. Run a test query
    test_query = "How does the attention mechanism work in transformers?"
    print(f"\nTest query: \"{test_query}\"\n")
    print("Top-3 search results:")
    print("-" * 60)

    results = search_index(test_query, index, chunks, model, top_k=3)
    for rank, (chunk, distance) in enumerate(results, start=1):
        preview = chunk[:150].replace("\n", " ")
        print(f"  {rank}. [dist={distance:.4f}] {preview}...")
        print()
