"""
chunking.py — Document loader and text chunker for the RAG pipeline.

This module reads .txt documents from the data/documents/ directory,
splits them into overlapping character-level chunks, and returns a
flat list of chunk strings ready for embedding and retrieval.
"""

import os
from typing import List, Tuple


# ── Default chunk parameters ──────────────────────────────────────────
CHUNK_SIZE: int = 400   # characters per chunk
OVERLAP: int = 50       # overlapping characters between consecutive chunks
DOCUMENTS_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "documents",
)


def load_documents(directory: str = DOCUMENTS_DIR) -> List[Tuple[str, str]]:
    """Load every .txt file from *directory*.

    Args:
        directory: Path to the folder that contains the raw text files.

    Returns:
        A list of (filename, content) tuples sorted by filename.
    """
    documents: List[Tuple[str, str]] = []

    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as fh:
                content = fh.read()
            documents.append((filename, content))

    return documents


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> List[str]:
    """Split *text* into overlapping chunks of roughly *chunk_size* characters.

    The function walks through the text with a sliding window.  Each
    window advances by ``chunk_size - overlap`` characters so that
    consecutive chunks share *overlap* characters of context.

    Args:
        text:       The source text to split.
        chunk_size: Maximum number of characters in each chunk.
        overlap:    Number of characters shared between adjacent chunks.

    Returns:
        A list of chunk strings.
    """
    if not text:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def chunk_documents(
    directory: str = DOCUMENTS_DIR,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> List[str]:
    """Load documents and split every one into overlapping chunks.

    This is the main entry-point that ties loading and chunking together.

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

    print(f"Generated {len(all_chunks)} chunks")
    return all_chunks


# ── CLI entry-point ───────────────────────────────────────────────────
if __name__ == "__main__":
    chunks = chunk_documents()

    if chunks:
        print(f"\nExample chunk:\n{chunks[0]}")
