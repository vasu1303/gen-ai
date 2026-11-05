
import json
from pathlib import Path

import faiss
import numpy as np

from src.embeddings import load_embedding_model


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "papers.index"
METADATA_PATH = VECTORSTORE_DIR / "chunks_metadata.json"


# --------------------------------------------------
# Load the vector store
# --------------------------------------------------

def load_vector_store():
    """Load the saved FAISS index and chunk metadata."""

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}. "
            "Run 'python -m src.vector_store' first."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Chunk metadata not found: {METADATA_PATH}"
        )

    # Load the FAISS index
    index = faiss.read_index(str(INDEX_PATH))

    # Load the corresponding metadata
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    # Verify index and metadata consistency
    if index.ntotal != len(metadata):
        raise ValueError(
            f"Mismatch: {index.ntotal} vectors but "
            f"{len(metadata)} metadata entries."
        )

    print(f"FAISS index loaded: {index.ntotal} vectors")
    print(f"Metadata loaded: {len(metadata)} chunks")

    return index, metadata


# --------------------------------------------------
# Retrieve relevant chunks
# --------------------------------------------------

def retrieve_relevant_chunks(
    question,
    index,
    metadata,
    model,
    top_k=3,
):
    """Retrieve the most relevant chunks for a question."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if index.ntotal == 0:
        return []

    # Add the recommended BGE instruction to the query
    query = (
        "Represent this sentence for searching relevant passages: "
        + question.strip()
    )

    # Convert the question into a normalized embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # FAISS expects a 2D float32 array
    query_embedding = np.asarray(
        [query_embedding],
        dtype=np.float32,
    )

    # Retrieve no more results than the index contains
    k = min(top_k, index.ntotal)

    scores, indices = index.search(query_embedding, k)

    results = []

    # Map retrieved vectors back to their original text
    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk = metadata[int(idx)]

        results.append(
            {
                "text": chunk["text"],
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "chunk": chunk["metadata"]["chunk"],
                "score": float(score),
            }
        )

    return results


# --------------------------------------------------
# Test retrieval
# --------------------------------------------------

if __name__ == "__main__":

    # Load the saved vector store
    index, metadata = load_vector_store()

    # Load the embedding model
    model = load_embedding_model()

    # Test question
    question = "How does the Transformer architecture use attention?"

    # Retrieve the three most relevant chunks
    results = retrieve_relevant_chunks(
        question=question,
        index=index,
        metadata=metadata,
        model=model,
        top_k=3,
    )

    print("\nQuestion:", question)
    print("\n--- Retrieved Chunks ---")

    for number, result in enumerate(results, start=1):
        print(f"\nResult {number}")
        print(f"Similarity score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Chunk: {result['chunk']}")
        print(f"Text: {result['text'][:500]}")
        print("-" * 60)
