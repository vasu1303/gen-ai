
import json
from pathlib import Path

import faiss
import numpy as np

from src.embeddings import load_embedding_model
from src.ingestion import extract_text_from_pdfs, chunk_documents


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "papers.index"
METADATA_PATH = VECTORSTORE_DIR / "chunks_metadata.json"


# --------------------------------------------------
# Create the vector store
# --------------------------------------------------

def create_vector_store():
    """Generate embeddings and save them in a FAISS index."""

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Step 1: Extract text and create chunks
    print("\nStep 1: Extracting PDF content...")

    documents = extract_text_from_pdfs()
    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were generated.")

    print(f"\nTotal chunks: {len(chunks)}")

    # Step 2: Load the embedding model
    print("\nStep 2: Loading embedding model...")

    model = load_embedding_model()

    # Step 3: Prepare chunk text
    texts = [
        chunk["page_content"]
        for chunk in chunks
    ]

    # Step 4: Generate embeddings
    print("\nStep 3: Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    print("Embedding matrix shape:", embeddings.shape)

    # Step 5: Create the FAISS index
    print("\nStep 4: Creating FAISS index...")

    dimension = embeddings.shape[1]

    # Normalized vectors + inner product = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # Step 6: Save the index
    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    # Step 7: Save text and metadata
    metadata = []

    for chunk in chunks:
        item = {
            "text": chunk["page_content"],
            "metadata": {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
            },
        }

        metadata.append(item)

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # Step 8: Display results
    print("\nVector store created successfully!")

    print(f"FAISS index saved to: {INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(f"Stored vectors: {index.ntotal}")
    print(f"Embedding dimension: {dimension}")

    return index, metadata


if __name__ == "__main__":
    create_vector_store()
