
import json
from pathlib import Path

import faiss
import numpy as np


METADATA_PATH = Path("data/image_metadata.json")

VECTORSTORE_DIR = Path("vectorstore")
IMAGE_INDEX_PATH = VECTORSTORE_DIR / "images.index"
IMAGE_METADATA_PATH = VECTORSTORE_DIR / "image_chunks_metadata.json"


def load_image_metadata():
    """Load metadata of all extracted images."""

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Image metadata not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def build_image_vector_store(model):
    """
    Generate embeddings for image-related PDF page text
    and store them in a separate FAISS index.
    """

    print("Building image vector store...")

    metadata = load_image_metadata()

    # Keep only images having associated page text
    image_metadata = [
        image for image in metadata
        if image.get("page_text", "").strip()
    ]

    if not image_metadata:
        raise ValueError("No image metadata with page text found.")

    # Extract page text for embedding
    texts = [
        image["page_text"]
        for image in image_metadata
    ]

    # Generate normalized embeddings
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # Create FAISS index using inner product
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save FAISS index and corresponding metadata
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(IMAGE_INDEX_PATH))

    with open(IMAGE_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(image_metadata, file, indent=4)

    print(f"Image FAISS index created: {index.ntotal} vectors")

    return index, image_metadata


def load_image_vector_store():
    """Load the saved image FAISS index and metadata."""

    if not IMAGE_INDEX_PATH.exists():
        raise FileNotFoundError(
            "Image FAISS index not found. Build it first."
        )

    if not IMAGE_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Image vector metadata not found. Build it first."
        )

    index = faiss.read_index(str(IMAGE_INDEX_PATH))

    with open(IMAGE_METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if index.ntotal != len(metadata):
        raise ValueError(
            "Image index and metadata are not aligned."
        )

    print(f"Image FAISS index loaded: {index.ntotal} vectors")

    return index, metadata


def retrieve_relevant_images(
    question: str,
    index,
    metadata,
    model,
    top_k: int = 3,
):
    """
    Retrieve images using semantic similarity
    between the question and associated PDF page text.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Convert question into an embedding
    query = (
        "Represent this sentence for searching relevant passages: "
        + question.strip()
    )

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    # Search for semantically similar page text
    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal),
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        image = metadata[int(idx)]

        results.append({
            **image,
            "score": float(score),
        })

    return results
