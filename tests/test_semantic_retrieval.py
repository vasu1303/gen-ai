
from src.embeddings import load_embedding_model
from src.image_retrieval import (
    build_image_vector_store,
    retrieve_relevant_images,
)


print("Loading embedding model...")
model = load_embedding_model()

print("\nBuilding image vector store...")
index, metadata = build_image_vector_store(model)

question = "Explain the Transformer architecture diagram"

print(f"\nQuestion: {question}")

images = retrieve_relevant_images(
    question=question,
    index=index,
    metadata=metadata,
    model=model,
    top_k=5,
)

print("\nSemantically retrieved images:")

for image in images:
    print(f"\nImage: {image['filename']}")
    print(f"PDF: {image['source']}")
    print(f"Page: {image['page']}")
    print(f"Similarity: {image['score']:.4f}")
