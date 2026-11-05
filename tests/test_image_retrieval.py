from src.image_retrieval import retrieve_relevant_images


question = "Explain the Transformer architecture and attention mechanism"

images = retrieve_relevant_images(question)

print("\nRelevant images:")

for image in images:
    print(f"\nImage: {image['filename']}")
    print(f"PDF: {image['source']}")
    print(f"Page: {image['page']}")
    print(f"Score: {image['score']}")
    print(f"Matched words: {image['matched_words']}")