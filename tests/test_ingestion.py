
from src.ingestion import (
    extract_text_from_pdfs,
    chunk_documents,
)


def main():
    print("=" * 60)
    print("TESTING PDF INGESTION")
    print("=" * 60)

    # Extract text and tables
    documents = extract_text_from_pdfs()

    print("\nFirst extracted document:")
    print("-" * 60)

    if documents:
        print("Source:", documents[0]["source"])
        print("Page:", documents[0]["page"])
        print("Content preview:")
        print(documents[0]["page_content"][:500])

    # Create chunks
    chunks = chunk_documents(documents)

    print("\nFirst chunk:")
    print("-" * 60)

    if chunks:
        print("Source:", chunks[0]["source"])
        print("Page:", chunks[0]["page"])
        print("Chunk number:", chunks[0]["chunk"])
        print(chunks[0]["page_content"][:500])

    print("\n" + "=" * 60)
    print("INGESTION TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
