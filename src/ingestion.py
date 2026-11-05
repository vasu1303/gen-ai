
from pathlib import Path #for file handling
import pymupdf #to extract data 
from langchain_text_splitters import RecursiveCharacterTextSplitter  #for chunking


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
EXTRACTED_IMAGES_DIR = PROJECT_ROOT / "data" / "extracted_images"

#extract texts and tables and images
def extract_text_from_pdfs():
    """
    Extract text and tables from all PDFs.

    Save embedded images separately.

    Each extracted document contains:
    - page_content: extracted text
    - source: PDF filename
    - page: page number
    """

    documents = []

    if not PAPERS_DIR.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {PAPERS_DIR}"
        )

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf")) #glob("*.pdf") returns a list of all pdf files in the PAPERS_DIR

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PAPERS_DIR}"
        )

    EXTRACTED_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        with pymupdf.open(pdf_path) as pdf:

            for page_number, page in enumerate(pdf, start=1):

                # Extract normal page text
                page_text = page.get_text("text").strip()

                content_parts = []

                if page_text:
                    content_parts.append(page_text)

                # ------------------------------------------
                # Extract tables
                # ------------------------------------------

                try:
                    tables = page.find_tables().tables #.tables is used to get the table data from the page

                    for table_number, table in enumerate(
                        tables,
                        start=1,
                    ):
                        table_data = table.extract()

                        if not table_data:
                            continue

                        table_lines = []

                        for row in table_data:
                            cells = []

                            for cell in row:
                                if cell is not None:
                                    cells.append(str(cell).strip())
                                else:
                                    cells.append("")

                            table_lines.append("\t".join(cells))

                        table_text = "\n".join(table_lines).strip()

                        if table_text:
                            content_parts.append(
                                f"\nTable {table_number}:\n"
                                f"{table_text}"
                            )

                except Exception as error:
                    print(
                        f"  Table extraction warning on page "
                        f"{page_number}: {error}"
                    )

                # ------------------------------------------
                # Extract embedded images
                # ------------------------------------------

                images = page.get_images(full=True)

                for image_number, image_info in enumerate(
                    images,
                    start=1,
                ):
                    image_xref = image_info[0]

                    try:
                        image_data = pdf.extract_image(image_xref)

                        image_bytes = image_data["image"]
                        image_extension = image_data["ext"]

                        image_filename = (
                            f"{pdf_path.stem}_page_{page_number}"
                            f"_image_{image_number}."
                            f"{image_extension}"
                        )

                        image_path = (
                            EXTRACTED_IMAGES_DIR / image_filename
                        )

                        image_path.write_bytes(image_bytes)

                    except Exception as error:
                        print(
                            f"  Image extraction warning on page "
                            f"{page_number}: {error}"
                        )

                # ------------------------------------------
                # Store page content and metadata
                # ------------------------------------------

                combined_text = "\n\n".join(content_parts).strip()

                if combined_text:
                    documents.append(
                        {
                            "page_content": combined_text,
                            "source": pdf_path.name,
                            "page": page_number,
                        }
                    )

        print(f"Finished: {pdf_path.name}")

    print("\nText and table extraction completed.")
    print(f"Pages with extracted content: {len(documents)}")
    print(f"Extracted images directory: {EXTRACTED_IMAGES_DIR}")

    return documents


# --------------------------------------------------
# Split extracted content into chunks
# --------------------------------------------------

def chunk_documents(
    documents,
    chunk_size=900,
    chunk_overlap=150,
):
    """
    Split extracted page content into smaller chunks.

    Preserve source PDF and page number in each chunk.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for document in documents:

        page_content = document["page_content"]

        split_texts = text_splitter.split_text(page_content)

        for chunk_number, chunk_text in enumerate(
            split_texts,
            start=1,
        ):
            chunks.append(
                {
                    "page_content": chunk_text,
                    "source": document["source"],
                    "page": document["page"],
                    "chunk": chunk_number,
                }
            )

    print(f"\nTotal chunks created: {len(chunks)}")

    return chunks
