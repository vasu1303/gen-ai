
from pathlib import Path
import pymupdf


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"


def inspect_pdfs():
    """Inspect text, images, drawings, and tables in each PDF."""

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {PAPERS_DIR.resolve()}"
        )

    print("=" * 70)
    print("PDF CONTENT INSPECTION")
    print("=" * 70)
    print(f"PDF folder: {PAPERS_DIR.resolve()}")
    print(f"Number of PDF files: {len(pdf_files)}")

    if len(pdf_files) != 5:
        print(
            f"\nWARNING: Expected 5 PDFs, but found "
            f"{len(pdf_files)}."
        )

    overall_pages = 0
    overall_text_pages = 0
    overall_images = 0
    overall_drawings = 0
    overall_tables = 0

    for pdf_path in pdf_files:
        print("\n" + "=" * 70)
        print(f"FILE: {pdf_path.name}")
        print("=" * 70)

        with pymupdf.open(pdf_path) as pdf:
            print(f"Total pages: {len(pdf)}")

            text_pages = 0
            pages_with_images = 0
            pages_with_drawings = 0
            pages_with_tables = 0

            total_images = 0
            total_drawings = 0
            total_tables = 0

            for page_number, page in enumerate(pdf, start=1):

                # 1. Extractable text
                text = page.get_text("text").strip()

                if text:
                    text_pages += 1

                # 2. Embedded images
                images = page.get_images(full=True)
                image_count = len(images)

                total_images += image_count

                if image_count > 0:
                    pages_with_images += 1

                # 3. Vector drawings
                # These may include diagram elements, charts,
                # lines, shapes, and other graphical objects.
                drawings = page.get_drawings()
                drawing_count = len(drawings)

                total_drawings += drawing_count

                if drawing_count > 0:
                    pages_with_drawings += 1

                # 4. Tables, if supported by this PyMuPDF version
                table_count = 0

                try:
                    if hasattr(page, "find_tables"):
                        table_result = page.find_tables()
                        table_count = len(table_result.tables)
                except Exception as error:
                    print(
                        f"Table detection issue on page "
                        f"{page_number}: {error}"
                    )

                total_tables += table_count

                if table_count > 0:
                    pages_with_tables += 1

                # Page-by-page summary
                print(
                    f"Page {page_number:>3}: "
                    f"text={'Yes' if text else 'No':<3} | "
                    f"characters={len(text):>5} | "
                    f"images={image_count:>3} | "
                    f"drawings={drawing_count:>4} | "
                    f"tables={table_count:>2}"
                )

            # Summary for this PDF
            print("\n--- FILE SUMMARY ---")
            print(f"Pages: {len(pdf)}")
            print(f"Pages containing text: {text_pages}")
            print(f"Pages containing embedded images: {pages_with_images}")
            print(f"Total embedded image occurrences: {total_images}")
            print(f"Pages containing vector drawings: {pages_with_drawings}")
            print(f"Total vector drawing objects: {total_drawings}")
            print(f"Pages with detected tables: {pages_with_tables}")
            print(f"Total detected tables: {total_tables}")

            # Update overall totals
            overall_pages += len(pdf)
            overall_text_pages += text_pages
            overall_images += total_images
            overall_drawings += total_drawings
            overall_tables += total_tables

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"PDF files inspected: {len(pdf_files)}")
    print(f"Total pages: {overall_pages}")
    print(f"Pages containing text: {overall_text_pages}")
    print(f"Embedded image occurrences: {overall_images}")
    print(f"Vector drawing objects: {overall_drawings}")
    print(f"Tables detected: {overall_tables}")
    print("\nInspection completed.")


if __name__ == "__main__":
    inspect_pdfs()
