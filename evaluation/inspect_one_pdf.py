
from pathlib import Path
import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "data" / "papers" / "1706.03762.pdf"
OUTPUT_DIR = PROJECT_ROOT / "inspection_output"


def inspect_pdf():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(PDF_PATH) as pdf:
        print(f"PDF: {PDF_PATH.name}")
        print(f"Total pages: {len(pdf)}")

        for page_number, page in enumerate(pdf, start=1):
            images = page.get_images(full=True)
            drawings = page.get_drawings()

            # Inspect pages containing embedded images,
            # many vector drawings, or detected tables.
            try:
                table_count = len(page.find_tables().tables)
            except Exception:
                table_count = 0

            if images or len(drawings) > 20 or table_count > 0:
                print(
                    f"Page {page_number}: "
                    f"images={len(images)}, "
                    f"drawings={len(drawings)}, "
                    f"tables={table_count}"
                )

                # Render the page to an image for visual inspection.
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(1.5, 1.5),
                    alpha=False,
                )

                image_path = (
                    OUTPUT_DIR / f"page_{page_number}.png"
                )
                pixmap.save(image_path)

                print(f"  Saved preview: {image_path}")

    print("\nInspection completed.")


if __name__ == "__main__":
    inspect_pdf()
