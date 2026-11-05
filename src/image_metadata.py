
import json
import re
from pathlib import Path

import pymupdf
from PIL import Image


IMAGE_DIR = Path("data/extracted_images")
PDF_DIR = Path("data/papers")
OUTPUT_PATH = Path("data/image_metadata.json")


def create_image_metadata():

    image_metadata = []

    # Cache PDF page text to avoid reopening the same PDF repeatedly
    pdf_page_text = {}

    for image_path in sorted(IMAGE_DIR.iterdir()):

        if not image_path.is_file():
            continue

        match = re.match(
            r"(.+)_page_(\d+)_image_(\d+)",
            image_path.stem
        )

        if not match:
            continue

        source = match.group(1) + ".pdf"
        page = int(match.group(2))
        image_number = int(match.group(3))

        try:
            with Image.open(image_path) as image:

                width, height = image.size

                if width < 200 or height < 200:
                    continue

            # Extract text from the corresponding PDF page
            if source not in pdf_page_text:

                pdf_path = PDF_DIR / source

                if not pdf_path.exists():
                    print(f"PDF not found: {pdf_path}")
                    continue

                pdf_document = pymupdf.open(pdf_path)

                pdf_page_text[source] = [
                    pdf_document[index].get_text("text")
                    for index in range(len(pdf_document))
                ]

                pdf_document.close()

            page_text = ""

            if page <= len(pdf_page_text[source]):
                page_text = pdf_page_text[source][page - 1]

            image_metadata.append({
                "filename": image_path.name,
                "source": source,
                "page": page,
                "image_number": image_number,
                "width": width,
                "height": height,
                "image_path": str(image_path),
                "page_text": page_text
            })

        except Exception as error:
            print(f"Could not process {image_path.name}: {error}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(image_metadata, file, indent=4)

    print(f"Image metadata created: {OUTPUT_PATH}")
    print(f"Total images indexed: {len(image_metadata)}")


if __name__ == "__main__":
    create_image_metadata()