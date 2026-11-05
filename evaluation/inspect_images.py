from pathlib import Path
from PIL import Image

IMAGE_DIR = Path("data/extracted_images")

for image_path in sorted(IMAGE_DIR.iterdir()):

    if not image_path.is_file():
        continue

    try:
        with Image.open(image_path) as image:
            width, height = image.size

            if width >= 200 and height >= 200:
                print(
                    f"{image_path.name} | "
                    f"Width: {width} | "
                    f"Height: {height}"
                )

    except Exception:
        continue