from pathlib import Path
from PIL import Image, ImageDraw
import math

IMAGE_DIR = Path("data/extracted_images")
OUTPUT_DIR = Path("data")

images = []

for path in sorted(IMAGE_DIR.iterdir()):
    try:
        image = Image.open(path).convert("RGB")

        width, height = image.size

        if width >= 200 and height >= 200:
            image.thumbnail((220, 160))
            images.append((path.name, image.copy()))

    except Exception:
        continue


# Contact sheet settings
columns = 3
cell_width = 260
cell_height = 210

rows = math.ceil(len(images) / columns)

sheet = Image.new(
    "RGB",
    (columns * cell_width, rows * cell_height),
    "white"
)

draw = ImageDraw.Draw(sheet)

for index, (name, image) in enumerate(images):

    x = (index % columns) * cell_width
    y = (index // columns) * cell_height

    image_x = x + (cell_width - image.width) // 2

    sheet.paste(image, (image_x, y + 5))

    draw.text(
        (x + 5, y + 170),
        name,
        fill="black"
    )


output_path = OUTPUT_DIR / "image_contact_sheet.jpg"

sheet.save(output_path)

print(f"Contact sheet created: {output_path}")
print(f"Total images included: {len(images)}")