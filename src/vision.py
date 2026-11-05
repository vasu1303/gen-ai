
from PIL import Image
import base64
from io import BytesIO
from pathlib import Path
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3-vl:4b-instruct"


def encode_image(image_path: str) -> str:
    """
    Resize an image and convert it into Base64 format.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as image:

        # Resize while maintaining the aspect ratio
        image.thumbnail((1024, 1024))

        # Convert to RGB for consistent image encoding
        image = image.convert("RGB")

        # Save resized image in memory
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)

        # Convert image bytes to Base64
        encoded_image = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

    return encoded_image


def analyze_image(
    image_path: str,
    question: str,
    conversation_history: str = "",
) -> str:
    """
    Generate an answer based on the provided image
    and previous conversation history.
    """

    encoded_image = encode_image(image_path)

    prompt = f"""
You are an assistant that understands research paper diagrams,
graphs and figures.

You are having a conversation with the user.

Previous conversation:
{conversation_history if conversation_history else "No previous conversation."}

Current question:
{question}

Instructions:

1. Analyze the provided image carefully.
2. Use the previous conversation to understand follow-up questions.
3. If the user refers to something discussed earlier,
   use the conversation history to understand the reference.
4. Explain the relevant visual information clearly.
5. Do not invent details that are not visible in the image.
6. If the question is a follow-up, answer in the context
   of the previous discussion.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [encoded_image],
        "stream": False,
        "options": {
            "num_predict": 700,
            "num_ctx": 4096
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    return response.json()["response"].strip()
