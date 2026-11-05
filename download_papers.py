
from pathlib import Path
import requests

# The five PDF URLs specified in the assignment
PAPERS = {
    "1706.03762.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "1810.04805.pdf": "https://arxiv.org/pdf/1810.04805.pdf",
    "2005.14165.pdf": "https://arxiv.org/pdf/2005.14165.pdf",
    "1907.11692.pdf": "https://arxiv.org/pdf/1907.11692.pdf",
    "1910.10683.pdf": "https://arxiv.org/pdf/1910.10683.pdf",
}

def download_papers():
    # Create the destination folder if it does not exist
    papers_dir = Path("data") / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in PAPERS.items():
        file_path = papers_dir / filename

        try:
            print(f"\nDownloading {filename}...")

            response = requests.get(url, timeout=90)
            response.raise_for_status()

            # Check that the response is actually a PDF
            if not response.content.startswith(b"%PDF"):
                raise ValueError("The downloaded file is not a valid PDF.")

            file_path.write_bytes(response.content)

            print(f"Saved: {file_path}")
            print(f"Size: {file_path.stat().st_size:,} bytes")

        except (requests.RequestException, ValueError) as error:
            print(f"Failed to download {filename}: {error}")

if __name__ == "__main__":
    download_papers()
