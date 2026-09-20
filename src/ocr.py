from pypdf import PdfReader
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a digital PDF using pypdf.

    Returns the concatenated text of all pages.
    For scanned PDFs (image-only), this returns an empty or near-empty string.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        pages.append(page_text)

    return "\n".join(pages).strip()


def is_probably_scanned(text: str, min_chars: int = 50) -> bool:
    """
    Heuristic: if extracted text is very short, the PDF is probably scanned.
    Real digital PDFs produce hundreds or thousands of characters per page.
    """
    return len(text.strip()) < min_chars