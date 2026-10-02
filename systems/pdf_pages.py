import os
import logging
from dataclasses import dataclass
from pypdf import PdfReader

@dataclass
class PageText:
    source: str
    page: int
    text: str

def read_pdf_pages(files: list[str]) -> list[PageText]:
    pages = []
    for filepath in files:
        if not os.path.isfile(filepath):
            logging.warning(f"File not found: {filepath}")
            continue

        filename = os.path.basename(filepath)
        try:
            reader = PdfReader(filepath)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text or not text.strip():
                    logging.warning(f"Extracted empty text from {filename} page {i + 1}")
                    text = ""
                pages.append(PageText(source=filename, page=i + 1, text=text))
        except Exception as e:
            logging.error(f"Failed to read PDF {filepath}: {e}")

    return pages

def format_pages(pages: list[PageText]) -> str:
    formatted = []
    for p in pages:
        formatted.append(f"[[{p.source} p.{p.page}]]\n{p.text}")
    return "\n\n".join(formatted)
