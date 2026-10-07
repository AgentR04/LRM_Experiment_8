"""Shared PDF -> text extraction helper (pypdf)."""
from pathlib import Path

from pypdf import PdfReader


def pdf_to_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        try:
            txt = page.extract_text() or ""
        except Exception as e:  # malformed page - keep going
            txt = f"[page {i} extraction error: {e}]"
        if txt.strip():
            pages.append(txt)
    return "\n\n".join(pages)
