"""OCR helpers used by the parser when text extraction fails."""
from __future__ import annotations

import time
from typing import Optional

from ._fitz import fitz

try:
    import pytesseract
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - optional dependency for environments lacking tesseract
    pytesseract = None  # type: ignore
    Image = None  # type: ignore

from .logging_utils import log_event


class OCRTimeout(Exception):
    """Raised when OCR exceeds the allotted time budget."""


def perform_page_ocr(
    pdf_path: str,
    page_index: int,
    time_budget_s: float,
    start_time: float,
    dpi: int = 150,
) -> Optional[str]:
    """Perform OCR on a single page using pytesseract as a lightweight fallback."""

    if pytesseract is None or Image is None:
        log_event("ocr_skipped", page=page_index, reason="pytesseract_missing")
        return None

    elapsed = time.time() - start_time
    if elapsed >= time_budget_s:
        raise OCRTimeout(f"Time budget exhausted before OCR on page {page_index}")

    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=dpi)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_start = time.time()
        text = pytesseract.image_to_string(image)
        log_event("ocr_performed", page=page_index, duration=time.time() - ocr_start)
        return text
    finally:
        doc.close()
