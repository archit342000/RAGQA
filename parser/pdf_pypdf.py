"""PDF extraction using pypdf."""

# This module houses the lightweight PDF path. It avoids heavy dependencies
# and works well for text-heavy documents; we fall back to unstructured for
# more complex layouts.

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


class PDFEncryptedError(RuntimeError):
    """Raised when the PDF is encrypted and cannot be processed."""


def extract_pdf_with_pypdf(file_path: str, *, max_pages: int | None = None) -> Tuple[List[str], Dict[str, str]]:
    """Extract raw text from each page using pypdf.

    Parameters
    ----------
    file_path:
        Location of the PDF to parse.
    max_pages:
        Hard upper bound that protects against extremely large uploads.

    Returns
    -------
    tuple[list[str], dict[str, str]]
        A list of raw page strings and a small metadata dictionary containing
        timing information that feeds into the higher-level driver metrics.
    """

    start = time.perf_counter()
    pdf_path = Path(file_path)

    try:
        reader = PdfReader(str(pdf_path))
    except PdfReadError as exc:  # pragma: no cover - exercised in integration
        raise RuntimeError(f"Unable to read PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # pragma: no cover - encryption uncommon in tests
            raise PDFEncryptedError("Encrypted PDF cannot be parsed without a password") from exc
        # Many PDFs have empty passwords; log the attempt so operators understand
        # why downstream pages may be empty.
        logger.info("PDF %s was encrypted; attempted empty password decryption.", pdf_path.name)

    pages_text: List[str] = []
    truncated = False

    for index, page in enumerate(reader.pages):
        if max_pages is not None and index >= max_pages:
            truncated = True
            break
        try:
            text = page.extract_text() or ""
        except ValueError:  # pragma: no cover - rare
            text = ""
        pages_text.append(text)

    elapsed = time.perf_counter() - start
    meta: Dict[str, str] = {
        "elapsed": f"{elapsed:.4f}",
        "page_count": str(len(pages_text)),
        "truncated": str(truncated),
    }
    logger.debug(
        "pypdf extraction for %s yielded %d pages in %.3fs (truncated=%s)",
        pdf_path.name,
        len(pages_text),
        elapsed,
        truncated,
    )
    return pages_text, meta
