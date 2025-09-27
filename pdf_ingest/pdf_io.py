"""Low-level PDF extraction utilities built on PyMuPDF and pdfplumber."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency guard
    import fitz  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    fitz = None  # type: ignore

try:
    import pdfplumber  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pdfplumber = None  # type: ignore

BudgetCheck = Callable[[str], bool]


@dataclass
class Line:
    """Normalized representation of a line of text on a PDF page."""

    page_index: int
    line_index: int
    text: str
    bbox: Tuple[float, float, float, float] | None
    x_center: float | None
    y_top: float | None

    def is_blank(self) -> bool:
        return not self.text.strip()

    @property
    def char_len(self) -> int:
        return len(self.text)


@dataclass
class PageExtraction:
    """Extraction results for a single page."""

    index: int
    glyphs: int
    lines: List[Line] = field(default_factory=list)

    @property
    def has_text(self) -> bool:
        return bool(self.lines)


def compute_glyph_counts(pdf_path: Path, max_pages: int, budget_check: BudgetCheck | None = None) -> List[int]:
    """Approximate glyph counts per page using pdfplumber as a lightweight probe."""

    counts: List[int] = []
    try:
        if pdfplumber is None:
            raise RuntimeError
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                if page_index >= max_pages:
                    break
                if budget_check and budget_check("glyph_probe"):
                    break
                counts.append(len(page.chars or []))
    except Exception:
        if fitz is None:  # pragma: no cover - environment guard
            raise RuntimeError("PyMuPDF (fitz) is required for glyph probing")
        with fitz.open(pdf_path) as doc:
            total = min(len(doc), max_pages)
            for page_index in range(total):
                if budget_check and budget_check("glyph_probe"):
                    break
                page = doc[page_index]
                counts.append(len(page.get_text("text")))
    return counts


def extract_pages(
    pdf_path: Path,
    max_pages: int,
    budget_check: BudgetCheck | None = None,
) -> List[PageExtraction]:
    """Extract lines and metadata for each page up to the configured limit."""

    pages: List[PageExtraction] = []
    if fitz is None:  # pragma: no cover - environment guard
        raise RuntimeError("PyMuPDF (fitz) is required for PDF extraction")

    with fitz.open(pdf_path) as doc:
        total = min(len(doc), max_pages)
        for page_index in range(total):
            if budget_check and budget_check("page_extract"):
                break
            page = doc[page_index]
            glyphs = len(page.get_text("text"))
            result = PageExtraction(index=page_index, glyphs=glyphs)
            text_dict = page.get_text("dict")
            line_counter = 0
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = " ".join(span.get("text", "") for span in spans).strip()
                    if not text:
                        continue
                    bbox = tuple(line.get("bbox", [])) if line.get("bbox") else None
                    x_center = None
                    y_top = None
                    if bbox:
                        x0, y0, x1, _ = bbox
                        x_center = (x0 + x1) / 2.0
                        y_top = y0
                    result.lines.append(
                        Line(
                            page_index=page_index,
                            line_index=line_counter,
                            text=text,
                            bbox=bbox,
                            x_center=x_center,
                            y_top=y_top,
                        )
                    )
                    line_counter += 1
            pages.append(result)
    return pages
