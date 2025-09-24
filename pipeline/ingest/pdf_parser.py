"""PyMuPDF-first PDF parser emitting detailed page graphs.

This module extracts fine-grained layout information from PDFs using
PyMuPDF (``fitz``). Each page is represented as a graph of blocks, lines,
and spans that expose bounding boxes, font metadata, and reading order. The
resulting structure feeds subsequent layout analysis, selective LayoutParser
routing, repair passes, and chunking.

The implementation intentionally keeps the dependency surface minimal: it is
safe to import even when PyMuPDF is unavailable, raising a clear error only
when parsing is attempted. This enables unit tests to stub the extractor
without importing heavy native extensions.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # PyMuPDF is required for the real parser but optional for tests.
    import fitz
except Exception:  # pragma: no cover - handled gracefully at runtime
    fitz = None  # type: ignore


# PyMuPDF encodes superscripts / subscripts via the second bit of ``flags``.
_SUPERSCRIPT_FLAG = 1 << 1


BBox = Tuple[float, float, float, float]


@dataclass(slots=True)
class PDFSpan:
    """Fine-grained span extracted by PyMuPDF."""

    text: str
    bbox: BBox
    font: str
    font_size: float
    flags: int
    color: int
    ascender: float | None = None
    descender: float | None = None

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class PDFLine:
    """Sequence of spans rendered on a single visual line."""

    spans: List[PDFSpan]
    bbox: BBox

    @property
    def text(self) -> str:
        return "".join(span.text for span in self.spans)


@dataclass(slots=True)
class PDFBlock:
    """Logical block detected by PyMuPDF (text, image, drawing)."""

    block_id: str
    bbox: BBox
    block_type: str
    lines: List[PDFLine] = field(default_factory=list)
    text: str = ""
    avg_font_size: float = 0.0
    dominant_fonts: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    char_start: int = 0
    char_end: int = 0
    class_hint: str = "prose"

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_text(self) -> bool:
        return self.block_type == "text"


@dataclass(slots=True)
class PageGraph:
    """Container for all blocks/spans belonging to a page."""

    page_number: int
    width: float
    height: float
    rotation: int
    blocks: List[PDFBlock]
    raw_dict: Dict[str, Any]
    char_start: int
    char_end: int

    @property
    def text_blocks(self) -> List[PDFBlock]:
        return [block for block in self.blocks if block.is_text and block.text]

    @property
    def page_text(self) -> str:
        return "\n".join(block.text for block in self.text_blocks)


@dataclass(slots=True)
class DocumentGraph:
    """Full-document representation composed of per-page graphs."""

    doc_id: str
    file_path: str
    pages: List[PageGraph]
    char_count: int
    metadata: Dict[str, Any]

    def iter_blocks(self) -> Iterator[PDFBlock]:
        for page in self.pages:
            yield from page.blocks

    def get_page(self, page_number: int) -> PageGraph:
        return self.pages[page_number - 1]


def _ensure_fitz_available() -> None:
    if fitz is None:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError(
            "PyMuPDF (fitz) is not installed. Install `PyMuPDF==1.24.*` to use the PDF parser."
        )


def _make_doc_id(file_path: Path) -> str:
    token = f"{file_path.name}:{file_path.stat().st_size}:{int(file_path.stat().st_mtime)}"
    digest = hashlib.sha1(token.encode("utf-8"), usedforsecurity=False)
    return digest.hexdigest()[:16]


def _normalise_text(text: str) -> str:
    """Normalise whitespace while preserving intentional line breaks."""

    if not text:
        return ""
    stripped = text.replace("\u00a0", " ")
    stripped = stripped.replace("\r", "")
    stripped = "\n".join(line.rstrip() for line in stripped.splitlines())
    return stripped.strip("\n")


def _infer_class_hint(block: PDFBlock, page_width: float) -> str:
    """Rudimentary structural hint used before LayoutParser refinement."""

    if not block.is_text or not block.text:
        return "aux"

    text = block.text.strip()
    line_count = len(block.lines)
    numeric_ratio = block.metadata.get("numeric_ratio", 0.0)
    bullet_ratio = block.metadata.get("bullet_ratio", 0.0)
    avg_line_chars = block.metadata.get("avg_line_chars", 0.0)
    width_ratio = block.width / max(page_width, 1.0)

    if numeric_ratio > 0.45 and width_ratio > 0.35:
        return "table"
    if bullet_ratio > 0.4 and line_count >= 3:
        return "list"
    if avg_line_chars < 15 and line_count <= 3 and width_ratio < 0.5:
        return "title"
    return "prose"


def _compute_line_metadata(line: PDFLine) -> Dict[str, float]:
    text = line.text.strip()
    length = len(text)
    digit_count = sum(ch.isdigit() for ch in text)
    bullet = 1 if text[:2].strip().startswith(("-", "*", "•")) else 0
    punctuation = sum(ch in {";", ":", "|", "-", "/"} for ch in text)
    return {
        "length": float(length),
        "digit_ratio": float(digit_count) / max(length, 1),
        "bullet": float(bullet),
        "punctuation_ratio": float(punctuation) / max(length, 1),
    }


def _enrich_block_statistics(block: PDFBlock) -> None:
    """Populate derived metrics used by downstream heuristics."""

    if not block.lines:
        return
    lengths: List[float] = []
    digit_ratios: List[float] = []
    bullets: List[float] = []
    punct_ratios: List[float] = []
    for line in block.lines:
        meta = _compute_line_metadata(line)
        lengths.append(meta["length"])
        digit_ratios.append(meta["digit_ratio"])
        bullets.append(meta["bullet"])
        punct_ratios.append(meta["punctuation_ratio"])
    if lengths:
        block.metadata["avg_line_chars"] = float(sum(lengths)) / len(lengths)
        block.metadata["max_line_chars"] = max(lengths)
    block.metadata["numeric_ratio"] = float(sum(digit_ratios)) / max(len(digit_ratios), 1)
    block.metadata["bullet_ratio"] = float(sum(bullets)) / max(len(bullets), 1)
    block.metadata["punct_ratio"] = float(sum(punct_ratios)) / max(len(punct_ratios), 1)


def _parse_blocks(
    *,
    page_dict: Dict[str, Any],
    page_number: int,
    page_width: float,
    char_offset: int,
) -> Tuple[List[PDFBlock], int]:
    blocks: List[PDFBlock] = []
    current_offset = char_offset
    text_blocks = 0

    for block_idx, block_dict in enumerate(page_dict.get("blocks", [])):
        block_type_raw = block_dict.get("type", 0)
        block_type = {
            0: "text",
            1: "image",
            2: "draw",
        }.get(block_type_raw, "unknown")
        bbox_tuple = tuple(float(v) for v in block_dict.get("bbox", (0, 0, 0, 0)))
        block_id = f"p{page_number}_b{block_idx}"
        block = PDFBlock(block_id=block_id, bbox=bbox_tuple, block_type=block_type)

        if block.is_text:
            font_weighted_total = 0.0
            font_weight_den = 0.0
            superscript_chars = 0
            for line_dict in block_dict.get("lines", []):
                spans: List[PDFSpan] = []
                line_bbox = tuple(float(v) for v in line_dict.get("bbox", bbox_tuple))
                for span_dict in line_dict.get("spans", []):
                    span_text = span_dict.get("text", "")
                    if span_text is None or span_text == "":
                        continue
                    span = PDFSpan(
                        text=span_text,
                        bbox=tuple(float(v) for v in span_dict.get("bbox", line_bbox)),
                        font=span_dict.get("font", ""),
                        font_size=float(span_dict.get("size", 0.0)),
                        flags=int(span_dict.get("flags", 0)),
                        color=int(span_dict.get("color", 0)),
                        ascender=float(span_dict.get("ascender", 0.0)) if span_dict.get("ascender") is not None else None,
                        descender=float(span_dict.get("descender", 0.0)) if span_dict.get("descender") is not None else None,
                    )
                    spans.append(span)
                    clean_text = span.text.replace("\n", "")
                    weight = max(len(clean_text.strip()), 1)
                    font_weighted_total += span.font_size * weight
                    font_weight_den += weight
                    if span.flags & _SUPERSCRIPT_FLAG:
                        superscript_chars += sum(ch.isdigit() for ch in span.text)
                    block.dominant_fonts[span.font] = block.dominant_fonts.get(span.font, 0) + weight
                if spans:
                    block.lines.append(PDFLine(spans=spans, bbox=line_bbox))
            block.text = _normalise_text("\n".join(line.text for line in block.lines))
            if font_weight_den:
                block.avg_font_size = font_weighted_total / font_weight_den
            text_len = len(block.text)
            block.metadata["char_count"] = text_len
            block.metadata["line_count"] = len(block.lines)
            block.metadata["span_count"] = sum(len(line.spans) for line in block.lines)
            block.metadata["superscript_count"] = superscript_chars
            block.metadata["char_density"] = (
                float(text_len) / max(block.area, 1.0) if block.area > 0 else 0.0
            )
            _enrich_block_statistics(block)
            block.class_hint = _infer_class_hint(block, page_width)
            block.char_start = current_offset
            current_offset += text_len
            block.char_end = current_offset
            current_offset += 1  # guard gap between blocks for downstream offsets
            text_blocks += 1
        blocks.append(block)

    logger.debug("Page %d extracted %d blocks (%d text)", page_number, len(blocks), text_blocks)
    return blocks, current_offset


def parse_pdf_with_pymupdf(
    file_path: str,
    *,
    doc_id: str | None = None,
    max_pages: Optional[int] = None,
) -> DocumentGraph:
    """Parse ``file_path`` into a :class:`DocumentGraph` using PyMuPDF."""

    _ensure_fitz_available()
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    resolved_doc_id = doc_id or _make_doc_id(pdf_path)
    document = fitz.open(pdf_path)  # type: ignore[arg-type]
    try:
        page_graphs: List[PageGraph] = []
        char_offset = 0
        page_limit = min(len(document), max_pages) if max_pages is not None else len(document)
        for page_index in range(page_limit):
            page = document.load_page(page_index)
            page_number = page_index + 1
            text_dict = page.get_text("dict")
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            blocks, new_offset = _parse_blocks(
                page_dict=text_dict,
                page_number=page_number,
                page_width=page_width,
                char_offset=char_offset,
            )
            char_start = char_offset
            char_end = new_offset
            char_offset = new_offset
            page_graphs.append(
                PageGraph(
                    page_number=page_number,
                    width=page_width,
                    height=page_height,
                    rotation=int(page.rotation),
                    blocks=blocks,
                    raw_dict=text_dict,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
    finally:
        document.close()

    metadata = {
        "page_count": len(page_graphs),
        "file_name": pdf_path.name,
    }
    char_count = page_graphs[-1].char_end if page_graphs else 0
    return DocumentGraph(
        doc_id=resolved_doc_id,
        file_path=str(pdf_path),
        pages=page_graphs,
        char_count=char_count,
        metadata=metadata,
    )


__all__ = [
    "PDFSpan",
    "PDFLine",
    "PDFBlock",
    "PageGraph",
    "DocumentGraph",
    "parse_pdf_with_pymupdf",
]
