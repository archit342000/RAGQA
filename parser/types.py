"""Type definitions for the CPU-first parser."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def as_list(self) -> List[float]:
        return [float(self.x0), float(self.y0), float(self.x1), float(self.y1)]


@dataclass
class LineSpan:
    """Represents a line of text extracted from a PDF page."""

    page_index: int
    line_index: int
    text: str
    bbox: Optional[BBox]
    char_start: int
    char_end: int
    is_caption: bool = False
    is_footnote: bool = False
    is_heading: bool = False
    is_list_item: bool = False

    def cleaned_text(self) -> str:
        return self.text.strip()


@dataclass
class CaptionSidecar:
    page_index: int
    anchor_line: int
    text: str
    bbox: Optional[BBox]


@dataclass
class TableExtraction:
    page_index: int
    csv_path: Optional[str]
    rows: int
    cols: int
    confidence: float
    skipped_reason: Optional[str] = None


@dataclass
class PageParseResult:
    page_index: int
    glyph_count: int
    had_text: bool
    ocr_performed: bool
    lines: List[LineSpan] = field(default_factory=list)
    captions: List[CaptionSidecar] = field(default_factory=list)
    tables: List[TableExtraction] = field(default_factory=list)
    noise_ratio: float = 0.0


@dataclass
class ParsedDocument:
    doc_id: str
    file_path: str
    pages: List[PageParseResult]
    config_used: dict
    parse_time_s: float
    content_hash: str

    def all_lines(self) -> Sequence[LineSpan]:
        for page in self.pages:
            for line in page.lines:
                yield line

    def caption_lines(self) -> Sequence[CaptionSidecar]:
        for page in self.pages:
            for caption in page.captions:
                yield caption

    def stats_summary(self) -> Tuple[int, int]:
        lines_total = sum(len(page.lines) for page in self.pages)
        captions_total = sum(len(page.captions) for page in self.pages)
        return lines_total, captions_total


@dataclass
class RunReport:
    total_docs: int
    total_pages: int
    truncated: bool
    skipped_docs: List[str]
    message: str = ""
