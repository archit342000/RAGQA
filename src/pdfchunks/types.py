"""Public data structures for the high-level pdfchunks driver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple


@dataclass(slots=True)
class ParsedPage:
    """Lightweight representation of a parsed page."""

    doc_id: str
    page_num: int
    text: str
    char_range: Tuple[int, int]
    metadata: Dict[str, str]
    offset_start: int | None = None
    offset_end: int | None = None


@dataclass(slots=True)
class ParsedDoc:
    """Parsed document with aggregate statistics and metadata."""

    doc_id: str
    doc_name: str
    pages: List[ParsedPage]
    total_chars: int
    parser_used: str
    stats: Dict[str, float]
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RunReport:
    """Summary of a multi-document parsing run."""

    total_docs: int
    parsed_docs: int
    total_pages: int
    skipped_docs: List[str]
    message: str = ""

    def _asdict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ChunkPayload:
    """Chunk emitted by the high-level driver."""

    id: str
    doc_id: str
    doc_name: str
    role: str
    section_seq: int
    section_title: str | None
    text: str
    token_len: int
    page_start: int
    page_end: int
    meta: Dict[str, str]


@dataclass(slots=True)
class PipelineResult:
    """Aggregate output from :func:`run_pipeline`."""

    docs: List[ParsedDoc]
    chunks: List[ChunkPayload]
    chunk_stats: Dict[str, Dict[str, float]]
    report: RunReport

