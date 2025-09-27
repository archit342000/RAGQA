"""CPU-first PDF parsing utilities for RAGQA."""

from .config import DEFAULT_CONFIG, ParserConfig, load_config
from .pdf_parser import PDFParser
from .types import (
    CaptionSidecar,
    LineSpan,
    PageParseResult,
    ParsedDocument,
    RunReport,
    TableExtraction,
)

__all__ = [
    "DEFAULT_CONFIG",
    "ParserConfig",
    "load_config",
    "PDFParser",
    "CaptionSidecar",
    "LineSpan",
    "PageParseResult",
    "ParsedDocument",
    "RunReport",
    "TableExtraction",
]
