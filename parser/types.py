"""Typed structures for parsed documents."""

# These NamedTuples act as the public contract for parser outputs
# and are shared between the core parsing logic, tests, and the UI layer.

from __future__ import annotations

from typing import Dict, List, NamedTuple


class ParsedPage(NamedTuple):
    """Represents one page extracted from a document."""

    doc_id: str
    page_num: int
    text: str
    char_range: tuple[int, int]
    metadata: Dict[str, str]


class ParsedDoc(NamedTuple):
    """Represents an entire parsed document."""

    doc_id: str
    pages: list[ParsedPage]
    total_chars: int
    parser_used: str
    stats: Dict[str, float]


class RunReport(NamedTuple):
    """Aggregate metadata for a multi-document parsing run.

    ``RunReport`` powers the UI banner that informs users when documents were
    skipped or truncated and helps developers understand how much work the
    parser performed.
    """

    total_docs: int
    total_pages: int
    truncated: bool
    skipped_docs: List[str]
    message: str
