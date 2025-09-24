"""Typed representations used by the chunking pipeline."""

from __future__ import annotations

from typing import Dict, NamedTuple


class Chunk(NamedTuple):
    """Final retrieval chunk emitting from the chunking driver."""

    doc_id: str
    doc_name: str
    page_start: int
    page_end: int
    section_title: str | None
    text: str
    token_len: int
    meta: Dict[str, str]
