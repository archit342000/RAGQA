"""Lightweight statistics helpers for the CPU-first parser."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence

from .types import ParsedDocument


def document_stats(document: ParsedDocument) -> Dict[str, int]:
    """Return per-document counters used by the UI debug panel."""

    counters = Counter()
    counters["pages"] = len(document.pages)
    for page in document.pages:
        counters["glyphs"] += page.glyph_count
        counters["lines"] += len(page.lines)
        counters["captions"] += len(page.captions)
        counters["tables"] += len(page.tables)
        if page.ocr_performed:
            counters["ocr_pages"] += 1
    return dict(counters)


def merge_doc_stats(documents: Sequence[ParsedDocument]) -> Dict[str, int]:
    """Aggregate counters across multiple documents."""

    aggregate = Counter()
    for document in documents:
        aggregate.update(document_stats(document))
    aggregate["documents"] = len(documents)
    return dict(aggregate)


__all__ = ["document_stats", "merge_doc_stats"]

