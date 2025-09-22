"""Utility functions for working with text spans."""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional, Sequence, Tuple

_SENTENCE_REGEX = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+|$)")


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Return `(start, end)` offsets for sentences in ``text``."""

    spans: List[Tuple[int, int]] = []
    if not text:
        return spans

    for match in _SENTENCE_REGEX.finditer(text):
        start, end = match.span()
        snippet = text[start:end].strip()
        if not snippet:
            continue
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def _extract_index(entry: Any) -> Optional[int]:
    if isinstance(entry, dict):
        value = entry.get("index")
    else:
        value = getattr(entry, "index", None)

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def resolve_evidence_spans(
    evidence: Iterable[Any],
    sentence_bounds: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Map evidence indices to character spans using ``sentence_bounds``."""

    spans: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    bounds_len = len(sentence_bounds)

    for entry in evidence:
        idx = _extract_index(entry)
        if idx is None or idx < 0 or idx >= bounds_len:
            continue
        span = sentence_bounds[idx]
        if span in seen:
            continue
        seen.add(span)
        spans.append(span)

    return spans


__all__ = ["sentence_spans", "resolve_evidence_spans"]

