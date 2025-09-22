"""Utility functions for working with text spans."""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional, Sequence, Tuple

_SENTENCE_REGEX = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+|$)")
_WS_RE = re.compile(r"\s+")


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


def resolve_evidence_spans(
    evidence: Iterable[Any],
    sentence_bounds: Sequence[Tuple[int, int]],
    window_text: str,
) -> List[Tuple[int, int]]:
    """Approximate evidence spans using ``window_text`` and ``sentence_bounds``."""

    spans: List[Tuple[int, int]] = []
    if not window_text:
        return spans

    normalized_text, index_map = _normalize_with_mapping(window_text)
    if not normalized_text:
        return spans

    sentence_norms: List[Tuple[str, Tuple[int, int]]] = []
    for start, end in sentence_bounds:
        if start >= end:
            continue
        snippet = window_text[start:end]
        norm = _normalize(snippet)
        if not norm:
            continue
        sentence_norms.append((norm, (start, end)))

    seen: set[Tuple[int, int]] = set()
    search_start = 0
    bounds_len = len(sentence_bounds)

    for entry in evidence:
        text = _extract_text(entry)
        span: Optional[Tuple[int, int]] = None
        next_search_start = search_start

        if text:
            normalized_entry = _normalize(text)
            if normalized_entry:
                idx = normalized_text.find(normalized_entry, search_start)
                if idx == -1:
                    idx = normalized_text.find(normalized_entry)
                if idx != -1:
                    span = _span_from_mapping(index_map, idx, len(normalized_entry))
                    if span is not None:
                        next_search_start = idx + len(normalized_entry)
                if span is None:
                    for sentence_norm, sentence_span in sentence_norms:
                        if normalized_entry in sentence_norm:
                            span = sentence_span
                            break
        else:
            idx = _extract_index(entry)
            if idx is not None and 0 <= idx < bounds_len:
                span = sentence_bounds[idx]

        search_start = next_search_start

        if span is None:
            continue
        start, end = span
        if start >= end:
            continue
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        spans.append((start, end))

    return spans


def _extract_text(entry: Any) -> Optional[str]:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        value = entry.get("text") or entry.get("snippet")
        if isinstance(value, str):
            return value
    else:
        for attr in ("text", "snippet"):
            value = getattr(entry, attr, None)
            if isinstance(value, str):
                return value
    return None


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


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


def _normalize_with_mapping(text: str) -> Tuple[str, List[int]]:
    normalized: List[str] = []
    index_map: List[int] = []
    last_was_space = True

    for idx, char in enumerate(text):
        if char.isspace():
            if last_was_space:
                continue
            normalized.append(" ")
            index_map.append(idx)
            last_was_space = True
            continue
        normalized.append(char.lower())
        index_map.append(idx)
        last_was_space = False

    if normalized and normalized[-1] == " ":
        normalized.pop()
        index_map.pop()

    return "".join(normalized), index_map


def _span_from_mapping(index_map: Sequence[int], start_idx: int, length: int) -> Optional[Tuple[int, int]]:
    if length <= 0 or start_idx < 0:
        return None
    end_idx = start_idx + length - 1
    if end_idx >= len(index_map):
        return None
    start = index_map[start_idx]
    end = index_map[end_idx] + 1
    if start >= end:
        return None
    return start, end


__all__ = ["sentence_spans", "resolve_evidence_spans"]

