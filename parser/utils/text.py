"""Utility functions for text analysis in the parser."""
from __future__ import annotations

import math
import re
from typing import Iterable, Tuple

JUNK_PATTERN = re.compile(r"[^\w\s.,;:'%$€£()\-+/\\]")
WHITESPACE_RE = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    cleaned = WHITESPACE_RE.sub(" ", text).strip()
    if not cleaned:
        return 0
    # Heuristic: assume ~4 characters per token for Latin scripts.
    return max(1, math.ceil(len(cleaned) / 4))


def junk_ratio(text: str) -> float:
    if not text:
        return 0.0
    junk_chars = len(JUNK_PATTERN.findall(text))
    return junk_chars / max(1, len(text))


def is_heading(text: str, max_length: int) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > max_length:
        return False
    if stripped.isupper():
        return True
    if stripped.endswith(":"):
        return True
    if stripped[0].isdigit() and stripped.replace(".", "").replace(" ", "").isalnum():
        return True
    return False


def is_list_item(text: str, markers: Iterable[str]) -> bool:
    stripped = text.lstrip()
    for marker in markers:
        if stripped.startswith(f"{marker} ") or stripped.startswith(f"{marker}\t"):
            return True
    if stripped[:2].isdigit() and stripped[2:3] in {".", ")"}:
        return True
    return False


def detect_caption(text: str, pattern: str) -> bool:
    return bool(re.match(pattern, text.strip(), flags=re.IGNORECASE))


def merge_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def window_pairs(values: Iterable[float]) -> Iterable[Tuple[float, float]]:
    iterator = iter(values)
    prev = None
    for value in iterator:
        if prev is not None:
            yield prev, value
        prev = value
