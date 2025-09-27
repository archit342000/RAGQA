"""Identify caption and footnote sidecars from extracted lines."""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

from .pdf_io import Line

CAPTION_PATTERNS = [re.compile(r"^Fig(ure)?\.?", re.IGNORECASE), re.compile(r"^Table", re.IGNORECASE)]
FOOTNOTE_PATTERN = re.compile(r"^(?:\d+|[a-z]{1,2})[\).\s]")


def split_sidecars(pages: Sequence[Sequence[Line]]) -> Tuple[List[List[Line]], List[Line], List[Line]]:
    """Separate caption/footnote lines from body text."""

    body_pages: List[List[Line]] = []
    captions: List[Line] = []
    footnotes: List[Line] = []

    for page_lines in pages:
        retained: List[Line] = []
        for idx, line in enumerate(page_lines):
            stripped = line.text.strip()
            if _is_caption(page_lines, idx):
                captions.append(line)
                continue
            if FOOTNOTE_PATTERN.match(stripped):
                footnotes.append(line)
                continue
            retained.append(line)
        body_pages.append(retained)
    return body_pages, captions, footnotes


def _is_caption(page_lines: Sequence[Line], idx: int) -> bool:
    stripped = page_lines[idx].text.strip()
    if any(pattern.match(stripped) for pattern in CAPTION_PATTERNS):
        return True
    window = range(max(0, idx - 3), min(len(page_lines), idx + 4))
    for j in window:
        if j == idx:
            continue
        neighbor = page_lines[j].text.strip()
        if any(pattern.match(neighbor) for pattern in CAPTION_PATTERNS):
            return True
    return False
