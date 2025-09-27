"""Helpers for routing captions and footnotes to sidecar chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .pdf_io import Line

CAPTION_PATTERNS = [re.compile(r"^Fig(?:ure)?\.?", re.IGNORECASE), re.compile(r"^Table", re.IGNORECASE)]
FOOTNOTE_PATTERN = re.compile(r"^(\d+|[a-zA-Z]\))")


@dataclass
class SidecarResult:
    body: List[Line]
    captions: List[Line]
    footnotes: List[Line]


def _is_caption_candidate(line: Line) -> bool:
    text = line.text.strip()
    return any(pattern.match(text) for pattern in CAPTION_PATTERNS)


def _is_footnote_candidate(line: Line) -> bool:
    text = line.text.strip()
    if not text:
        return False
    return bool(FOOTNOTE_PATTERN.match(text)) and len(text.split()) < 20


def split_sidecars(lines: Sequence[Line]) -> SidecarResult:
    captions: List[Line] = []
    footnotes: List[Line] = []
    body: List[Line] = []
    caption_indices: set[int] = set()
    for idx, line in enumerate(lines):
        if _is_caption_candidate(line):
            captions.append(line)
            caption_indices.add(idx)
            for delta in (-3, -2, -1, 1, 2, 3):
                j = idx + delta
                if 0 <= j < len(lines) and j not in caption_indices and _is_caption_candidate(lines[j]):
                    captions.append(lines[j])
                    caption_indices.add(j)

    for idx, line in enumerate(lines):
        if idx in caption_indices:
            continue
        if _is_footnote_candidate(line):
            footnotes.append(line)
        else:
            body.append(line)
    return SidecarResult(body=body, captions=captions, footnotes=footnotes)

