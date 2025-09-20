"""Page segmentation and coarse layout classification utilities.

The chunking pipeline treats each page as a collection of coarse "blocks"
such as headings, paragraphs, tables, and lists. This module provides the
lightweight heuristics required to build those blocks and emit simple layout
metrics that guide the downstream semantic/fixed chunking strategy. The goal
is not to be perfect, but to provide fast, deterministic signals that work on
CPU-only Spaces without extra dependencies.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from chunking.types import Block

_LINES_PATTERN = re.compile(r"\r?\n")
# Matches paragraph-like spans separated by one or more blank lines. The lazy
# quantifier keeps headings and short sentences intact while avoiding runaway
# matches on dense documents.
_PARAGRAPH_PATTERN = re.compile(r"(?s)(.+?)(?:\n\s*\n|$)")


def _analyze_page(text: str) -> dict[str, float | bool]:
    """Extract lightweight layout signals used to choose a chunking strategy.

    Parameters
    ----------
    text:
        Cleaned page text after the parser/cleaner passes.

    Returns
    -------
    dict
        Metrics describing the proportion of short lines, numeric/table-like
        lines, and gaps. These values are consumed by the driver to determine
        whether semantic chunking is safe or if we should drop back to fixed
        windows for a given page.
    """

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return {"short_line_ratio": 0.0, "digit_symbol_ratio": 0.0, "gap_line_ratio": 1.0, "table_heavy": False}

    short_lines = sum(1 for line in lines if len(line.strip()) < 40)
    gap_lines = text.count("\n\n")

    def _digit_symbol_score(line: str) -> float:
        stripped = line.strip()
        if not stripped:
            return 0.0
        digits = sum(ch.isdigit() for ch in stripped)
        symbols = sum(ch in "|:/-+%," for ch in stripped)
        return (digits + symbols) / len(stripped)

    digit_symbol_lines = sum(1 for line in lines if _digit_symbol_score(line) > 0.6)

    total = len(lines)
    short_ratio = short_lines / total
    digit_symbol_ratio = digit_symbol_lines / total
    gap_ratio = gap_lines / max(1, text.count("\n"))

    table_heavy = digit_symbol_ratio > 0.5 or short_ratio > 0.6

    return {
        "short_line_ratio": float(short_ratio),
        "digit_symbol_ratio": float(digit_symbol_ratio),
        "gap_line_ratio": float(gap_ratio),
        "table_heavy": table_heavy,
    }


def _classify_block(raw: str) -> str:
    """Return a best-effort guess of what kind of text a block represents.

    The heuristics intentionally favour recall over precision: downstream
    components simply need to know whether a block *might* be a heading/list
    so they can set sensible metadata on chunks.
    """
    stripped = raw.strip()
    if not stripped:
        return "other"
    lines = stripped.splitlines()
    if len(lines) == 1:
        candidate = lines[0]
        if len(candidate) <= 60 and candidate == candidate.upper() and any(ch.isalpha() for ch in candidate):
            return "heading"
        if candidate.endswith(":" ) and len(candidate) <= 80:
            return "heading"
    if any(line.lstrip().startswith(('- ', '* ', '•', '1.', 'a.')) for line in lines):
        return "list"
    if any('|' in line or '\t' in line for line in lines):
        return "table"
    digit_ratio = sum(ch.isdigit() for ch in stripped) / max(1, len(stripped))
    if digit_ratio > 0.4:
        return "table"
    if '```' in stripped or stripped.startswith('    '):
        return "code"
    return "paragraph"


def segment_page(doc_id: str, doc_name: str, page_num: int, text: str) -> Tuple[List[Block], dict[str, float | bool]]:
    """Split a page into coarse blocks and compute layout diagnostics."""

    profile = _analyze_page(text)
    blocks: List[Block] = []

    for match in _PARAGRAPH_PATTERN.finditer(text):
        raw = match.group(1)
        char_start = match.start(1)
        char_end = match.end(1)
        kind = _classify_block(raw)
        blocks.append(
            Block(
                doc_id=doc_id,
                doc_name=doc_name,
                page_num=page_num,
                kind=kind,
                text=raw.strip(),
                char_start=char_start,
                char_end=char_end,
            )
        )

    if not blocks:
        # Some pages (e.g. scanned tables) may not include paragraph breaks.
        # Emit a single "other" block so downstream stages still receive text.
        blocks.append(
            Block(
                doc_id=doc_id,
                doc_name=doc_name,
                page_num=page_num,
                kind="other",
                text=text.strip(),
                char_start=0,
                char_end=len(text),
            )
        )

    return blocks, profile
