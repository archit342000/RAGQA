"""Utilities for cleaning extracted document text."""

# The functions in this module are shared by every parser backend. They focus
# on turning slightly noisy PDF/text extraction output into more uniform blocks
# that downstream RAG pipelines can tokenize reliably.

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List

# Repeated headers/footers that appear on at least 60% of the pages are
# considered boilerplate and will be stripped from the final output.
_HEADER_FOOTER_THRESHOLD = 0.6


_DEHYPHEN_PATTERN = re.compile(r"(?<=\w)-\n(?=[a-zA-Z])")
_MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
_TRAILING_SPACE_PATTERN = re.compile(r"[ \t]+$", re.MULTILINE)


def clean_text_block(text: str) -> str:
    """Normalise a chunk of text extracted from a document page.

    The routine removes carriage returns, joins common hyphenation breaks,
    collapses repeated whitespace, and trims trailing spaces on each line. This
    keeps the token streams deterministic across different parser backends.
    Parameters
    ----------
    text:
        Raw string extracted from a PDF page.

    Returns
    -------
    str
        A cleaned-up block ready for downstream ingestion.
    """

    if not text:
        return ""

    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    dehyphenated = _DEHYPHEN_PATTERN.sub("", normalised)
    collapsed_spaces = _MULTI_SPACE_PATTERN.sub(" ", dehyphenated)
    trimmed = _TRAILING_SPACE_PATTERN.sub("", collapsed_spaces)
    collapsed_newlines = _MULTI_NEWLINE_PATTERN.sub("\n\n", trimmed)
    return collapsed_newlines.strip("\n")


def _normalise_line(line: str) -> str:
    return line.strip().lower()


def remove_headers_footers(pages: Iterable[str]) -> List[str]:
    """Remove repeating header/footer lines observed across most pages.

    We scan the first and last non-empty lines on each page, then drop any
    candidate that appears on a majority of pages. This heuristic performs well
    for corporate reports that stamp the same banner on every page.
    Parameters
    ----------
    pages:
        Sequence of page strings to inspect.

    Returns
    -------
    list[str]
        Pages with common headers/footers stripped.
    """

    pages_list = list(pages)
    if not pages_list:
        return []

    first_lines: Counter[str] = Counter()
    last_lines: Counter[str] = Counter()

    for page in pages_list:
        lines = page.splitlines()
        # find first meaningful line
        for line in lines:
            stripped = line.strip()
            if stripped:
                first_lines[_normalise_line(stripped)] += 1
                break
        for line in reversed(lines):
            stripped = line.strip()
            if stripped:
                last_lines[_normalise_line(stripped)] += 1
                break

    page_count = len(pages_list)
    threshold = max(2, math.ceil(_HEADER_FOOTER_THRESHOLD * page_count))

    headers_to_remove = {
        line
        for line, count in first_lines.items()
        if count >= threshold and line
    }
    footers_to_remove = {
        line
        for line, count in last_lines.items()
        if count >= threshold and line
    }

    cleaned_pages: List[str] = []
    for page in pages_list:
        lines = page.splitlines()
        start_idx = 0
        end_idx = len(lines)

        # remove header
        while start_idx < end_idx:
            candidate = lines[start_idx].strip()
            if candidate and _normalise_line(candidate) in headers_to_remove:
                start_idx += 1
            else:
                break

        # remove footer
        while end_idx > start_idx:
            candidate = lines[end_idx - 1].strip()
            if candidate and _normalise_line(candidate) in footers_to_remove:
                end_idx -= 1
            else:
                break

        sliced = lines[start_idx:end_idx]
        cleaned_pages.append("\n".join(sliced))

    return cleaned_pages
