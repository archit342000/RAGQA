"""Metrics utilities for parsed documents."""

# This module focuses on lightweight heuristics that feed status/debug panels
# for the hybrid parser without depending on heavy third-party metrics.

from __future__ import annotations

import statistics
from typing import Dict, Iterable, List

from .types import ParsedDoc


def page_character_counts(pages: Iterable[str]) -> List[int]:
    """Compute character counts per page (including whitespace)."""

    return [len(page) for page in pages]


def compute_density_metrics(pages: Iterable[str], *, min_chars_per_page: int) -> Dict[str, float]:
    """Return basic density-related statistics for the supplied pages."""
    pages_list = list(pages)
    counts = page_character_counts(pages_list)
    total_pages = len(counts)
    total_chars = sum(counts)
    empty_or_low = sum(1 for count in counts if count < min_chars_per_page)

    avg_chars = (total_chars / total_pages) if total_pages else 0.0
    median_chars = float(statistics.median(counts)) if counts else 0.0
    empty_ratio = (empty_or_low / total_pages) if total_pages else 0.0

    return {
        "total_pages": float(total_pages),
        "total_chars": float(total_chars),
        "avg_chars_per_page": float(avg_chars),
        "median_chars_per_page": float(median_chars),
        "empty_or_low_ratio": float(empty_ratio),
    }


def compute_layout_features(pages: Iterable[str]) -> Dict[str, float]:
    """Estimate how complex a document layout is based on simple heuristics."""
    total_lines = 0
    short_lines = 0
    gap_lines = 0
    digit_symbol_lines = 0

    for page in pages:
        for raw_line in page.splitlines():
            total_lines += 1
            stripped = raw_line.strip()
            if not stripped:
                gap_lines += 1
                continue
            if len(stripped) < 40:
                short_lines += 1
            non_alpha = sum(1 for ch in stripped if not ch.isalpha())
            frac_non_alpha = (non_alpha / len(stripped)) if stripped else 0.0
            if frac_non_alpha >= 0.6:
                digit_symbol_lines += 1

    if total_lines == 0:
        return {
            "short_line_ratio": 0.0,
            "gap_line_ratio": 0.0,
            "digit_symbol_ratio": 0.0,
            "layout_complexity_score": 0.0,
        }

    short_ratio = short_lines / total_lines
    gap_ratio = gap_lines / total_lines
    digit_ratio = digit_symbol_lines / total_lines
    complex_score = 0.4 * short_ratio + 0.3 * gap_ratio + 0.3 * digit_ratio

    return {
        "short_line_ratio": float(short_ratio),
        "gap_line_ratio": float(gap_ratio),
        "digit_symbol_ratio": float(digit_ratio),
        "layout_complexity_score": float(complex_score),
    }


def char_count_histogram(pages: Iterable[str], *, bin_size: int = 200) -> Dict[str, float]:
    """Bucket page character counts into coarse bins for later inspection."""
    counts = page_character_counts(pages)
    if not counts:
        return {}

    totals: Dict[str, float] = {}
    total_pages = float(len(counts))
    for count in counts:
        bucket_start = (count // bin_size) * bin_size
        bucket_end = bucket_start + bin_size - 1
        key = f"char_hist_{bucket_start}_{bucket_end}"
        totals[key] = totals.get(key, 0.0) + 1.0

    for key in list(totals.keys()):
        totals[key] = totals[key] / total_pages

    return totals


def compute_document_metrics(pages: Iterable[str], *, min_chars_per_page: int) -> Dict[str, float]:
    """Combine density, layout, and histogram features into one dictionary."""
    pages_list = list(pages)
    density = compute_density_metrics(pages_list, min_chars_per_page=min_chars_per_page)
    layout = compute_layout_features(pages_list)
    histogram = char_count_histogram(pages_list)
    combined: Dict[str, float] = {}
    combined.update(density)
    combined.update(layout)
    combined.update(histogram)
    return combined


def merge_doc_stats(docs: List[ParsedDoc]) -> Dict[str, float]:
    """Aggregate stats across documents for debugging displays."""

    if not docs:
        return {}

    aggregate: Dict[str, float] = {
        "total_docs": float(len(docs)),
        "total_pages": float(sum(len(doc.pages) for doc in docs)),
        "total_chars": float(sum(doc.total_chars for doc in docs)),
    }

    if docs:
        aggregate["avg_pages_per_doc"] = aggregate["total_pages"] / len(docs)

    durations = [doc.stats.get("parse_duration_seconds", 0.0) for doc in docs]
    aggregate["total_parse_seconds"] = float(sum(durations))

    return aggregate
