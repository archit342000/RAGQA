"""Layout signal computation for selective LayoutParser routing."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from pipeline.ingest.pdf_parser import DocumentGraph, PDFBlock, PageGraph


SignalName = str

_SIGNAL_WEIGHTS: Dict[SignalName, float] = {
    "CIS": 0.18,
    "OGR": 0.14,
    "BXS": 0.12,
    "DAS": 0.10,
    "FVS": 0.08,
    "ROJ": 0.12,
    "TFI": 0.10,
    "MSA": 0.08,
    "FNL": 0.08,
}


@dataclass(slots=True)
class PageSignalExtras:
    column_count: int
    column_assignments: Dict[str, int]
    table_overlap_ratio: float
    figure_overlap_ratio: float
    dominant_font_size: float
    footnote_block_ids: List[str]


@dataclass(slots=True)
class PageLayoutSignals:
    page_number: int
    raw: Dict[SignalName, float]
    normalized: Dict[SignalName, float]
    page_score: float
    extras: PageSignalExtras


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        return values_sorted[0]
    pos = (len(values_sorted) - 1) * (q / 100.0)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return values_sorted[int(pos)]
    lower_val = values_sorted[lower]
    upper_val = values_sorted[upper]
    return lower_val + (upper_val - lower_val) * (pos - lower)


def _robust_normalise(value: float, median: float, iqr: float) -> float:
    if iqr <= 1e-6:
        diff = value - median
        if diff == 0:
            return 0.5
        return max(0.0, min(1.0, 0.5 + 0.5 * math.tanh(diff)))
    z = (value - median) / (iqr * 1.5 + 1e-6)
    normalised = 0.5 + 0.5 * math.tanh(z)
    return max(0.0, min(1.0, normalised))


def _intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def _estimate_columns(page: PageGraph) -> Tuple[int, Dict[str, int], float]:
    text_blocks = page.text_blocks
    if not text_blocks:
        return 1, {}, 0.0
    columns: List[Tuple[float, float]] = []
    assignments: Dict[str, int] = {}
    tolerance = max(page.width * 0.12, 18.0)
    for block in sorted(text_blocks, key=lambda b: (b.bbox[0], b.bbox[1])):
        center = (block.bbox[0] + block.bbox[2]) / 2.0
        assigned_index = None
        for idx, (col_center, count) in enumerate(columns):
            if abs(center - col_center) <= tolerance:
                new_center = (col_center * count + center) / (count + 1)
                columns[idx] = (new_center, count + 1)
                assigned_index = idx
                break
        if assigned_index is None:
            columns.append((center, 1))
            assigned_index = len(columns) - 1
        assignments[block.block_id] = assigned_index
    column_count = max(1, len(columns))
    if column_count <= 1:
        cis = 0.0
    else:
        centers = [center for center, _ in columns]
        norm_centers = [center / max(page.width, 1.0) for center in centers]
        diffs = [abs(norm_centers[idx] - norm_centers[idx - 1]) for idx in range(1, len(norm_centers))]
        cis = min(1.0, (sum(diffs) / max(len(diffs), 1)) * column_count)
    return column_count, assignments, cis


def _open_gap_ratio(page: PageGraph) -> float:
    page_area = max(page.width * page.height, 1.0)
    text_area = sum(block.area for block in page.text_blocks)
    aux_area = sum(block.area for block in page.blocks if not block.is_text)
    whitespace = max(page_area - text_area - aux_area, 0.0)
    return min(1.0, whitespace / page_area)


def _block_width_spread(page: PageGraph) -> float:
    widths = [block.width / max(page.width, 1.0) for block in page.text_blocks if block.width > 0]
    if len(widths) <= 1:
        return 0.0
    spread = statistics.pstdev(widths)
    return min(1.0, spread * math.sqrt(len(widths)))


def _density_anomaly_score(page: PageGraph) -> float:
    densities: List[float] = []
    for block in page.text_blocks:
        area = max(block.area, 1.0)
        char_count = block.metadata.get("char_count", len(block.text))
        densities.append(char_count / area)
    if len(densities) <= 1:
        return 0.0
    mean_density = sum(densities) / len(densities)
    variance = sum((density - mean_density) ** 2 for density in densities) / len(densities)
    return min(1.0, math.sqrt(variance) * 8.0)


def _font_variance_score(page: PageGraph) -> float:
    font_sizes = [block.avg_font_size for block in page.text_blocks if block.avg_font_size > 0]
    if len(font_sizes) <= 1:
        return 0.0
    mean_size = sum(font_sizes) / len(font_sizes)
    if mean_size <= 0:
        return 0.0
    variance = sum((size - mean_size) ** 2 for size in font_sizes) / len(font_sizes)
    return min(1.0, math.sqrt(variance) / max(mean_size, 1e-6))


def _raggedness_score(page: PageGraph) -> float:
    if not page.text_blocks:
        return 0.0
    right_edges = [(page.width - block.bbox[2]) / max(page.width, 1.0) for block in page.text_blocks]
    if len(right_edges) <= 1:
        return right_edges[0] if right_edges else 0.0
    spread = statistics.pstdev(right_edges)
    return min(1.0, spread * 3.0)


def _table_figure_indicator(page: PageGraph) -> Tuple[float, float, float, List[PDFBlock]]:
    page_area = max(page.width * page.height, 1.0)
    table_candidates: List[PDFBlock] = []
    for block in page.text_blocks:
        if block.class_hint == "table" or block.metadata.get("numeric_ratio", 0.0) > 0.35:
            table_candidates.append(block)
    figure_area = sum(block.area for block in page.blocks if block.block_type == "image")
    table_area = sum(block.area for block in table_candidates)
    table_ratio = min(1.0, table_area / page_area)
    figure_ratio = min(1.0, figure_area / page_area)
    tfi = min(1.0, table_ratio + figure_ratio)
    return tfi, table_ratio, figure_ratio, table_candidates


def _multi_structure_arrangement(page: PageGraph) -> float:
    if not page.text_blocks:
        return 0.0
    structured = [block for block in page.text_blocks if block.class_hint in {"list", "title"}]
    return min(1.0, len(structured) / max(len(page.text_blocks), 1))


def _footnote_load(page: PageGraph) -> Tuple[float, List[str], float]:
    if not page.text_blocks:
        return 0.0, [], 0.0
    avg_sizes = [block.avg_font_size for block in page.text_blocks if block.avg_font_size > 0]
    dominant = sum(avg_sizes) / len(avg_sizes) if avg_sizes else 0.0
    footnotes: List[PDFBlock] = []
    total_chars = sum(block.metadata.get("char_count", len(block.text)) for block in page.text_blocks)
    for block in page.text_blocks:
        if dominant and block.avg_font_size > 0 and block.avg_font_size < dominant * 0.75:
            if block.bbox[1] > page.height * 0.75:
                footnotes.append(block)
    if not footnotes or total_chars == 0:
        return 0.0, [], dominant
    char_ratio = sum(block.metadata.get("char_count", len(block.text)) for block in footnotes) / total_chars
    score = min(1.0, 0.5 * char_ratio + 0.5)
    return score, [block.block_id for block in footnotes], dominant


def _table_prose_overlap(page: PageGraph, tables: List[PDFBlock]) -> float:
    if not tables:
        return 0.0
    page_area = max(page.width * page.height, 1.0)
    overlap = 0.0
    prose_blocks = [block for block in page.text_blocks if block not in tables]
    for table in tables:
        for block in prose_blocks:
            overlap += _intersection_area(table.bbox, block.bbox)
    return min(1.0, overlap / page_area)


def compute_layout_signals(document: DocumentGraph) -> List[PageLayoutSignals]:
    """Compute raw and normalised layout signals for ``document``."""

    raw_signals: List[Dict[SignalName, float]] = []
    extras_list: List[PageSignalExtras] = []

    for page in document.pages:
        column_count, assignments, cis = _estimate_columns(page)
        ogr = _open_gap_ratio(page)
        bxs = _block_width_spread(page)
        das = _density_anomaly_score(page)
        fvs = _font_variance_score(page)
        roj = _raggedness_score(page)
        tfi, table_ratio, figure_ratio, table_blocks = _table_figure_indicator(page)
        msa = _multi_structure_arrangement(page)
        fnl, footnote_ids, dominant_font = _footnote_load(page)
        overlap_ratio = _table_prose_overlap(page, table_blocks)

        raw = {
            "CIS": cis,
            "OGR": ogr,
            "BXS": bxs,
            "DAS": das,
            "FVS": fvs,
            "ROJ": roj,
            "TFI": tfi,
            "MSA": msa,
            "FNL": fnl,
        }
        raw_signals.append(raw)
        extras_list.append(
            PageSignalExtras(
                column_count=column_count,
                column_assignments=assignments,
                table_overlap_ratio=overlap_ratio,
                figure_overlap_ratio=figure_ratio,
                dominant_font_size=dominant_font,
                footnote_block_ids=footnote_ids,
            )
        )

    medians: Dict[SignalName, float] = {}
    iqrs: Dict[SignalName, float] = {}
    reference_pages = raw_signals[: min(10, len(raw_signals))]
    for signal_name in _SIGNAL_WEIGHTS:
        values = [page_raw[signal_name] for page_raw in reference_pages]
        medians[signal_name] = _percentile(values, 50.0)
        q1 = _percentile(values, 25.0)
        q3 = _percentile(values, 75.0)
        iqrs[signal_name] = max(0.0, q3 - q1)

    results: List[PageLayoutSignals] = []
    for page, raw, extras in zip(document.pages, raw_signals, extras_list):
        normalized = {
            name: _robust_normalise(raw[name], medians[name], iqrs[name]) for name in _SIGNAL_WEIGHTS
        }
        score = sum(_SIGNAL_WEIGHTS[name] * normalized[name] for name in _SIGNAL_WEIGHTS)
        results.append(
            PageLayoutSignals(
                page_number=page.page_number,
                raw=raw,
                normalized=normalized,
                page_score=score,
                extras=extras,
            )
        )
    return results


__all__ = ["PageLayoutSignals", "PageSignalExtras", "compute_layout_signals"]
