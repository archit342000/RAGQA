"""Layout signal computation for selective LayoutParser routing."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from pipeline.ingest.pdf_parser import DocumentGraph, PDFBlock, PageGraph


SignalName = str

_SIGNAL_NAMES: Tuple[SignalName, ...] = (
    "CIS",
    "OGR",
    "BXS",
    "DAS",
    "FVS",
    "ROJ",
    "TFI",
    "MSA",
    "FNL",
)

_NORMALISATION_WINDOW = 32


@dataclass(slots=True)
class PageSignalExtras:
    column_count: int
    column_assignments: Dict[str, int]
    table_overlap_ratio: float
    figure_overlap_ratio: float
    dominant_font_size: float
    footnote_block_ids: List[str]
    superscript_spans: int
    total_line_count: int
    has_normal_density: bool
    char_density: float
    structural_score: float = 0.0
    intrusion_ratio: float = 0.0


@dataclass(slots=True)
class PageLayoutSignals:
    page_number: int
    raw: Dict[SignalName, float]
    normalized: Dict[SignalName, float]
    page_score: float
    extras: PageSignalExtras


def _percentile(values: Sequence[float], q: float) -> float:
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
        return 1.0 if value > median else 0.0
    lower = median - 1.5 * iqr
    upper = median + 1.5 * iqr
    if value <= lower:
        return 0.0
    if value >= upper:
        return 1.0
    return (value - lower) / (upper - lower)


def _page_area(page: PageGraph) -> float:
    return max(page.width * page.height, 1.0)


def _filtered_text_blocks(page: PageGraph) -> List[PDFBlock]:
    page_area = _page_area(page)
    min_area = page_area * 0.005
    return [block for block in page.text_blocks if block.area >= min_area]


def _intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def _estimate_columns(
    page: PageGraph, blocks: Sequence[PDFBlock]
) -> Tuple[int, Dict[str, int], List[Tuple[float, float, float, float]], float]:
    tolerance = max(page.width * 0.08, 24.0)
    if not blocks:
        assignments = {block.block_id: 0 for block in page.text_blocks}
        return 1, assignments, [], tolerance

    columns: List[Dict[str, float]] = []
    assignments: Dict[str, int] = {}
    for block in sorted(blocks, key=lambda b: (b.bbox[0], b.bbox[1])):
        center = (block.bbox[0] + block.bbox[2]) / 2.0
        assigned_idx = None
        for idx, column in enumerate(columns):
            if abs(center - column["center"]) <= tolerance:
                weight = column["count"] + 1
                column["center"] = (column["center"] * column["count"] + center) / weight
                column["left"] = min(column["left"], block.bbox[0])
                column["right"] = max(column["right"], block.bbox[2])
                column["count"] = weight
                assigned_idx = idx
                break
        if assigned_idx is None:
            columns.append(
                {
                    "center": center,
                    "left": block.bbox[0],
                    "right": block.bbox[2],
                    "count": 1,
                }
            )
            assigned_idx = len(columns) - 1
        assignments[block.block_id] = assigned_idx

    order = sorted(range(len(columns)), key=lambda idx: columns[idx]["center"])
    remap = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
    columns_sorted: List[Tuple[float, float, float, float]] = []
    for new_idx, old_idx in enumerate(order):
        column = columns[old_idx]
        columns_sorted.append(
            (
                column["center"],
                column["left"],
                column["right"],
                float(column["count"]),
            )
        )
    for block_id, idx in list(assignments.items()):
        assignments[block_id] = remap[idx]

    for block in page.text_blocks:
        if block.block_id in assignments:
            continue
        if not columns_sorted:
            assignments[block.block_id] = 0
            continue
        center = (block.bbox[0] + block.bbox[2]) / 2.0
        best_idx = min(
            range(len(columns_sorted)),
            key=lambda idx: abs(center - columns_sorted[idx][0]),
        )
        assignments[block.block_id] = best_idx

    column_count = max(1, len(columns_sorted)) if columns_sorted else 1
    return column_count, assignments, columns_sorted, tolerance


def _column_inconsistency(
    blocks: Sequence[PDFBlock],
    columns: Sequence[Tuple[float, float, float, float]],
    assignments: Dict[str, int],
    tolerance: float,
    page_width: float,
) -> float:
    if len(columns) <= 1 or not blocks:
        return 0.0
    total_area = sum(block.area for block in blocks)
    if total_area <= 0:
        return 0.0
    deviation = 0.0
    slack = max(tolerance * 1.2, page_width * 0.02)
    for block in blocks:
        idx = assignments.get(block.block_id, 0)
        target = columns[idx][0]
        center = (block.bbox[0] + block.bbox[2]) / 2.0
        diff = abs(center - target)
        deviation += min(1.0, diff / slack) * block.area
    weighted = deviation / total_area
    column_centers = [col[0] / max(page_width, 1.0) for col in columns]
    spread = statistics.pstdev(column_centers) if len(column_centers) > 1 else 0.0
    return min(1.0, weighted * len(columns) + spread * 0.5)


def _off_grid_ratio(
    blocks: Sequence[PDFBlock],
    columns: Sequence[Tuple[float, float, float, float]],
    assignments: Dict[str, int],
    tolerance: float,
    page_width: float,
) -> float:
    if not blocks or not columns:
        return 0.0
    total_area = sum(block.area for block in blocks)
    if total_area <= 0:
        return 0.0
    allowed = max(tolerance * 0.75, page_width * 0.03)
    off_area = 0.0
    for block in blocks:
        idx = assignments.get(block.block_id, 0)
        center = (block.bbox[0] + block.bbox[2]) / 2.0
        target = columns[idx][0]
        if abs(center - target) > allowed:
            off_area += block.area
    return min(1.0, off_area / total_area)


def _boxiness_score(blocks: Sequence[PDFBlock], page_width: float) -> float:
    widths = [
        block.width / max(page_width, 1.0)
        for block in blocks
        if block.class_hint != "table" and block.width > 0
    ]
    if len(widths) <= 1:
        return 0.0
    spread = statistics.pstdev(widths)
    return min(1.0, spread * math.sqrt(len(widths)) * 0.8)


def _density_anomaly_score(blocks: Sequence[PDFBlock]) -> float:
    densities = [
        block.metadata.get("char_density", 0.0)
        for block in blocks
        if block.metadata.get("char_density", 0.0) > 0.0
    ]
    if len(densities) <= 1:
        return 0.0
    median = statistics.median(densities)
    if median <= 0:
        return 0.0
    deviations = [abs(density - median) for density in densities]
    mad = statistics.median(deviations)
    return min(1.0, (mad / max(median, 1e-6)) * 3.0)


def _font_variance_score(blocks: Sequence[PDFBlock]) -> float:
    font_sizes = [block.avg_font_size for block in blocks if block.avg_font_size > 0]
    if len(font_sizes) <= 1:
        return 0.0
    mean_size = sum(font_sizes) / len(font_sizes)
    variance = sum((size - mean_size) ** 2 for size in font_sizes) / len(font_sizes)
    return min(1.0, math.sqrt(variance) / max(mean_size, 1e-6))


def _reading_order_jumps(
    page: PageGraph,
    blocks: Sequence[PDFBlock],
    assignments: Dict[str, int],
) -> float:
    if len(blocks) <= 1:
        return 0.0
    sorted_blocks = sorted(blocks, key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))
    jumps = 0
    transitions = 0
    page_height = max(page.height, 1.0)
    page_width = max(page.width, 1.0)
    for prev, curr in zip(sorted_blocks, sorted_blocks[1:]):
        transitions += 1
        prev_col = assignments.get(prev.block_id, 0)
        curr_col = assignments.get(curr.block_id, 0)
        vertical_gap = max(0.0, (curr.bbox[1] - prev.bbox[3]) / page_height)
        horizontal_shift = abs(curr.bbox[0] - prev.bbox[0]) / page_width
        if prev_col != curr_col and horizontal_shift > 0.18 and vertical_gap < 0.45:
            jumps += 1
        elif vertical_gap < 0.02 and horizontal_shift > 0.25:
            jumps += 1
    if transitions == 0:
        return 0.0
    return min(1.0, jumps / transitions)


def _table_figure_intrusion(
    page: PageGraph, blocks: Sequence[PDFBlock]
) -> Tuple[float, float, float]:
    page_area = _page_area(page)
    min_area = page_area * 0.005
    prose_blocks = [block for block in blocks if block.class_hint != "table"]
    prose_area = sum(block.area for block in prose_blocks)
    if prose_area <= 0:
        return 0.0, 0.0, 0.0

    table_blocks = [
        block
        for block in page.text_blocks
        if block.class_hint == "table" and block.area >= min_area
    ]
    figure_blocks = [
        block
        for block in page.blocks
        if block.block_type == "image" and block.area >= min_area
    ]

    table_overlap = 0.0
    figure_overlap = 0.0
    for table in table_blocks:
        for prose in prose_blocks:
            table_overlap += _intersection_area(table.bbox, prose.bbox)
    for figure in figure_blocks:
        for prose in prose_blocks:
            figure_overlap += _intersection_area(figure.bbox, prose.bbox)

    table_ratio = min(1.0, table_overlap / max(prose_area, 1.0))
    figure_ratio = min(1.0, figure_overlap / max(prose_area, 1.0))
    intrusion = min(1.0, table_ratio + figure_ratio)
    return table_ratio, figure_ratio, intrusion


def _margin_small_area(page: PageGraph) -> float:
    page_area = _page_area(page)
    if page_area <= 0:
        return 0.0
    margin_band = max(page.width * 0.12, 36.0)
    min_region_area = page_area * 0.008
    margin_area = 0.0
    for block in page.blocks:
        if block.area < min_region_area:
            continue
        left, _, right, _ = block.bbox
        if right <= margin_band or left >= (page.width - margin_band):
            margin_area += block.area
    return min(1.0, (margin_area / page_area) * 2.5)


def _footnote_likelihood(
    page: PageGraph,
) -> Tuple[float, List[str], float, int]:
    text_blocks = page.text_blocks
    if not text_blocks:
        return 0.0, [], 0.0, 0
    font_sizes = [block.avg_font_size for block in text_blocks if block.avg_font_size > 0]
    dominant = sum(font_sizes) / len(font_sizes) if font_sizes else 0.0
    superscripts = sum(int(block.metadata.get("superscript_count", 0)) for block in text_blocks)
    page_area = _page_area(page)
    total_chars = sum(block.metadata.get("char_count", len(block.text)) for block in text_blocks)
    footnote_blocks: List[PDFBlock] = []
    for block in text_blocks:
        if block.avg_font_size <= 0 or block.metadata.get("char_count", 0) == 0:
            continue
        if block.bbox[1] < page.height * 0.7:
            continue
        if dominant and block.avg_font_size >= dominant * 0.85:
            continue
        if block.area / page_area < 0.002:
            continue
        footnote_blocks.append(block)
    if superscripts < 3 or not footnote_blocks or total_chars == 0:
        return 0.0, [], dominant, superscripts
    footnote_chars = sum(block.metadata.get("char_count", len(block.text)) for block in footnote_blocks)
    char_ratio = footnote_chars / max(total_chars, 1)
    score = min(1.0, char_ratio * 3.0)
    return score, [block.block_id for block in footnote_blocks], dominant, superscripts


def _page_line_density(page: PageGraph) -> Tuple[int, bool, float]:
    total_lines = sum(block.metadata.get("line_count", 0) for block in page.text_blocks)
    total_chars = sum(block.metadata.get("char_count", len(block.text)) for block in page.text_blocks)
    density = total_chars / _page_area(page)
    has_normal = 7.5e-4 <= density <= 3.5e-3
    return total_lines, has_normal, density


def compute_layout_signals(document: DocumentGraph) -> List[PageLayoutSignals]:
    """Compute raw and normalised layout signals for ``document``."""

    raw_signals: List[Dict[SignalName, float]] = []
    extras_list: List[PageSignalExtras] = []

    for page in document.pages:
        filtered_blocks = _filtered_text_blocks(page)
        column_count, assignments, columns, tolerance = _estimate_columns(page, filtered_blocks)
        cis = _column_inconsistency(filtered_blocks, columns, assignments, tolerance, page.width)
        ogr = _off_grid_ratio(filtered_blocks, columns, assignments, tolerance, page.width)
        bxs = _boxiness_score(filtered_blocks, page.width)
        das = _density_anomaly_score(filtered_blocks)
        fvs = _font_variance_score(filtered_blocks)
        roj = _reading_order_jumps(page, filtered_blocks, assignments)
        table_overlap, figure_overlap, intrusion = _table_figure_intrusion(page, filtered_blocks)
        msa = _margin_small_area(page)
        fnl, footnote_ids, dominant_font, superscripts = _footnote_likelihood(page)

        raw = {
            "CIS": cis,
            "OGR": ogr,
            "BXS": bxs,
            "DAS": das,
            "FVS": fvs,
            "ROJ": roj,
            "TFI": intrusion,
            "MSA": msa,
            "FNL": fnl,
        }
        raw_signals.append(raw)

        total_lines, has_normal_density, char_density = _page_line_density(page)
        extras_list.append(
            PageSignalExtras(
                column_count=column_count,
                column_assignments=assignments,
                table_overlap_ratio=table_overlap,
                figure_overlap_ratio=figure_overlap,
                dominant_font_size=dominant_font,
                footnote_block_ids=footnote_ids,
                superscript_spans=superscripts,
                total_line_count=total_lines,
                has_normal_density=has_normal_density,
                char_density=char_density,
                intrusion_ratio=intrusion,
            )
        )

    medians: Dict[SignalName, float] = {}
    iqrs: Dict[SignalName, float] = {}
    reference_pages = raw_signals[: min(_NORMALISATION_WINDOW, len(raw_signals))]
    for name in _SIGNAL_NAMES:
        values = [page_raw[name] for page_raw in reference_pages]
        medians[name] = _percentile(values, 50.0)
        q1 = _percentile(values, 25.0)
        q3 = _percentile(values, 75.0)
        iqrs[name] = max(0.0, q3 - q1)

    results: List[PageLayoutSignals] = []
    for page, raw, extras in zip(document.pages, raw_signals, extras_list):
        normalized = {
            name: _robust_normalise(raw[name], medians[name], iqrs[name])
            for name in _SIGNAL_NAMES
        }
        structural = max(normalized["CIS"], normalized["ROJ"], normalized["TFI"])
        extras.structural_score = structural
        composite = (
            normalized["OGR"]
            + normalized["BXS"]
            + normalized["DAS"]
            + normalized["FVS"]
            + normalized["MSA"]
            + normalized["FNL"]
        ) / 6.0
        score_prime = max(0.0, min(1.0, 0.7 * structural + 0.3 * composite))
        results.append(
            PageLayoutSignals(
                page_number=page.page_number,
                raw=raw,
                normalized=normalized,
                page_score=score_prime,
                extras=extras,
            )
        )
    return results


__all__ = ["PageLayoutSignals", "PageSignalExtras", "compute_layout_signals"]
