"""Merge cheap and strong extraction outputs."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .utils import (
    Block,
    DocumentLayout,
    PageLayout,
    bbox_union,
    load_config,
    symmetric_band,
)


def merge_layouts(
    cheap: DocumentLayout,
    strong: Optional[DocumentLayout],
    config: Optional[Dict[str, object]] = None,
) -> DocumentLayout:
    cfg = config or load_config()
    strong_pages = {page.page_number: page for page in strong.pages} if strong else {}
    merged_pages: List[PageLayout] = []
    for cheap_page in cheap.pages:
        strong_page = strong_pages.get(cheap_page.page_number)
        if strong_page:
            blocks = reconcile_blocks(cheap_page, strong_page)
        else:
            blocks = [block.copy() for block in cheap_page.blocks]
        merged_page = PageLayout(
            page_number=cheap_page.page_number,
            width=cheap_page.width,
            height=cheap_page.height,
            blocks=blocks,
            meta=dict(cheap_page.meta),
        )
        apply_false_wraparound_fix(merged_page, cfg)
        merged_pages.append(merged_page)
    return DocumentLayout(pages=merged_pages, meta=dict(cheap.meta))


def reconcile_blocks(cheap_page: PageLayout, strong_page: PageLayout) -> List[Block]:
    strong_blocks = [block.copy() for block in strong_page.blocks]
    cheap_blocks = [block.copy() for block in cheap_page.blocks]
    matched_strong = set()
    for block in cheap_blocks:
        best_idx = None
        best_overlap = 0.0
        for idx, strong_block in enumerate(strong_blocks):
            overlap = _iou(block.bbox, strong_block.bbox)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        if best_idx is not None and best_overlap >= 0.3:
            strong_block = strong_blocks[best_idx]
            matched_strong.add(best_idx)
            if len(strong_block.lines) >= len(block.lines):
                block.lines = strong_block.lines
                block.bbox = strong_block.bbox
            block.attrs.update(strong_block.attrs)
    leftovers = [strong_blocks[idx] for idx in range(len(strong_blocks)) if idx not in matched_strong]
    return cheap_blocks + leftovers


def apply_false_wraparound_fix(page: PageLayout, config: Dict[str, object]) -> None:
    wrapping = config.get("wrapping", {})
    width_threshold = float(wrapping.get("wrap_text_width_max_pct_of_col", 0.6))
    tolerance = float(wrapping.get("symmetric_band_tolerance_pct", 0.1))
    text_blocks = [block for block in page.blocks if block.block_type == "text"]
    consumed: List[Block] = []
    for block in list(text_blocks):
        width_pct = block.width / page.width if page.width else 0.0
        if width_pct >= width_threshold:
            continue
        if not (
            symmetric_band(block, page.width, tolerance)
            or block.attrs.get("col_id") == 1
        ):
            continue
        neighbor = _nearest_column_block(page, block, consumed)
        if neighbor is None:
            continue
        neighbor.lines.extend(block.lines)
        neighbor.bbox = bbox_union(neighbor.bbox, block.bbox)
        consumed.append(block)
    if consumed:
        page.blocks = [block for block in page.blocks if block not in consumed]


def _nearest_column_block(
    page: PageLayout, band_block: Block, exclude: List[Block]
) -> Optional[Block]:
    candidates = []
    for block in page.blocks:
        if block is band_block or block in exclude:
            continue
        if block.block_type != "text":
            continue
        vertical_overlap = min(block.bottom, band_block.bottom) - max(block.top, band_block.top)
        if vertical_overlap <= 0:
            continue
        distance = min(abs(block.left - band_block.left), abs(block.right - band_block.right))
        candidates.append((distance, -vertical_overlap, block))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union
