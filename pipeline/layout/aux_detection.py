"""Auxiliary block detection heuristics tuned for textbook-style PDFs."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pipeline.ingest.pdf_parser import PDFBlock, PageGraph

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]

_AUX_LEXICON = (
    "fig.",
    "figure",
    "table",
    "source",
    "activity",
    "discuss",
    "think",
    "imagine",
    "let's",
    "lets",
    "keywords",
)


@dataclass(slots=True)
class AuxBlockRecord:
    block_id: str
    category: str
    confidence: float
    reasons: List[str] = field(default_factory=list)
    figure_bbox: Optional[BBox] = None
    anchor_hint: Optional[str] = None
    is_caption: bool = False
    is_footnote: bool = False


@dataclass(slots=True)
class AuxDetectionResult:
    blocks: Dict[str, AuxBlockRecord]
    caption_links: Dict[str, BBox]
    footnote_ids: List[str]


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _lexical_score(text: str, lexicon: Sequence[str]) -> Tuple[float, Optional[str]]:
    lowered = _normalise(text)
    for token in lexicon:
        if lowered.startswith(token):
            if token.startswith("table"):
                return 0.6, "table"
            if token.startswith("fig") or token.startswith("figure"):
                return 0.6, "figure"
            return 0.5, None
    return 0.0, None


def _column_width(page: PageGraph, column_assignments: Mapping[str, int] | None) -> float:
    if not column_assignments:
        return page.width
    column_count = (max(column_assignments.values()) + 1) if column_assignments else 1
    column_count = max(1, column_count)
    return page.width / column_count


def _column_positions(page: PageGraph, column_assignments: Mapping[str, int] | None) -> Dict[int, float]:
    if not column_assignments:
        return {}
    positions: Dict[int, List[float]] = {}
    for block in page.text_blocks:
        column = column_assignments.get(block.block_id)
        if column is None:
            continue
        positions.setdefault(column, []).append(block.bbox[0])
    return {column: sum(xs) / len(xs) for column, xs in positions.items() if xs}


def _iou(a: BBox, b: BBox) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    if union <= 0:
        return 0.0
    return max(0.0, min(1.0, inter / union))


def _distance_to_bbox(block: PDFBlock, bbox: BBox) -> float:
    cx = (block.bbox[0] + block.bbox[2]) / 2.0
    cy = (block.bbox[1] + block.bbox[3]) / 2.0
    bx = (bbox[0] + bbox[2]) / 2.0
    by = (bbox[1] + bbox[3]) / 2.0
    return math.sqrt((cx - bx) ** 2 + (cy - by) ** 2)


def detect_auxiliary_blocks(
    page: PageGraph,
    *,
    body_font_size: float,
    column_assignments: Mapping[str, int] | None = None,
    figure_regions: Sequence[BBox] | None = None,
    lexicon: Sequence[str] = _AUX_LEXICON,
) -> AuxDetectionResult:
    """Classify auxiliary blocks for ``page`` using lexical and geometric cues."""

    figure_regions = figure_regions or []
    column_centres = _column_positions(page, column_assignments)
    column_width = _column_width(page, column_assignments)
    result: Dict[str, AuxBlockRecord] = {}
    caption_links: Dict[str, BBox] = {}
    footnote_ids: List[str] = []

    for block in page.text_blocks:
        if block.metadata.get("suppressed"):
            continue
        text = block.text.strip()
        if not text or len(text) < 2:
            continue

        score = 0.0
        reasons: List[str] = []
        lexical_score, lexical_target = _lexical_score(text, lexicon)
        if lexical_score:
            score += lexical_score
            reasons.append("lexical")

        font_ratio = block.avg_font_size / max(body_font_size or 1.0, 1.0)
        if font_ratio <= 0.85:
            score += 0.2
            reasons.append("small-font")

        width_ratio = block.width / max(column_width, 1.0)
        if width_ratio <= 0.7:
            score += 0.15
            reasons.append("narrow")

        column = column_assignments.get(block.block_id) if column_assignments else None
        off_grid = False
        if column is not None and column in column_centres:
            offset = abs(block.bbox[0] - column_centres[column])
            if offset > column_width * 0.2:
                off_grid = True
        elif column is None and width_ratio < 0.5:
            off_grid = True
        if off_grid:
            score += 0.15
            reasons.append("off-grid")

        centre_x = (block.bbox[0] + block.bbox[2]) / 2.0
        centred = abs(centre_x - (page.width / 2.0)) <= page.width * 0.12
        if centred:
            score += 0.1
            reasons.append("centred")

        nearest_bbox: Optional[BBox] = None
        if figure_regions:
            ious = [(_iou(block.bbox, bbox), bbox) for bbox in figure_regions]
            if ious:
                top_iou, bbox = max(ious, key=lambda item: item[0])
                if top_iou >= 0.05:
                    nearest_bbox = bbox
                    score += 0.2
                    reasons.append("figure-proximity")
                else:
                    # fallback to distance weighting
                    closest = min(figure_regions, key=lambda bbox: _distance_to_bbox(block, bbox))
                    dist = _distance_to_bbox(block, closest)
                    if dist < max(page.width, page.height) * 0.15:
                        nearest_bbox = closest
                        score += 0.15
                        reasons.append("figure-near")

        is_bottom_band = block.bbox[1] >= page.height * 0.82
        has_superscript = block.metadata.get("superscript_count", 0) >= 3
        if is_bottom_band and font_ratio <= 0.8:
            score = max(score, 0.6)
            reasons.append("footnote-band")
            if has_superscript:
                reasons.append("superscript")
            footnote_ids.append(block.block_id)

        if score < 0.5 and block.block_id not in footnote_ids:
            continue

        category = "aux"
        is_caption = False
        is_footnote = block.block_id in footnote_ids
        if is_footnote:
            category = "footnote"
        elif lexical_target == "table":
            category = "table_caption"
            is_caption = True
        elif lexical_target == "figure":
            category = "figure_caption"
            is_caption = True
        elif nearest_bbox is not None:
            category = "figure_caption"
            is_caption = True
        elif off_grid or centred:
            category = "callout"

        record = AuxBlockRecord(
            block_id=block.block_id,
            category=category,
            confidence=min(1.0, score),
            reasons=reasons,
            figure_bbox=nearest_bbox,
            anchor_hint="after",
            is_caption=is_caption,
            is_footnote=is_footnote,
        )
        result[block.block_id] = record
        if nearest_bbox is not None:
            caption_links[block.block_id] = nearest_bbox

    return AuxDetectionResult(blocks=result, caption_links=caption_links, footnote_ids=footnote_ids)


__all__ = ["AuxBlockRecord", "AuxDetectionResult", "detect_auxiliary_blocks"]
