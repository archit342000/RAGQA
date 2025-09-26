"""Fuse detector regions with grouped blocks."""
from __future__ import annotations

from typing import Dict, Iterable, List

from .grouping import Block


def _iou(b1, b2) -> float:
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def fuse_regions(
    blocks: Iterable[Block],
    regions: List[Dict[str, object]],
    cfg: Dict[str, object],
) -> List[Block]:
    """Assign region tags to blocks using IoU-majority."""

    priority = [
        "title",
        "caption",
        "sidebar",
        "table",
        "figure",
        "text",
    ]
    for block in blocks:
        best_tag = None
        best_score = 0.0
        for region in regions:
            bbox = tuple(region.get("bbox", (0, 0, 0, 0)))
            score = _iou(block.bbox, bbox)
            if score > 0:
                cls = str(region.get("class"))
                rank = priority.index(cls) if cls in priority else len(priority)
                cand_rank = priority.index(best_tag) if best_tag in priority else len(priority)
                if score > best_score or (score == best_score and rank < cand_rank):
                    best_score = score
                    best_tag = cls
        if best_tag:
            block.region_tag = best_tag
        else:
            block.region_tag = block.region_tag or "text"
    return list(blocks)
