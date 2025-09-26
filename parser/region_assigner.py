"""Utilities for mapping detector regions to layout blocks."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .utils import Block, PageLayout

BBox = Tuple[float, float, float, float]

PRIORITY = [
    "title",
    "caption",
    "sidebar",
    "table",
    "figure",
    "paragraph",
    "list",
]


def _bbox_iou(box_a: BBox, box_b: BBox) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def _para_like(block: Block) -> bool:
    text = (block.text or "").strip()
    if not text:
        return False
    width = block.width
    height = block.height
    if width <= 0 or height <= 0:
        return False
    token_count = len(text.split())
    return token_count >= 4 and width >= height * 0.6


def _best_region(score_map: Dict[str, float]) -> Optional[str]:
    if not score_map:
        return None
    sorted_pairs = sorted(
        score_map.items(),
        key=lambda kv: (-kv[1], PRIORITY.index(kv[0]) if kv[0] in PRIORITY else len(PRIORITY)),
    )
    return sorted_pairs[0][0]


def assign_regions(
    page: PageLayout,
    detections: Sequence[Dict[str, object]],
    *,
    allow_override: bool = True,
) -> None:
    """Annotate blocks on ``page`` with detector provided region tags.

    ``detections`` is expected to contain dictionaries with keys ``cls`` and
    ``bbox_pdf``.  The assignment uses IoU-majority with tie breaking based on
    :data:`PRIORITY`.  When no detector region overlaps a block, the function
    defaults to ``paragraph`` for paragraph-like text blocks.
    """

    block_lookup: List[Block] = list(page.blocks)
    if not block_lookup:
        return

    detection_entries: List[Dict[str, object]] = []
    for idx, det in enumerate(detections or []):
        cls = str(det.get("cls", ""))
        bbox = det.get("bbox_pdf") or det.get("bbox")
        if not cls or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        detection_entries.append(
            {
                "id": f"det_{page.page_number}_{idx}",
                "cls": cls,
                "score": float(det.get("score", 0.0)),
                "bbox": (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ),
            }
        )

    for block in block_lookup:
        if not allow_override and block.attrs.get("region_tag"):
            continue
        block.attrs.pop("region_tag", None)
        block.attrs.pop("region_score", None)
        block.attrs.pop("region_id", None)
        block.attrs.pop("region_bbox", None)

        scores: Dict[str, float] = {}
        best_detection: Optional[Dict[str, object]] = None
        best_overlap = 0.0
        for det in detection_entries:
            overlap = _bbox_iou(block.bbox, det["bbox"])  # type: ignore[arg-type]
            if overlap <= 0:
                continue
            cls = str(det["cls"])
            scores[cls] = scores.get(cls, 0.0) + overlap
            if overlap > best_overlap:
                best_overlap = overlap
                best_detection = det

        region = _best_region(scores)
        if region is None and _para_like(block):
            region = "paragraph"

        if region:
            block.attrs["region_tag"] = region
        if best_detection:
            block.attrs["region_score"] = float(best_detection.get("score", 0.0))
            block.attrs["region_id"] = best_detection.get("id")
            block.attrs["region_bbox"] = best_detection.get("bbox")

