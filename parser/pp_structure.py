"""PP-Structure layout inference helpers and region tagging utilities."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import logging

try:  # pragma: no cover - optional dependency for runtime environments with paddleocr
    from paddleocr import PPStructure  # type: ignore
except Exception:  # pragma: no cover - paddleocr may be unavailable in tests
    PPStructure = None  # type: ignore

try:  # pragma: no cover - optional dependency for rendering PDF pages
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None  # type: ignore

from .utils import PageLayout

LOGGER = logging.getLogger(__name__)

REGION_KEYS: Tuple[str, ...] = ("title", "text", "figure", "table", "list")


def _normalise_boxes(boxes: Iterable[Sequence[float]]) -> List[List[float]]:
    normalised: List[List[float]] = []
    for box in boxes:
        if len(box) != 4:
            continue
        try:
            x0, y0, x1, y1 = [float(coord) for coord in box]
        except (TypeError, ValueError):  # pragma: no cover - guard against malformed input
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        normalised.append([x0, y0, x1, y1])
    return normalised


def _empty_regions() -> Dict[str, List[List[float]]]:
    return {key: [] for key in REGION_KEYS}


@lru_cache(maxsize=1)
def _load_pp_structure() -> Optional["PPStructure"]:
    if PPStructure is None:  # pragma: no cover - dependency may not be installed
        return None
    try:  # pragma: no cover - heavy runtime path, skipped in tests
        return PPStructure(layout=True, show_log=False)
    except Exception:  # pragma: no cover - safety guard
        LOGGER.exception("Failed to initialise PP-Structure; continuing without it")
        return None


def _render_page(pdf_path: Path, page_number: int, dpi: int = 144) -> Optional["fitz.Pixmap"]:
    if fitz is None:  # pragma: no cover - optional dependency missing
        return None
    try:  # pragma: no cover - heavy runtime path
        doc = fitz.open(pdf_path)
        try:
            page = doc.load_page(page_number)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            return pix
        finally:
            doc.close()
    except Exception:  # pragma: no cover - guard against rendering issues
        LOGGER.exception("Failed to render page %s for PP-Structure", page_number)
        return None


def _extract_regions_from_runtime(pix: "fitz.Pixmap") -> Dict[str, List[List[float]]]:
    model = _load_pp_structure()
    if model is None:  # pragma: no cover - runtime fallback
        return _empty_regions()
    try:  # pragma: no cover - heavy runtime path
        img = pix.tobytes("ppm")
        result = model(img)
    except Exception:  # pragma: no cover - guard for paddleocr runtime issues
        LOGGER.exception("PP-Structure inference failed; returning empty regions")
        return _empty_regions()
    layout_regions: Dict[str, List[List[float]]] = _empty_regions()
    for item in result:
        label = str(item.get("type", "text")).lower()
        box = item.get("bbox") or item.get("box")
        if label not in layout_regions:
            continue
        boxes = layout_regions[label]
        boxes.extend(_normalise_boxes([box]))
    return layout_regions


def pp_structure_layout(
    page: PageLayout,
    pdf_path: Optional[Path] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, List[List[float]]]:
    """Return PP-Structure regions for *page*.

    The helper first checks whether the page already carries ``pp_regions`` in
    its metadata (allowing tests to inject deterministic fixtures).  When the
    optional :mod:`paddleocr` dependency is available the function attempts to
    render the page and run the PP-Structure layout model.  Failures degrade to
    an empty set of regions so the downstream pipeline can continue using
    margin-based heuristics.
    """

    if isinstance(page.meta, Mapping):
        existing = page.meta.get("pp_regions")
        if isinstance(existing, Mapping):
            return {key: _normalise_boxes(existing.get(key, [])) for key in REGION_KEYS}
    if pdf_path is None:
        return _empty_regions()
    pix = _render_page(pdf_path, page.page_number)
    if pix is None:  # pragma: no cover - runtime fallback when rendering fails
        return _empty_regions()
    regions = _extract_regions_from_runtime(pix)
    page.meta["pp_regions"] = regions
    return regions


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _is_prose_like(block: Any) -> bool:
    if getattr(block, "block_type", "") != "text":
        return False
    text = getattr(block, "text", "") or ""
    if len(text.split()) < 5:
        return False
    width = getattr(block, "width", 0.0) or 0.0
    if width <= 0:
        return False
    indent = float(block.attrs.get("indent", 0.0)) if hasattr(block, "attrs") else 0.0
    return indent <= width * 0.25


def tag_blocks_with_regions(
    blocks: Sequence[Any],
    regions: Optional[Mapping[str, Sequence[Sequence[float]]]] = None,
) -> List[Any]:
    """Assign ``region_tag`` to each block using IoU-majority voting."""

    regions = regions or {}
    prepared = {key: _normalise_boxes(regions.get(key, [])) for key in REGION_KEYS}
    for block in blocks:
        bbox = getattr(block, "bbox", None)
        if bbox is None:
            continue
        votes: Dict[str, float] = {}
        for label, boxes in prepared.items():
            if not boxes:
                continue
            best = max((_bbox_iou(bbox, box) for box in boxes), default=0.0)
            if best > 0.0:
                votes[label] = best
        chosen: Optional[str] = None
        if votes:
            # Identify label with maximum IoU; resolve ties by favouring text for prose-like blocks.
            best_score = max(votes.values())
            top_labels = [label for label, score in votes.items() if abs(score - best_score) < 1e-6]
            if len(top_labels) == 1:
                chosen = top_labels[0]
            elif "text" in top_labels and _is_prose_like(block):
                chosen = "text"
            else:
                # deterministic ordering for ties
                chosen = sorted(top_labels)[0]
        elif _is_prose_like(block):
            chosen = "text"
        if hasattr(block, "attrs"):
            if chosen is None:
                block.attrs.pop("region_tag", None)
            else:
                block.attrs["region_tag"] = chosen
    return list(blocks)
