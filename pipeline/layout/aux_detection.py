"""Auxiliary block detection heuristics tuned for textbook-style PDFs."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from pipeline.ingest.pdf_parser import PDFBlock, PageGraph

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "parser.yaml"


def _default_config() -> Dict[str, object]:
    return {
        "font_ratio_threshold": 0.85,
        "width_ratio_threshold": 0.7,
        "center_tolerance_ratio": 0.12,
        "off_grid_tolerance_ratio": 0.2,
        "bottom_band_ratio": 0.15,
        "min_confidence": 0.5,
        "figure_iou_threshold": 0.05,
        "figure_distance_ratio": 0.15,
        "lexicon": [
            "Fig.",
            "Figure",
            "Table",
            "Source",
            "Activity",
            "Discuss",
            "Think",
            "Imagine",
            "Let's",
            "Let’s",
            "Keywords",
        ],
    }


@lru_cache(maxsize=1)
def _aux_config() -> Dict[str, object]:
    config = _default_config()
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        data = {}
    except Exception as exc:  # pragma: no cover - configuration errors are non-fatal
        logger.debug("Failed to load parser config %s: %s", _CONFIG_PATH, exc)
        data = {}
    aux_cfg = data.get("aux_detection") if isinstance(data, dict) else None
    if isinstance(aux_cfg, dict):
        for key, value in aux_cfg.items():
            if key == "lexicon" and isinstance(value, list):
                config["lexicon"] = [str(item) for item in value]
            elif key in config:
                config[key] = value
    return config


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
    owner_section_id: Optional[str] = None
    adjacent_to_figure: bool = False
    references: List[str] = field(default_factory=list)


@dataclass(slots=True)
class AuxDetectionResult:
    blocks: Dict[str, AuxBlockRecord]
    caption_links: Dict[str, BBox]
    footnote_ids: List[str]
def _lexical_category(text: str, lexicon: Sequence[str]) -> Optional[str]:
    stripped = text.strip()
    if not stripped:
        return None
    for token in lexicon:
        pattern = rf"^{re.escape(token)}\b"
        if re.match(pattern, stripped, re.IGNORECASE):
            lowered = token.lower()
            if lowered.startswith("table"):
                return "table"
            if lowered.startswith("fig"):
                return "figure"
            if lowered.startswith("source"):
                return "source"
            if lowered.startswith("activity"):
                return "activity"
            if lowered.startswith("discuss"):
                return "discuss"
            if lowered.startswith("think"):
                return "think"
            if lowered.startswith("imagine"):
                return "imagine"
            if "keyword" in lowered:
                return "keywords"
            return "aux"
    return None


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
    section_map: Mapping[str, str] | None = None,
) -> AuxDetectionResult:
    """Classify auxiliary blocks for ``page`` using lexical and geometric cues."""

    config = _aux_config()
    lexicon = config.get("lexicon", [])
    figure_regions = figure_regions or []
    column_centres = _column_positions(page, column_assignments)
    column_width = _column_width(page, column_assignments)
    off_grid_tol = max(column_width * float(config.get("off_grid_tolerance_ratio", 0.2)), 1.0)
    centre_tol = page.width * float(config.get("center_tolerance_ratio", 0.12))
    bottom_band_threshold = page.height * (1.0 - float(config.get("bottom_band_ratio", 0.15)))
    min_confidence = float(config.get("min_confidence", 0.5))
    distance_threshold = max(page.width, page.height) * float(config.get("figure_distance_ratio", 0.15))
    iou_threshold = float(config.get("figure_iou_threshold", 0.05))

    result: Dict[str, AuxBlockRecord] = {}
    caption_links: Dict[str, BBox] = {}
    footnote_ids: List[str] = []

    for block in page.text_blocks:
        if block.metadata.get("suppressed"):
            continue
        text = block.text.strip()
        if len(text) < 2:
            continue

        reasons: List[str] = []
        score = 0.0
        lexical_category = _lexical_category(text, lexicon) if isinstance(lexicon, Iterable) else None
        if lexical_category:
            score += 0.4
            reasons.append("lexical")

        font_ratio = block.avg_font_size / max(body_font_size or 1.0, 1.0)
        if font_ratio <= float(config.get("font_ratio_threshold", 0.85)):
            score += 0.25
            reasons.append("small-font")

        width_ratio = block.width / max(column_width, 1.0)
        if width_ratio <= float(config.get("width_ratio_threshold", 0.7)):
            score += 0.2
            reasons.append("narrow")

        column = column_assignments.get(block.block_id) if column_assignments else None
        off_grid = False
        if column is not None and column in column_centres:
            offset = abs(block.bbox[0] - column_centres[column])
            if offset > off_grid_tol:
                off_grid = True
        elif column is None and width_ratio < 0.5:
            off_grid = True
        if off_grid:
            score += 0.15
            reasons.append("off-grid")

        centre_x = (block.bbox[0] + block.bbox[2]) / 2.0
        centred = abs(centre_x - (page.width / 2.0)) <= centre_tol
        if centred:
            score += 0.1
            reasons.append("centred")

        nearest_bbox: Optional[BBox] = None
        if figure_regions:
            ious = [(_iou(block.bbox, bbox), bbox) for bbox in figure_regions]
            if ious:
                top_iou, bbox = max(ious, key=lambda item: item[0])
                if top_iou >= iou_threshold:
                    nearest_bbox = bbox
                    score += 0.2
                    reasons.append("figure-proximity")
                else:
                    closest = min(figure_regions, key=lambda bbox: _distance_to_bbox(block, bbox))
                    dist = _distance_to_bbox(block, closest)
                    if dist <= distance_threshold:
                        nearest_bbox = closest
                        score += 0.15
                        reasons.append("figure-near")

        superscripts = block.metadata.get("superscript_count", 0)
        is_bottom_band = block.bbox[1] >= bottom_band_threshold
        is_footnote = False
        if is_bottom_band and font_ratio <= 0.8 and superscripts >= 3:
            is_footnote = True
            score = max(score, 0.75)
            reasons.append("footnote-band")
            reasons.append("superscripts")
            footnote_ids.append(block.block_id)

        if score < min_confidence and not is_footnote:
            continue

        category = "aux"
        is_caption = False
        if is_footnote:
            category = "footnote"
        elif lexical_category == "table":
            category = "table_caption"
            is_caption = True
        elif lexical_category == "figure":
            category = "figure_caption"
            is_caption = True
        elif nearest_bbox is not None:
            category = "figure_caption"
            is_caption = True
        elif lexical_category in {"activity", "discuss", "think", "imagine", "keywords", "source"}:
            category = "callout"
        elif off_grid or centred:
            category = "callout"

        owner_section = None
        if section_map and block.block_id in section_map:
            owner_section = section_map[block.block_id]
        else:
            owner_section = block.metadata.get("section_id")

        references = []
        footnote_refs = block.metadata.get("footnote_refs")
        if isinstance(footnote_refs, list):
            references = [str(ref) for ref in footnote_refs]

        record = AuxBlockRecord(
            block_id=block.block_id,
            category=category,
            confidence=min(1.0, score),
            reasons=reasons,
            figure_bbox=nearest_bbox,
            anchor_hint="section-end",
            is_caption=is_caption,
            is_footnote=is_footnote,
            owner_section_id=str(owner_section) if owner_section else None,
            adjacent_to_figure=nearest_bbox is not None,
            references=references,
        )
        result[block.block_id] = record
        block.metadata["owner_section_id"] = record.owner_section_id
        if nearest_bbox is not None:
            caption_links[block.block_id] = nearest_bbox

    return AuxDetectionResult(blocks=result, caption_links=caption_links, footnote_ids=footnote_ids)


__all__ = ["AuxBlockRecord", "AuxDetectionResult", "detect_auxiliary_blocks"]
