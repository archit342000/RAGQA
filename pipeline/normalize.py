from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Dict, Any, Optional, TYPE_CHECKING

import re

from .config import PipelineConfig

from .docling_adapter import DoclingBlock
from .ids import make_block_id

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from .telemetry import Telemetry


@dataclass(slots=True)
class Block:
    doc_id: str
    block_id: str
    page: int
    order: int
    type: str
    text: str
    bbox: dict
    heading_level: int | None
    heading_path: List[str]
    source: dict
    aux: dict
    role: str
    aux_subtype: str | None
    parent_block_id: str | None
    role_confidence: float


def normalise_blocks(
    doc_id: str,
    blocks: Sequence[DoclingBlock],
    config: PipelineConfig,
    telemetry: "Telemetry | None" = None,
) -> List[Block]:
    normalised: List[Block] = []
    header_footer_stats = _compute_header_footer_stats(blocks, config)
    for order, block in enumerate(blocks):
        bbox = None
        if block.bbox:
            x0, y0, x1, y1 = block.bbox
            bbox = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        else:
            bbox = {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}
        block_type = block.block_type if block.block_type in {
            "heading",
            "paragraph",
            "list",
            "item",
            "table",
            "figure",
            "caption",
            "code",
            "footnote",
        } else "paragraph"
        role, aux_subtype, parent_block_id, confidence, drop, flag = _classify_block(
            block_type,
            block.text,
            block.page_number,
            bbox,
            header_footer_stats,
            config,
        )
        if drop:
            if telemetry is not None:
                telemetry.inc("aux_discarded")
            if flag and telemetry is not None:
                telemetry.flag(flag)
            continue
        normalised.append(
            Block(
                doc_id=doc_id,
                block_id=make_block_id(block.page_number, order + 1),
                page=block.page_number,
                order=order,
                type=block_type,
                text=block.text,
                bbox=bbox,
                heading_level=block.heading_level,
                heading_path=list(block.heading_path),
                source={
                    "stage": block.source_stage,
                    "tool": block.source_tool,
                    "version": block.source_version,
                },
                aux=dict(block.aux),
                role=role,
                aux_subtype=aux_subtype,
                parent_block_id=parent_block_id,
                role_confidence=confidence,
            )
        )
        if flag and telemetry is not None:
            telemetry.flag(flag)
    return normalised


def _compute_header_footer_stats(
    blocks: Sequence[DoclingBlock], config: PipelineConfig
) -> Dict[str, Any]:
    y_band = config.aux.y_band_pct
    page_extents: Dict[int, Tuple[float, float]] = {}
    for block in blocks:
        if not block.bbox:
            continue
        _, y0, _, y1 = block.bbox
        stats = page_extents.setdefault(block.page_number, (y0, y1))
        page_extents[block.page_number] = (min(stats[0], y0), max(stats[1], y1))
    header_counts: Dict[str, set[int]] = {}
    footer_counts: Dict[str, set[int]] = {}
    for block in blocks:
        text = (block.text or "").strip()
        if not text:
            continue
        if not block.bbox:
            continue
        _, y0, _, y1 = block.bbox
        ymin, ymax = page_extents.get(block.page_number, (0.0, 1.0))
        span = max(ymax - ymin, 1.0)
        norm_y0 = (y0 - ymin) / span
        norm_y1 = (y1 - ymin) / span
        if norm_y1 <= y_band:
            header_counts.setdefault(text, set()).add(block.page_number)
        if norm_y0 >= (1.0 - y_band):
            footer_counts.setdefault(text, set()).add(block.page_number)
    total_pages = max((block.page_number for block in blocks), default=0)
    threshold = max(1, int(total_pages * config.aux.header_footer.repetition_threshold))
    repeated_headers = {
        text for text, pages in header_counts.items() if len(pages) >= threshold
    }
    repeated_footers = {
        text for text, pages in footer_counts.items() if len(pages) >= threshold
    }
    return {
        "headers": repeated_headers,
        "footers": repeated_footers,
        "page_extents": page_extents,
    }


_CAPTION_RE = re.compile(r"^(fig(ure)?\.?|table\.?)", re.IGNORECASE)
_ACTIVITY_RE = re.compile(r"^(activity|exercise|try.?it|let.?s\s+(recall|discuss))", re.IGNORECASE)
_SOURCE_RE = re.compile(r"^source\s+\d+", re.IGNORECASE)
_PEDAGOGY_RE = re.compile(r"let.?s\s+(recall|discuss)", re.IGNORECASE)
_HEADER_FOOTER_RE = re.compile(r"reprint\s+\d{4}-\d{2,}", re.IGNORECASE)


def _classify_block(
    block_type: str,
    text: str,
    page_number: int,
    bbox: Dict[str, float],
    stats: Dict[str, Any],
    config: PipelineConfig,
) -> Tuple[str, Optional[str], Optional[str], float, bool, Optional[str]]:
    cleaned = (text or "").strip()
    role = "main"
    aux_subtype: Optional[str] = None
    parent: Optional[str] = None
    confidence = 0.7
    drop = False
    flag: Optional[str] = None

    if block_type in {"figure", "table"}:
        role = "auxiliary"
        aux_subtype = "sidebar"
        confidence = 0.85
        return role, aux_subtype, parent, confidence, drop, flag

    if block_type == "caption" or _CAPTION_RE.match(cleaned):
        role = "auxiliary"
        aux_subtype = "caption"
        confidence = 0.95
        return role, aux_subtype, parent, confidence, drop, flag

    if _SOURCE_RE.match(cleaned):
        role = "auxiliary"
        aux_subtype = "source"
        confidence = 0.9
        return role, aux_subtype, parent, confidence, drop, flag

    if _ACTIVITY_RE.match(cleaned):
        role = "auxiliary"
        aux_subtype = "activity"
        confidence = 0.9
        return role, aux_subtype, parent, confidence, drop, flag

    y_band = config.aux.y_band_pct
    ymin, ymax = stats.get("page_extents", {}).get(page_number, (0.0, 1.0))
    span = max(ymax - ymin, 1.0)
    norm_y0 = (bbox.get("y0", 0.0) - ymin) / span
    norm_y1 = (bbox.get("y1", 0.0) - ymin) / span

    if block_type == "footnote" or norm_y0 >= (1.0 - y_band):
        role = "auxiliary"
        aux_subtype = "footnote"
        confidence = 0.8
        return role, aux_subtype, parent, confidence, drop, flag

    if cleaned and cleaned in stats.get("headers", set()):
        if block_type == "heading":
            # Collision between header detection and true heading
            flag = "AUX03"
        else:
            role = "auxiliary"
            aux_subtype = "header"
            confidence = 0.95
            drop = True
            if _PEDAGOGY_RE.search(cleaned):
                flag = "AUX03"
            return role, aux_subtype, parent, confidence, drop, flag

    if cleaned and cleaned in stats.get("footers", set()):
        role = "auxiliary"
        aux_subtype = "footer"
        confidence = 0.95
        drop = True
        return role, aux_subtype, parent, confidence, drop, flag

    if _HEADER_FOOTER_RE.search(cleaned):
        role = "auxiliary"
        aux_subtype = "footer"
        confidence = 0.6
        drop = True
        return role, aux_subtype, parent, confidence, drop, flag

    if block_type == "heading":
        confidence = 0.9
        return role, aux_subtype, parent, confidence, drop, flag

    if not cleaned:
        role = "auxiliary"
        aux_subtype = "other"
        confidence = 0.3
        flag = "AUX01"
        return role, aux_subtype, parent, confidence, drop, flag

    return role, aux_subtype, parent, confidence, drop, flag
