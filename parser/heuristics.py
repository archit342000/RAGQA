"""Heuristic helpers for hotfix rules."""
from __future__ import annotations

from math import fabs
from statistics import pstdev
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import re

from .utils import slugify

CAPTION_RE = re.compile(r"^(Fig\.|Figure|Plate)\s*\d+\b|^(Map|Table)\s*\d+\b", re.I)
ACTIVITY_RE = re.compile(r"^(Activity|Discuss|Think|Did you know)\b", re.I)
PAGE_NO_RE = re.compile(r"^\s*\d{1,3}\s*$")


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _distance_to_bbox(point: Tuple[float, float], bbox: Sequence[float]) -> float:
    px, py = point
    x0, y0, x1, y1 = bbox
    if px < x0:
        dx = x0 - px
    elif px > x1:
        dx = px - x1
    else:
        dx = 0.0
    if py < y0:
        dy = y0 - py
    elif py > y1:
        dy = py - y1
    else:
        dy = 0.0
    return (dx ** 2 + dy ** 2) ** 0.5


def in_caption_ring(
    bbox: Sequence[float], fig_bbox: Sequence[float], Rin: float = 32.0, Rout: float = 120.0
) -> bool:
    """Return True if block bbox lies within the caption ring of a figure."""

    if not bbox or not fig_bbox:
        return False
    cx, cy = _bbox_center(bbox)
    distance = _distance_to_bbox((cx, cy), fig_bbox)
    return Rin <= distance <= Rout


def looks_like_body(block: Dict[str, Any], body_stats: Any) -> bool:
    meta = block.get("meta", {})
    if block.get("block_type") != "text":
        return False
    font_size = meta.get("font_size") or 0.0
    indent = meta.get("indent") or 0.0
    token_count = meta.get("token_count", 0)
    if token_count < 4:
        return False
    if fabs(font_size - (body_stats.font_size or 0.0)) > max(1.5, body_stats.font_size * 0.15):
        return False
    if fabs(indent - (body_stats.indent or 0.0)) > max(6.0, body_stats.indent * 0.5):
        return False
    return True


def aligns_with_body_indent(block: Dict[str, Any], body_stats: Any, tol_px: float = 6.0) -> bool:
    meta = block.get("meta", {})
    if block.get("block_type") != "text":
        return False
    indent = meta.get("indent") or 0.0
    return fabs(indent - (body_stats.indent or 0.0)) <= tol_px


def callout_cues(block: Dict[str, Any], body_stats: Any, cfg: Dict[str, Any]) -> Tuple[int, Dict[str, bool]]:
    meta = block.get("meta", {})
    text = (block.get("text") or "").strip()
    cues = {
        "keyword": bool(ACTIVITY_RE.match(text)),
        "inset": False,
        "leading": False,
        "caps_title": False,
    }
    inset_min = float(cfg.get("callout", {}).get("min_inset_px", 12))
    width_ratio_max = float(cfg.get("callout", {}).get("max_width_ratio_vs_body", 0.85))
    leading_delta = float(cfg.get("callout", {}).get("min_leading_ratio_delta", 0.10))

    indent = meta.get("indent") or 0.0
    col_width = body_stats.col_width or 1.0
    width = meta.get("width") or 0.0
    if indent - (body_stats.indent or 0.0) >= inset_min or (width / col_width) <= width_ratio_max:
        cues["inset"] = True

    line_height = meta.get("line_height") or 0.0
    if body_stats.font_size:
        leading_ratio = line_height / body_stats.font_size
        cues["leading"] = leading_ratio >= (1.0 + leading_delta)

    if text and len(text.split()) <= 6 and text.split()[0].isupper():
        cues["caps_title"] = text.split()[0].isupper() and text.split()[0].isalpha()

    count = sum(1 for v in cues.values() if v)
    return count, cues


def is_running_header(text: str, patterns: Dict[str, float]) -> bool:
    key = slugify(text)
    return bool(key and key in patterns)


def is_running_footer(text: str, patterns: Dict[str, float]) -> bool:
    key = slugify(text)
    return bool(key and key in patterns)


def normalise_band_text(text: str) -> str:
    return slugify(text or "")


def compute_running_patterns(
    doc: Any, cfg: Dict[str, Any]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    bands = cfg.get("bands", {})
    header_pct = float(bands.get("header_band_h_pct", 0.12))
    footer_pct = float(bands.get("footer_band_h_pct", 0.12))
    repetition_cfg = cfg.get("repetition", {})
    min_pages = int(repetition_cfg.get("min_pages", 3))
    jitter_px = float(repetition_cfg.get("max_vertical_jitter_px", 20.0))

    header_hits: Dict[str, list] = {}
    footer_hits: Dict[str, list] = {}

    for page in getattr(doc, "pages", []):
        height = page.height or 1.0
        header_cutoff = header_pct * height
        footer_cutoff = (1.0 - footer_pct) * height
        for block in page.blocks:
            if block.block_type != "text":
                continue
            text = (block.text or "").strip()
            if not text:
                continue
            key = normalise_band_text(text)
            if not key:
                continue
            y0, y1 = block.bbox[1], block.bbox[3]
            y_center = (y0 + y1) / 2.0
            if y_center <= header_cutoff:
                header_hits.setdefault(key, []).append(y_center)
            elif y_center >= footer_cutoff:
                footer_hits.setdefault(key, []).append(y_center)

    def _filter_hits(counts: Dict[str, list]) -> Dict[str, float]:
        patterns: Dict[str, float] = {}
        for key, ys in counts.items():
            if len(ys) < min_pages:
                continue
            if len(ys) == 1:
                patterns[key] = ys[0]
                continue
            if pstdev(ys) <= jitter_px:
                patterns[key] = sum(ys) / len(ys)
        return patterns

    return _filter_hits(header_hits), _filter_hits(footer_hits)


__all__ = [
    "CAPTION_RE",
    "ACTIVITY_RE",
    "PAGE_NO_RE",
    "in_caption_ring",
    "looks_like_body",
    "aligns_with_body_indent",
    "callout_cues",
    "is_running_header",
    "is_running_footer",
    "compute_running_patterns",
]
