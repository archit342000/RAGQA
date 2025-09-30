from __future__ import annotations

from typing import Dict
import re

from ..normalize import Block

DENY_PREFIX = re.compile(
    r"^(?i:fig(ure)?\.?|table\.?|source(\s+\d+)?|activity|exercise|try.?it|let.?s\s+(recall|discuss))"
)


def continuity_score(bi: Block, bj: Block, weights: Dict[str, float]) -> float:
    """Compute the continuity score between two candidate blocks."""

    w_style = weights.get("style", 3.0)
    w_indent = weights.get("indent", 2.0)
    w_xalign = weights.get("xalign", 2.0)
    w_heading = weights.get("heading", 4.0)
    w_gap = weights.get("gap", -1.0)
    w_sent = weights.get("sent", 2.0)

    style = _style_similarity(bi, bj)
    indent = _indent_similarity(bi, bj)
    xalign = _width_similarity(bi, bj)
    heading = 1.0 if tuple(bi.heading_path) == tuple(bj.heading_path) else 0.0
    gap = -_gap_penalty(bi, bj)
    sent = _sentence_continuation(bi, bj)

    score = (
        w_style * style
        + w_indent * indent
        + w_xalign * xalign
        + w_heading * heading
        + w_gap * gap
        + w_sent * sent
    )

    if _is_aux_like(bj):
        score -= 4.0
    return score


def _style_similarity(a: Block, b: Block) -> float:
    font_a = _font_size(a)
    font_b = _font_size(b)
    if font_a is None or font_b is None:
        return 1.0 if a.type == b.type else 0.0
    return 1.0 if abs(font_a - font_b) <= 0.5 else 0.0


def _indent_similarity(a: Block, b: Block) -> float:
    if not a.bbox or not b.bbox:
        return 0.0
    ax = a.bbox["x0"]
    bx = b.bbox["x0"]
    width = max(b.bbox["x1"] - b.bbox["x0"], 1.0)
    return 1.0 if abs(ax - bx) <= max(4.0, 0.02 * width) else 0.0


def _width_similarity(a: Block, b: Block) -> float:
    if not a.bbox or not b.bbox:
        return 0.0
    aw = a.bbox["x1"] - a.bbox["x0"]
    bw = b.bbox["x1"] - b.bbox["x0"]
    if max(aw, bw) <= 0:
        return 0.0
    return 1.0 if abs(aw - bw) <= 0.1 * max(aw, bw) else 0.0


def _gap_penalty(a: Block, b: Block) -> float:
    if not a.bbox or not b.bbox:
        return 1.0
    dy = b.bbox["y0"] - a.bbox["y1"]
    page_gap = max(0, b.page - a.page)
    gap = max(0.0, dy) + page_gap * 50.0
    return gap / 100.0


def _sentence_continuation(a: Block, b: Block) -> float:
    a_text = (a.text or "").rstrip()
    b_text = (b.text or "").lstrip()
    if not a_text or not b_text:
        return 0.0
    if a_text.endswith((".", "?", "!", ";", ":")):
        return 0.0
    return 1.0 if b_text[:1].islower() else 0.0


def _font_size(block: Block) -> float | None:
    font = block.aux.get("font_size") if block.aux else None
    if font is None:
        return None
    try:
        return float(font)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _is_aux_like(block: Block) -> bool:
    text = (block.text or "").strip()
    if DENY_PREFIX.match(text):
        return True
    if block.aux:
        if block.aux.get("near_figure_or_table_within_lh_1_5"):
            return True
        if block.aux.get("narrow_width_lt_0_6col"):
            return True
    return False
