from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict

from .normalize import Block
from .config import PipelineConfig

_DENY_RE = re.compile(
    r"^(fig(ure)?\.?\s*\d+|table\s*\d+|source(\s+\d+)?|activity|exercise|try.?it|let.?s\s+(recall|discuss)|key\s+takeaways|summary)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Gate5Decision:
    allow: bool
    reason: str
    details: Dict[str, Any]


def evaluate_gate5(block: Block, config: PipelineConfig) -> Gate5Decision:
    """Evaluate Gate-5 rules for a block. Returns a decision with reason."""

    lowered_type = (block.type or "").lower()
    if lowered_type not in {"paragraph", "list", "item", "code", "equation"}:
        return Gate5Decision(False, "not_narrative_type", {"type": lowered_type})

    text = (block.text or "").strip()
    if _DENY_RE.match(text):
        return Gate5Decision(False, "regex_match", {"text": text[:64]})

    bbox = block.bbox or {}
    page_height = float(block.aux.get("page_height") or 0.0)
    if page_height > 0:
        y0 = float(bbox.get("y0", 0.0))
        y1 = float(bbox.get("y1", 0.0))
        band_pct = config.gate5.header_footer.y_band_pct
        band_height = page_height * band_pct
        if y0 <= band_height or y1 >= (page_height - band_height):
            return Gate5Decision(False, "header_footer_band", {"y0": y0, "y1": y1})

    if block.aux_subtype in {"caption", "header", "footer"}:
        reason = "caption_zone" if block.aux_subtype == "caption" else "header_footer_band"
        return Gate5Decision(False, reason, {"aux_subtype": block.aux_subtype})

    near_float_flag = bool(
        block.aux.get("near_figure_or_table_within_lh_1_5")
        or block.aux.get("near_float_within_lh_x1_5")
        or block.aux.get("near_float")
    )
    if near_float_flag:
        return Gate5Decision(False, "caption_zone", {"near_float": True})

    column_width = float(block.aux.get("column_width") or 0.0)
    if column_width <= 0.0:
        column_width = float(block.aux.get("column_span_width") or 0.0)
    width = float(bbox.get("x1", 0.0)) - float(bbox.get("x0", 0.0))
    if column_width <= 0.0:
        column_width = width if width > 0 else 1.0
    min_fraction = config.gate5.sidebar.min_column_width_fraction
    if width < (min_fraction * column_width):
        return Gate5Decision(False, "narrow_box", {"width": width, "column_width": column_width})

    return Gate5Decision(True, "", {})
