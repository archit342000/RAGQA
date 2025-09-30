from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .normalize import Block

CAPTION_RE = re.compile(r"^(?i:fig(ure)?\.?|table)")
LEXICAL_AUX_RE = re.compile(
    r"(?i)^(source(\s+\d+)?|activity|exercise|try.?it|let.?s\s+(recall|discuss)|key\s+takeaways|summary)\b"
)


def main_gate(block: Block | Dict[str, Any], context: Dict[str, Any] | None = None) -> Tuple[bool, List[str]]:
    """Strict narrative filter used prior to Flow-First packing.

    Parameters
    ----------
    block:
        Canonical block dictionary (typically produced by ``normalise_blocks``) containing
        the standard schema fields along with ``aux`` geometry/style metadata.
    context:
        Optional context dictionary; currently only ``config`` may be provided.

    Returns
    -------
    tuple
        ``(passed, rejection_reasons)`` where ``passed`` indicates the block can remain
        in the narrative stream. When ``passed`` is ``False`` the block should be routed
        to the auxiliary buffer and the recorded reasons persisted for telemetry/QA.
    """

    reasons: List[str] = []
    if isinstance(block, dict):
        block_dict = block
    else:
        block_dict = {
            "type": block.type,
            "text": block.text,
            "aux": block.aux,
        }

    block_type = (block_dict.get("type") or "").lower()
    if block_type not in {"paragraph", "list", "item", "code", "equation"}:
        reasons.append("type_not_narrative")

    text = (block_dict.get("text") or "").strip()
    if text:
        if CAPTION_RE.match(text):
            reasons.append("caption_prefix")
        if LEXICAL_AUX_RE.match(text):
            reasons.append("lexical_aux")

    meta = block_dict.get("aux") or {}
    if meta.get("near_figure_or_table_within_lh_1_5"):
        reasons.append("near_float")
    if meta.get("callout_box") or meta.get("narrow_width_lt_0_6col"):
        reasons.append("callout_box_or_style_shift")
    if meta.get("font_band_shift_abrupt"):
        reasons.append("callout_box_or_style_shift")
    if meta.get("small_font_band_outlier"):
        reasons.append("small_font_band")

    passed = not reasons
    return passed, reasons
