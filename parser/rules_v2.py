"""Flow-safe rule cascade for main vs aux classification."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .grouping import Block

CAPTION_RE = re.compile(r"^(Fig(?:ure)?\.?|Table|Map|Plate)\s?\d+", re.I)
PAGE_NO_RE = re.compile(r"^\s*\d{1,3}\s*$")
SIDENOTE_RE = re.compile(r"^(Activity|Discuss|Think|Did you know\??|Recall)", re.I)


@dataclass
class ClassifierState:
    header_counts: Dict[str, int] = field(default_factory=dict)
    footer_counts: Dict[str, int] = field(default_factory=dict)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _in_band(bbox, height: float, pct: float, top: bool) -> bool:
    if height <= 0:
        return False
    band = height * pct
    if top:
        return bbox[1] <= band
    return bbox[3] >= height - band


def rules_v2_classify(
    page,
    blocks: List[Block],
    regions: List[Dict[str, object]],
    cfg: Dict[str, object],
    state: Optional[ClassifierState] = None,
) -> List[Dict[str, object]]:
    """Classify blocks into main/aux/control according to ordered rules."""

    state = state or ClassifierState()
    thresholds = cfg.get("thresholds", {})
    tau_main = float(thresholds.get("tau_main", 0.60))
    tau_low = float(thresholds.get("tau_fail_safe_low", 0.52))
    bias_to_main = bool(cfg.get("bias", {}).get("bias_to_main", True)) if isinstance(cfg.get("bias"), dict) else True
    headers_cfg = cfg.get("headers_footers", {})
    repetition_pages = int(headers_cfg.get("repetition_pages", 3)) if isinstance(headers_cfg, dict) else 3
    header_pct = float(cfg.get("bands", {}).get("header_pct", cfg.get("bands", {}).get("header_y_pct", 0.12)))
    footer_pct = float(cfg.get("bands", {}).get("footer_pct", cfg.get("bands", {}).get("footer_y_pct", 0.12)))

    forced_map = {
        "caption": "caption",
        "sidebar": "sidebar",
        "figure": "figure",
        "table": "table",
        "page-header": "header",
        "page-footer": "footer",
        "footnote": "footnote",
    }

    classified: List[Dict[str, object]] = []
    height = getattr(page, "height", 0.0)

    for block in blocks:
        text = block.text
        norm_text = _norm(text)
        region_tag = block.region_tag or "text"
        record: Dict[str, object] = {
            "id": block.block_id,
            "type": "main",
            "subtype": "paragraph",
            "text": text,
            "bbox": list(block.bbox),
            "links": [],
            "page_references": [],
            "ro_index": block.ro_index,
            "flow_id": None,
            "region_tag": region_tag,
            "confidence": 0.9,
            "aux_shadow": False,
            "inline_caption": False,
            "quarantined": False,
            "meta": dict(block.meta),
            "reason": [],
            "ms": 0.0,
        }

        forced_subtype = forced_map.get(region_tag)
        if forced_subtype:
            record["type"] = "aux"
            record["subtype"] = forced_subtype
            record["reason"].append(f"H0:Region={region_tag}")
            classified.append(record)
            continue

        is_header_band = _in_band(block.bbox, height, header_pct, top=True)
        is_footer_band = _in_band(block.bbox, height, footer_pct, top=False)

        if is_header_band:
            state.header_counts[norm_text] = state.header_counts.get(norm_text, 0) + 1
            if state.header_counts[norm_text] >= repetition_pages:
                record["type"] = "aux"
                record["subtype"] = "header"
                record["reason"].append("H1:HeaderRepetition")
                classified.append(record)
                continue
        if is_footer_band:
            state.footer_counts[norm_text] = state.footer_counts.get(norm_text, 0) + 1
            if state.footer_counts[norm_text] >= repetition_pages or PAGE_NO_RE.match(text):
                record["type"] = "aux"
                record["subtype"] = "footer" if not PAGE_NO_RE.match(text) else "page_number"
                record["reason"].append("H1:FooterRepetition")
                classified.append(record)
                continue

        if CAPTION_RE.match(text):
            record["type"] = "aux"
            record["subtype"] = "caption"
            record["reason"].append("H3:CaptionCue")
            classified.append(record)
            continue

        if region_tag == "text" and SIDENOTE_RE.match(text):
            record["type"] = "aux"
            record["subtype"] = "sidebar"
            record["reason"].append("H4:SidebarCue")
            classified.append(record)
            continue

        width = max(1.0, block.bbox[2] - block.bbox[0])
        col_width = max(1.0, float(block.meta.get("col_width", width)))
        width_ratio = min(1.0, width / col_width)
        line_count = int(block.meta.get("line_count", 1))
        text_len = len(text.strip())
        ms = 0.4 + 0.3 * width_ratio
        if text_len > 80:
            ms += 0.1
            record["reason"].append("H2:LongText")
        if line_count > 1:
            ms += 0.05
        ms = max(0.0, min(ms, 1.0))
        record["ms"] = ms

        if ms >= tau_main:
            record["reason"].append("H2:AboveTau")
            classified.append(record)
            continue
        if tau_low <= ms < tau_main and bias_to_main:
            record["reason"].append("H2:BiasToMain")
            classified.append(record)
            continue

        record["type"] = "aux"
        record["subtype"] = "other"
        record["reason"].append("H7:FallbackAux")
        classified.append(record)

    return classified
