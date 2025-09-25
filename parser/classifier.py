"""Block classification with aux prepass, mainness scoring, and section control blocks."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import re
import statistics

from .utils import (
    Block,
    DocumentLayout,
    PageLayout,
    detect_footnote_marker,
    ends_with_terminal_punct,
    font_stats,
    is_all_caps,
    line_spacing,
    load_config,
    slugify,
    symmetric_band,
    text_density,
    token_count,
    whitespace_halo,
)


@dataclass
class BodyStats:
    font_size: float
    col_width: float
    indent: float


@dataclass
class BlockPrediction:
    page: int
    index: int
    kind: str
    aux_type: Optional[str]
    subtype: Optional[str]
    text: str
    bbox: Tuple[float, float, float, float]
    ms: float
    hs: Optional[float]
    reason: List[str]
    meta: Dict[str, object]
    flow: str
    confidence: float
    source_block: Block
    expect_continuation: bool = False
    quarantined: bool = False
    anchor_hint: Optional[Tuple[int, int]] = None

    def copy(self) -> "BlockPrediction":
        return replace(self, reason=list(self.reason), meta=dict(self.meta))


def _is_in_active_section(state: Dict[str, Any]) -> bool:
    return state.get("section_state", "OUT_OF_SECTION") != "OUT_OF_SECTION"


def _bias_to_main_allowed(state: Dict[str, Any]) -> bool:
    return _is_in_active_section(state)


def _page_context(page: PageLayout, cfg: Dict[str, object]) -> Dict[str, Any]:
    bands = cfg.get("bands", {})
    thresholds = cfg.get("thresholds", {})
    margin = bands.get("margin_x_pct", [0.0, 0.09, 0.91, 1.0])
    header_band = float(bands.get("header_y_pct", 0.12))
    footer_band = float(bands.get("footer_y_pct", 0.12))
    return {
        "page": page,
        "width": float(page.width or 1.0),
        "height": float(page.height or 1.0),
        "margin_left_pct": float(margin[1]),
        "margin_right_pct": float(margin[2]),
        "header_band": header_band,
        "footer_band": 1.0 - footer_band,
        "thresholds": thresholds,
        "header_templates": set(slugify(t) for t in page.meta.get("header_templates", []) if t),
        "footer_templates": set(slugify(t) for t in page.meta.get("footer_templates", []) if t),
        "recent_figures": [],
    }


def _page_font_stats(page: PageLayout) -> Tuple[float, float]:
    font_sizes: List[float] = []
    for block in page.blocks:
        font_sizes.extend(block.font_sizes)
    if not font_sizes:
        return 0.0, 0.0
    mean = statistics.mean(font_sizes)
    stdev = statistics.pstdev(font_sizes) if len(font_sizes) > 1 else 0.0
    return mean, stdev


def _block_record(
    page: PageLayout,
    block: Block,
    index: int,
    ctx: Dict[str, Any],
    font_mean: float,
    font_std: float,
) -> Dict[str, Any]:
    x0, y0, x1, y1 = block.bbox
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    page_width = ctx["width"]
    page_height = ctx["height"]
    mean_size, std_size = font_stats(block)
    density = text_density(block)
    halo_above, halo_below = whitespace_halo(page, block)
    meta: Dict[str, Any] = {
        "font_size": mean_size,
        "font_size_std": std_size,
        "font_z": 0.0 if font_std == 0 else (mean_size - font_mean) / (font_std or 1.0),
        "line_height": line_spacing(block),
        "width": width,
        "height": height,
        "width_pct": width / page_width if page_width else 0.0,
        "height_pct": height / page_height if page_height else 0.0,
        "left_pct": x0 / page_width if page_width else 0.0,
        "right_pct": x1 / page_width if page_width else 0.0,
        "top_pct": y0 / page_height if page_height else 0.0,
        "bottom_pct": y1 / page_height if page_height else 0.0,
        "indent": block.attrs.get("indent", max(0.0, x0 - page.meta.get("text_left", 0.0))),
        "token_count": token_count(block.text),
        "char_count": len(block.text or ""),
        "text_density": density,
        "halo_above": halo_above,
        "halo_below": halo_below,
        "col_id": block.attrs.get("col_id"),
        "wrap_protected": False,
        "is_continuation": False,
        "open_paragraph_id": None,
        "page_width": page_width,
        "page_height": page_height,
        "region_tag": block.attrs.get("region_tag"),
    }
    record = {
        "page": page.page_number,
        "index": index,
        "order_index": float(index),
        "kind": "unknown",
        "aux_type": None,
        "subtype": None,
        "text": block.text,
        "bbox": block.bbox,
        "block_type": block.block_type,
        "meta": meta,
        "region_tag": meta.get("region_tag"),
        "flow": "main",
        "reason": [],
        "ms": 0.0,
        "hs": None,
        "source": block,
    }
    return record


def prepass_aux(blocks: List[Dict[str, Any]], page_ctx: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    thresholds = cfg.get("thresholds", {})
    caption_cfg = cfg.get("caption", {})
    sidenote_cfg = cfg.get("sidenote", {})
    caption_overlap_min = float(
        caption_cfg.get(
            "horiz_overlap_min",
            thresholds.get("caption_h_overlap_min", 0.30),
        )
    )
    caption_extend_chars = int(caption_cfg.get("extended_merge_max_chars", 220))
    margin_left = float(page_ctx.get("margin_left_pct", 0.05))
    margin_right = float(page_ctx.get("margin_right_pct", 0.95))
    header_band = float(page_ctx.get("header_band", 0.12))
    footer_band = float(page_ctx.get("footer_band", 0.88))
    page_width = float(page_ctx.get("width", 1.0) or 1.0)
    header_templates = page_ctx.get("header_templates", set())
    footer_templates = page_ctx.get("footer_templates", set())

    sidenote_max_width = float(
        sidenote_cfg.get(
            "max_width_ratio", thresholds.get("sidenote_max_width_ratio", 0.50)
        )
    )
    sidenote_font_z_max = float(
        sidenote_cfg.get(
            "font_z_max", thresholds.get("sidenote_font_z_max", -0.40)
        )
    )

    activity_patterns = [
        re.compile(r"\b(activity|try\s+this|project|practice)\b", re.IGNORECASE),
        re.compile(r"\b(let's\s+do|hands[- ]on)\b", re.IGNORECASE),
    ]
    activity_cfg = cfg.get("activity", {})
    cue_regex = activity_cfg.get("cue_regex")
    if isinstance(cue_regex, str):
        try:
            activity_patterns.insert(0, re.compile(cue_regex, re.IGNORECASE))
        except re.error:
            pass
    source_pattern = re.compile(r"^(source|courtesy|credit|from)\b", re.IGNORECASE)
    page_number_pattern = re.compile(r"^(\d+|[ivxlcdm]+)$", re.IGNORECASE)
    boxed_pattern = re.compile(r"\b(callout|remember|note)\b", re.IGNORECASE)

    figure_records: Dict[int, Dict[str, Any]] = {}
    for offset, fig in enumerate(page_ctx.get("recent_figures", [])):
        figure_records[-(offset + 1)] = fig
    caption_indices: List[int] = []

    for idx, block in enumerate(blocks):
        meta = block["meta"]
        region_tag = meta.get("region_tag")
        text = (block.get("text") or "").strip()
        text_slug = slugify(text)
        text_lower = text.lower()
        top_pct = float(meta.get("top_pct", 1.0))
        bottom_pct = float(meta.get("bottom_pct", 0.0))
        width_pct = float(meta.get("width_pct", 0.0))
        reasons = block.setdefault("reason", [])

        if block["block_type"] in {"figure", "image"}:
            block["kind"] = "aux"
            block["aux_type"] = "figure"
            block["flow"] = "aux"
            block["confidence"] = 0.92
            reasons.append("PrepassA:Figure")
            figure_records[idx] = block
            continue
        if block["block_type"] == "table":
            block["kind"] = "aux"
            block["aux_type"] = "table"
            block["flow"] = "aux"
            block["confidence"] = 0.90
            reasons.append("PrepassA:Table")
            figure_records[idx] = block
            continue

        if not text:
            continue

        token_count = int(meta.get("token_count", 0))
        char_count = int(meta.get("char_count", len(text)))

        font_z = meta.get("font_z", 0.0)

        if top_pct <= header_band and region_tag in {None, "title", "text"}:
            if page_number_pattern.match(text.replace(" ", "")):
                block["kind"] = "aux"
                block["aux_type"] = "page_number"
                block["flow"] = "aux"
                block["confidence"] = 0.88
                reasons.append("PrepassA:HeaderPageNumber")
                continue
            if slugify(text) in header_templates and token_count <= 18 and font_z <= 0.2:
                block["kind"] = "aux"
                block["aux_type"] = "header"
                block["flow"] = "aux"
                block["confidence"] = 0.85
                reasons.append("PrepassA:HeaderTemplate")
                continue
            if token_count <= 15 and char_count <= 90 and font_z <= 0.2:
                block["kind"] = "aux"
                block["aux_type"] = "header"
                block["flow"] = "aux"
                block["confidence"] = 0.75
                reasons.append("PrepassA:HeaderBand")
                continue

        if bottom_pct >= footer_band and region_tag in {None, "title", "text"}:
            if page_number_pattern.match(text.replace(" ", "")):
                block["kind"] = "aux"
                block["aux_type"] = "page_number"
                block["flow"] = "aux"
                block["confidence"] = 0.88
                reasons.append("PrepassA:FooterPageNumber")
                continue
            if detect_footnote_marker(text):
                block["kind"] = "aux"
                block["aux_type"] = "footnote"
                block["flow"] = "aux"
                block["confidence"] = 0.86
                reasons.append("PrepassA:FootnoteMarker")
                continue
            if slugify(text) in footer_templates and token_count <= 18 and font_z <= 0.2:
                block["kind"] = "aux"
                block["aux_type"] = "footer"
                block["flow"] = "aux"
                block["confidence"] = 0.83
                reasons.append("PrepassA:FooterTemplate")
                continue
            if token_count <= 20 and char_count <= 120 and font_z <= 0.2:
                block["kind"] = "aux"
                block["aux_type"] = "footer"
                block["flow"] = "aux"
                block["confidence"] = 0.72
                reasons.append("PrepassA:FooterBand")
                continue

        if source_pattern.match(text):
            block["kind"] = "aux"
            block["aux_type"] = "source_label"
            block["flow"] = "aux"
            block["confidence"] = 0.82
            reasons.append("PrepassA:SourceLabel")
            continue

        if detect_footnote_marker(text) and top_pct >= footer_band - 0.1:
            block["kind"] = "aux"
            block["aux_type"] = "footnote"
            block["flow"] = "aux"
            block["confidence"] = 0.86
            reasons.append("PrepassA:FootnoteMarker")
            continue

        if width_pct <= sidenote_max_width and (
            meta.get("left_pct", 0.0) <= margin_left
            or meta.get("right_pct", 1.0) >= margin_right
        ):
            if meta.get("font_z", 0.0) <= sidenote_font_z_max:
                block["kind"] = "aux"
                block["aux_type"] = "sidenote"
                block["flow"] = "aux"
                block["confidence"] = 0.78
                reasons.append("PrepassA:SidenoteBand")
                continue
            if any(pattern.search(text) for pattern in activity_patterns):
                block["kind"] = "aux"
                block["aux_type"] = "callout"
                block["flow"] = "aux"
                block["confidence"] = 0.76
                reasons.append("PrepassA:ActivityMargin")
                continue

        if any(pattern.search(text) for pattern in activity_patterns):
            if width_pct <= 0.65 or meta.get("indent", 0.0) <= 12.0:
                block["kind"] = "aux"
                block["aux_type"] = "callout"
                block["flow"] = "aux"
                block["confidence"] = 0.74
                reasons.append("PrepassA:ActivityCue")
                continue

        if boxed_pattern.search(text_slug) and width_pct <= 0.65:
            block["kind"] = "aux"
            block["aux_type"] = "callout"
            block["flow"] = "aux"
            block["confidence"] = 0.7
            reasons.append("PrepassA:CalloutLexical")
            continue

        if text_lower.startswith(("figure", "fig.", "table")) and region_tag in {None, "text", "figure", "list"}:
            nearest: Optional[Dict[str, Any]] = None
            overlap = 0.0
            for fig_idx, fig in figure_records.items():
                fig_box = fig["bbox"]
                fig_width = max(1e-6, fig_box[2] - fig_box[0])
                inter_left = max(block["bbox"][0], fig_box[0])
                inter_right = min(block["bbox"][2], fig_box[2])
                inter_width = max(0.0, inter_right - inter_left)
                horiz = inter_width / max(min(meta.get("width", fig_width), fig_width), 1e-6)
                vertical_gap = max(0.0, block["bbox"][1] - fig_box[3])
                if horiz >= caption_overlap_min and vertical_gap <= page_width * 0.05:
                    if horiz > overlap:
                        overlap = horiz
                        nearest = fig
            if nearest and meta.get("font_z", 0.0) <= 0.05:
                block["kind"] = "aux"
                block["aux_type"] = "caption"
                block["flow"] = "aux"
                block["confidence"] = 0.84
                block["anchor_hint"] = (nearest["page"], nearest["index"])
                reasons.append("PrepassA:CaptionProximity")
                caption_indices.append(idx)
                continue
            if nearest is None and meta.get("font_z", 0.0) <= 0.0 and token_count <= 22:
                block["kind"] = "aux"
                block["aux_type"] = "caption"
                block["flow"] = "aux"
                block["confidence"] = 0.72
                reasons.append("PrepassA:CaptionLexical")
                caption_indices.append(idx)
                continue

    # Extended caption merging: include subsequent small-font lines that align horizontally
    for idx in caption_indices:
        if idx >= len(blocks):
            continue
        caption_block = blocks[idx]
        if caption_block.get("kind") != "aux" or caption_block.get("aux_type") != "caption":
            continue
        accumulated = len((caption_block.get("text") or ""))
        anchor_hint = caption_block.get("anchor_hint")
        base_bbox = caption_block["bbox"]
        base_width = max(1e-6, base_bbox[2] - base_bbox[0])
        for look_ahead in range(idx + 1, len(blocks)):
            next_block = blocks[look_ahead]
            if accumulated >= caption_extend_chars:
                break
            if next_block.get("kind") == "aux":
                if next_block.get("aux_type") == "caption":
                    accumulated += len(next_block.get("text") or "")
                continue
            if next_block["block_type"] != "text":
                break
            next_meta = next_block["meta"]
            if next_meta.get("font_size", 0.0) > caption_block["meta"].get("font_size", 0.0) * 1.1:
                break
            inter_left = max(base_bbox[0], next_block["bbox"][0])
            inter_right = min(base_bbox[2], next_block["bbox"][2])
            inter_width = max(0.0, inter_right - inter_left)
            horiz = inter_width / base_width
            vertical_gap = max(0.0, next_block["bbox"][1] - base_bbox[3])
            if horiz < caption_overlap_min or vertical_gap > max(6.0, caption_block["meta"].get("line_height", 12.0) * 1.5):
                break
            next_block["kind"] = "aux"
            next_block["aux_type"] = "caption"
            next_block["flow"] = "aux"
            next_block["confidence"] = 0.7
            next_block.setdefault("reason", []).append("PrepassA:CaptionExtended")
            if anchor_hint:
                next_block["anchor_hint"] = anchor_hint
            accumulated += len(next_block.get("text") or "")
            base_bbox = (
                min(base_bbox[0], next_block["bbox"][0]),
                base_bbox[1],
                max(base_bbox[2], next_block["bbox"][2]),
                next_block["bbox"][3],
            )


def estimate_body_stats(blocks: List[Dict[str, Any]], page_ctx: Dict[str, Any]) -> BodyStats:
    font_sizes = [b["meta"]["font_size"] for b in blocks if b["block_type"] == "text" and b["meta"]["font_size"]]
    widths = [b["meta"]["width"] for b in blocks if b["kind"] != "aux" and b["block_type"] == "text" and b["meta"]["width"]]
    indents = [b["meta"].get("indent", 0.0) or 0.0 for b in blocks if b["block_type"] == "text"]
    font_size = statistics.median(font_sizes) if font_sizes else 11.0
    col_width = statistics.median(widths) if widths else page_ctx["width"] * 0.45
    indent = statistics.median(indents) if indents else 12.0
    return BodyStats(font_size=font_size, col_width=col_width, indent=indent)


def _should_stitch(prev_block: Dict[str, Any], block: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    if prev_block["block_type"] != "text" or block["block_type"] != "text":
        return False
    stitching_cfg = cfg.get("stitching", {})
    top_lookahead = float(stitching_cfg.get("top_lookahead_pct", 0.30))
    prev_text = prev_block.get("text", "") or ""
    if _expect_continuation(prev_block):
        return True
    prev_meta = prev_block["meta"]
    block_meta = block["meta"]
    indent_delta = abs((block_meta.get("indent") or 0.0) - (prev_meta.get("indent") or 0.0))
    if prev_block["page"] != block["page"]:
        if block_meta.get("top_pct", 1.0) <= top_lookahead + 0.05:
            if indent_delta <= max(12.0, (prev_meta.get("indent", 0.0) or 0.0) * 0.6):
                return True
    if not ends_with_terminal_punct(prev_text) and indent_delta <= max(10.0, (prev_meta.get("indent", 0.0) or 0.0) * 0.5):
        return True
    if prev_text.endswith("-"):
        return True
    return False


def stitch_paragraphs(blocks: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    max_page_gap = 2
    last_text: Optional[Dict[str, Any]] = None
    current_para_id: Optional[str] = None
    for block in blocks:
        if block["block_type"] != "text":
            continue
        if last_text is None or block["page"] - last_text["page"] > max_page_gap:
            current_para_id = None
        is_continuation = False
        if last_text is not None:
            if block["page"] - last_text["page"] <= max_page_gap and _should_stitch(last_text, block, cfg):
                is_continuation = True
                current_para_id = last_text["meta"].get("open_paragraph_id") or current_para_id
        if not current_para_id:
            current_para_id = f"p{block['page']}_{block['index']}"
        block.setdefault("meta", {})
        block["meta"]["open_paragraph_id"] = current_para_id
        block["meta"]["is_continuation"] = is_continuation
        block["is_continuation"] = is_continuation
        last_text = block


def assign_columns_and_wrap(
    blocks: List[Dict[str, Any]],
    page_ctx: Dict[str, Any],
    cfg: Dict[str, Any],
    stats: BodyStats,
) -> None:
    caption_cfg = cfg.get("caption", {})
    thresholds = cfg.get("thresholds", {})
    near_image_multiplier = float(
        caption_cfg.get(
            "near_image_lh_multiplier",
            thresholds.get("near_image_lh_multiplier", 1.5),
        )
    )
    figures = [b for b in blocks if b["block_type"] in {"figure", "image", "table"}]
    page_width = page_ctx["width"]

    for block in blocks:
        meta = block["meta"]
        if meta.get("col_id") is None:
            center = (block["bbox"][0] + block["bbox"][2]) / 2.0
            if center < page_width * 0.33:
                meta["col_id"] = 0
            elif center > page_width * 0.66:
                meta["col_id"] = 2
            else:
                meta["col_id"] = 1
        if block["block_type"] != "text" or not figures:
            continue
        lh = meta.get("line_height") or 0.0
        if not lh:
            continue
        ratio = lh / (stats.font_size or 1.0)
        if ratio < near_image_multiplier:
            continue
        block_box = block["bbox"]
        for fig in figures:
            fig_box = fig["bbox"]
            horiz_overlap = not (
                block_box[2] <= fig_box[0] or block_box[0] >= fig_box[2]
            )
            if not horiz_overlap:
                continue
            vertical_gap = min(
                abs(block_box[1] - fig_box[3]),
                abs(fig_box[1] - block_box[3]),
            )
            if vertical_gap <= stats.font_size * 2:
                meta["wrap_protected"] = True
                block["reason"].append("WrapProtect")
                break


def mainness_score(
    block: Dict[str, Any], stats: BodyStats, page_ctx: Dict[str, Any], cfg: Dict[str, Any]
) -> Tuple[float, List[str]]:
    meta = block["meta"]
    reasons: List[str] = []
    score = 0.1
    text = (block.get("text") or "").strip()
    if block["block_type"] == "text":
        score += 0.3
        if meta.get("token_count", 0) >= 8:
            score += 0.1
            reasons.append("TokenCount")
        if meta.get("width", 0.0) >= stats.col_width * 0.7:
            score += 0.12
            reasons.append("ColumnWidth")
        if abs((meta.get("indent") or 0.0) - stats.indent) <= max(6.0, stats.indent * 0.4):
            score += 0.08
            reasons.append("IndentMatch")
        if meta.get("wrap_protected"):
            score += 0.08
            reasons.append("WrapProtect")
        if meta.get("text_density", 0.0) >= 0.25:
            score += 0.05
            reasons.append("Density")
        if meta.get("is_continuation"):
            score += 0.04
            reasons.append("Continuation")
    else:
        score -= 0.18

    margin_left = page_ctx.get("margin_left_pct", 0.05)
    margin_right = page_ctx.get("margin_right_pct", 0.95)
    if meta.get("left_pct", 0.0) >= margin_left and meta.get("right_pct", 1.0) <= margin_right:
        score += 0.06
        reasons.append("InsideMargins")
    else:
        score -= 0.05
        reasons.append("MarginBand")

    if detect_footnote_marker(text):
        score -= 0.32
        reasons.append("FootnoteMarker")
    if re.match(r"^(figure|fig\.|table)\b", text, re.IGNORECASE):
        score -= 0.25
        reasons.append("CaptionCue")
    if re.match(r"^(source|courtesy|credit)\b", text, re.IGNORECASE):
        score -= 0.22
        reasons.append("SourceCue")
    if "activity" in text.lower():
        score -= 0.12
    if meta.get("width_pct", 0.0) <= 0.25:
        score -= 0.1
        reasons.append("Narrow")
    band_top = page_ctx.get("header_band", 0.12)
    band_bottom = page_ctx.get("footer_band", 0.88)
    if meta.get("top_pct", 1.0) <= band_top or meta.get("bottom_pct", 0.0) >= band_bottom:
        score -= 0.18
        reasons.append("HeaderFooterBand")

    score = max(0.0, min(1.0, score))
    return score, reasons


def _geometry_continuity(
    last_meta: Optional[Dict[str, Any]], current_meta: Dict[str, Any], stats: BodyStats
) -> bool:
    if not last_meta:
        return False
    indent_delta = abs((current_meta.get("indent") or 0.0) - (last_meta.get("indent") or 0.0))
    if indent_delta > max(12.0, stats.indent * 0.6):
        return False
    last_col = last_meta.get("col_id")
    cur_col = current_meta.get("col_id")
    if last_col is not None and cur_col is not None and last_col != cur_col:
        return False
    return True


def flow_safe_decision(
    block: Dict[str, Any],
    state: Dict[str, Any],
    stats: BodyStats,
    cfg: Dict[str, Any],
    page_ctx: Dict[str, Any],
) -> Tuple[str, float, List[str]]:
    thresholds = cfg.get("thresholds", {})
    tau = float(thresholds.get("tau_main", 0.6))
    tau_low = float(thresholds.get("tau_fail_safe_low", 0.52))
    tau_bias_high = float(thresholds.get("tau_bias_high", 0.65))
    tau_bias_high = max(tau_bias_high, tau)
    score = block.get("ms")
    reasons: List[str] = []
    if score is None:
        score, ms_reasons = mainness_score(block, stats, page_ctx, cfg)
        block["ms"] = score
        reasons.extend(ms_reasons)
    if score >= tau:
        reasons.append("AboveTauMain")
        return "main", score, reasons
    if tau_low <= score < tau_bias_high and not block.get("aux_type"):
        reasons.append("FailSafe")
        if _bias_to_main_allowed(state):
            meta = block.get("meta", {})
            continuation_pair = bool(meta.get("is_continuation") and state.get("continuing_paragraph"))
            geometry_ok = _geometry_continuity(state.get("last_main_meta"), meta, stats)
            if continuation_pair or geometry_ok:
                reasons.append("BiasToMainWindow")
        return "main", score, reasons
    reasons.append("BelowTauMain")
    return "aux", score, reasons


def maybe_quarantine(block: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    thresholds = cfg.get("thresholds", {})
    qmin = float(thresholds.get("quarantined_aux_conf_min", 0.65))
    if block.get("kind") == "aux" and (block.get("confidence") or 0.0) < qmin:
        block["quarantined"] = True
        block.setdefault("reason", []).append("QuarantinedLowConf")


def _bbox_distance_sq(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    ax = (b1[0] + b1[2]) / 2.0
    ay = (b1[1] + b1[3]) / 2.0
    bx = (b2[0] + b2[2]) / 2.0
    by = (b2[1] + b2[3]) / 2.0
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def _nearest_figure(
    block: Dict[str, Any],
    figures: List[Dict[str, Any]],
    distance_limit: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_dist: Optional[float] = None
    for figure in figures:
        dist = _bbox_distance_sq(block["bbox"], figure["bbox"])
        if distance_limit is not None and dist > distance_limit * distance_limit:
            continue
        if best is None or dist < (best_dist or float("inf")):
            best = figure
            best_dist = dist
    return best


def anchor_aux(
    block: Dict[str, Any],
    figures: List[Dict[str, Any]],
    last_main: Optional[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> None:
    aux_type = block.get("aux_type")
    if aux_type == "caption":
        if block.get("anchor_hint") is None:
            nearest = _nearest_figure(block, figures)
            if nearest:
                block["anchor_hint"] = (nearest["page"], nearest["index"])
    elif aux_type in {"sidenote", "callout", "footnote"}:
        if last_main:
            block["anchor_hint"] = (last_main["page"], last_main["index"])
    elif aux_type == "source_label":
        strategy_cfg = cfg.get("source_label", {})
        strategy = strategy_cfg.get("strategy", "free_aux")
        if strategy == "attach_to_nearest_figure":
            max_dist = float(strategy_cfg.get("distance_px_max", 120))
            nearest = _nearest_figure(block, figures, max_dist)
            if nearest:
                block["anchor_hint"] = (nearest["page"], nearest["index"])
        elif strategy == "attach_to_nearest_excerpt" and last_main:
            block["anchor_hint"] = (last_main["page"], last_main["index"])


def subtype_aux(block: Dict[str, Any], page_ctx: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[str]:
    if block["aux_type"]:
        return block["aux_type"]
    meta = block["meta"]
    text = (block.get("text") or "").strip()
    top_pct = meta.get("top_pct", 0.0)
    bottom_pct = meta.get("bottom_pct", 0.0)
    if block["block_type"] == "table":
        return "table"
    if block["block_type"] in {"figure", "image"}:
        return "figure"
    if detect_footnote_marker(text) and bottom_pct >= page_ctx.get("footer_band", 0.88) - 0.05:
        return "footnote"
    if top_pct <= page_ctx.get("header_band", 0.12):
        if text.isdigit() and len(text) <= 4:
            return "page_number"
        return "header"
    if bottom_pct >= page_ctx.get("footer_band", 0.88):
        if text.isdigit() and len(text) <= 4:
            return "page_number"
        return "footer"
    if text.lower().startswith(("figure", "fig.", "table")):
        return "caption"
    if re.match(r"^(source|courtesy|credit)\b", text, re.IGNORECASE):
        return "source_label"
    margin_left = page_ctx.get("margin_left_pct", 0.05)
    margin_right = page_ctx.get("margin_right_pct", 0.95)
    if meta.get("left_pct", 0.0) <= margin_left or meta.get("right_pct", 1.0) >= margin_right:
        if meta.get("width_pct", 1.0) <= float(cfg.get("thresholds", {}).get("sidenote_max_width_ratio", 0.5)):
            return "sidenote"
    if len(text) <= 35 and is_all_caps(text):
        return "callout"
    return "other"


def detect_implicit_section_start(
    window_blocks: List[Dict[str, Any]],
    doc_state: Dict[str, Any],
    cfg: Dict[str, Any],
    stats: BodyStats,
) -> Optional[Dict[str, Any]]:
    settings = cfg.get("implicit_section", {})
    if not settings.get("enable", True) or not window_blocks:
        return None
    if settings.get("only_when_no_section", True) and _is_in_active_section(doc_state):
        return None
    if doc_state.get("continuing_paragraph"):
        return None
    candidate = window_blocks[0]
    if candidate.get("kind") != "main" or candidate.get("meta", {}).get("is_continuation"):
        return None
    threshold = float(settings.get("score_threshold", 0.75))
    meta = candidate["meta"]
    score = 0.0
    if meta.get("top_pct", 1.0) <= float(settings.get("top_region_pct", 0.25)):
        score += 0.35
    if meta.get("char_count", 0) >= int(settings.get("min_opening_chars", 160)):
        score += 0.25
    if meta.get("halo_above", 0.0) >= settings.get("whitespace_halo_min_lh", 1.2) * stats.font_size:
        score += 0.15
    if settings.get("dropcap_detect", True) and candidate.get("text"):
        first = candidate["text"][0]
        if first.isalpha() and first.isupper():
            score += 0.1
    indent_delta = abs((meta.get("indent") or 0.0) - stats.indent)
    if indent_delta <= float(settings.get("first_line_indent_delta_px", 8)):
        score += 0.1
    if score >= threshold:
        control = {
            "page": candidate["page"],
            "index": candidate["index"],
            "order_index": candidate.get("order_index", float(candidate["index"])) - 0.5,
            "kind": "control",
            "flow": "control",
            "aux_type": None,
            "subtype": "implicit_section_start",
            "text": "",
            "bbox": candidate["bbox"],
            "block_type": "control",
            "meta": {
                "font_size": stats.font_size,
                "col_id": candidate["meta"].get("col_id"),
                "halo_above": candidate["meta"].get("halo_above"),
                "open_paragraph_id": None,
            },
            "reason": ["ImplicitSectionStart"],
            "ms": 1.0,
            "hs": None,
            "source": candidate["source"],
        }
        return control
    return None


def _heading_score(block: Dict[str, Any], page: PageLayout, stats: BodyStats, thresholds: Dict[str, Any]) -> Tuple[Optional[float], List[str]]:
    if block["block_type"] != "text" or not block.get("text"):
        return None, []
    reasons: List[str] = []
    meta = block["meta"]
    score = 0.0
    if meta["font_size"] >= stats.font_size * 1.1:
        score += min(0.5, (meta["font_size"] - stats.font_size) / (stats.font_size or 1.0))
        reasons.append("FontJump")
    if meta["font_size_std"] == 0 and meta["token_count"] <= 12:
        score += 0.15
        reasons.append("ShortLine")
    if block["text"].isupper():
        score += 0.15
        reasons.append("AllCaps")
    if block["meta"]["halo_above"] >= 12 and block["meta"]["halo_below"] >= 12:
        score += 0.1
        reasons.append("WhitespaceHalo")
    if detect_footnote_marker(block["text"]):
        score -= 0.3
    if ends_with_terminal_punct(block["text"]):
        score -= 0.2
    if score <= 0:
        return None, reasons
    return max(0.0, min(1.0, score)), reasons


def _expect_continuation(block: Dict[str, Any]) -> bool:
    text = block.get("text", "")
    if not text:
        return False
    if text.endswith("-"):
        return True
    if not ends_with_terminal_punct(text) and block["meta"].get("token_count", 0) >= 4:
        return True
    return False


def _confidence(kind: str, ms: float, hs: Optional[float]) -> float:
    if kind == "main":
        return max(ms, hs or 0.5)
    if kind == "control":
        return 1.0
    return max(0.6, 1.0 - ms / 2.0)


def _build_predictions(blocks: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[BlockPrediction]:
    predictions: List[BlockPrediction] = []
    for block in sorted(blocks, key=lambda b: (b["page"], b.get("order_index", float(b["index"])) )):
        reasons = list(dict.fromkeys(block.get("reason", [])))
        confidence = block.get("confidence")
        if confidence is None:
            confidence = _confidence(block.get("kind", "main"), block.get("ms", 0.0), block.get("hs"))
        prediction = BlockPrediction(
            page=block["page"],
            index=int(block.get("index", 0)),
            kind=block["kind"],
            aux_type=block.get("aux_type"),
            subtype=block.get("subtype"),
            text=block.get("text", ""),
            bbox=block["bbox"],
            ms=block.get("ms", 0.0),
            hs=block.get("hs"),
            reason=reasons,
            meta=dict(block.get("meta", {})),
            flow=block.get("flow", block.get("kind", "main")),
            confidence=confidence,
            source_block=block["source"],
            expect_continuation=bool(block.get("expect_continuation", False)),
            quarantined=bool(block.get("quarantined", False)),
            anchor_hint=block.get("anchor_hint"),
        )
        prediction.meta.setdefault("col_id", block["meta"].get("col_id"))
        prediction.meta.setdefault("open_paragraph_id", block["meta"].get("open_paragraph_id"))
        prediction.meta.setdefault("region_tag", block.get("region_tag"))
        predictions.append(prediction)
    return predictions


def classify_blocks(doc: DocumentLayout, config: Optional[Dict[str, object]] = None) -> List[BlockPrediction]:
    cfg: Dict[str, Any] = config or load_config()
    thresholds = cfg.get("thresholds", {})
    predictions: List[Dict[str, Any]] = []
    doc_state: Dict[str, Any] = {
        "section_state": "OUT_OF_SECTION",
        "continuing_paragraph": False,
        "last_main_meta": None,
    }
    aux_buffer: List[Dict[str, Any]] = []
    last_main_record: Optional[Dict[str, Any]] = None

    recent_figures: List[Dict[str, Any]] = []
    for page in doc.pages:
        page_ctx = _page_context(page, cfg)
        page_ctx["recent_figures"] = list(recent_figures)
        font_mean, font_std = _page_font_stats(page)
        block_records: List[Dict[str, Any]] = []
        figures_for_next: List[Dict[str, Any]] = []
        for idx, block in enumerate(page.blocks):
            record = _block_record(page, block, idx, page_ctx, font_mean, font_std)
            block_records.append(record)
            if block.block_type in {"figure", "image", "table"}:
                figures_for_next.append({"bbox": record["bbox"], "index": record["index"], "page": page.page_number})

        prepass_aux(block_records, page_ctx, cfg)
        stitch_paragraphs(block_records, cfg)
        stats = estimate_body_stats(block_records, page_ctx)
        assign_columns_and_wrap(block_records, page_ctx, cfg, stats)

        figures_on_page = [b for b in block_records if b["block_type"] in {"figure", "image", "table"}]
        for record in block_records:
            record.setdefault("reason", [])
            hs, heading_reasons = _heading_score(record, page, stats, thresholds)
            if hs is not None:
                record["hs"] = hs
                record["reason"].extend(heading_reasons)
                if hs >= float(thresholds.get("tau_heading_h12", 0.75)):
                    record["meta"]["heading_level"] = 1
                    record["kind"] = "main"
                    record["flow"] = "main"
                elif hs >= float(thresholds.get("tau_heading_h3", 0.6)):
                    record["meta"]["heading_level"] = 2
                    record["kind"] = "main"
                    record["flow"] = "main"

            ms, ms_reasons = mainness_score(record, stats, page_ctx, cfg)
            record["ms"] = ms
            record["reason"].extend(ms_reasons)

            is_heading = bool(record["meta"].get("heading_level"))
            if is_heading:
                record["ms"] = max(record["ms"], 0.9)
                record["confidence"] = max(record.get("confidence", 0.0), record["ms"])

            if record.get("kind") == "aux" or record["block_type"] != "text":
                record["kind"] = "aux"
                record["aux_type"] = subtype_aux(record, page_ctx, cfg)
                record["flow"] = "aux"
                record.setdefault("confidence", 0.72)
                anchor_aux(record, figures_on_page, last_main_record, cfg)
                maybe_quarantine(record, cfg)
                aux_buffer.append(record)
                continue

            if is_heading:
                decision = "main"
                score = record["ms"]
                decision_reasons: List[str] = []
            else:
                decision, score, decision_reasons = flow_safe_decision(record, doc_state, stats, cfg, page_ctx)
                record["ms"] = score
                record["reason"].extend(decision_reasons)

            if decision == "aux":
                record["kind"] = "aux"
                record["aux_type"] = subtype_aux(record, page_ctx, cfg)
                record["flow"] = "aux"
                record.setdefault("confidence", 0.58)
                anchor_aux(record, figures_on_page, last_main_record, cfg)
                maybe_quarantine(record, cfg)
                aux_buffer.append(record)
                continue

            record["kind"] = "main"
            record["flow"] = "main"
            record["confidence"] = max(record.get("confidence", 0.0), score)
            record["expect_continuation"] = _expect_continuation(record)

            control_block = None
            if not _is_in_active_section(doc_state):
                control_block = detect_implicit_section_start([record], doc_state, cfg, stats)
                if control_block:
                    predictions.extend(aux_buffer)
                    aux_buffer.clear()
                    predictions.append(control_block)
                    doc_state["section_state"] = "IN_SECTION"
                    doc_state["continuing_paragraph"] = False

            is_heading = bool(record["meta"].get("heading_level"))
            if is_heading or not _is_in_active_section(doc_state):
                predictions.extend(aux_buffer)
                aux_buffer.clear()

            record["reason"] = list(dict.fromkeys(record["reason"]))
            predictions.append(record)
            doc_state["section_state"] = "IN_SECTION"
            doc_state["continuing_paragraph"] = bool(record["meta"].get("is_continuation"))
            doc_state["last_main_meta"] = record.get("meta")
            last_main_record = record

        recent_figures = figures_for_next

    if aux_buffer:
        doc_state["section_state"] = "OUT_OF_SECTION"
        predictions.extend(aux_buffer)
        aux_buffer.clear()

    return _build_predictions(predictions, cfg)
