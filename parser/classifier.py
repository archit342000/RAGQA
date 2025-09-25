"""Block classification with aux prepass, mainness scoring, and section control blocks."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

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

    def copy(self) -> "BlockPrediction":
        return replace(self, reason=list(self.reason), meta=dict(self.meta))


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
        "flow": "main",
        "reason": [],
        "ms": 0.0,
        "hs": None,
        "source": block,
    }
    return record


def prepass_aux(blocks: List[Dict[str, Any]], page_ctx: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    thresholds = cfg.get("thresholds", {})
    caption_overlap_min = float(thresholds.get("caption_h_overlap_min", 0.3))
    sidenote_max_width = float(thresholds.get("sidenote_max_width_ratio", 0.5))
    sidenote_font_z_max = float(thresholds.get("sidenote_font_z_max", -0.4))
    page_width = page_ctx["width"]
    header_band = page_ctx["header_band"]
    footer_band = page_ctx["footer_band"]
    margin_left = page_ctx["margin_left_pct"]
    margin_right = page_ctx["margin_right_pct"]
    figures = [b for b in blocks if b["block_type"] in {"figure", "image", "table"}]

    for block in blocks:
        meta = block["meta"]
        text = block.get("text", "") or ""
        text_slug = slugify(text)
        text_lower = text.strip().lower()
        top_pct = meta["top_pct"]
        bottom_pct = meta["bottom_pct"]
        width_pct = meta["width_pct"]
        reasons = block["reason"]

        if block["block_type"] in {"figure", "image"}:
            block["kind"] = "aux"
            block["aux_type"] = "figure"
            block["flow"] = "aux"
            reasons.append("PrepassA:Figure")
            continue
        if block["block_type"] == "table":
            block["kind"] = "aux"
            block["aux_type"] = "table"
            block["flow"] = "aux"
            reasons.append("PrepassA:Table")
            continue

        if not text:
            continue

        if top_pct <= header_band:
            if len(text) <= 8 and text.replace(" ", "").isdigit():
                block["kind"] = "aux"
                block["aux_type"] = "page_number"
                block["flow"] = "aux"
                reasons.append("PrepassA:HeaderPageNumber")
                continue
            if meta["token_count"] <= 15:
                block["kind"] = "aux"
                block["aux_type"] = "header"
                block["flow"] = "aux"
                reasons.append("PrepassA:HeaderBand")
                continue
        if bottom_pct >= footer_band:
            if detect_footnote_marker(text):
                block["kind"] = "aux"
                block["aux_type"] = "footnote"
                block["flow"] = "aux"
                reasons.append("PrepassA:FootnoteBand")
                continue
            if len(text) <= 8 and text.replace(" ", "").isdigit():
                block["kind"] = "aux"
                block["aux_type"] = "page_number"
                block["flow"] = "aux"
                reasons.append("PrepassA:FooterPageNumber")
                continue
            if meta["token_count"] <= 20:
                block["kind"] = "aux"
                block["aux_type"] = "footer"
                block["flow"] = "aux"
                reasons.append("PrepassA:FooterBand")
                continue

        if detect_footnote_marker(text) and meta["top_pct"] > footer_band - 0.1:
            block["kind"] = "aux"
            block["aux_type"] = "footnote"
            block["flow"] = "aux"
            reasons.append("PrepassA:FootnoteMarker")
            continue

        if width_pct <= sidenote_max_width and (
            meta["left_pct"] <= margin_left or meta["right_pct"] >= margin_right
        ):
            if meta["font_z"] <= sidenote_font_z_max:
                block["kind"] = "aux"
                block["aux_type"] = "sidenote"
                block["flow"] = "aux"
                reasons.append("PrepassA:SidenoteBand")
                continue

        if text_lower.startswith(("figure", "fig.", "table")):
            nearest: Optional[Dict[str, Any]] = None
            overlap = 0.0
            for fig in figures + page_ctx.get("recent_figures", []):
                fig_box = fig["bbox"]
                fig_width = max(1e-6, fig_box[2] - fig_box[0])
                inter_left = max(block["bbox"][0], fig_box[0])
                inter_right = min(block["bbox"][2], fig_box[2])
                inter_width = max(0.0, inter_right - inter_left)
                horiz = inter_width / max(min(meta["width"], fig_width), 1e-6)
                vert_gap = max(0.0, block["bbox"][1] - fig_box[3])
                if horiz >= caption_overlap_min and vert_gap <= page_width * 0.05:
                    if horiz > overlap:
                        overlap = horiz
                        nearest = fig
            if nearest and meta["font_z"] <= 0.0:
                block["kind"] = "aux"
                block["aux_type"] = "caption"
                block["flow"] = "aux"
                block["meta"]["anchor_fig"] = nearest["index"]
                reasons.append("PrepassA:CaptionProximity")
                continue
            if nearest is None and meta["font_z"] <= 0.0 and meta["token_count"] <= 20:
                block["kind"] = "aux"
                block["aux_type"] = "caption"
                block["flow"] = "aux"
                reasons.append("PrepassA:CaptionLexical")
                continue

        if "callout" in text_slug and width_pct <= 0.6:
            block["kind"] = "aux"
            block["aux_type"] = "callout"
            block["flow"] = "aux"
            reasons.append("PrepassA:CalloutLexical")
            continue


def estimate_body_stats(blocks: List[Dict[str, Any]], page_ctx: Dict[str, Any]) -> BodyStats:
    font_sizes = [b["meta"]["font_size"] for b in blocks if b["block_type"] == "text" and b["meta"]["font_size"]]
    widths = [b["meta"]["width"] for b in blocks if b["kind"] != "aux" and b["block_type"] == "text" and b["meta"]["width"]]
    indents = [b["meta"].get("indent", 0.0) or 0.0 for b in blocks if b["block_type"] == "text"]
    font_size = statistics.median(font_sizes) if font_sizes else 11.0
    col_width = statistics.median(widths) if widths else page_ctx["width"] * 0.45
    indent = statistics.median(indents) if indents else 12.0
    return BodyStats(font_size=font_size, col_width=col_width, indent=indent)


def assign_columns_and_wrap(
    blocks: List[Dict[str, Any]],
    page_ctx: Dict[str, Any],
    cfg: Dict[str, Any],
    stats: BodyStats,
) -> None:
    thresholds = cfg.get("thresholds", {})
    near_image_multiplier = float(thresholds.get("near_image_lh_multiplier", 1.5))
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
    block: Dict[str, Any], stats: BodyStats, page_ctx: Dict[str, Any]
) -> Tuple[float, List[str]]:
    meta = block["meta"]
    reasons: List[str] = []
    score = 0.2
    if block["block_type"] == "text":
        score += 0.25
        if meta["token_count"] >= 8:
            score += 0.1
            reasons.append("TokenCount")
        if meta["width"] >= stats.col_width * 0.7:
            score += 0.12
            reasons.append("ColumnWidth")
        if abs((meta["indent"] or 0.0) - stats.indent) <= max(6.0, stats.indent * 0.4):
            score += 0.08
            reasons.append("IndentMatch")
        if meta["wrap_protected"]:
            score += 0.07
            reasons.append("WrapProtected")
        if meta["text_density"] >= 0.25:
            score += 0.05
            reasons.append("Density")
    else:
        score -= 0.15
    if meta["left_pct"] > page_ctx.get("margin_left_pct", 0.05) and meta["right_pct"] < page_ctx.get("margin_right_pct", 0.95):
        score += 0.06
        reasons.append("InsideMargins")
    if detect_footnote_marker(block.get("text", "")):
        score -= 0.3
        reasons.append("FootnoteMarker")
    if block["block_type"] != "text" and block["kind"] != "aux":
        score -= 0.2
    if meta["width_pct"] <= 0.25:
        score -= 0.12
        reasons.append("Narrow")
    if meta["top_pct"] <= page_ctx.get("header_band", 0.12) or meta["bottom_pct"] >= page_ctx.get("footer_band", 0.88):
        score -= 0.18
        reasons.append("HeaderFooterBand")
    score = max(0.0, min(1.0, score))
    return score, reasons


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
    if meta.get("left_pct", 0.0) <= page_ctx.get("margin_left_pct", 0.05) or meta.get("right_pct", 1.0) >= page_ctx.get("margin_right_pct", 0.95):
        if meta.get("width_pct", 1.0) <= float(cfg.get("thresholds", {}).get("sidenote_max_width_ratio", 0.5)):
            return "sidenote"
    if len(text) <= 35 and is_all_caps(text):
        return "callout"
    return "other"


def detect_implicit_section_start(
    window_blocks: List[Dict[str, Any]],
    page_ctx: Dict[str, Any],
    cfg: Dict[str, Any],
    stats: BodyStats,
) -> Optional[Dict[str, Any]]:
    settings = cfg.get("implicit_section", {})
    if not settings.get("enable", True) or not window_blocks:
        return None
    threshold = float(settings.get("score_threshold", 0.75))
    candidate = window_blocks[0]
    if candidate["kind"] != "main" or candidate.get("subtype") == "implicit_section_start":
        return None
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
            reason=list(block.get("reason", [])),
            meta=dict(block.get("meta", {})),
            flow=block.get("flow", block.get("kind", "main")),
            confidence=_confidence(block.get("kind", "main"), block.get("ms", 0.0), block.get("hs")),
            source_block=block["source"],
            expect_continuation=bool(block.get("expect_continuation", False)),
        )
        prediction.meta.setdefault("col_id", block["meta"].get("col_id"))
        prediction.meta.setdefault("open_paragraph_id", block["meta"].get("open_paragraph_id"))
        predictions.append(prediction)
    return predictions


def classify_blocks(doc: DocumentLayout, config: Optional[Dict[str, object]] = None) -> List[BlockPrediction]:
    cfg: Dict[str, Any] = config or load_config()
    predictions: List[Dict[str, Any]] = []
    thresholds = cfg.get("thresholds", {})
    tau_main = float(thresholds.get("tau_main", 0.6))
    tau_confident = float(thresholds.get("tau_main_page_confident", tau_main))
    tau_fail_safe_low = float(thresholds.get("tau_fail_safe_low", 0.52))

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
        stats = estimate_body_stats(block_records, page_ctx)
        assign_columns_and_wrap(block_records, page_ctx, cfg, stats)

        for record in block_records:
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
            ms, ms_reasons = mainness_score(record, stats, page_ctx)
            record["ms"] = ms
            record["reason"].extend(ms_reasons)
            if record["kind"] == "aux":
                record["aux_type"] = subtype_aux(record, page_ctx, cfg)
                record["flow"] = "aux"
                continue
            if record["block_type"] != "text":
                record["kind"] = "aux"
                record["aux_type"] = subtype_aux(record, page_ctx, cfg)
                record["flow"] = "aux"
                continue
            tau = tau_main if font_mean else tau_confident
            if ms >= tau:
                record["kind"] = "main"
                record["flow"] = "main"
            elif tau_fail_safe_low <= ms < tau:
                record["kind"] = "main"
                record["flow"] = "main"
                record["reason"].append("FailSafe")
            else:
                record["kind"] = "aux"
                record["aux_type"] = subtype_aux(record, page_ctx, cfg)
                record["flow"] = "aux"
            record["expect_continuation"] = _expect_continuation(record)
            if record["expect_continuation"]:
                record["meta"]["open_paragraph_id"] = f"p{record['page']}_{record['index']}"

        main_blocks = [b for b in block_records if b["kind"] == "main"]
        implicit_control = detect_implicit_section_start(main_blocks[:1], page_ctx, cfg, stats)
        if implicit_control:
            predictions.append(implicit_control)
            if main_blocks:
                main_blocks[0]["reason"].append("ImplicitSectionLinked")
                main_blocks[0]["meta"]["section_start"] = True
        predictions.extend(block_records)
        recent_figures = figures_for_next

    return _build_predictions(predictions, cfg)
