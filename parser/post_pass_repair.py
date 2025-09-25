"""Post-classification layout repair utilities."""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

from .classifier import BlockPrediction
from .utils import bbox_union

CAPTION_RE = re.compile(r"^((Fig\.|Figure)\s*\d+|Table\s+\d+)\b", re.IGNORECASE)


def layout_repair(
    predictions: Sequence[BlockPrediction],
    config: Optional[Dict[str, object]] = None,
) -> List[BlockPrediction]:
    """Apply mandatory layout fixes on a copy of *predictions*.

    The helper follows the order supplied in ``config['post_pass']['order']``.  The
    individual stages operate on best-effort heuristics – when a rule cannot be
    applied confidently the original block is preserved to avoid false
    negatives.
    """

    if not predictions:
        return []
    cfg = config or {}
    post_cfg = cfg.get("post_pass", {}) if isinstance(cfg, dict) else {}
    if not post_cfg.get("enable", True):
        return list(predictions)
    stages = post_cfg.get(
        "order",
        [
            "stitch_first",
            "peel_captions",
            "shrink_caption_bounds",
            "demote_headers",
            "group_activity",
            "dedup_quarantine",
        ],
    )
    working = [deepcopy(pred) for pred in predictions]
    for stage in stages:
        if stage == "stitch_first":
            continue  # already handled upstream
        if stage == "peel_captions":
            working = _peel_captions(working)
        elif stage == "shrink_caption_bounds":
            working = _shrink_caption_bounds(working)
        elif stage == "demote_headers":
            working = _demote_headers(working, cfg)
        elif stage == "group_activity":
            working = _group_activity(working)
        elif stage == "dedup_quarantine":
            working = _dedup_quarantine(working, cfg)
    return working


def _clone(pred: BlockPrediction) -> BlockPrediction:
    copy = deepcopy(pred)
    copy.reason = list(pred.reason)
    copy.meta = dict(pred.meta)
    return copy


def _nearest_figure(
    block: BlockPrediction,
    peers: Sequence[BlockPrediction],
) -> Optional[BlockPrediction]:
    candidates: List[Tuple[float, float, BlockPrediction]] = []
    bx0, by0, bx1, by1 = block.bbox
    for peer in peers:
        if peer.page != block.page:
            continue
        if peer is block:
            continue
        if peer.kind != "aux" or peer.aux_type not in {"figure", "table"}:
            continue
        px0, py0, px1, py1 = peer.bbox
        horiz_overlap = min(bx1, px1) - max(bx0, px0)
        if horiz_overlap <= 0:
            continue
        vert_gap = max(0.0, by0 - py1)
        dist = vert_gap + abs((bx0 + bx1) / 2 - (px0 + px1) / 2)
        candidates.append((dist, -horiz_overlap, peer))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _peel_captions(predictions: Sequence[BlockPrediction]) -> List[BlockPrediction]:
    peeled: List[BlockPrediction] = []
    for pred in predictions:
        text = (pred.text or "").strip()
        figure = _nearest_figure(pred, predictions)
        if (
            pred.kind == "main"
            and text
            and CAPTION_RE.match(text)
            and (pred.meta.get("region_tag") in {"figure", "table"} or figure is not None)
        ):
            caption = _clone(pred)
            caption.kind = "aux"
            caption.flow = "aux"
            caption.aux_type = "caption"
            caption.reason.append("PostPass:Peel")
            caption.confidence = max(pred.confidence, 0.8)
            if figure is not None:
                caption.anchor_hint = (figure.page, figure.index)
            peeled.append(caption)
            continue
        peeled.append(_clone(pred))
    return peeled


def _split_caption_text(text: str) -> Tuple[str, str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return text, ""
    if len(sentences) == 1:
        return sentences[0], ""
    first = sentences[0]
    remainder = " ".join(sentences[1:]).strip()
    return first.strip(), remainder


def _shrink_caption_bounds(predictions: Sequence[BlockPrediction]) -> List[BlockPrediction]:
    adjusted: List[BlockPrediction] = []
    remainders: List[BlockPrediction] = []
    for pred in predictions:
        if pred.kind == "aux" and pred.aux_type == "caption" and pred.text:
            first, remainder = _split_caption_text(pred.text)
            caption = _clone(pred)
            caption.text = first
            if remainder:
                caption.reason.append("PostPass:CaptionTrim")
                remainder_block = _clone(pred)
                remainder_block.kind = "main"
                remainder_block.flow = "main"
                remainder_block.aux_type = None
                remainder_block.text = remainder
                remainder_block.reason.append("PostPass:CaptionRemainder")
                remainder_block.quarantined = False
                remainder_block.confidence = max(pred.confidence, 0.6)
                remainders.append(remainder_block)
            adjusted.append(caption)
        else:
            adjusted.append(_clone(pred))
    adjusted.extend(remainders)
    adjusted.sort(key=lambda p: (p.page, p.index, 0 if p.kind == "main" else 1))
    return adjusted


def _demote_headers(
    predictions: Sequence[BlockPrediction],
    cfg: Dict[str, object],
) -> List[BlockPrediction]:
    demoted: List[BlockPrediction] = []
    thresholds = cfg.get("thresholds", {}) if isinstance(cfg, dict) else {}
    header_band = cfg.get("bands", {}).get("header_y_pct", 0.12) if isinstance(cfg, dict) else 0.12
    footer_band = cfg.get("bands", {}).get("footer_y_pct", 0.12) if isinstance(cfg, dict) else 0.12
    for pred in predictions:
        if pred.kind == "aux" and pred.aux_type in {"header", "footer"}:
            tokens = len(pred.text.split()) if pred.text else 0
            line_count = (pred.text.count("\n") + 1) if pred.text else 0
            top_pct = float(pred.meta.get("top_pct", 1.0))
            bottom_pct = float(pred.meta.get("bottom_pct", 0.0))
            region_tag = pred.meta.get("region_tag")
            in_band = (
                (pred.aux_type == "header" and top_pct <= header_band + 0.02)
                or (pred.aux_type == "footer" and bottom_pct >= 1.0 - footer_band - 0.02)
            )
            if tokens > 12 or line_count > 2 or (region_tag not in {"title", "text"} and not in_band):
                demoted_block = _clone(pred)
                demoted_block.kind = "main"
                demoted_block.flow = "main"
                demoted_block.aux_type = None
                demoted_block.reason.append("PostPass:DemoteHeader")
                demoted_block.confidence = max(pred.confidence, float(thresholds.get("tau_fail_safe_low", 0.52)))
                demoted.append(demoted_block)
                continue
        demoted.append(_clone(pred))
    return demoted


def _group_activity(predictions: Sequence[BlockPrediction]) -> List[BlockPrediction]:
    grouped: List[BlockPrediction] = []
    buffer: Optional[BlockPrediction] = None
    for pred in sorted(predictions, key=lambda p: (p.page, p.index)):
        if pred.kind == "aux" and pred.aux_type in {"callout", "sidenote"}:
            if buffer and buffer.page == pred.page and buffer.aux_type == pred.aux_type:
                buffer.text = "\n".join(filter(None, [buffer.text, pred.text]))
                buffer.bbox = bbox_union(buffer.bbox, pred.bbox)
                buffer.reason = list(dict.fromkeys(buffer.reason + ["PostPass:GroupActivity"]))
                buffer.confidence = max(buffer.confidence, pred.confidence)
                continue
            if buffer:
                grouped.append(buffer)
            buffer = _clone(pred)
            buffer.reason.append("PostPass:GroupActivity")
            continue
        if buffer:
            grouped.append(buffer)
            buffer = None
        grouped.append(_clone(pred))
    if buffer:
        grouped.append(buffer)
    grouped.sort(key=lambda p: (p.page, p.index))
    return grouped


def _dedup_quarantine(
    predictions: Sequence[BlockPrediction],
    cfg: Optional[Dict[str, object]] = None,
) -> List[BlockPrediction]:
    thresholds = (cfg or {}).get("thresholds", {}) if isinstance(cfg, dict) else {}
    qmin = float(thresholds.get("quarantined_aux_conf_min", 0.65))
    seen: Dict[Tuple[int, str], BlockPrediction] = {}
    deduped: List[BlockPrediction] = []
    for pred in predictions:
        key = (pred.page, (pred.text or "").strip())
        if pred.kind == "aux" and key in seen:
            existing = seen[key]
            if pred.confidence > existing.confidence:
                existing.quarantined = True
                pred.quarantined = False
                deduped.append(_clone(pred))
                continue
            pred.quarantined = True
            deduped.append(_clone(pred))
            continue
        clone = _clone(pred)
        if clone.kind == "aux" and clone.confidence < qmin:
            clone.quarantined = True
            if "PostPass:Quarantine" not in clone.reason:
                clone.reason.append("PostPass:Quarantine")
        deduped.append(clone)
        if clone.kind == "aux":
            seen[key] = clone
    deduped.sort(key=lambda p: (p.page, p.index))
    return deduped
