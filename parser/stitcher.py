"""Paragraph stitching and auxiliary buffering."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .classifier import BlockPrediction
from .utils import bbox_union


@dataclass
class StitchedBlock:
    page: int
    kind: str
    aux_type: Optional[str]
    subtype: Optional[str]
    text: str
    bbox: Tuple[float, float, float, float]
    flow: str
    ms: float
    hs: Optional[float]
    reason: List[str]
    confidence: float
    meta: Dict[str, object]
    sources: List[Tuple[int, int]] = field(default_factory=list)
    attached_across_pages: bool = False
    anchor_source: Optional[Tuple[int, int]] = None
    quarantined: bool = False
    anchor_hint: Optional[Tuple[int, int]] = None


class ParagraphStitcher:
    def __init__(self, config: Dict[str, object], ttl_pages: int = 2) -> None:
        self.config = config
        self.ttl_pages = ttl_pages
        thresholds = self.config.get("thresholds", {})
        self.tau_heading_h12 = float(thresholds.get("tau_heading_h12", 0.75))
        self.tau_heading_h3 = float(thresholds.get("tau_heading_h3", 0.60))

    def stitch(self, predictions: List[BlockPrediction]) -> List[StitchedBlock]:
        ordered = sorted(predictions, key=lambda p: (p.page, p.index))
        stitched: List[StitchedBlock] = []
        pending: Optional[StitchedBlock] = None
        pending_ttl = self.ttl_pages
        aux_buffer: List[BlockPrediction] = []
        last_main: Optional[StitchedBlock] = None
        current_page: Optional[int] = None
        for prediction in ordered:
            if current_page is None or prediction.page != current_page:
                if current_page is not None and pending is not None:
                    delta = prediction.page - current_page
                    pending_ttl -= delta
                    if pending_ttl < 0:
                        stitched.append(pending)
                        last_main = pending
                        pending = None
                        pending_ttl = self.ttl_pages
                current_page = prediction.page
            if prediction.kind == "control":
                if pending is not None:
                    stitched.append(pending)
                    last_main = pending
                    pending = None
                self._flush_aux_buffer(aux_buffer, last_main, stitched)
                aux_buffer = []
                control_block = self._from_prediction(prediction)
                control_block.sources.append((prediction.page, prediction.index))
                stitched.append(control_block)
                pending_ttl = self.ttl_pages
                continue
            if prediction.kind == "main":
                if pending and self._should_continue(pending, prediction):
                    pending.text = self._join_text(pending.text, prediction.text)
                    pending.bbox = bbox_union(pending.bbox, prediction.bbox)
                    pending.ms = max(pending.ms, prediction.ms)
                    pending.hs = prediction.hs or pending.hs
                    pending.reason = self._merge_reasons(pending.reason, prediction.reason)
                    pending.sources.append((prediction.page, prediction.index))
                    pending.attached_across_pages |= pending.page != prediction.page
                    pending.meta.setdefault("continuations", 0)
                    pending.meta["continuations"] += 1
                    pending.meta["open_continuation"] = prediction.expect_continuation
                    pending_ttl = self.ttl_pages
                else:
                    if pending is not None:
                        stitched.append(pending)
                        last_main = pending
                        self._flush_aux_buffer(aux_buffer, last_main, stitched)
                        aux_buffer = []
                    pending = self._from_prediction(prediction)
                    pending.sources.append((prediction.page, prediction.index))
                    pending.meta["open_continuation"] = prediction.expect_continuation
                    pending_ttl = self.ttl_pages
                    last_main = pending
                    if self._is_heading(prediction):
                        pending.meta["open_continuation"] = False
                        pending.meta["is_heading"] = True
                        self._flush_aux_buffer(aux_buffer, last_main, stitched)
                        aux_buffer = []
                continue
            aux_buffer.append(prediction)
        if pending is not None:
            stitched.append(pending)
            last_main = pending
        self._flush_aux_buffer(aux_buffer, last_main, stitched)
        return stitched

    def _from_prediction(self, prediction: BlockPrediction) -> StitchedBlock:
        meta = dict(prediction.meta)
        meta.setdefault("sources", [])
        if prediction.subtype is not None:
            meta.setdefault("subtype", prediction.subtype)
        return StitchedBlock(
            page=prediction.page,
            kind=prediction.kind,
            aux_type=prediction.aux_type,
            subtype=getattr(prediction, "subtype", None),
            text=prediction.text,
            bbox=prediction.bbox,
            flow=prediction.flow,
            ms=prediction.ms,
            hs=prediction.hs,
            reason=list(prediction.reason),
            confidence=prediction.confidence,
            meta=meta,
            sources=[],
            quarantined=prediction.quarantined,
            anchor_hint=prediction.anchor_hint,
        )

    def _join_text(self, left: str, right: str) -> str:
        if not left:
            return right
        if left.endswith("-"):
            return left[:-1] + right
        if left.endswith(" "):
            return left + right
        return left + " " + right

    def _merge_reasons(self, left: List[str], right: List[str]) -> List[str]:
        merged = list(left)
        for reason in right:
            if reason not in merged:
                merged.append(reason)
        return merged

    def _should_continue(self, block: StitchedBlock, prediction: BlockPrediction) -> bool:
        if prediction.meta.get("heading_level"):
            return False
        if block.meta.get("is_heading"):
            return False
        if block.meta.get("open_continuation"):
            return True
        if prediction.page - block.page > self.ttl_pages:
            return False
        indent = prediction.meta.get("indent", 0.0) or 0.0
        if indent <= 2.0:
            return True
        if prediction.text and prediction.text[0].islower():
            return True
        return False

    def _is_heading(self, prediction: BlockPrediction) -> bool:
        if prediction.hs is None:
            return False
        if prediction.hs >= self.tau_heading_h12:
            return True
        if prediction.hs >= self.tau_heading_h3 and prediction.meta.get("heading_level"):
            return True
        return False

    def _flush_aux_buffer(
        self,
        aux_buffer: List[BlockPrediction],
        anchor: Optional[StitchedBlock],
        stitched: List[StitchedBlock],
    ) -> None:
        if not aux_buffer:
            return
        if anchor is None and stitched:
            anchor = self._last_main(stitched)
        created: Dict[Tuple[int, int], StitchedBlock] = {}
        for prediction in aux_buffer:
            block = self._from_prediction(prediction)
            source_key = (prediction.page, prediction.index)
            block.sources.append(source_key)
            created[source_key] = block
            target_anchor = None
            if prediction.anchor_hint:
                target_anchor = created.get(prediction.anchor_hint)
                if target_anchor is None:
                    target_anchor = self._resolve_anchor(prediction.anchor_hint, stitched)
            if target_anchor is None:
                target_anchor = anchor
            if target_anchor is None and stitched:
                target_anchor = self._last_main(stitched)
            if target_anchor and target_anchor.sources:
                block.anchor_source = target_anchor.sources[0]
                block.attached_across_pages = target_anchor.page != prediction.page
            stitched.append(block)
        aux_buffer.clear()

    def _resolve_anchor(
        self, hint: Tuple[int, int], stitched: List[StitchedBlock]
    ) -> Optional[StitchedBlock]:
        for block in reversed(stitched):
            if hint in block.sources:
                return block
        return None

    def _last_main(self, stitched: List[StitchedBlock]) -> Optional[StitchedBlock]:
        for block in reversed(stitched):
            if block.kind == "main":
                return block
        return None


def stitch_predictions(
    predictions: List[BlockPrediction],
    config: Dict[str, object],
) -> List[StitchedBlock]:
    stitcher = ParagraphStitcher(config=config)
    return stitcher.stitch(predictions)
