from __future__ import annotations

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import PipelineConfig
from .flow_chunker import FlowChunkPlan, build_flow_chunk_plan
from .flow_fence import flow_fence_tail_sanitize
from .main_gate import main_gate
from .gate5 import Gate5Decision, evaluate_gate5
from .diagnostics import make_diagnostic_row
from .normalize import Block


@dataclass(slots=True)
class SegmentChunk:
    segment_id: str
    segment_seq: int
    heading_path: List[str]
    text: str
    page_span: List[int]
    token_count: int
    evidence_spans: List[dict]
    sidecars: List[dict]
    quality: dict
    aux_groups: Dict[str, List[dict]]
    notes: List[str]
    limits: dict
    flow_overflow: int
    closed_at_boundary: str
    aux_in_followup: bool
    link_prev_index: Optional[int]
    link_next_index: Optional[int]
    is_main_only: bool
    is_aux_only: bool
    aux_subtypes_present: List[str]
    aux_group_seq: Optional[int]
    debug_blocks: List[Block] = field(default_factory=list)


@dataclass(slots=True)
class _Segment:
    segment_id: str
    seq: int
    heading_path: List[str]
    start_page: int
    last_page: int
    main_blocks: List[Block] = field(default_factory=list)
    aux_blocks: List[Block] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    anchors: List[Dict[str, Optional[str]]] = field(default_factory=list)
    notes: set[str] = field(default_factory=set)
    all_blocks: List[Block] = field(default_factory=list)


class Segmentor:
    """Two-pass segment buffer implementing Auxiliary Isolation."""

    def __init__(
        self,
        doc_id: str,
        config: PipelineConfig,
        telemetry,
        token_counter: Callable[[str], int],
    ) -> None:
        self.doc_id = doc_id
        self.config = config
        self.telemetry = telemetry
        self.token_counter = token_counter
        self.active: Optional[_Segment] = None
        self._segment_index = 0
        self._pre_segment_aux: List[Block] = []
        self._last_main_by_heading: Dict[Tuple[str, ...], Block] = {}
        self._last_main_by_page_heading: Dict[Tuple[int, Tuple[str, ...]], Block] = {}
        self._last_main_block: Optional[Block] = None

    # ------------------------------------------------------------------
    def add_block(self, block: Block) -> List[SegmentChunk]:
        """Feed blocks in reading order; returns emitted chunk payloads."""
        passed, reasons = main_gate(block)
        block.main_gate_passed = passed
        block.rejection_reasons = reasons
        if not passed and block.role == "main":
            block.role = "auxiliary"
            if block.aux_subtype is None:
                block.aux_subtype = "other"

        if block.type == "heading":
            heading_path = list(block.heading_path)
            if not heading_path and block.text.strip():
                heading_path = [block.text.strip()]
            if self.active is None:
                self._start_segment(block)
            else:
                self.active.heading_path = heading_path
                self.active.last_page = max(self.active.last_page, block.page)
            return []

        if self.active is None:
            if passed and block.role == "main":
                self._start_segment(block)
            else:
                self._queue_pre_segment_aux(block)
                return []

        assert self.active is not None  # for type checkers
        emitted: List[SegmentChunk] = []

        if self._needs_soft_flush(block):
            self.telemetry.inc("soft_boundaries")
            emitted.extend(self._flush(soft=True))
            if self.active is None:
                if passed and block.role == "main":
                    self._start_segment(block)
                else:
                    self._queue_pre_segment_aux(block)
                return emitted

        segment = self.active
        segment.last_page = max(segment.last_page, block.page)
        if block.page not in segment.pages:
            segment.pages.append(block.page)
        segment.all_blocks.append(block)

        if passed and block.role == "main":
            decision = evaluate_gate5(block, self.config)
            self._record_gate5(block, decision)
            if decision.allow:
                self._register_main(block)
                segment.main_blocks.append(block)
                block.aux.setdefault("diagnostic_color", "green")
                self.telemetry.inc("main_blocks_kept")
            else:
                block.aux.setdefault("diagnostic_color", "orange")
                block.aux.setdefault("gate5_reason", decision.reason)
                self.telemetry.inc("gate5_denies")
                if block.aux_subtype not in {"header", "footer"}:
                    self.telemetry.inc("aux_buffered")
                    self.telemetry.inc("blocks_diverted_to_aux")
                self._anchor_aux(block, segment)
                if block.aux_subtype is None:
                    block.aux_subtype = "other"
                segment.aux_blocks.append(block)
        else:
            if block.aux_subtype not in {"header", "footer"}:
                self.telemetry.inc("aux_buffered")
                if not passed:
                    self.telemetry.inc("blocks_diverted_to_aux")
            self._anchor_aux(block, segment)
            segment.aux_blocks.append(block)

        return emitted

    def boundary(self, hard: bool) -> List[SegmentChunk]:
        if self.active is None:
            return []
        if hard:
            self.telemetry.inc("hard_boundaries")
        else:
            self.telemetry.inc("soft_boundaries")
        return self._flush(soft=not hard)

    def finish(self) -> List[SegmentChunk]:
        return self._flush(soft=False)

    # ------------------------------------------------------------------
    def _start_segment(self, block: Block) -> None:
        self._segment_index += 1
        heading = list(block.heading_path)
        if not heading and block.type == "heading" and block.text.strip():
            heading = [block.text.strip()]
        segment_id = f"{self.doc_id}-seg{self._segment_index:03d}"
        segment = _Segment(
            segment_id=segment_id,
            seq=self._segment_index - 1,
            heading_path=heading,
            start_page=block.page,
            last_page=block.page,
            main_blocks=[],
            aux_blocks=list(self._pre_segment_aux),
        )
        segment.pages.append(block.page)
        if segment.aux_blocks:
            segment.all_blocks.extend(segment.aux_blocks)
        self._pre_segment_aux.clear()
        self.active = segment

    def _queue_pre_segment_aux(self, block: Block) -> None:
        if block.aux_subtype not in {"header", "footer"}:
            self.telemetry.inc("aux_buffered")
        self._pre_segment_aux.append(block)

    def _needs_soft_flush(self, block: Block) -> bool:
        if self.active is None:
            return False
        max_span = self.config.aux.soft_boundary_max_deferred_pages
        span = block.page - self.active.start_page + 1
        if span > max_span:
            self.active.notes.add("soft-flush")
            self.telemetry.flag("AUX04")
            return True
        return False

    def _register_main(self, block: Block) -> None:
        key = (block.page, tuple(block.heading_path))
        self._last_main_by_page_heading[key] = block
        self._last_main_by_heading[tuple(block.heading_path)] = block
        self._last_main_block = block
        if self.active:
            if block.heading_path:
                self.active.heading_path = list(block.heading_path)

    def _anchor_aux(self, block: Block, segment: _Segment) -> None:
        status = "orphan"
        parent: Optional[Block] = None
        if block.aux_subtype in {"header", "footer"}:
            segment.anchors.append({"aux_block_id": block.block_id, "parent_block_id": None, "status": status})
            return
        key = (block.page, tuple(block.heading_path))
        parent = self._last_main_by_page_heading.get(key)
        if parent is not None:
            status = "prev"
        else:
            prev_key = (block.page - 1, tuple(block.heading_path))
            parent = self._last_main_by_page_heading.get(prev_key)
            if parent is not None:
                status = "prev_page"
        if parent is None and self._last_main_block is not None:
            parent = self._last_main_block
            status = "prev"
        if parent is not None:
            block.parent_block_id = parent.block_id
        segment.anchors.append(
            {
                "aux_block_id": block.block_id,
                "parent_block_id": parent.block_id if parent else None,
                "status": status,
            }
        )

    def _flush(self, soft: bool) -> List[SegmentChunk]:
        if self.active is None:
            return []
        segment = self.active
        self.active = None

        payloads: List[SegmentChunk] = []
        if self._should_activate_paragraph_only(segment):
            self._activate_paragraph_only(segment)
        plans = build_flow_chunk_plan(segment.main_blocks, self.config, self.token_counter)
        seq_counter = 0
        for plan in plans:
            if not plan.blocks:
                continue
            ordered_blocks: List[Block] = []
            for fragment in plan.blocks:
                block = fragment.block
                if not ordered_blocks or ordered_blocks[-1] is not block:
                    ordered_blocks.append(block)
            kept, diverted, hits = flow_fence_tail_sanitize(ordered_blocks, {"config": self.config})
            if hits:
                self.telemetry.inc("flow_fence_hits", hits)
            if diverted:
                for extra in diverted:
                    if extra not in segment.aux_blocks:
                        segment.aux_blocks.append(extra)
                        if extra.aux_subtype not in {"header", "footer"}:
                            self.telemetry.inc("aux_buffered")
                            self.telemetry.inc("blocks_diverted_to_aux")
            allowed = {block.block_id for block in kept}
            plan.blocks = [fragment for fragment in plan.blocks if fragment.block.block_id in allowed]
            if not plan.blocks:
                continue
            chunk = self._build_chunk_from_plan(plan, segment)
            chunk.segment_id = segment.segment_id
            chunk.segment_seq = seq_counter
            chunk.is_main_only = True
            chunk.is_aux_only = False
            chunk.aux_subtypes_present = []
            chunk.aux_group_seq = None
            if soft or "soft-flush" in segment.notes:
                if "soft-flush" not in chunk.notes:
                    chunk.notes.append("soft-flush")
            payloads.append(chunk)
            seq_counter += 1

        segment.aux_blocks.extend(self._pre_segment_aux)
        self._pre_segment_aux.clear()

        aux_groups, legacy_sidecars, aux_notes, aux_tokens, aux_subtypes = self._group_aux(segment.aux_blocks)
        if payloads and aux_tokens:
            last = payloads[-1]
            last.sidecars.extend(legacy_sidecars)
            if aux_notes:
                last.notes.extend([note for note in aux_notes if note not in last.notes])
            hard = self.config.flow.limits.hard
            if last.token_count + aux_tokens <= hard:
                merged = last.aux_groups
                merged.setdefault("sidecars", []).extend(aux_groups["sidecars"])
                merged.setdefault("footnotes", []).extend(aux_groups["footnotes"])
                merged.setdefault("other", []).extend(aux_groups["other"])
                last.aux_subtypes_present = sorted(aux_subtypes)
            else:
                aux_chunk = self._build_aux_chunk(segment, seq_counter, aux_groups, aux_notes, aux_subtypes)
                aux_chunk.link_prev_index = len(payloads) - 1
                payloads[-1].aux_in_followup = True
                payloads[-1].link_next_index = len(payloads)
                payloads.append(aux_chunk)
        elif aux_groups["sidecars"] or aux_groups["footnotes"] or aux_groups["other"]:
            aux_chunk = self._build_aux_chunk(segment, seq_counter, aux_groups, aux_notes, aux_subtypes)
            payloads.append(aux_chunk)
        elif payloads:
            payloads[-1].sidecars.extend(legacy_sidecars)

        self._validate_invariants(payloads)
        if payloads:
            self.telemetry.inc("segments")
        return payloads

    # ------------------------------------------------------------------
    def _build_chunk_from_plan(self, plan: FlowChunkPlan, segment: _Segment) -> SegmentChunk:
        evidence_map: Dict[str, dict] = {}
        evidence_order: List[str] = []
        combined = ""
        pages = set()
        notes = set()
        last_block_id: Optional[str] = None
        ordered_blocks: List[Block] = []
        for fragment in plan.blocks:
            block = fragment.block
            if not ordered_blocks or ordered_blocks[-1] is not block:
                ordered_blocks.append(block)
        for fragment in plan.blocks:
            block = fragment.block
            snippet = fragment.text.strip()
            if not snippet:
                continue
            if combined:
                if fragment.is_continuation or block.block_id == last_block_id:
                    combined += " "
                else:
                    combined += "\n\n"
            start = len(combined)
            combined += snippet
            end = len(combined)
            if block.block_id not in evidence_map:
                evidence_order.append(block.block_id)
                evidence_map[block.block_id] = {
                    "para_block_id": block.block_id,
                    "start": start,
                    "end": end,
                }
            else:
                evidence_map[block.block_id]["end"] = end
            pages.add(block.page)
            last_block_id = block.block_id
        if plan.forced_split:
            notes.add("forced-split")
            self.telemetry.flag("FLOW_FORCED_SPLIT")
        token_count = self.token_counter(combined)
        evidence_spans = [evidence_map[bid] for bid in evidence_order]
        ocr_pages = {
            fragment.block.page
            for fragment in plan.blocks
            if fragment.block.source.get("stage") == "ocr"
        }
        rescued = any(fragment.block.source.get("stage") == "layout" for fragment in plan.blocks)
        chunk = SegmentChunk(
            segment_id=segment.segment_id,
            segment_seq=0,
            heading_path=list(segment.heading_path),
            text=combined,
            page_span=[min(pages), max(pages)] if pages else [segment.start_page, segment.last_page],
            token_count=token_count,
            evidence_spans=evidence_spans,
            sidecars=[],
            quality={
                "ocr_pages": len(ocr_pages),
                "rescued": rescued,
                "notes": "",
            },
            aux_groups={"sidecars": [], "footnotes": [], "other": []},
            notes=list(segment.notes),
            limits={
                "target": self.config.flow.limits.target,
                "soft": self.config.flow.limits.soft,
                "hard": self.config.flow.limits.hard,
                "min": self.config.flow.limits.minimum,
            },
            flow_overflow=max(0, token_count - self.config.flow.limits.target),
            closed_at_boundary=plan.closed_at,
            aux_in_followup=False,
            link_prev_index=None,
            link_next_index=None,
            is_main_only=True,
            is_aux_only=False,
            aux_subtypes_present=[],
            aux_group_seq=None,
            debug_blocks=ordered_blocks,
        )
        if notes:
            joined = ",".join(sorted(notes))
            chunk.quality["notes"] = joined
            for note in notes:
                if note not in chunk.notes:
                    chunk.notes.append(note)
        return chunk

    def _build_aux_chunk(
        self,
        segment: _Segment,
        seq_counter: int,
        aux_groups: Dict[str, List[dict]],
        aux_notes: List[str],
        aux_subtypes: List[str],
    ) -> SegmentChunk:
        chunk = SegmentChunk(
            segment_id=segment.segment_id,
            segment_seq=seq_counter,
            heading_path=list(segment.heading_path),
            text="",
            page_span=[segment.start_page, segment.last_page],
            token_count=0,
            evidence_spans=[],
            sidecars=[],
            quality={"ocr_pages": 0, "rescued": False, "notes": ""},
            aux_groups=aux_groups,
            notes=list(set(list(segment.notes) + aux_notes)),
            limits={
                "target": self.config.flow.limits.target,
                "soft": self.config.flow.limits.soft,
                "hard": self.config.flow.limits.hard,
                "min": self.config.flow.limits.minimum,
            },
            flow_overflow=0,
            closed_at_boundary="EOF",
            aux_in_followup=True,
            link_prev_index=None,
            link_next_index=None,
            is_main_only=False,
            is_aux_only=True,
            aux_subtypes_present=sorted(aux_subtypes),
            aux_group_seq=1,
            debug_blocks=[],
        )
        return chunk

    def _group_aux(self, aux_blocks: Sequence[Block]):
        sidecars: List[dict] = []
        legacy_sidecars: List[dict] = []
        footnotes: List[dict] = []
        other: List[dict] = []
        notes: List[str] = []
        text_samples: List[str] = []
        subtypes: set[str] = set()
        for block in aux_blocks:
            text = (block.text or "").strip()
            subtype = block.aux_subtype or "other"
            if subtype:
                subtypes.add(subtype)
            if block.type in {"figure", "table"} or subtype == "caption":
                entry_type = "table" if block.type == "table" or text.lower().startswith("table") else "figure"
                sidecar_entry = {
                    "parent_block_id": block.parent_block_id,
                    "type": entry_type,
                    "page": block.page,
                    "text": text,
                }
                sidecars.append(sidecar_entry)
                legacy_sidecars.append({"type": entry_type, "page": block.page, "text": text})
                if subtype == "caption" and block.parent_block_id is None:
                    notes.append("orphan-caption")
                if text:
                    text_samples.append(text)
                self.telemetry.inc("aux_emitted")
                continue
            if subtype == "footnote":
                footnotes.append({"ref_id": block.block_id, "text": text})
                if text:
                    text_samples.append(text)
                self.telemetry.inc("aux_emitted")
                continue
            other.append({"aux_subtype": subtype, "text": text})
            if text:
                text_samples.append(text)
            self.telemetry.inc("aux_emitted")
        aux_tokens = self.token_counter("\n\n".join(text_samples)) if text_samples else 0
        return (
            {"sidecars": sidecars, "footnotes": footnotes, "other": other},
            legacy_sidecars,
            notes,
            aux_tokens,
            list(subtypes),
        )

    def _validate_invariants(self, chunks: Sequence[SegmentChunk]) -> None:
        if not chunks:
            return
        deny_re = re.compile(r"^(fig|figure|table|source|activity|let.?s)", re.IGNORECASE)
        segment_id = chunks[0].segment_id
        aux_seen = False
        main_seen = False
        for chunk in chunks:
            if chunk.segment_id != segment_id:
                self._raise_invariant("I2", "segment mismatch")
            if chunk.is_aux_only:
                aux_seen = True
                continue
            main_seen = True
            if aux_seen:
                self._raise_invariant("I2", "main chunk after aux chunk")
            for line in (chunk.text or "").splitlines():
                if deny_re.match(line.strip()):
                    self._raise_invariant("I1", "deny regex text in main chunk")
            for block in chunk.debug_blocks:
                if not self._width_ok(block):
                    self._raise_invariant("I3", f"width floor violation {block.block_id}")

    # ------------------------------------------------------------------
    def _record_gate5(self, block: Block, decision: Gate5Decision) -> None:
        if hasattr(self.telemetry, "record_gate5_decision"):
            label = "APPEND_OK" if decision.allow else f"DENY(G5:{decision.reason})"
            row = make_diagnostic_row(block, label, decision.reason)
            self.telemetry.record_gate5_decision(row)

    def _should_activate_paragraph_only(self, segment: _Segment) -> bool:
        if segment.main_blocks:
            min_blocks = self.config.paragraph_only.min_blocks_across_pages
            window = self.config.paragraph_only.window_pages
            if len(segment.main_blocks) >= min_blocks:
                return False
            page_span = segment.last_page - segment.start_page + 1
            return page_span >= window
        return True

    def _activate_paragraph_only(self, segment: _Segment) -> None:
        allowed_types = {"paragraph", "list", "item", "code", "equation"}
        seen_main: set[str] = set()
        seen_aux: set[str] = {block.block_id for block in segment.aux_blocks}
        fallback_main: List[Block] = []
        for block in segment.all_blocks:
            lowered = (block.type or "").lower()
            if lowered not in allowed_types:
                if block.block_id not in seen_aux:
                    segment.aux_blocks.append(block)
                    seen_aux.add(block.block_id)
                continue
            decision = evaluate_gate5(block, self.config)
            self._record_gate5(block, decision)
            if decision.allow:
                if block.block_id not in seen_main:
                    fallback_main.append(block)
                    seen_main.add(block.block_id)
                    block.aux.setdefault("diagnostic_color", "green")
            else:
                block.aux.setdefault("diagnostic_color", "orange")
                block.aux.setdefault("gate5_reason", decision.reason)
                self.telemetry.inc("gate5_denies")
                if block.block_id not in seen_aux:
                    if block.aux_subtype not in {"header", "footer"}:
                        self.telemetry.inc("aux_buffered")
                    if block.aux_subtype is None:
                        block.aux_subtype = "other"
                    segment.aux_blocks.append(block)
                    seen_aux.add(block.block_id)
        segment.main_blocks = fallback_main
        segment.notes.add("paragraph-only")
        self.telemetry.inc("paragraph_only_activations")

    def _width_ok(self, block: Block) -> bool:
        bbox = block.bbox or {}
        width = float(bbox.get("x1", 0.0)) - float(bbox.get("x0", 0.0))
        if width <= 0:
            return True
        column_width = float(block.aux.get("column_width") or 0.0)
        if column_width <= 0:
            column_width = width
        min_fraction = self.config.gate5.sidebar.min_column_width_fraction
        return width >= (min_fraction * column_width)

    def _raise_invariant(self, code: str, message: str) -> None:
        self.telemetry.inc("invariant_violations")
        raise RuntimeError(f"{code} violated: {message}")
