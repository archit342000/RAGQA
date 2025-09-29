from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .config import PipelineConfig
from .normalize import Block
from .flow_chunker import FlowChunkPlan, build_flow_chunk_plan


@dataclass(slots=True)
class SegmentChunk:
    heading_path: List[str]
    text: str
    page_span: List[int]
    token_count: int
    evidence_spans: List[dict]
    sidecars: List[dict]
    quality: dict
    aux_groups: dict
    notes: List[str]
    limits: dict
    flow_overflow: int
    closed_at_boundary: str
    aux_in_followup: bool
    link_prev_index: Optional[int]
    link_next_index: Optional[int]


@dataclass(slots=True)
class _Segment:
    heading_path: List[str]
    start_page: int
    last_page: int
    main_blocks: List[Block] = field(default_factory=list)
    aux_buffer: List[Block] = field(default_factory=list)
    notes: set[str] = field(default_factory=set)
    last_main_page: int = 0


class Segmentor:
    """Buffer auxiliaries per segment and emit chunks on boundaries."""

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
        self._pre_segment_aux: List[Block] = []
        self._last_main_by_page_heading: Dict[Tuple[int, Tuple[str, ...]], Block] = {}
        self._last_main_by_heading: Dict[Tuple[str, ...], Block] = {}
        self._last_main_block: Optional[Block] = None
        self._pending_soft_flush = False

    # ---- public API -------------------------------------------------
    def boundary(self, hard: bool) -> List[SegmentChunk]:
        if self.active is None:
            return []
        if hard:
            self.telemetry.inc("hard_boundaries")
        else:
            self.telemetry.inc("soft_boundaries")
        return self._flush(soft=not hard)

    def add_block(self, block: Block) -> List[SegmentChunk]:
        if self.active is None and block.role == "auxiliary":
            self._pre_segment_aux.append(block)
            if block.aux_subtype not in {"header", "footer"}:
                self.telemetry.inc("aux_buffered")
            return []

        if block.role == "main":
            if self.active is None and self._eligible_for_segment0(block):
                self._start_segment(block)
            elif self.active is None:
                return []
        elif self.active is None:
            return []

        emitted: List[SegmentChunk] = []
        if self._should_soft_flush(block):
            emitted.extend(self.boundary(hard=False))

        if self.active is None:
            if block.role == "main":
                self._start_segment(block)
            else:
                self._pre_segment_aux.append(block)
                if block.aux_subtype not in {"header", "footer"}:
                    self.telemetry.inc("aux_buffered")
            return emitted

        self.active.last_page = max(self.active.last_page, block.page)

        if block.role == "main":
            self._register_main(block)
            if block.type != "heading":
                self.active.main_blocks.append(block)
        else:
            if block.aux_subtype not in {"header", "footer"}:
                self.telemetry.inc("aux_buffered")
            self._assign_parent(block)
            self.active.aux_buffer.append(block)

        return emitted

    def finish(self) -> List[SegmentChunk]:
        return self._flush(soft=False)

    # ---- helpers ----------------------------------------------------
    def _eligible_for_segment0(self, block: Block) -> bool:
        text_len = len((block.text or "").strip())
        if text_len >= self.config.aux.segment0_min_chars:
            return True
        if block.heading_level is not None and block.heading_level <= 3:
            return True
        if text_len > 0:
            return True
        return False

    def _start_segment(self, block: Block) -> None:
        heading_path = list(block.heading_path) or (
            [block.text.strip()] if block.type == "heading" and block.text.strip() else []
        )
        start_page = block.page
        segment = _Segment(
            heading_path=heading_path,
            start_page=start_page,
            last_page=start_page,
            main_blocks=[],
            aux_buffer=list(self._pre_segment_aux),
            notes=set(),
            last_main_page=start_page,
        )
        self._pre_segment_aux.clear()
        self.active = segment

    def _register_main(self, block: Block) -> None:
        key = (block.page, tuple(block.heading_path))
        self._last_main_by_page_heading[key] = block
        self._last_main_by_heading[tuple(block.heading_path)] = block
        self._last_main_block = block
        if self.active:
            self.active.last_main_page = block.page
            if block.heading_path:
                self.active.heading_path = list(block.heading_path)

    def _assign_parent(self, block: Block) -> None:
        if block.aux_subtype in {"header", "footer"}:
            return
        key = (block.page, tuple(block.heading_path))
        parent = self._last_main_by_page_heading.get(key)
        if parent is None:
            parent = self._last_main_by_heading.get(tuple(block.heading_path))
        if parent is None and self._last_main_block is not None:
            parent = self._last_main_block
        if parent is not None:
            block.parent_block_id = parent.block_id
        elif block.aux_subtype == "caption":
            self.telemetry.flag("AUX02")

    def _should_soft_flush(self, block: Block) -> bool:
        if self.active is None:
            return False
        max_pages = self.config.aux.soft_boundary_max_deferred_pages
        span = block.page - self.active.start_page + 1
        if span > max_pages:
            self.telemetry.flag("AUX04")
            self.active.notes.add("soft-flush")
            return True
        if (
            block.role == "auxiliary"
            and self.active.last_main_page
            and block.page - self.active.last_main_page >= 1
        ):
            self.active.notes.add("soft-flush")
            return True
        return False

    def _flush(self, soft: bool) -> List[SegmentChunk]:
        if self.active is None:
            return []
        segment = self.active
        self.active = None
        if not segment.main_blocks and not segment.aux_buffer:
            return []
        chunks = self._pack_main_chunks(segment)
        if not chunks:
            # build placeholder chunk for auxiliary-only segments
            chunks = [
                SegmentChunk(
                    heading_path=list(segment.heading_path),
                    text="",
                    page_span=[segment.start_page, segment.last_page],
                    token_count=0,
                    evidence_spans=[],
                    sidecars=[],
                    quality={"ocr_pages": 0, "rescued": False, "notes": ""},
                    aux_groups={"sidecars": [], "footnotes": [], "other": []},
                    notes=list(segment.notes),
                    limits={
                        "target": self.config.flow.limits.target,
                        "soft": self.config.flow.limits.soft,
                        "hard": self.config.flow.limits.hard,
                        "min": self.config.flow.limits.minimum,
                    },
                    flow_overflow=0,
                    closed_at_boundary="EOF",
                    aux_in_followup=False,
                    link_prev_index=None,
                    link_next_index=None,
                )
            ]
        aux_groups, legacy_sidecars, aux_notes, aux_tokens = self._group_aux(
            segment.aux_buffer
        )
        last_chunk = chunks[-1]
        last_chunk.sidecars.extend(legacy_sidecars)
        if soft or "soft-flush" in segment.notes:
            if "soft-flush" not in last_chunk.notes:
                last_chunk.notes.append("soft-flush")
        last_chunk.notes.extend(aux_notes)
        hard_limit = self.config.flow.limits.hard
        if aux_tokens and last_chunk.token_count + aux_tokens > hard_limit:
            aux_chunk = SegmentChunk(
                heading_path=list(segment.heading_path),
                text="",
                page_span=list(last_chunk.page_span),
                token_count=0,
                evidence_spans=[],
                sidecars=[],
                quality={"ocr_pages": 0, "rescued": False, "notes": ""},
                aux_groups=aux_groups,
                notes=list(segment.notes),
                limits=last_chunk.limits,
                flow_overflow=0,
                closed_at_boundary="EOF",
                aux_in_followup=True,
                link_prev_index=len(chunks) - 1,
                link_next_index=None,
            )
            last_chunk.aux_in_followup = True
            last_chunk.link_next_index = len(chunks)
            chunks.append(aux_chunk)
        else:
            merged_groups = last_chunk.aux_groups
            merged_groups.setdefault("sidecars", []).extend(aux_groups["sidecars"])
            merged_groups.setdefault("footnotes", []).extend(aux_groups["footnotes"])
            merged_groups.setdefault("other", []).extend(aux_groups["other"])
        self.telemetry.inc("segments")
        return chunks

    # ---- chunk creation ---------------------------------------------
    def _pack_main_chunks(self, segment: _Segment) -> List[SegmentChunk]:
        if not segment.main_blocks:
            return []
        plans = build_flow_chunk_plan(segment.main_blocks, self.config, self.token_counter)
        return [self._build_chunk_from_plan(plan, segment) for plan in plans]

    def _build_chunk_from_plan(self, plan: FlowChunkPlan, segment: _Segment) -> SegmentChunk:
        evidence_spans: List[dict] = []
        pages = set()
        combined = ""
        notes = set()
        for block in plan.blocks:
            snippet = (block.text or "").strip()
            if not snippet:
                continue
            if combined:
                combined += "\n\n"
            start = len(combined)
            combined += snippet
            end = len(combined)
            evidence_spans.append({
                "para_block_id": block.block_id,
                "start": start,
                "end": end,
            })
            pages.add(block.page)
            if block.source.get("notes"):
                notes.add(block.source.get("notes"))
        token_count = self.token_counter(combined)
        ocr_pages = {
            block.page
            for block in plan.blocks
            if block.source.get("stage") == "ocr"
        }
        rescued = any(block.source.get("stage") == "layout" for block in plan.blocks)
        chunk = SegmentChunk(
            heading_path=list(segment.heading_path),
            text=combined,
            page_span=[min(pages), max(pages)] if pages else [segment.start_page, segment.last_page],
            token_count=token_count,
            evidence_spans=evidence_spans,
            sidecars=[],
            quality={
                "ocr_pages": len(ocr_pages),
                "rescued": rescued,
                "notes": ",".join(sorted(notes)) if notes else "",
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
        )
        return chunk

    def _group_aux(self, aux_blocks: Iterable[Block]):
        sidecars: List[dict] = []
        legacy_sidecars: List[dict] = []
        footnotes: List[dict] = []
        other: List[dict] = []
        notes: List[str] = []
        text_samples: List[str] = []
        for block in aux_blocks:
            text = (block.text or "").strip()
            subtype = block.aux_subtype or "other"
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
                self.telemetry.inc("aux_emitted")
                text_samples.append(text)
                continue
            if subtype == "footnote":
                footnotes.append({"ref_id": block.block_id, "text": text})
                self.telemetry.inc("aux_emitted")
                text_samples.append(text)
                continue
            other.append({"aux_subtype": subtype, "text": text})
            self.telemetry.inc("aux_emitted")
            if text:
                text_samples.append(text)
        aux_tokens = (
            self.token_counter("\n\n".join(sample for sample in text_samples if sample))
            if text_samples
            else 0
        )
        return (
            {"sidecars": sidecars, "footnotes": footnotes, "other": other},
            legacy_sidecars,
            notes,
            aux_tokens,
        )
