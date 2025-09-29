from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import re

from .config import PipelineConfig
from .normalize import Block


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
                    notes=[],
                )
            ]
        aux_groups, legacy_sidecars, aux_notes = self._group_aux(segment.aux_buffer)
        last_chunk = chunks[-1]
        last_chunk.sidecars.extend(legacy_sidecars)
        merged_groups = last_chunk.aux_groups
        merged_groups.setdefault("sidecars", []).extend(aux_groups["sidecars"])
        merged_groups.setdefault("footnotes", []).extend(aux_groups["footnotes"])
        merged_groups.setdefault("other", []).extend(aux_groups["other"])
        if soft or "soft-flush" in segment.notes:
            last_chunk.notes.append("soft-flush")
        last_chunk.notes.extend(aux_notes)
        self.telemetry.inc("segments")
        return chunks

    # ---- chunk creation ---------------------------------------------
    def _pack_main_chunks(self, segment: _Segment) -> List[SegmentChunk]:
        chunks: List[SegmentChunk] = []
        if not segment.main_blocks:
            return chunks
        target = self.config.chunk.tokens.target
        maximum = self.config.chunk.tokens.maximum

        current_text: List[str] = []
        current_blocks: List[Tuple[Block, str]] = []

        def _flush_current() -> None:
            if not current_blocks:
                return
            text = "\n\n".join(part for _, part in current_blocks if part.strip())
            pages = {blk.page for blk, _ in current_blocks}
            evidence_spans: List[dict] = []
            cursor = 0
            combined_text = ""
            for blk, part in current_blocks:
                snippet = part.strip()
                if not snippet:
                    continue
                if combined_text:
                    combined_text += "\n\n"
                start = len(combined_text)
                combined_text += snippet
                end = len(combined_text)
                evidence_spans.append({
                    "para_block_id": blk.block_id,
                    "start": start,
                    "end": end,
                })
            ocr_pages = {
                blk.page
                for blk, _ in current_blocks
                if blk.source.get("stage") == "ocr"
            }
            rescued = any(blk.source.get("stage") == "layout" for blk, _ in current_blocks)
            notes = {
                blk.source.get("notes")
                for blk, _ in current_blocks
                if blk.source.get("notes")
            }
            chunk = SegmentChunk(
                heading_path=list(segment.heading_path),
                text=combined_text,
                page_span=[min(pages), max(pages)] if pages else [segment.start_page, segment.last_page],
                token_count=self.token_counter(combined_text),
                evidence_spans=evidence_spans,
                sidecars=[],
                quality={
                    "ocr_pages": len(ocr_pages),
                    "rescued": rescued,
                    "notes": ",".join(sorted(notes)) if notes else "",
                },
                aux_groups={"sidecars": [], "footnotes": [], "other": []},
                notes=[],
            )
            chunks.append(chunk)
            current_blocks.clear()
            current_text.clear()

        for block in segment.main_blocks:
            if not block.text.strip():
                continue
            segments = self._split_text(block.text)
            for piece in segments:
                candidate = list(current_blocks)
                candidate.append((block, piece))
                text = "\n\n".join(part for _, part in candidate if part.strip())
                tokens = self.token_counter(text)
                if tokens > maximum and current_blocks:
                    _flush_current()
                    candidate = [(block, piece)]
                    text = piece.strip()
                    tokens = self.token_counter(text)
                current_blocks = candidate
                if tokens >= target:
                    _flush_current()
        if current_blocks:
            _flush_current()
        return chunks

    def _split_text(self, text: str) -> List[str]:
        if not text:
            return []
        maximum = self.config.chunk.tokens.maximum
        queue: List[str] = [segment for segment in text.split("\n\n") if segment.strip()]
        if not queue:
            queue = [text.strip()]
        result: List[str] = []
        while queue:
            current = queue.pop(0).strip()
            if not current:
                continue
            if self.token_counter(current) <= maximum:
                result.append(current)
                continue
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", current) if s.strip()]
            if len(sentences) > 1:
                buffer = ""
                for sentence in sentences:
                    candidate = f"{buffer} {sentence}".strip() if buffer else sentence
                    if self.token_counter(candidate) > maximum and buffer:
                        queue.append(buffer.strip())
                        buffer = sentence
                    else:
                        buffer = candidate
                if buffer:
                    queue.append(buffer.strip())
                continue
            words = current.split()
            for idx in range(0, len(words), maximum):
                queue.append(" ".join(words[idx : idx + maximum]).strip())
        return result

    def _group_aux(self, aux_blocks: Iterable[Block]):
        sidecars: List[dict] = []
        legacy_sidecars: List[dict] = []
        footnotes: List[dict] = []
        other: List[dict] = []
        notes: List[str] = []
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
                continue
            if subtype == "footnote":
                footnotes.append({"ref_id": block.block_id, "text": text})
                self.telemetry.inc("aux_emitted")
                continue
            other.append({"aux_subtype": subtype, "text": text})
            self.telemetry.inc("aux_emitted")
        return (
            {"sidecars": sidecars, "footnotes": footnotes, "other": other},
            legacy_sidecars,
            notes,
        )
