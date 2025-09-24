"""Textbook-safe threading of fused blocks with delayed auxiliary anchoring."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import yaml

from pipeline.audit.thread_audit import run_thread_audit
from pipeline.ingest.pdf_parser import DocumentGraph
from pipeline.layout.aux_detection import AuxBlockRecord, AuxDetectionResult, detect_auxiliary_blocks
from pipeline.layout.lp_fuser import FusedBlock, FusedDocument
from pipeline.layout.signals import PageLayoutSignals
from pipeline.repair.dehyphenate import DehyphenationResult, apply_dehyphenation

logger = logging.getLogger(__name__)

try:  # Optional dependency for higher quality sentence segmentation.
    import spacy
except Exception:  # pragma: no cover - environment without spaCy
    spacy = None  # type: ignore


@dataclass(slots=True)
class ThreadingReport:
    queued_aux: int = 0
    placed_aux: int = 0
    carried_aux: int = 0
    dehyphenated_pairs: int = 0
    audit_fixes: int = 0
    delayed_aux: int = 0
    post_section_aux: int = 0
    page_end_flushes: int = 0
    section_flushes: int = 0


@dataclass(slots=True)
class _PendingAux:
    aux: FusedBlock
    record: Optional[AuxBlockRecord] = None
    section_id: Optional[str] = None
    adjacent_to_figure: bool = False


@dataclass(slots=True)
class ThreadingConfig:
    anchor_template: str = "[AUX-{page}-{index:03d}]"
    allow_adjacent_figure_aux: bool = False


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "parser.yaml"


@lru_cache(maxsize=1)
def _threading_config() -> ThreadingConfig:
    default = ThreadingConfig()
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        data = {}
    except Exception as exc:  # pragma: no cover - configuration errors are non-fatal
        logger.debug("Failed to load parser config %s: %s", _CONFIG_PATH, exc)
        data = {}
    thread_cfg = data.get("threading") if isinstance(data, dict) else None
    anchor_template = default.anchor_template
    allow_adjacent = default.allow_adjacent_figure_aux
    if isinstance(thread_cfg, dict):
        if "anchor_template" in thread_cfg:
            anchor_template = str(thread_cfg["anchor_template"])
        if "allow_adjacent_figure_aux" in thread_cfg:
            allow_adjacent = bool(thread_cfg["allow_adjacent_figure_aux"])
    return ThreadingConfig(anchor_template=anchor_template, allow_adjacent_figure_aux=allow_adjacent)


class _Sentencizer:
    def __init__(self) -> None:
        self._nlp = None
        if spacy is not None:
            try:
                self._nlp = spacy.blank("en")
                if "sentencizer" not in self._nlp.pipe_names:
                    self._nlp.add_pipe("sentencizer")
            except Exception:  # pragma: no cover - spaCy misconfigured
                self._nlp = None

    def segment(self, text: str) -> List[tuple[int, int]]:
        if not text.strip():
            return []
        if self._nlp is not None:
            doc = self._nlp(text)
            return [(sent.start_char, sent.end_char) for sent in doc.sents if sent.text.strip()]
        matches = list(re.finditer(r"[^.!?]+[.!?]?", text, re.MULTILINE))
        spans: List[tuple[int, int]] = []
        for match in matches:
            span_text = match.group().strip()
            if not span_text:
                continue
            spans.append(match.span())
        return spans


@dataclass(slots=True)
class SectionState:
    section_id: str
    level: int
    queue: List[_PendingAux] = field(default_factory=list)
    last_block: Optional[FusedBlock] = None
    heading_block_id: Optional[str] = None
    lead_block_id: Optional[str] = None
    paragraphs_seen: int = 0


def _estimate_body_font(document: DocumentGraph) -> float:
    fonts: List[float] = []
    for block in document.iter_blocks():
        if not block.is_text or block.metadata.get("suppressed"):
            continue
        if block.avg_font_size > 0:
            fonts.append(block.avg_font_size)
    if not fonts:
        return 11.0
    fonts.sort()
    return fonts[len(fonts) // 2]


def _is_sealed(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped[-1] in {".", "?", "!", "\u201d"}:
        return True
    if stripped.endswith("--"):
        return False
    if stripped.endswith("-"):
        return False
    return False


class Threader:
    def __init__(self, *, sentencizer: Optional[_Sentencizer] = None) -> None:
        self.sentencizer = sentencizer or _Sentencizer()
        self._anchor_counter = 0
        self.config = _threading_config()

    def _next_anchor_marker(self, page_number: int) -> str:
        self._anchor_counter += 1
        template = self.config.anchor_template
        try:
            return template.format(page=page_number, index=self._anchor_counter)
        except Exception:  # pragma: no cover - guard against malformed templates
            return f"[AUX-{page_number}-{self._anchor_counter:03d}]"

    def _append_anchor(self, block: FusedBlock, marker: str) -> None:
        text = block.text.rstrip()
        if text and text[-1] not in {".", "?", "!", "\u201d"}:
            text = text + "."
        block.text = (text + " " + marker).strip()
        anchors = block.metadata.get("anchors")
        if not isinstance(anchors, list):
            anchors = []
        if marker not in anchors:
            anchors.append(marker)
        block.metadata["anchors"] = anchors
        block.metadata["has_anchor_refs"] = True

    def _flush_queue_to_block(
        self,
        block: Optional[FusedBlock],
        queue: List[_PendingAux],
        section_id: Optional[str],
    ) -> int:
        if block is None or not queue:
            return 0
        placed = 0
        anchors = block.metadata.get("anchors")
        if not isinstance(anchors, list):
            anchors = []
        for pending in queue:
            marker = pending.aux.anchor or self._next_anchor_marker(block.page_number)
            if marker not in block.text:
                self._append_anchor(block, marker)
            if marker not in anchors:
                anchors.append(marker)
            block.metadata["anchors"] = anchors
            block.metadata["has_anchor_refs"] = True
            pending.aux.anchor = marker
            owner_section = section_id or block.metadata.get("section_id")
            pending.aux.metadata["anchored_to"] = block.block_id
            pending.aux.metadata["section_id"] = owner_section
            pending.aux.metadata["owner_section_id"] = owner_section
            pending.aux.metadata["post_section_stream"] = True
            if pending.record and pending.record.references:
                pending.aux.metadata["references"] = pending.record.references
            placed += 1
        queue.clear()
        return placed

    def _flush_section_state(
        self,
        state: SectionState,
        fallback: Optional[FusedBlock],
        report: ThreadingReport,
        parent: Optional[SectionState],
    ) -> None:
        if not state.queue:
            return
        target = state.last_block or fallback
        if target is None and parent is not None:
            parent.queue.extend(state.queue)
            report.carried_aux += len(state.queue)
            state.queue.clear()
            return
        if target is None:
            report.carried_aux += len(state.queue)
            state.queue.clear()
            return
        placed = self._flush_queue_to_block(target, state.queue, state.section_id)
        if placed:
            report.placed_aux += placed
            report.post_section_aux += placed
            report.section_flushes += 1

    def _close_sections(
        self,
        stack: List[SectionState],
        new_level: int,
        report: ThreadingReport,
        fallback: Optional[FusedBlock],
    ) -> None:
        while len(stack) > 1 and stack[-1].level >= new_level:
            state = stack.pop()
            parent = stack[-1] if stack else None
            parent_block = parent.last_block if parent else None
            fallback_block = state.last_block or fallback or parent_block
            self._flush_section_state(state, fallback_block, report, parent)

    def _recompute_offsets(self, document: FusedDocument) -> None:
        offset = 0
        for page in document.pages:
            for block in page.main_flow:
                block.char_start = offset
                block.char_end = block.char_start + len(block.text)
                offset = block.char_end + 1 if block.text else block.char_end
            for aux in page.auxiliaries:
                aux.char_start = offset
                aux.char_end = aux.char_start + len(aux.text)
                offset = aux.char_end + 1 if aux.text else aux.char_end

    def _aux_records(
        self,
        document: DocumentGraph,
        signals: Mapping[int, PageLayoutSignals],
        body_font: float,
    ) -> Dict[int, AuxDetectionResult]:
        per_page: Dict[int, AuxDetectionResult] = {}
        for page in document.pages:
            signal = signals.get(page.page_number)
            dominant = body_font
            columns: Mapping[str, int] | None = None
            if signal is not None:
                dominant = signal.extras.dominant_font_size or dominant
                columns = signal.extras.column_assignments
            section_map = {
                block.block_id: str(block.metadata.get("section_id"))
                for block in page.blocks
                if block.metadata.get("section_id") is not None
            }
            figure_regions = [
                block.bbox
                for block in page.blocks
                if block.block_type in {"image", "draw"} or block.metadata.get("figure_candidate")
            ]
            per_page[page.page_number] = detect_auxiliary_blocks(
                page,
                body_font_size=dominant,
                column_assignments=columns,
                figure_regions=figure_regions,
                section_map=section_map,
            )
        return per_page

    def thread_document(
        self,
        document: DocumentGraph,
        fused: FusedDocument,
        signals: Sequence[PageLayoutSignals],
    ) -> tuple[FusedDocument, ThreadingReport]:
        page_signals = {signal.page_number: signal for signal in signals}
        report = ThreadingReport()
        if not fused.pages:
            return fused, report

        body_font = _estimate_body_font(document)
        detections = self._aux_records(document, page_signals, body_font)

        pdf_lookup = {block.block_id: block for block in document.iter_blocks()}

        section_states: Dict[str, SectionState] = {}
        root_state = SectionState(section_id="0", level=0)
        section_states[root_state.section_id] = root_state
        section_stack: List[SectionState] = [root_state]

        last_block: Optional[FusedBlock] = None
        carry_open = False

        for page in fused.pages:
            detection = detections.get(page.page_number)
            extras = page_signals.get(page.page_number)
            page_body_font = extras.extras.dominant_font_size if extras else body_font

            if last_block and page.main_flow and carry_open:
                change = apply_dehyphenation(last_block, page.main_flow[0])
                if isinstance(change, DehyphenationResult):
                    report.dehyphenated_pairs += 1
                    last_block = page.main_flow[0]

            target_map: Dict[str, List[_PendingAux]] = defaultdict(list)

            if detection:
                for aux in page.auxiliaries:
                    record = detection.blocks.get(aux.block_id)
                    if record:
                        aux.metadata.setdefault("aux_detect_confidence", record.confidence)
                        aux.metadata.setdefault("aux_reasons", record.reasons)
                        if record.figure_bbox:
                            aux.metadata["caption_target_bbox"] = record.figure_bbox
                        if record.is_footnote:
                            aux.metadata["is_footnote"] = True
                        if record.category:
                            aux.aux_category = aux.aux_category or record.category
                    pending = _PendingAux(aux=aux, record=record)
                    owner_section = (
                        (record.owner_section_id if record else None)
                        or str(aux.metadata.get("owner_section_id") or "")
                        or section_stack[-1].section_id
                    )
                    pending.section_id = owner_section
                    pending.adjacent_to_figure = bool(record.adjacent_to_figure if record else False)
                    aux.metadata["owner_section_id"] = owner_section
                    state = section_states.get(owner_section)
                    if state is None:
                        state = SectionState(
                            section_id=owner_section,
                            level=section_stack[-1].level if section_stack else 0,
                        )
                        section_states[owner_section] = state
                    state.queue.append(pending)
                    report.queued_aux += 1
                    report.delayed_aux += 1
                    anchor_target = aux.metadata.get("anchored_to")
                    if anchor_target:
                        target_map[anchor_target].append(pending)
            else:
                for aux in page.auxiliaries:
                    owner_section = str(aux.metadata.get("owner_section_id") or section_stack[-1].section_id)
                    pending = _PendingAux(aux=aux, record=None, section_id=owner_section)
                    state = section_states.get(owner_section)
                    if state is None:
                        state = SectionState(
                            section_id=owner_section,
                            level=section_stack[-1].level if section_stack else 0,
                        )
                        section_states[owner_section] = state
                    state.queue.append(pending)
                    report.queued_aux += 1
                    report.delayed_aux += 1
                    anchor_target = aux.metadata.get("anchored_to")
                    if anchor_target:
                        target_map[anchor_target].append(pending)

            for block in page.main_flow:
                source = pdf_lookup.get(block.block_id)
                if source:
                    meta_section_id = str(source.metadata.get("section_id") or section_stack[-1].section_id)
                    meta_level = int(source.metadata.get("section_level") or 0)
                    is_heading = bool(source.metadata.get("is_heading"))
                    if "section_lead" in source.metadata:
                        block.metadata["section_lead"] = bool(source.metadata["section_lead"])
                    if "keep_with_next" in source.metadata:
                        block.metadata["keep_with_next"] = bool(source.metadata["keep_with_next"])
                    if "paragraph_id" in source.metadata:
                        block.metadata["paragraph_id"] = source.metadata["paragraph_id"]
                else:
                    meta_section_id = section_stack[-1].section_id
                    meta_level = section_stack[-1].level
                    is_heading = False

                block.metadata["section_id"] = meta_section_id
                block.metadata["section_level"] = meta_level
                block.metadata["is_heading"] = is_heading

                if is_heading and meta_level:
                    self._close_sections(section_stack, meta_level, report, last_block)
                    state = section_states.get(meta_section_id)
                    if state is None:
                        state = SectionState(section_id=meta_section_id, level=meta_level)
                        section_states[meta_section_id] = state
                    else:
                        state.level = meta_level
                        state.last_block = None
                    state.heading_block_id = block.block_id
                    state.lead_block_id = None
                    state.paragraphs_seen = 0
                    section_stack.append(state)
                    current_state = state
                else:
                    current_state = section_stack[-1]
                    if current_state.section_id != meta_section_id:
                        existing = None
                        for candidate in reversed(section_stack):
                            if candidate.section_id == meta_section_id:
                                existing = candidate
                                break
                        if existing is not None:
                            while section_stack[-1] is not existing:
                                closing = section_stack.pop()
                                parent = section_stack[-1] if section_stack else None
                                fallback_block = closing.last_block or (parent.last_block if parent else last_block)
                                self._flush_section_state(closing, fallback_block, report, parent)
                            current_state = section_stack[-1]
                        else:
                            state = section_states.get(meta_section_id)
                            if state is None:
                                state = SectionState(
                                    section_id=meta_section_id,
                                    level=current_state.level,
                                )
                                section_states[meta_section_id] = state
                            section_stack.append(state)
                            current_state = state
                    current_state.level = max(current_state.level, meta_level)
                    if source and source.metadata.get("section_lead"):
                        current_state.lead_block_id = block.block_id
                    if not is_heading:
                        current_state.last_block = block
                        current_state.paragraphs_seen += 1

                if block.block_id in target_map:
                    for pending in target_map[block.block_id]:
                        marker = pending.aux.anchor
                        if marker and marker in block.text:
                            block.text = block.text.replace(marker, "").strip()
                            anchors = block.metadata.get("anchors")
                            if isinstance(anchors, list) and marker in anchors:
                                anchors.remove(marker)

                sealed = _is_sealed(block.text)
                block.metadata["paragraph_sealed"] = sealed
                block.metadata["paragraph_open"] = not sealed

                if (
                    self.config.allow_adjacent_figure_aux
                    and block.block_id in target_map
                    and sealed
                ):
                    state = section_states.get(current_state.section_id)
                    if state:
                        flush_candidates = [
                            pending
                            for pending in list(state.queue)
                            if pending.adjacent_to_figure and pending in target_map[block.block_id]
                        ]
                        if flush_candidates:
                            for pending in flush_candidates:
                                state.queue.remove(pending)
                            placed = self._flush_queue_to_block(block, flush_candidates, current_state.section_id)
                            if placed:
                                report.placed_aux += placed
                                report.post_section_aux += placed
                                report.section_flushes += 1

                carry_open = not sealed
                last_block = block

        self._close_sections(section_stack, 1, report, last_block)
        root_state = section_stack[0]
        fallback_block = last_block or root_state.last_block
        if root_state.queue:
            placed = self._flush_queue_to_block(fallback_block, root_state.queue, root_state.section_id)
            if placed:
                report.placed_aux += placed
                report.post_section_aux += placed
                report.section_flushes += 1
            else:
                report.carried_aux += len(root_state.queue)
                root_state.queue.clear()

        findings = run_thread_audit(fused)
        report.audit_fixes = len(findings)
        if findings:
            logger.debug("Thread audit applied %d fixes", len(findings))

        self._recompute_offsets(fused)
        return fused, report


__all__ = ["Threader", "ThreadingReport"]
