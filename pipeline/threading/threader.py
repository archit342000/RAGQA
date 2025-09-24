"""Textbook-safe threading of fused blocks with delayed auxiliary anchoring."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

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


@dataclass(slots=True)
class _PendingAux:
    aux: FusedBlock
    record: Optional[AuxBlockRecord] = None


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


class _SectionTracker:
    def __init__(self) -> None:
        self._path: List[int] = []

    def enter(self, level: int) -> str:
        level = max(1, level)
        while len(self._path) >= level:
            self._path.pop()
        while len(self._path) < level:
            self._path.append(0)
        self._path[level - 1] += 1
        for idx in range(level, len(self._path)):
            self._path[idx] = 0
        return ".".join(str(num) for num in self._path if num > 0)

    def current(self) -> str:
        return ".".join(str(num) for num in self._path if num > 0) or "0"


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


def _heading_level(block: Optional[FusedBlock], fallback_size: float) -> int:
    if block is None or not block.text.strip():
        return 0
    ratio = block.avg_font_size / max(fallback_size, 1.0)
    if ratio >= 1.6:
        return 1
    if ratio >= 1.35:
        return 2
    if ratio >= 1.18:
        return 3
    if ratio >= 1.1 and re.match(r"^[0-9]+(\.[0-9]+)*\s", block.text.strip()):
        return 2
    return 0


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

    def _next_anchor_marker(self, page_number: int) -> str:
        self._anchor_counter += 1
        return f"[AUX-{page_number}-{self._anchor_counter:03d}]"

    def _flush_aux(self, block: FusedBlock, queue: List[_PendingAux], section_id: str) -> int:
        if not queue:
            return 0
        placed = 0
        for pending in queue:
            marker = pending.aux.anchor or self._next_anchor_marker(block.page_number)
            if marker not in block.text:
                text = block.text.rstrip()
                if text and text[-1] not in {".", "?", "!", "\u201d"}:
                    text = text + "."
                block.text = (text + " " + marker).strip()
            block.metadata.setdefault("anchors", [])
            if marker not in block.metadata["anchors"]:
                block.metadata["anchors"].append(marker)
            block.metadata["has_anchor_refs"] = True
            pending.aux.anchor = marker
            pending.aux.metadata["anchored_to"] = block.block_id
            pending.aux.metadata["section_id"] = section_id
            placed += 1
        queue.clear()
        return placed

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
            per_page[page.page_number] = detect_auxiliary_blocks(
                page,
                body_font_size=dominant,
                column_assignments=columns,
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
        tracker = _SectionTracker()

        pending_aux: List[_PendingAux] = []
        carry_open = False
        last_block: Optional[FusedBlock] = None
        protect_until: Optional[str] = None

        pdf_lookup = {block.block_id: block for block in document.iter_blocks()}

        for page in fused.pages:
            detection = detections.get(page.page_number)
            section_id = tracker.current()
            extras = page_signals.get(page.page_number)
            page_body_font = extras.extras.dominant_font_size if extras else body_font
            target_map: Dict[str, List[_PendingAux]] = {}
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
                    target_id = aux.metadata.get("anchored_to")
                    pending = _PendingAux(aux=aux, record=record)
                    if target_id:
                        target_map.setdefault(target_id, []).append(pending)
                    else:
                        pending_aux.append(pending)
                        report.queued_aux += 1
            else:
                for aux in page.auxiliaries:
                    target_id = aux.metadata.get("anchored_to")
                    pending = _PendingAux(aux=aux, record=None)
                    if target_id:
                        target_map.setdefault(target_id, []).append(pending)
                    else:
                        pending_aux.append(pending)
                        report.queued_aux += 1

            if last_block and page.main_flow and carry_open:
                change = apply_dehyphenation(last_block, page.main_flow[0])
                if isinstance(change, DehyphenationResult):
                    report.dehyphenated_pairs += 1
                    last_block = page.main_flow[0]

            protect_next = False
            for block in page.main_flow:
                source = pdf_lookup.get(block.block_id)
                heading_level = _heading_level(source, page_body_font)
                if heading_level:
                    section_id = tracker.enter(heading_level)
                    block.metadata["section_id"] = section_id
                    block.metadata["section_level"] = heading_level
                    block.metadata["is_heading"] = True
                    block.metadata.pop("section_lead", None)
                    block.metadata.pop("keep_with_next", None)
                    protect_next = True
                    carry_open = False
                else:
                    block.metadata["section_id"] = section_id or tracker.current()
                    block.metadata["section_level"] = heading_level or 0
                    block.metadata["is_heading"] = False

                new_aux = target_map.get(block.block_id, [])
                if new_aux:
                    report.queued_aux += len(new_aux)
                    for pending in new_aux:
                        if pending.aux.anchor and pending.aux.anchor in block.text:
                            block.text = block.text.replace(pending.aux.anchor, "").strip()
                            anchors = block.metadata.get("anchors")
                            if isinstance(anchors, list) and pending.aux.anchor in anchors:
                                anchors.remove(pending.aux.anchor)
                        if not pending.aux.anchor:
                            pending.aux.anchor = self._next_anchor_marker(page.page_number)
                        pending_aux.append(pending)

                if protect_next and not block.metadata.get("is_heading"):
                    block.metadata["section_lead"] = True
                    block.metadata["keep_with_next"] = True
                    protect_until = block.block_id
                    protect_next = False
                elif not block.metadata.get("is_heading"):
                    block.metadata["section_lead"] = False
                    block.metadata["keep_with_next"] = False

                sealed = _is_sealed(block.text)
                block.metadata["paragraph_sealed"] = sealed
                block.metadata["paragraph_open"] = not sealed

                if not sealed:
                    carry_open = True
                else:
                    placed = self._flush_aux(block, pending_aux, block.metadata.get("section_id", section_id))
                    if placed:
                        report.placed_aux += placed
                    if protect_until and protect_until == block.block_id:
                        protect_until = None
                    carry_open = False

                last_block = block

            if page.main_flow and pending_aux and not carry_open:
                last_main = page.main_flow[-1]
                placed = self._flush_aux(last_main, pending_aux, last_main.metadata.get("section_id", section_id))
                if placed:
                    report.placed_aux += placed
                    carry_open = False

        if pending_aux and last_block:
            report.carried_aux += len(pending_aux)
            placed = self._flush_aux(last_block, pending_aux, last_block.metadata.get("section_id", tracker.current()))
            report.placed_aux += placed
            pending_aux.clear()

        findings = run_thread_audit(fused)
        report.audit_fixes = len(findings)
        if findings:
            logger.debug("Thread audit applied %d fixes", len(findings))

        self._recompute_offsets(fused)
        return fused, report


__all__ = ["Threader", "ThreadingReport"]
