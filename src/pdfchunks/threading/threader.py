"""Thread MAIN paragraphs while isolating AUX payloads."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from ..config import ThreadingConfig
from ..parsing.ownership import SectionAssignment
from .transactions import TransactionManager

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ThreadUnit:
    """Serialized unit emitted by the threader."""

    doc_id: str
    section_seq: int
    para_seq: int
    sent_seq: int
    emit_phase: int
    text: str
    role: str
    subtype: str | None = None
    block_ids: List[str] = field(default_factory=list)

    @property
    def order_key(self) -> tuple[str, int, int, int, int]:
        return (self.doc_id, self.section_seq, self.para_seq, self.sent_seq, self.emit_phase)


@dataclass(slots=True)
class ParagraphBuilder:
    """Accumulates text across blocks until sealing conditions are met."""

    parts: List[str] = field(default_factory=list)
    block_ids: List[str] = field(default_factory=list)

    def add(self, text: str, block_id: str) -> None:
        self.parts.append(text)
        self.block_ids.append(block_id)

    def clear(self) -> None:
        self.parts.clear()
        self.block_ids.clear()

    def text(self) -> str:
        return " ".join(part for part in self.parts if part).strip()

    def empty(self) -> bool:
        return not self.parts


class Threader:
    """Thread MAIN text while deferring AUX emission."""

    def __init__(self, config: ThreadingConfig):
        self.config = config
        self._sentence_pattern = re.compile(config.sentence_regex)
        self._paragraph_end_pattern = re.compile(config.paragraph_end_regex)

    def thread(self, assignments: Iterable[SectionAssignment], doc_id: str) -> "ThreadResult":
        tm = TransactionManager()
        paragraph_builders: Dict[int, ParagraphBuilder] = defaultdict(ParagraphBuilder)
        aux_counters: Dict[int, int] = defaultdict(int)
        units: List[ThreadUnit] = []
        active_section: int | None = None

        def finalize_section(section_seq: int, force: bool = False) -> None:
            if section_seq is None:
                return
            builder = paragraph_builders.get(section_seq)
            if builder and (force or not builder.empty()):
                if not builder.empty():
                    LOGGER.debug("Forcing paragraph seal for section %s", section_seq)
                    _emit_paragraph(section_seq, force=True)
            tm.seal(section_seq)
            for aux_assignment in tm.drain_aux(section_seq):
                owner = aux_assignment.owner_section_seq
                state = tm.section_state(owner)
                para_seq = state.para_seq
                aux_counters[owner] += 1
                wrapped = self._wrap_aux_text(aux_assignment.label.block.text)
                units.append(
                    ThreadUnit(
                        doc_id=doc_id,
                        section_seq=owner,
                        para_seq=para_seq,
                        sent_seq=aux_counters[owner],
                        emit_phase=1,
                        text=wrapped,
                        role="AUX",
                        subtype=aux_assignment.label.subtype or "auxiliary",
                        block_ids=[aux_assignment.label.block.block_id],
                    )
                )

        def _emit_paragraph(section_seq: int, force: bool = False) -> None:
            builder = paragraph_builders[section_seq]
            if builder.empty():
                return
            text = builder.text()
            if not text:
                builder.clear()
                return
            if not force and not self._paragraph_end_pattern.search(text):
                return
            state = tm.register_main(section_seq)
            sentences = self._sentencize(text)
            for idx, sentence in enumerate(sentences):
                units.append(
                    ThreadUnit(
                        doc_id=doc_id,
                        section_seq=section_seq,
                        para_seq=state.para_seq,
                        sent_seq=idx,
                        emit_phase=0,
                        text=sentence,
                        role="MAIN",
                        block_ids=list(builder.block_ids),
                    )
                )
            builder.clear()

        for assignment in assignments:
            label = assignment.label
            section_seq = assignment.section_seq or assignment.owner_section_seq

            if label.role == "MAIN" and label.is_heading:
                if active_section is not None and section_seq != active_section:
                    finalize_section(active_section)
                tm.begin(section_seq, heading=assignment)
                active_section = section_seq
                heading_text = self._clean_main_text(label.block.text)
                if heading_text:
                    units.append(
                        ThreadUnit(
                            doc_id=doc_id,
                            section_seq=section_seq,
                            para_seq=0,
                            sent_seq=0,
                            emit_phase=0,
                            text=heading_text,
                            role="MAIN",
                            subtype="heading",
                            block_ids=[label.block.block_id],
                        )
                    )
                continue

            if label.role == "MAIN":
                if active_section is None:
                    active_section = section_seq
                elif section_seq != active_section:
                    finalize_section(active_section)
                    active_section = section_seq
                tm.begin(section_seq)
                builder = paragraph_builders[section_seq]
                cleaned = self._clean_main_text(label.block.text)
                if cleaned:
                    builder.add(cleaned, label.block.block_id)
                _emit_paragraph(section_seq)
                continue

            # AUX flow
            tm.enqueue_aux(assignment)

        if active_section is not None:
            finalize_section(active_section, force=True)

        # Flush any remaining sections that never became active but accumulated AUX.
        for section_seq in list(tm.delayed_aux_flush.keys()):
            finalize_section(section_seq, force=True)

        units.sort(key=lambda unit: unit.order_key)
        return ThreadResult(units=units, delayed_aux_counts=dict(tm.delayed_aux_flush))

    def _clean_main_text(self, text: str) -> str:
        text = text.replace("\u00ad", "")
        if self.config.dehyphenate:
            text = text.replace("-\n", "")
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _sentencize(self, text: str) -> List[str]:
        if not text:
            return []
        matches = list(self._sentence_pattern.finditer(text))
        sentences: List[str] = []
        start = 0
        for match in matches:
            end = match.end()
            piece = text[start:end].strip()
            if piece:
                sentences.append(piece)
            start = end
        tail = text[start:].strip()
        if tail:
            sentences.append(tail)
        if not sentences:
            sentences = [text.strip()]
        return sentences

    def _wrap_aux_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text.strip())
        return f"<aux>{cleaned}</aux>"


@dataclass(slots=True)
class ThreadResult:
    """Return value from :class:`Threader` executions."""

    units: List[ThreadUnit]
    delayed_aux_counts: Dict[int, int]

