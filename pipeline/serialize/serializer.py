"""Serialize fused documents into section-first ordered units."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

from pipeline.layout.lp_fuser import FusedBlock, FusedDocument
from pipeline.threading.transactions import SectionTransaction, SectionTransactions

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SerializedUnit:
    """Smallest emission unit (sentence or auxiliary block)."""

    doc_id: str
    block: FusedBlock
    text: str
    section_seq: int
    section_id: str
    para_seq: int
    sent_seq: int
    emit_phase: int
    char_start: int
    char_end: int
    page_number: int
    order_key: Tuple[str, int, int, int, int]
    unit_kind: str
    is_paragraph_start: bool
    owner_section_seq: int | None
    paragraph_id: str
    references: List[str]
    segment_type: str


@dataclass(slots=True)
class SerializationResult:
    units: List[SerializedUnit]
    delayed_aux_count: int
    flush_events: List[Tuple[int, int]]


_SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]?", re.MULTILINE)


def _split_sentences(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    matches = list(_SENTENCE_PATTERN.finditer(text))
    spans: List[Tuple[str, Tuple[int, int]]] = []
    for match in matches:
        raw = match.group()
        if not raw.strip():
            continue
        leading = len(raw) - len(raw.lstrip())
        trailing_len = len(raw.rstrip())
        start = match.start() + leading
        end = match.start() + trailing_len
        if start >= end:
            continue
        spans.append((raw.strip(), (start, end)))
    if not spans and text.strip():
        stripped = text.strip()
        start = text.find(stripped)
        spans.append((stripped, (start, start + len(stripped))))
    return spans


def _is_procedure_block(block: FusedBlock) -> bool:
    lines = [line.strip() for line in block.text.splitlines() if line.strip()]
    if not lines:
        return False
    procedure_lines = sum(1 for line in lines if re.match(r"^(\d+\.|[A-Z]\.|[-*•])\s", line))
    return procedure_lines >= max(2, len(lines) // 2)


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "parser.yaml"


@lru_cache(maxsize=1)
def _serializer_config() -> Dict[str, object]:
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # pragma: no cover - configuration errors are non-fatal
        logger.debug("Failed to load serializer config %s: %s", _CONFIG_PATH, exc)
        return {}
    section = data.get("serializer") if isinstance(data, dict) else None
    if isinstance(section, dict):
        return section
    return {}


class Serializer:
    """Serializer producing section-first ordered units from a fused document."""

    def __init__(self, log_preview: int | None = None) -> None:
        config = _serializer_config()
        if log_preview is None:
            try:
                log_preview = int(config.get("log_preview", 12))
            except (TypeError, ValueError):
                log_preview = 12
        self.log_preview = max(0, int(log_preview))

    def _sentence_units(
        self,
        txn: SectionTransaction,
        block: FusedBlock,
        *,
        doc_id: str,
    ) -> List[SerializedUnit]:
        text = block.text or ""
        if not text.strip():
            return []
        section_id = str(block.metadata.get("section_id") or txn.section_id)
        try:
            para_seq = int(block.metadata.get("para_seq") or 0)
        except (TypeError, ValueError):
            para_seq = 0
        txn.record_paragraph(para_seq)
        sentences = _split_sentences(text)
        if not sentences:
            return []
        segment_type = "procedure" if _is_procedure_block(block) else "prose"
        paragraph_id = str(block.metadata.get("paragraph_id") or block.block_id)
        references = block.metadata.get("references")
        if not isinstance(references, list):
            references = []
        units: List[SerializedUnit] = []
        is_start = True
        for sentence_text, (start, end) in sentences:
            sent_seq = txn.next_sentence()
            abs_start = block.char_start + start
            abs_end = block.char_start + end
            unit = SerializedUnit(
                doc_id=doc_id,
                block=block,
                text=sentence_text.strip(),
                section_seq=txn.section_seq,
                section_id=section_id,
                para_seq=para_seq,
                sent_seq=sent_seq,
                emit_phase=0,
                char_start=abs_start,
                char_end=abs_end,
                page_number=block.page_number,
                order_key=(doc_id, txn.section_seq, para_seq, sent_seq, 0),
                unit_kind="sentence",
                is_paragraph_start=is_start,
                owner_section_seq=None,
                paragraph_id=paragraph_id,
                references=list(references),
                segment_type=segment_type,
            )
            units.append(unit)
            is_start = False
        return units

    def _aux_units(
        self,
        txn: SectionTransaction,
        blocks: Sequence[FusedBlock],
        *,
        doc_id: str,
    ) -> List[SerializedUnit]:
        units: List[SerializedUnit] = []
        for block in blocks:
            text = block.text.strip()
            if not text:
                continue
            para_seq = txn.claim_aux_para_seq()
            sent_seq = txn.next_sentence()
            references = block.metadata.get("references")
            if not isinstance(references, list):
                references = []
            owner_seq = txn.section_seq
            unit = SerializedUnit(
                doc_id=doc_id,
                block=block,
                text=text,
                section_seq=txn.section_seq,
                section_id=str(block.metadata.get("owner_section_id") or txn.section_id),
                para_seq=para_seq,
                sent_seq=sent_seq,
                emit_phase=1,
                char_start=block.char_start,
                char_end=block.char_end,
                page_number=block.page_number,
                order_key=(doc_id, txn.section_seq, para_seq, sent_seq, 1),
                unit_kind=block.aux_category or block.region_label,
                is_paragraph_start=True,
                owner_section_seq=owner_seq,
                paragraph_id=str(block.metadata.get("paragraph_id") or block.block_id),
                references=list(references),
                segment_type="aux",
            )
            units.append(unit)
        return units

    def serialize(self, document: FusedDocument) -> SerializationResult:
        aux_by_section: Dict[int, List[FusedBlock]] = {}
        for page in document.pages:
            for aux in page.auxiliaries:
                owner_seq = aux.metadata.get("owner_section_seq")
                if owner_seq is None:
                    owner_seq = aux.metadata.get("section_seq")
                try:
                    owner_int = int(owner_seq) if owner_seq is not None else 0
                except (TypeError, ValueError):
                    owner_int = 0
                aux_by_section.setdefault(owner_int, []).append(aux)

        manager = SectionTransactions(document.doc_id, aux_by_section)
        units: List[SerializedUnit] = []

        for page in document.pages:
            for block in page.main_flow:
                try:
                    section_seq = int(block.metadata.get("section_seq") or 0)
                except (TypeError, ValueError):
                    section_seq = 0
                section_id = str(block.metadata.get("section_id") or str(section_seq))
                try:
                    level = int(block.metadata.get("section_level") or 0)
                except (TypeError, ValueError):
                    level = 0
                is_heading = bool(block.metadata.get("is_heading"))
                txn, flushed = manager.observe(
                    section_seq=section_seq,
                    section_id=section_id,
                    level=level,
                    is_heading=is_heading,
                )
                if flushed:
                    grouped: Dict[int, Tuple[SectionTransaction, List[FusedBlock]]] = {}
                    for flush_txn, flush_block in flushed:
                        key = id(flush_txn)
                        if key not in grouped:
                            grouped[key] = (flush_txn, [])
                        grouped[key][1].append(flush_block)
                    for flush_txn, blocks in grouped.values():
                        units.extend(self._aux_units(flush_txn, blocks, doc_id=document.doc_id))
                sentence_units = self._sentence_units(txn, block, doc_id=document.doc_id)
                units.extend(sentence_units)

        grouped_flush: Dict[int, Tuple[SectionTransaction, List[FusedBlock]]] = {}
        for flush_txn, flush_block in manager.finalize():
            key = id(flush_txn)
            if key not in grouped_flush:
                grouped_flush[key] = (flush_txn, [])
            grouped_flush[key][1].append(flush_block)
        for flush_txn, blocks in grouped_flush.values():
            units.extend(self._aux_units(flush_txn, blocks, doc_id=document.doc_id))

        if self.log_preview > 0 and units:
            preview = ", ".join(str(unit.order_key) for unit in units[: self.log_preview])
            logger.debug("order preview: %s", preview)
        logger.debug("delayed_aux_count=%d", manager.delayed_aux_count)
        if manager.flush_events:
            logger.debug("section flush sizes: %s", manager.flush_events)

        return SerializationResult(
            units=units,
            delayed_aux_count=manager.delayed_aux_count,
            flush_events=list(manager.flush_events),
        )


__all__ = ["SerializationResult", "SerializedUnit", "Serializer"]

