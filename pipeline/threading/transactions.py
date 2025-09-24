"""Section-scoped auxiliary transaction manager for section-first serialization."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Mapping, Sequence, Tuple

from pipeline.layout.lp_fuser import FusedBlock

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SectionTransaction:
    """Track per-section sequencing and delayed auxiliaries."""

    section_seq: int
    section_id: str
    level: int
    aux_queue: Deque[FusedBlock] = field(default_factory=deque)
    last_para_seq: int = 0
    next_para_seq: int = 1
    next_sent_seq: int = 1
    sealed: bool = False
    flush_log: List[int] = field(default_factory=list)

    def record_paragraph(self, para_seq: int) -> None:
        if para_seq <= 0:
            return
        if para_seq > self.last_para_seq:
            self.last_para_seq = para_seq
            self.next_para_seq = para_seq + 1
        else:
            self.next_para_seq = max(self.next_para_seq, para_seq + 1)

    def next_sentence(self) -> int:
        value = self.next_sent_seq
        self.next_sent_seq += 1
        return value

    def claim_aux_para_seq(self) -> int:
        value = max(self.next_para_seq, self.last_para_seq + 1)
        self.last_para_seq = value
        self.next_para_seq = value + 1
        return value

    def ensure_paragraph_seq(self, para_seq: int, *, is_heading: bool = False) -> int:
        """Normalise ``para_seq`` so emitted units remain monotonic."""

        if para_seq > 0:
            self.record_paragraph(para_seq)
            return para_seq

        if is_heading and self.last_para_seq == 0:
            self.next_para_seq = max(self.next_para_seq, 1)
            return 0

        value = self.claim_aux_para_seq()
        logger.debug(
            "allocated fallback para_seq for section %s: %s",
            self.section_seq,
            value,
        )
        return value


class SectionTransactions:
    """Manage BEGIN/SEAL/FLUSH operations for section-scoped auxiliaries."""

    def __init__(
        self,
        doc_id: str,
        aux_by_section: Mapping[int, Sequence[FusedBlock]] | None = None,
    ) -> None:
        self.doc_id = doc_id
        self._transactions: Dict[int, SectionTransaction] = {}
        self._stack: List[SectionTransaction] = []
        self._aux_lookup: Dict[int, Deque[FusedBlock]] = {
            section: deque(list(blocks)) for section, blocks in (aux_by_section or {}).items()
        }
        self.delayed_aux_count = sum(len(blocks) for blocks in (aux_by_section or {}).values())
        root = self._ensure_transaction(section_seq=0, section_id="0", level=0)
        if not self._stack:
            self._stack.append(root)
        self.flush_events: List[Tuple[int, int]] = []

    def _ensure_transaction(self, section_seq: int, section_id: str, level: int) -> SectionTransaction:
        txn = self._transactions.get(section_seq)
        if txn is None:
            queue = self._aux_lookup.get(section_seq, deque())
            txn = SectionTransaction(
                section_seq=section_seq,
                section_id=section_id,
                level=level,
                aux_queue=queue,
            )
            self._transactions[section_seq] = txn
        else:
            txn.section_id = section_id
            txn.level = level
        return txn

    def _seal_section(self, txn: SectionTransaction) -> List[Tuple[SectionTransaction, FusedBlock]]:
        if txn.sealed:
            return []
        txn.sealed = True
        flushed: List[Tuple[SectionTransaction, FusedBlock]] = []
        if txn.aux_queue:
            size = len(txn.aux_queue)
            txn.flush_log.append(size)
            self.flush_events.append((txn.section_seq, size))
        while txn.aux_queue:
            block = txn.aux_queue.popleft()
            flushed.append((txn, block))
        return flushed

    def _seal_to_level(self, level: int) -> List[Tuple[SectionTransaction, FusedBlock]]:
        flushed: List[Tuple[SectionTransaction, FusedBlock]] = []
        while len(self._stack) > 1 and self._stack[-1].level >= level:
            txn = self._stack.pop()
            flushed.extend(self._seal_section(txn))
        return flushed

    def _pop_until(
        self,
        section_seq: int,
        level: int,
    ) -> List[Tuple[SectionTransaction, FusedBlock]]:
        flushed: List[Tuple[SectionTransaction, FusedBlock]] = []
        while len(self._stack) > 1 and self._stack[-1].section_seq != section_seq:
            top = self._stack[-1]
            if top.level >= level:
                txn = self._stack.pop()
                flushed.extend(self._seal_section(txn))
            else:
                break
        return flushed

    def observe(
        self,
        *,
        section_seq: int,
        section_id: str,
        level: int,
        is_heading: bool,
    ) -> Tuple[SectionTransaction, List[Tuple[SectionTransaction, FusedBlock]]]:
        """Observe a block belonging to ``section_seq`` and manage stack transitions."""

        if section_seq < 0:
            section_seq = 0
        flushed: List[Tuple[SectionTransaction, FusedBlock]] = []
        if is_heading:
            flushed.extend(self._seal_to_level(level))
        else:
            flushed.extend(self._pop_until(section_seq, level))

        txn = self._ensure_transaction(section_seq, section_id, level)
        if not self._stack or self._stack[-1].section_seq != section_seq:
            self._stack.append(txn)
        return txn, flushed

    def attach_auxiliary(self, block: FusedBlock, owner_section_seq: int | None) -> None:
        seq = owner_section_seq if owner_section_seq is not None else 0
        txn = self._ensure_transaction(seq, section_id=str(block.metadata.get("owner_section_id") or "0"), level=0)
        txn.aux_queue.append(block)
        self.delayed_aux_count += 1

    def finalize(self) -> List[Tuple[SectionTransaction, FusedBlock]]:
        flushed: List[Tuple[SectionTransaction, FusedBlock]] = []
        for section_seq, queue in list(self._aux_lookup.items()):
            if section_seq not in self._transactions and queue:
                block = queue[0]
                section_id = str(block.metadata.get("owner_section_id") or section_seq)
                self._ensure_transaction(section_seq, section_id, level=0)
        while self._stack:
            txn = self._stack.pop()
            flushed.extend(self._seal_section(txn))
        return flushed


__all__ = ["SectionTransaction", "SectionTransactions"]

