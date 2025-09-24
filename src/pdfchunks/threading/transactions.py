"""Section transaction helpers to enforce AUX isolation."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from ..parsing.ownership import SectionAssignment


@dataclass(slots=True)
class SectionTransaction:
    """Tracks state for a section while threading main paragraphs."""

    section_seq: int
    heading_seen: bool = False
    lead_paragraph_seen: bool = False
    para_seq: int = 0
    aux_queue: Deque[SectionAssignment] = field(default_factory=deque)
    sealed: bool = False

    def register_main(self) -> None:
        if not self.lead_paragraph_seen:
            self.lead_paragraph_seen = True
        self.para_seq += 1

    def enqueue_aux(self, assignment: SectionAssignment) -> None:
        if not self.lead_paragraph_seen:
            raise ValueError("AUX cannot appear before the lead paragraph in a section")
        self.aux_queue.append(assignment)

    def seal(self) -> None:
        self.sealed = True


class TransactionManager:
    """Manage section-scoped AUX queues and guardrails."""

    def __init__(self):
        self._sections: Dict[int, SectionTransaction] = {}
        self._current_section: Optional[int] = None
        self.delayed_aux_flush: Dict[int, int] = defaultdict(int)
        self._last_section_with_lead: Optional[int] = None

    def begin(self, section_seq: int, heading: Optional[SectionAssignment] = None) -> SectionTransaction:
        section = self._sections.get(section_seq)
        if section is None:
            section = SectionTransaction(section_seq=section_seq, heading_seen=bool(heading))
            self._sections[section_seq] = section
        else:
            section.heading_seen = section.heading_seen or bool(heading)
        if self._current_section is not None and self._current_section != section_seq:
            self.seal(self._current_section)
        self._current_section = section_seq
        return section

    def register_main(self, section_seq: int) -> SectionTransaction:
        section = self._sections.setdefault(section_seq, SectionTransaction(section_seq=section_seq))
        section.register_main()
        self._last_section_with_lead = section_seq
        return section

    def enqueue_aux(self, assignment: SectionAssignment) -> None:
        owner_seq = assignment.owner_section_seq
        section = self._sections.setdefault(owner_seq, SectionTransaction(section_seq=owner_seq))

        if not section.lead_paragraph_seen:
            if owner_seq == 0:
                section.lead_paragraph_seen = True
            else:
                fallback_seq = self._last_section_with_lead
                if fallback_seq is None or fallback_seq == owner_seq:
                    raise ValueError("AUX cannot appear before the lead paragraph in a section")
                section = self._sections.setdefault(
                    fallback_seq, SectionTransaction(section_seq=fallback_seq)
                )
                if not section.lead_paragraph_seen:
                    section.lead_paragraph_seen = True
                assignment = SectionAssignment(
                    label=assignment.label,
                    section_seq=assignment.section_seq,
                    owner_section_seq=fallback_seq,
                )
                owner_seq = fallback_seq

        section.enqueue_aux(assignment)
        self.delayed_aux_flush[owner_seq] = len(section.aux_queue)

    def seal(self, section_seq: int) -> None:
        section = self._sections.get(section_seq)
        if not section:
            return
        section.seal()
        self._current_section = None

    def drain_aux(self, section_seq: int) -> List[SectionAssignment]:
        section = self._sections.get(section_seq)
        if not section or not section.sealed:
            return []
        drained = list(section.aux_queue)
        section.aux_queue.clear()
        self.delayed_aux_flush[section_seq] = 0
        return drained

    def flush_all(self) -> List[SectionAssignment]:
        drained: List[SectionAssignment] = []
        for section_seq, section in sorted(self._sections.items()):
            if not section.sealed:
                section.seal()
            drained.extend(self.drain_aux(section_seq))
        return drained

    def section_state(self, section_seq: int) -> SectionTransaction:
        return self._sections.setdefault(section_seq, SectionTransaction(section_seq=section_seq))

