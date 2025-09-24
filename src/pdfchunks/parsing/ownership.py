"""Ownership assignment for AUX blocks and section sequencing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from .classifier import BlockLabel


@dataclass(slots=True)
class SectionAssignment:
    """Mapping of blocks to section/owner identifiers."""

    label: BlockLabel
    section_seq: int
    owner_section_seq: int


def assign_sections(labels: Iterable[BlockLabel]) -> List[SectionAssignment]:
    """Assign sequential section identifiers to labelled blocks."""

    assignments: List[SectionAssignment] = []
    section_stack: Deque[Tuple[int, int]] = deque()  # (level, section_seq)
    current_section_seq = 0

    for label in labels:
        if label.role != "MAIN":
            assignments.append(
                SectionAssignment(label=label, section_seq=current_section_seq, owner_section_seq=current_section_seq)
            )
            continue

        if label.is_heading:
            while section_stack and section_stack[-1][0] >= label.section_level:
                section_stack.pop()
            current_section_seq = current_section_seq + 1 if assignments else 1
            section_stack.append((label.section_level, current_section_seq))
            assignments.append(
                SectionAssignment(label=label, section_seq=current_section_seq, owner_section_seq=current_section_seq)
            )
        else:
            if section_stack:
                current_section_seq = section_stack[-1][1]
            assignments.append(
                SectionAssignment(label=label, section_seq=current_section_seq, owner_section_seq=current_section_seq)
            )

    return assignments


def assign_aux_ownership(assignments: Iterable[SectionAssignment]) -> List[SectionAssignment]:
    """Adjust AUX ownership using nearest preceding header metadata."""

    final: List[SectionAssignment] = []
    last_header_for_section: Dict[int, SectionAssignment] = {}
    last_section_with_main: Optional[int] = None
    sections_with_main: set[int] = set()

    for assignment in assignments:
        label = assignment.label
        if label.role == "MAIN" and label.is_heading:
            last_header_for_section[assignment.section_seq] = assignment
            final.append(assignment)
            continue

        if label.role == "MAIN":
            final.append(assignment)
            sections_with_main.add(assignment.section_seq)
            last_section_with_main = assignment.section_seq
            continue

        owner_seq = assignment.owner_section_seq
        if label.subtype == "caption" and "figure_section_seq" in label.block.metadata:
            owner_seq = int(label.block.metadata.get("figure_section_seq", owner_seq))
        elif label.subtype == "footnote" and "reference_section_seq" in label.block.metadata:
            owner_seq = int(label.block.metadata.get("reference_section_seq", owner_seq))
        elif owner_seq == 0 and last_section_with_main is not None:
            owner_seq = last_section_with_main
        elif owner_seq == 0 and last_header_for_section:
            owner_seq = max(last_header_for_section)

        if owner_seq not in sections_with_main and last_section_with_main is not None:
            owner_seq = last_section_with_main

        final.append(
            SectionAssignment(label=label, section_seq=assignment.section_seq, owner_section_seq=owner_seq)
        )

    return final

