from __future__ import annotations

import pytest

from pdfchunks.parsing.block_extractor import Block
from pdfchunks.parsing.classifier import BlockLabel
from pdfchunks.parsing.ownership import SectionAssignment
from pdfchunks.threading.transactions import TransactionManager


def make_block(block_id: str, text: str) -> Block:
    return Block(
        block_id=block_id,
        text=text,
        font_name="Times",
        font_size=10.0,
        line_height=12.0,
        bbox=(50.0, 100.0, 550.0, 150.0),
        page_num=0,
        page_width=600.0,
        page_height=800.0,
    )


def test_aux_before_lead_paragraph_raises():
    tm = TransactionManager()
    heading = SectionAssignment(
        label=BlockLabel(block=make_block("h", "Heading"), role="MAIN", is_heading=True, section_level=1),
        section_seq=1,
        owner_section_seq=1,
    )
    tm.begin(1, heading=heading)
    aux_assignment = SectionAssignment(
        label=BlockLabel(block=make_block("a", "<aux>oops</aux>"), role="AUX", subtype="caption"),
        section_seq=1,
        owner_section_seq=1,
    )
    with pytest.raises(ValueError):
        tm.enqueue_aux(aux_assignment)


def test_enqueue_and_drain_after_seal():
    tm = TransactionManager()
    heading = SectionAssignment(
        label=BlockLabel(block=make_block("h", "Heading"), role="MAIN", is_heading=True, section_level=1),
        section_seq=2,
        owner_section_seq=2,
    )
    tm.begin(2, heading=heading)
    tm.register_main(2)
    aux_assignment = SectionAssignment(
        label=BlockLabel(block=make_block("a", "Aux text"), role="AUX", subtype="caption"),
        section_seq=2,
        owner_section_seq=2,
    )
    tm.enqueue_aux(aux_assignment)
    tm.seal(2)
    drained = tm.drain_aux(2)
    assert drained == [aux_assignment]

