from __future__ import annotations

from pdfchunks.parsing.block_extractor import Block
from pdfchunks.parsing.classifier import BlockLabel
from pdfchunks.parsing.ownership import SectionAssignment, assign_aux_ownership


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


def test_aux_before_lead_is_reassigned_to_prior_section():
    heading_one = SectionAssignment(
        label=BlockLabel(
            block=make_block("h1", "Heading 1"),
            role="MAIN",
            is_heading=True,
            section_level=1,
        ),
        section_seq=1,
        owner_section_seq=1,
    )
    main_one = SectionAssignment(
        label=BlockLabel(block=make_block("m1", "Paragraph one."), role="MAIN"),
        section_seq=1,
        owner_section_seq=1,
    )
    heading_two = SectionAssignment(
        label=BlockLabel(
            block=make_block("h2", "Heading 2"),
            role="MAIN",
            is_heading=True,
            section_level=1,
        ),
        section_seq=2,
        owner_section_seq=2,
    )
    pre_lead_aux = SectionAssignment(
        label=BlockLabel(block=make_block("a", "Caption text"), role="AUX", subtype="caption"),
        section_seq=2,
        owner_section_seq=2,
    )

    final = assign_aux_ownership([heading_one, main_one, heading_two, pre_lead_aux])

    reassigned = final[-1]
    assert reassigned.owner_section_seq == 1

