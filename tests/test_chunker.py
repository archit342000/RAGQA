from __future__ import annotations

from pipeline.chunker import chunk_blocks
from pipeline.config import PipelineConfig
from pipeline.normalize import Block


def _block(
    *,
    doc_id: str = "doc",
    block_id: str,
    page: int,
    order: int,
    type: str,
    text: str,
    heading_level: int | None = None,
    heading_path: list[str] | None = None,
    source_stage: str = "docling",
    role: str = "main",
    aux_subtype: str | None = None,
    safe_split_after: bool | None = None,
    boundary_kind: str | None = None,
) -> Block:
    tokens = len(text.split()) if text else 0
    if safe_split_after is None:
        safe_split_after = type in {"paragraph", "item", "list"}
    if boundary_kind is None:
        if type == "heading":
            if heading_level == 1:
                boundary_kind = "H1"
            elif heading_level == 2:
                boundary_kind = "H2"
            elif heading_level == 3:
                boundary_kind = "H3"
            else:
                boundary_kind = "Sub"
        elif type in {"list", "item"}:
            boundary_kind = "List"
        elif type == "paragraph":
            boundary_kind = "Para"
        else:
            boundary_kind = "None"
    if heading_path is None:
        if type == "heading":
            resolved_heading_path = [text]
        else:
            resolved_heading_path = []
    else:
        resolved_heading_path = heading_path
    return Block(
        doc_id=doc_id,
        block_id=block_id,
        page=page,
        order=order,
        type=type,
        text=text,
        bbox={"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
        heading_level=heading_level,
        heading_path=resolved_heading_path,
        source={"stage": source_stage, "tool": "stub", "version": "1.0"},
        aux={},
        role=role,
        aux_subtype=aux_subtype,
        parent_block_id=None,
        role_confidence=0.9,
        safe_split_after=safe_split_after,
        boundary_kind=boundary_kind,
        est_tokens=tokens,
    )


def test_chunker_emits_section_first_with_aux_followups() -> None:
    config = PipelineConfig.from_mapping(
        {"flow": {"limits": {"target": 10, "soft": 12, "hard": 14, "min": 5}}}
    )
    blocks = [
        _block(block_id="b1", page=1, order=0, type="heading", text="Heading 1", heading_level=1),
        _block(block_id="b2", page=1, order=1, type="paragraph", text="alpha beta gamma delta", heading_path=["Heading 1"]),
        _block(
            block_id="b3",
            page=1,
            order=2,
            type="table",
            text="Table contents",
            role="auxiliary",
            aux_subtype="sidebar",
            heading_path=["Heading 1"],
        ),
        _block(block_id="b4", page=2, order=3, type="heading", text="Section", heading_level=2, heading_path=["Heading 1", "Section"]),
        _block(
            block_id="b5",
            page=2,
            order=4,
            type="figure",
            text="Figure contents",
            role="auxiliary",
            aux_subtype="sidebar",
            heading_path=["Heading 1", "Section"],
        ),
        _block(
            block_id="b6",
            page=2,
            order=5,
            type="paragraph",
            text="one two three four five six seven",
            heading_path=["Heading 1", "Section"],
        ),
    ]

    chunks = chunk_blocks("doc", blocks, config)
    main_chunks = [chunk for chunk in chunks if chunk.is_main_only]
    aux_chunks = [chunk for chunk in chunks if chunk.is_aux_only]

    assert len(main_chunks) == 2
    assert len(aux_chunks) == 2

    first_main = main_chunks[0]
    first_aux = next(chunk for chunk in aux_chunks if chunk.section_id == first_main.section_id)
    assert first_main.heading_path[-1] == "Heading 1"
    assert first_main.aux_groups["sidecars"] == []
    assert any(entry.get("text") == "Table contents" for entry in first_aux.aux_groups["other"])

    second_main = main_chunks[1]
    second_aux = next(chunk for chunk in aux_chunks if chunk.section_id == second_main.section_id)
    assert second_main.heading_path[-1] == "Section"
    assert any(entry.get("text") == "Figure contents" for entry in second_aux.aux_groups["other"])
    assert second_main.token_count <= config.flow.limits.hard


def test_chunker_splits_long_text() -> None:
    config = PipelineConfig.from_mapping(
        {"flow": {"limits": {"target": 5, "soft": 6, "hard": 7, "min": 3}}}
    )
    text = " ".join(str(i) for i in range(40))
    blocks = [_block(block_id="bp", page=1, order=0, type="paragraph", text=text)]

    chunks = chunk_blocks("doc", blocks, config)
    assert chunks
    assert all(chunk.token_count <= config.flow.limits.hard for chunk in chunks)
    assert any(chunk.notes and "forced-split" in chunk.notes for chunk in chunks)
    for chunk in chunks:
        assert all(span["para_block_id"] == "bp" for span in chunk.evidence_spans)
