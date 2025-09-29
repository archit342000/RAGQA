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
) -> Block:
    return Block(
        doc_id=doc_id,
        block_id=block_id,
        page=page,
        order=order,
        type=type,
        text=text,
        bbox={"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
        heading_level=heading_level,
        heading_path=heading_path or [],
        source={"stage": source_stage, "tool": "stub", "version": "1.0"},
        aux={},
        role=role,
        aux_subtype=aux_subtype,
        parent_block_id=None,
        role_confidence=0.9,
    )


def test_chunker_respects_headings_and_sidecars() -> None:
    config = PipelineConfig.from_mapping({"chunk": {"tokens": {"target": 10, "min": 5, "max": 12}}})
    blocks = [
        _block(block_id="b1", page=1, order=0, type="heading", text="Heading 1", heading_level=1),
        _block(block_id="b2", page=1, order=1, type="paragraph", text="alpha beta gamma delta"),
        _block(block_id="b3", page=1, order=2, type="table", text="Table contents", role="auxiliary", aux_subtype="sidebar"),
        _block(block_id="b4", page=2, order=3, type="heading", text="Section", heading_level=2),
        _block(block_id="b5", page=2, order=4, type="figure", text="Figure contents", role="auxiliary", aux_subtype="sidebar"),
        _block(block_id="b6", page=2, order=5, type="paragraph", text="one two three four five six seven"),
    ]

    chunks = chunk_blocks("doc", blocks, config)
    assert len(chunks) == 2

    first, second = chunks
    assert first.heading_path[-1] == "Heading 1"
    assert first.sidecars and first.sidecars[0]["type"] == "table"
    assert first.aux_groups["sidecars"]
    assert first.evidence_spans and first.evidence_spans[0]["para_block_id"] == "b2"

    assert second.heading_path[-1] == "Section"
    assert second.sidecars and second.sidecars[0]["type"] == "figure"
    assert second.aux_groups["sidecars"]
    assert second.evidence_spans[0]["start"] < second.evidence_spans[0]["end"]
    assert second.token_count <= config.chunk.tokens.maximum


def test_chunker_splits_long_text() -> None:
    config = PipelineConfig.from_mapping({"chunk": {"tokens": {"target": 5, "min": 3, "max": 6}}})
    text = " ".join(str(i) for i in range(40))
    blocks = [_block(block_id="bp", page=1, order=0, type="paragraph", text=text)]

    chunks = chunk_blocks("doc", blocks, config)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= config.chunk.tokens.maximum
        assert chunk.evidence_spans[0]["para_block_id"] == "bp"
