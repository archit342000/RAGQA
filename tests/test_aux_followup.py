from pipeline.chunker import chunk_blocks
from pipeline.config import PipelineConfig
from pipeline.normalize import Block


def _block(
    block_id: str,
    text: str,
    *,
    block_type: str = "paragraph",
    role: str = "main",
    aux_subtype: str | None = None,
    boundary_kind: str = "Para",
    page: int = 1,
) -> Block:
    tokens = len(text.split()) if text else 0
    return Block(
        doc_id="doc",
        block_id=block_id,
        page=page,
        order=0,
        type=block_type,
        text=text,
        bbox={"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
        heading_level=1 if block_type == "heading" else None,
        heading_path=[text] if block_type == "heading" else [],
        source={"stage": "docling", "tool": "stub", "version": "1"},
        aux={},
        role=role,
        aux_subtype=aux_subtype,
        parent_block_id=None,
        role_confidence=0.9,
        safe_split_after=block_type in {"paragraph", "list", "item"},
        boundary_kind=boundary_kind,
        est_tokens=tokens,
    )


def test_aux_followup_chunk_links_to_main() -> None:
    config = PipelineConfig.from_mapping(
        {
            "flow": {
                "limits": {"target": 10, "soft": 12, "hard": 14, "min": 5},
            }
        }
    )
    blocks = [
        _block("h1", "Section", block_type="heading", boundary_kind="H1"),
        _block("p1", "one two three four five six seven eight nine ten"),
        _block(
            "f1",
            "footnote text that is quite long to force follow up",
            role="auxiliary",
            aux_subtype="footnote",
            block_type="footnote",
            boundary_kind="Sent",
        ),
    ]

    chunks = chunk_blocks("doc", blocks, config)

    assert len(chunks) == 2
    main, aux = chunks
    assert main.aux_in_followup is True
    assert aux.aux_in_followup is True
    assert aux.link_prev == main.chunk_id
    assert main.link_next == aux.chunk_id
    assert aux.aux_groups["footnotes"], "Aux chunk should contain deferred footnotes"
