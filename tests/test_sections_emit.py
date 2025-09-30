from pipeline.reorder_stitch.sections import assemble_sections
from pipeline.normalize import Block


def _block(block_id: str, heading: list[str]) -> Block:
    return Block(
        doc_id="d",
        block_id=block_id,
        page=1,
        order=0,
        type="paragraph",
        text="text",
        bbox={"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 10.0},
        heading_level=None,
        heading_path=heading,
        source={"stage": "docling", "tool": "stub", "version": "1"},
        aux={},
        role="main",
        aux_subtype=None,
        parent_block_id=None,
        role_confidence=1.0,
        safe_split_after=True,
        boundary_kind="Para",
        est_tokens=10,
    )


def test_assemble_sections_picks_primary_thread() -> None:
    blocks = {
        "t1a": _block("t1a", ["H"]),
        "t1b": _block("t1b", ["H"]),
        "t2": _block("t2", ["H"]),
    }
    threads = [
        {"thread_id": "thr_0000", "section": ["H"], "blocks": ["t1a", "t1b"], "cohesion": 2.0, "tokens": 50},
        {"thread_id": "thr_0001", "section": ["H"], "blocks": ["t2"], "cohesion": 1.6, "tokens": 10},
    ]
    sections, aux_pool = assemble_sections(threads, blocks)
    assert len(sections) == 1
    section = sections[0]
    assert section["blocks"][:2] == ["t1a", "t1b"]
    assert "t2" in section["blocks"]
    assert not aux_pool


def test_assemble_sections_returns_aux_pool() -> None:
    blocks = {"b": _block("b", ["H"])}
    threads = []
    sections, aux_pool = assemble_sections(threads, blocks)
    assert not sections
    assert aux_pool == [blocks["b"]]
