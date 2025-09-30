from pipeline.reorder_stitch.columns import detect_columns
from pipeline.normalize import Block


def _block(block_id: str, x0: float, x1: float) -> Block:
    return Block(
        doc_id="d",
        block_id=block_id,
        page=1,
        order=0,
        type="paragraph",
        text="text",
        bbox={"x0": x0, "y0": 0.0, "x1": x1, "y1": 10.0},
        heading_level=None,
        heading_path=[],
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


def test_detect_columns_two_clusters() -> None:
    blocks = [_block("b1", 0.0, 100.0), _block("b2", 120.0, 220.0)]
    columns, assignments = detect_columns(blocks, page_width=240.0)
    assert len(columns) == 2
    assert assignments["b1"] != assignments["b2"]


def test_detect_columns_rejects_narrow_column() -> None:
    wide = _block("wide", 0.0, 200.0)
    narrow = _block("narrow", 210.0, 240.0)
    columns, assignments = detect_columns([wide, narrow], page_width=400.0)
    assert len(columns) == 1
    assert assignments["wide"] == assignments["narrow"] == 0
