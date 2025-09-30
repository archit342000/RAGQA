from pipeline.reorder_stitch.threads import build_threads
from pipeline.normalize import Block


def _block(block_id: str, page: int, y: float, text: str = "para") -> Block:
    return Block(
        doc_id="d",
        block_id=block_id,
        page=page,
        order=0,
        type="paragraph",
        text=text,
        bbox={"x0": 0.0, "y0": y, "x1": 100.0, "y1": y + 10.0},
        heading_level=None,
        heading_path=["H"],
        source={"stage": "docling", "tool": "stub", "version": "1"},
        aux={},
        role="main",
        aux_subtype=None,
        parent_block_id=None,
        role_confidence=1.0,
        safe_split_after=True,
        boundary_kind="Para",
        est_tokens=20,
    )


def test_build_threads_creates_single_thread() -> None:
    blocks = [_block("b1", 1, 0.0), _block("b2", 1, 20.0)]
    columns = {"b1": 0, "b2": 0}
    weights = {"style": 3.0, "indent": 2.0, "xalign": 2.0, "heading": 4.0, "gap": -1.0, "sent": 2.0}
    threads, unthreaded = build_threads(blocks, columns, weights, 1.5)
    assert len(threads) == 1
    assert threads[0]["blocks"] == ["b1", "b2"]
    assert not unthreaded


def test_build_threads_filters_short_paths() -> None:
    block = _block("b1", 1, 0.0, text="short")
    columns = {"b1": 0}
    weights = {"style": 3.0, "indent": 2.0, "xalign": 2.0, "heading": 4.0, "gap": -1.0, "sent": 2.0}
    threads, unthreaded = build_threads([block], columns, weights, 1.5)
    assert not threads
    assert unthreaded == [block]
