from pipeline.reorder_stitch.continuity import continuity_score
from pipeline.normalize import Block


def _block(block_id: str, text: str, x0: float, x1: float) -> Block:
    return Block(
        doc_id="d",
        block_id=block_id,
        page=1,
        order=0,
        type="paragraph",
        text=text,
        bbox={"x0": x0, "y0": 0.0, "x1": x1, "y1": 10.0},
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
        est_tokens=len(text.split()),
    )


def test_continuity_prefers_sentence_continuation() -> None:
    weights = {"style": 3.0, "indent": 2.0, "xalign": 2.0, "heading": 4.0, "gap": -1.0, "sent": 2.0}
    first = _block("b1", "This is a", 0.0, 100.0)
    second = _block("b2", "continuation", 0.0, 100.0)
    third = _block("b3", "New sentence.", 0.0, 100.0)

    cont_score = continuity_score(first, second, weights)
    new_score = continuity_score(first, third, weights)
    assert cont_score > new_score


def test_continuity_penalises_aux_prefix() -> None:
    weights = {"style": 3.0, "indent": 2.0, "xalign": 2.0, "heading": 4.0, "gap": -1.0, "sent": 2.0}
    first = _block("b1", "Some text", 0.0, 100.0)
    caption = _block("b2", "Figure 1. Caption", 0.0, 100.0)
    normal = _block("b3", "Another paragraph", 0.0, 100.0)
    score_caption = continuity_score(first, caption, weights)
    score_normal = continuity_score(first, normal, weights)
    assert score_caption < score_normal
