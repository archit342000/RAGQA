from pipeline.config import PipelineConfig
from pipeline.flow_chunker import build_flow_chunk_plan
from pipeline.normalize import Block


def _block(text: str, boundary: str = "Para") -> Block:
    tokens = len(text.split())
    return Block(
        doc_id="doc",
        block_id=f"b{tokens}",
        page=1,
        order=tokens,
        type="paragraph",
        text=text,
        bbox={"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
        heading_level=None,
        heading_path=[],
        source={"stage": "docling", "tool": "stub", "version": "1"},
        aux={},
        role="main",
        aux_subtype=None,
        parent_block_id=None,
        role_confidence=0.9,
        safe_split_after=True,
        boundary_kind=boundary,
        est_tokens=tokens,
    )


def test_flow_chunk_plan_allows_boundary_overflow() -> None:
    config = PipelineConfig.from_mapping(
        {
            "flow": {
                "limits": {"target": 10, "soft": 12, "hard": 16, "min": 5},
                "boundary_slack_tokens": 4,
            }
        }
    )
    blocks = [_block("alpha beta gamma delta epsilon"), _block("beta gamma delta"), _block("section", "H1"), _block("more text here" )]
    plans = build_flow_chunk_plan(blocks, config, lambda text: len(text.split()))
    assert plans, "Expected at least one chunk plan"
    first = plans[0]
    assert len(first.blocks) >= 2
    assert first.closed_at in {"H1", "EOF"}


def test_flow_chunk_plan_splits_long_paragraph() -> None:
    config = PipelineConfig.from_mapping(
        {
            "flow": {
                "limits": {"target": 6, "soft": 8, "hard": 10, "min": 3},
                "boundary_slack_tokens": 2,
            }
        }
    )
    text = "Sentence one is lengthy. Sentence two is also quite long. Sentence three keeps going."
    block = _block(text)

    plans = build_flow_chunk_plan([block], config, lambda value: len(value.split()))

    assert plans, "Plan should exist"
    total_fragments = sum(len(plan.blocks) for plan in plans)
    assert total_fragments >= 2
    assert any(plan.forced_split for plan in plans)
    for plan in plans:
        for fragment in plan.blocks:
            assert fragment.tokens <= config.flow.limits.hard
