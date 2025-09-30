from pipeline.config import PipelineConfig
from pipeline.gate5 import evaluate_gate5
from pipeline.normalize import Block


def make_block(text: str, **kwargs) -> Block:
    return Block(
        doc_id="d",
        block_id=kwargs.get("block_id", "b1"),
        page=kwargs.get("page", 1),
        order=kwargs.get("order", 0),
        type=kwargs.get("type", "paragraph"),
        text=text,
        bbox=kwargs.get("bbox", {"x0": 0.0, "y0": 50.0, "x1": 100.0, "y1": 150.0}),
        heading_level=None,
        heading_path=[],
        source={},
        aux={},
        role="main",
        aux_subtype=None,
        parent_block_id=None,
        role_confidence=1.0,
        safe_split_after=True,
        boundary_kind="Para",
        est_tokens=len(text.split()),
        main_gate_passed=True,
        rejection_reasons=[],
    )


def test_gate5_blocks_caption_by_regex():
    config = PipelineConfig()
    block = make_block("Figure 3. Overview", bbox={"x0": 0.0, "y0": 0.2, "x1": 100.0, "y1": 120.0})
    decision = evaluate_gate5(block, config)
    assert not decision.allow
    assert decision.reason == "regex_match"


def test_gate5_width_floor_blocks_sidebar():
    config = PipelineConfig()
    block = make_block(
        "Normal paragraph",
        bbox={"x0": 0.0, "y0": 300.0, "x1": 10.0, "y1": 350.0},
    )
    block.aux["page_height"] = 1000.0
    block.aux["column_width"] = 100.0
    decision = evaluate_gate5(block, config)
    assert not decision.allow
    assert decision.reason == "narrow_box"
