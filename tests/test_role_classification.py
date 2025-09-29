from pipeline.config import PipelineConfig
from pipeline.normalize import _classify_block


def _stats():
    return {"headers": set(), "footers": set(), "page_extents": {1: (0.0, 1.0), 2: (0.0, 1.0)}}


def test_caption_regex() -> None:
    config = PipelineConfig.from_mapping({})
    role, subtype, _, _, _, _ = _classify_block(
        "paragraph",
        "Figure 1. Sample",
        1,
        {"y0": 0.2, "y1": 0.3},
        _stats(),
        config,
    )
    assert role == "auxiliary" and subtype == "caption"


def test_activity_regex() -> None:
    config = PipelineConfig.from_mapping({})
    _, subtype, _, _, _, _ = _classify_block(
        "paragraph",
        "Let's discuss",
        1,
        {"y0": 0.2, "y1": 0.3},
        _stats(),
        config,
    )
    assert subtype == "activity"


def test_header_detection_drops() -> None:
    config = PipelineConfig.from_mapping({"aux": {"header_footer": {"repetition_threshold": 0.5}}})
    stats = {"headers": {"Any repeated header"}, "footers": set(), "page_extents": {2: (0.0, 1.0)}}
    _, subtype, _, _, drop, _ = _classify_block(
        "paragraph",
        "Any repeated header",
        2,
        {"y0": 0.0, "y1": 0.02},
        stats,
        config,
    )
    assert subtype == "header" and drop
