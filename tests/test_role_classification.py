from pipeline.config import PipelineConfig
from pipeline.normalize import _classify_block, normalise_blocks
from pipeline.docling_adapter import DoclingBlock
from pipeline.telemetry import Telemetry


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


def test_normalisation_relaxes_header_dropcap() -> None:
    config = PipelineConfig.from_mapping({})
    telemetry = Telemetry(doc_id="doc", file_name="doc.pdf")
    blocks = [
        DoclingBlock(page_number=1, block_type="paragraph", text="Header", bbox=(0.0, 0.0, 1.0, 0.02)),
        DoclingBlock(page_number=1, block_type="paragraph", text="Body", bbox=(0.0, 0.2, 1.0, 0.4)),
        DoclingBlock(page_number=1, block_type="paragraph", text="Footer", bbox=(0.0, 0.98, 1.0, 1.0)),
    ]
    normalised = normalise_blocks("doc", blocks, config, telemetry)

    assert len(normalised) == 3
    assert telemetry.was_filter_relaxed(1)
