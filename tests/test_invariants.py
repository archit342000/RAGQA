import pytest

from pipeline.config import PipelineConfig
from pipeline.normalize import Block
from pipeline.segmentor import SegmentChunk, Segmentor
from pipeline.telemetry import Telemetry


def make_block(text: str, **kwargs) -> Block:
    return Block(
        doc_id="d",
        block_id=kwargs.get("block_id", "b1"),
        page=kwargs.get("page", 1),
        order=kwargs.get("order", 0),
        type=kwargs.get("type", "paragraph"),
        text=text,
        bbox=kwargs.get("bbox", {"x0": 0.0, "y0": 0.2, "x1": 100.0, "y1": 200.0}),
        heading_level=None,
        heading_path=[],
        source={},
        aux={"column_width": 100.0},
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


def test_invariant_flags_deny_regex():
    config = PipelineConfig()
    telemetry = Telemetry(doc_id="d", file_name="doc.pdf")
    segmentor = Segmentor("d", config, telemetry, lambda text: len(text.split()))
    block = make_block("Paragraph text")
    chunk = SegmentChunk(
        segment_id="d-seg001",
        segment_seq=0,
        heading_path=[],
        text="Figure 5. Appears",
        page_span=[1, 1],
        token_count=3,
        evidence_spans=[],
        sidecars=[],
        quality={},
        aux_groups={"sidecars": [], "footnotes": [], "other": []},
        notes=[],
        limits={"target": 1600, "soft": 2000, "hard": 2400, "min": 900},
        flow_overflow=0,
        closed_at_boundary="Para",
        aux_in_followup=False,
        link_prev_index=None,
        link_next_index=None,
        is_main_only=True,
        is_aux_only=False,
        aux_subtypes_present=[],
        aux_group_seq=None,
        debug_blocks=[block],
    )
    with pytest.raises(RuntimeError):
        segmentor._validate_invariants([chunk])
    assert telemetry.invariant_violations >= 1
