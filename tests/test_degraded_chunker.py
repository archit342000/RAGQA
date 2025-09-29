from pipeline.chunker import chunk_blocks
from pipeline.config import PipelineConfig
from pipeline.normalize import Block
from pipeline.telemetry import Telemetry


def make_block(text: str, block_type: str = "heading") -> Block:
    return Block(
        doc_id="doc",
        block_id="b1",
        page=1,
        order=0,
        type=block_type,
        text=text,
        bbox={"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
        heading_level=1 if block_type == "heading" else None,
        heading_path=[text] if block_type == "heading" else [],
        source={"stage": "triage", "tool": "pymupdf", "version": "1"},
        aux={},
        role="main",
        aux_subtype=None,
        parent_block_id=None,
        role_confidence=0.5,
    )


def test_chunker_emits_degraded_chunk_when_no_main_blocks():
    config = PipelineConfig()
    telemetry = Telemetry(doc_id="doc", file_name="doc.pdf")
    blocks = [make_block("Heading", "heading")]

    chunks = chunk_blocks("doc", blocks, config, telemetry=telemetry)

    assert chunks, "Degraded chunker should emit at least one chunk"
    assert telemetry.fallbacks_used["degraded"] >= 1
