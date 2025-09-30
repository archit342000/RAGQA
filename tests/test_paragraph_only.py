from pipeline.chunker import chunk_blocks
from pipeline.config import PipelineConfig
from pipeline.normalize import Block
from pipeline.telemetry import Telemetry


def make_block(text: str, **kwargs) -> Block:
    aux = kwargs.get("aux", {})
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
        aux=aux,
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


def test_paragraph_only_activates_when_no_main_blocks():
    config = PipelineConfig()
    telemetry = Telemetry(doc_id="d", file_name="doc.pdf")
    blocks = [
        make_block(
            "Main paragraph on top",
            block_id="b1",
            page=1,
            bbox={"x0": 0.0, "y0": 0.0, "x1": 80.0, "y1": 40.0},
            aux={"page_height": 100.0},
        ),
        make_block(
            "Continuation paragraph",
            block_id="b2",
            page=2,
            bbox={"x0": 0.0, "y0": 0.0, "x1": 80.0, "y1": 40.0},
            aux={"page_height": 100.0},
        ),
    ]
    chunks = chunk_blocks("d", blocks, config, telemetry)
    assert telemetry.paragraph_only_activations >= 1
    assert chunks
    assert all(chunk.is_aux_only or chunk.text for chunk in chunks)
