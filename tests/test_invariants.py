from pipeline.config import PipelineConfig
from pipeline.normalize import Block
from pipeline.segmentor import Segmentor


class _TelemetryStub:
    def __init__(self) -> None:
        self.metrics: dict[str, int] = {}
        self.flags: list[str] = []

    def inc(self, key: str, amount: int = 1) -> None:
        self.metrics[key] = self.metrics.get(key, 0) + amount

    def flag(self, code: str) -> None:
        self.flags.append(code)


def _block(block_id: str, text: str, *, role: str = "main", aux_subtype: str | None = None) -> Block:
    return Block(
        doc_id="doc",
        block_id=block_id,
        page=1,
        order=0,
        type="paragraph",
        text=text,
        bbox={"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
        heading_level=None,
        heading_path=[],
        source={"stage": "docling", "tool": "stub", "version": "1"},
        aux={},
        role=role,
        aux_subtype=aux_subtype,
        parent_block_id=None,
        role_confidence=0.9,
        safe_split_after=True,
        boundary_kind="Para",
        est_tokens=len(text.split()),
    )


def test_segment_chunks_emit_aux_after_main() -> None:
    config = PipelineConfig.from_mapping({"flow": {"limits": {"target": 10, "soft": 12, "hard": 14, "min": 5}}})
    telemetry = _TelemetryStub()
    segmentor = Segmentor("doc", config, telemetry, lambda s: len(s.split()))

    segmentor.add_block(_block("m1", "Main body text"))
    segmentor.add_block(_block("m2", "Continuation of the main narrative."))
    segmentor.add_block(_block("fn1", "Footnote text", role="auxiliary", aux_subtype="footnote"))

    chunks = segmentor.finish()
    assert chunks
    main_chunks = [chunk for chunk in chunks if not chunk.is_aux_only]
    aux_chunks = [chunk for chunk in chunks if chunk.is_aux_only]
    assert main_chunks
    if aux_chunks:
        assert all(chunk.segment_seq < aux_chunks[0].segment_seq for chunk in main_chunks)
    else:
        assert any(chunk.aux_subtypes_present for chunk in main_chunks)
