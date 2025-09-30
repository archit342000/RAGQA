from pipeline.config import PipelineConfig
from pipeline.normalize import Block
from pipeline.segmentor import Segmentor


class DummyTelemetry:
    def __init__(self) -> None:
        self.metrics = {"aux_buffered": 0}
        self.flags: list[str] = []

    def inc(self, key: str, amount: int = 1) -> None:
        self.metrics[key] = self.metrics.get(key, 0) + amount

    def flag(self, code: str) -> None:
        self.flags.append(code)


def _token_counter(text: str) -> int:
    return len(text.split())


def _block(block_id: str, text: str, page: int = 1, role: str = "main") -> Block:
    tokens = len(text.split()) if text else 0
    if role == "main":
        safe_split = True
        boundary = "Para"
    else:
        safe_split = False
        boundary = "None"
    return Block(
        doc_id="doc",
        block_id=block_id,
        page=page,
        order=0,
        type="paragraph" if role == "main" else "caption",
        text=text,
        bbox={"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
        heading_level=1,
        heading_path=["H1"],
        source={"stage": "docling", "tool": "stub", "version": "1.0"},
        aux={},
        role=role,
        aux_subtype="caption" if role == "auxiliary" else None,
        parent_block_id=None,
        role_confidence=0.9,
        safe_split_after=safe_split,
        boundary_kind=boundary,
        est_tokens=tokens,
    )


def test_soft_flush_flags_aux04() -> None:
    config = PipelineConfig.from_mapping({"chunk": {"tokens": {"target": 5, "min": 1, "max": 10}}, "aux": {"soft_boundary": {"max_deferred_pages": 1}}})
    telemetry = DummyTelemetry()
    segmentor = Segmentor("doc", config, telemetry, _token_counter)
    main_block = _block("b1", "alpha beta gamma", page=1)
    segmentor.add_block(main_block)
    aux_block = _block("b2", "Figure caption", page=3, role="auxiliary")
    segmentor.add_block(aux_block)
    segmentor.finish()
    assert "AUX04" in telemetry.flags
