from __future__ import annotations

from typing import List

from pipeline.config import PipelineConfig
from pipeline.normalize import Block
from pipeline.segmentor import Segmentor


class _TelemetryStub:
    def __init__(self) -> None:
        self.metrics: dict[str, int] = {}
        self.flags: List[str] = []

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


def test_caption_anchors_to_previous_main_block() -> None:
    config = PipelineConfig()
    telemetry = _TelemetryStub()
    segmentor = Segmentor("doc", config, telemetry, lambda s: len(s.split()))

    main_block = _block("b1", "Main narrative content")
    caption_block = _block("c1", "Figure 1. Example", role="auxiliary", aux_subtype="caption")

    segmentor.add_block(main_block)
    segmentor.add_block(caption_block)
    chunks = segmentor.finish()
    assert chunks
    last_chunk = chunks[-1]
    sidecars = last_chunk.aux_groups.get("sidecars", [])
    assert any(entry.get("parent_block_id") == "b1" for entry in sidecars)
