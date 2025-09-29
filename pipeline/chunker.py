from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .config import PipelineConfig
from .ids import make_chunk_id
from .normalize import Block
from .segmentor import SegmentChunk, Segmentor
from .degraded_chunker import build_degraded_segment_chunks
from .token_utils import count_tokens

logger = logging.getLogger(__name__)

BOUNDARY_RE = re.compile(r"^(references|appendix|glossary|index|chapter\s+|part\s+)", re.IGNORECASE)

@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    page_span: List[int]
    heading_path: List[str]
    text: str
    token_count: int
    sidecars: List[dict]
    evidence_spans: List[dict]
    quality: dict
    aux_groups: dict
    notes: str | None


class _NullTelemetry:
    def inc(self, *args, **kwargs) -> None:  # pragma: no cover - fallback
        return None

    def flag(self, *args, **kwargs) -> None:  # pragma: no cover - fallback
        return None


def _convert_chunks(doc_id: str, payloads: Iterable[SegmentChunk]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for payload in payloads:
        notes = sorted(set(payload.notes))
        chunk = Chunk(
            chunk_id=make_chunk_id(),
            doc_id=doc_id,
            page_span=list(payload.page_span),
            heading_path=list(payload.heading_path),
            text=payload.text,
            token_count=payload.token_count,
            sidecars=list(payload.sidecars),
            evidence_spans=list(payload.evidence_spans),
            quality=dict(payload.quality),
            aux_groups={
                "sidecars": list(payload.aux_groups.get("sidecars", [])),
                "footnotes": list(payload.aux_groups.get("footnotes", [])),
                "other": list(payload.aux_groups.get("other", [])),
            },
            notes=",".join(notes) if notes else None,
        )
        chunks.append(chunk)
    return chunks


def _is_hard_boundary(block: Block) -> bool:
    text = (block.text or "").strip()
    if block.role == "auxiliary" and block.aux_subtype == "activity":
        return True
    if block.type == "heading" and block.heading_level is not None and block.heading_level <= 3:
        return True
    if BOUNDARY_RE.match(text):
        return True
    return False


def chunk_blocks(
    doc_id: str,
    blocks: Sequence[Block],
    config: PipelineConfig,
    telemetry=None,
) -> List[Chunk]:
    telemetry = telemetry or _NullTelemetry()
    if _should_use_degraded(blocks):
        degraded_payloads = build_degraded_segment_chunks(doc_id, blocks, config)
        if telemetry is not None:
            telemetry.flag("DEGRADED_PATH")
        return _convert_chunks(doc_id, degraded_payloads)
    segmentor = Segmentor(doc_id, config, telemetry, count_tokens)
    chunks: List[Chunk] = []

    for block in blocks:
        if _is_hard_boundary(block):
            flushed = segmentor.boundary(hard=True)
            if flushed:
                chunks.extend(_convert_chunks(doc_id, flushed))
        emitted = segmentor.add_block(block)
        if emitted:
            chunks.extend(_convert_chunks(doc_id, emitted))

    tail = segmentor.finish()
    if tail:
        chunks.extend(_convert_chunks(doc_id, tail))

    if not chunks:
        degraded_payloads = build_degraded_segment_chunks(doc_id, blocks, config)
        if degraded_payloads:
            if hasattr(telemetry, "fallbacks_used"):
                telemetry.fallbacks_used["degraded"] = telemetry.fallbacks_used.get(
                    "degraded", 0
                ) + 1
            telemetry.flag("DEGRADED_CHUNKER")
            chunks.extend(_convert_chunks(doc_id, degraded_payloads))

    return chunks


def _should_use_degraded(blocks: Sequence[Block]) -> bool:
    if not blocks:
        return False
    if any(block.type == "heading" for block in blocks):
        return False
    permitted_sources = {"extractor", "triage", "ocr"}
    return all(block.source.get("stage") in permitted_sources for block in blocks)
