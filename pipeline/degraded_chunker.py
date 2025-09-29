from __future__ import annotations

from typing import List, Sequence

from .config import PipelineConfig
from .normalize import Block
from .segmentor import SegmentChunk
from .token_utils import count_tokens


def build_degraded_segment_chunks(
    doc_id: str,
    blocks: Sequence[Block],
    config: PipelineConfig,
) -> List[SegmentChunk]:
    """Fallback chunker that produces SegmentChunk payloads."""

    paragraphs: List[tuple[Block | None, str]] = []
    for block in blocks:
        text = (block.text or "").strip()
        if not text:
            continue
        if block.type == "heading":
            continue
        paragraphs.append((block, _normalise_paragraph(text)))

    if not paragraphs:
        stub = "No structured text recovered; emitting triage fallback."
        paragraphs.append((None, stub))

    target = config.chunk.tokens.target
    minimum = config.chunk.tokens.minimum
    maximum = config.chunk.tokens.maximum
    chunks: List[SegmentChunk] = []
    buffer: List[tuple[Block | None, str]] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        text_parts = [part for _, part in buffer if part]
        combined = "\n\n".join(text_parts)
        tokens = count_tokens(combined)
        pages = [blk.page for blk, _ in buffer if blk is not None]
        heading = []
        quality = {"ocr_pages": 0, "rescued": False, "notes": "degraded"}
        evidence: List[dict] = []
        cursor = 0
        for blk, part in buffer:
            if blk is None:
                continue
            start = combined.find(part, cursor)
            if start < 0:
                start = cursor
            end = start + len(part)
            evidence.append({"para_block_id": blk.block_id, "start": start, "end": end})
            cursor = end
        chunk = SegmentChunk(
            heading_path=heading,
            text=combined,
            page_span=[min(pages) if pages else 1, max(pages) if pages else 1],
            token_count=tokens,
            evidence_spans=evidence,
            sidecars=[],
            quality=quality,
            aux_groups={"sidecars": [], "footnotes": [], "other": []},
            notes=["degraded"],
        )
        chunks.append(chunk)
        buffer = []

    for paragraph in paragraphs:
        candidate = buffer + [paragraph]
        text_candidate = "\n\n".join(part for _, part in candidate if part)
        tokens = count_tokens(text_candidate)
        if tokens > maximum and buffer:
            flush_buffer()
            candidate = [paragraph]
            text_candidate = paragraph[1]
            tokens = count_tokens(text_candidate)
        buffer = candidate
        if tokens >= target and tokens >= minimum:
            flush_buffer()

    if buffer:
        flush_buffer()

    if not chunks:
        flush_buffer()

    return chunks


def _normalise_paragraph(text: str) -> str:
    lines = [line.strip("-") for line in text.splitlines()]
    return " ".join(segment.strip() for segment in lines if segment.strip())
