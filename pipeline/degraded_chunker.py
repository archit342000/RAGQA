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

    target = config.chunk.degraded_target
    minimum = config.chunk.degraded_minimum
    maximum = config.chunk.degraded_maximum
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
            segment_id=f"{doc_id}-degraded",
            segment_seq=len(chunks),
            heading_path=heading,
            text=combined,
            page_span=[min(pages) if pages else 1, max(pages) if pages else 1],
            token_count=tokens,
            evidence_spans=evidence,
            sidecars=[],
            quality=quality,
            aux_groups={"sidecars": [], "footnotes": [], "other": []},
            notes=["degraded"],
            limits={
                "target": config.flow.limits.target,
                "soft": config.flow.limits.soft,
                "hard": config.flow.limits.hard,
                "min": config.flow.limits.minimum,
            },
            flow_overflow=max(0, tokens - config.flow.limits.target),
            closed_at_boundary="EOF",
            aux_in_followup=False,
            link_prev_index=None,
            link_next_index=None,
            is_main_only=True,
            is_aux_only=False,
            aux_subtypes_present=[],
            aux_group_seq=None,
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
    lines = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            lines.append("\n")
            continue
        if stripped.endswith("-") and len(stripped) > 1 and not stripped.endswith("--"):
            stripped = stripped[:-1]
        lines.append(stripped)
    paragraph = []
    buffer: List[str] = []
    for token in lines:
        if token == "\n":
            if buffer:
                paragraph.append(" ".join(buffer))
                buffer = []
            continue
        buffer.append(token)
    if buffer:
        paragraph.append(" ".join(buffer))
    if not paragraph:
        paragraph = [" ".join(token for token in lines if token and token != "\n")]
    return "\n\n".join(segment for segment in paragraph if segment)
