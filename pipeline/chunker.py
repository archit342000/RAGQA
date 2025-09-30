from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from .config import PipelineConfig
from .ids import make_chunk_id
from .normalize import Block
from .degraded_chunker import build_degraded_segment_chunks
from .segmentor import SegmentChunk
from .flow_chunker import FlowChunkPlan, build_flow_chunk_plan
from .token_utils import count_tokens
from .microfixes.cross_page_stitch import cross_page_stitch
from .microfixes.sentence_gate import sentence_closure_gate
from .reorder_stitch import (
    assemble_sections,
    build_threads,
    detect_columns,
    group_auxiliary_blocks,
)

logger = logging.getLogger(__name__)

FLOAT_TYPES = {"table", "figure", "caption", "footnote"}


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
    limits: dict
    flow_overflow: int
    closed_at_boundary: str
    aux_in_followup: bool
    link_prev: str | None
    link_next: str | None
    segment_id: str | None
    segment_seq: int | None
    is_main_only: bool
    is_aux_only: bool
    aux_subtypes_present: List[str]
    aux_group_seq: int | None
    section_id: str | None
    thread_id: str | None
    debug_block_ids: List[str] = field(default_factory=list)


class _NullTelemetry:
    def __init__(self) -> None:
        self.metrics: Dict[str, int] = {}

    def inc(self, *args, **kwargs) -> None:  # pragma: no cover - fallback
        return None

    def flag(self, *args, **kwargs) -> None:  # pragma: no cover - fallback
        return None


def chunk_blocks(
    doc_id: str,
    blocks: Sequence[Block],
    config: PipelineConfig,
    telemetry=None,
) -> List[Chunk]:
    telemetry = telemetry or _NullTelemetry()

    if _should_use_degraded(blocks):
        degraded_payloads = build_degraded_segment_chunks(doc_id, blocks, config)
        converted = _convert_chunks(doc_id, degraded_payloads)
        _record_flow_metrics(converted, telemetry)
        if hasattr(telemetry, "flag"):
            telemetry.flag("DEGRADED_PATH")
        if hasattr(telemetry, "inc"):
            telemetry.inc("emitted_chunks", len(converted))
        return converted

    block_lookup: Dict[str, Block] = {block.block_id: block for block in blocks}
    if not block_lookup:
        return []

    narrative_types = {"paragraph", "list", "item", "code", "equation"}
    main_candidates = [
        block
        for block in blocks
        if block.role == "main"
        and block.type in narrative_types
        and (block.text or "").strip()
    ]
    if not main_candidates:
        degraded_payloads = build_degraded_segment_chunks(doc_id, blocks, config)
        converted = _convert_chunks(doc_id, degraded_payloads)
        _record_flow_metrics(converted, telemetry)
        if hasattr(telemetry, "flag"):
            telemetry.flag("DEGRADED_CHUNKER")
        if hasattr(telemetry, "inc"):
            telemetry.inc("emitted_chunks", len(converted))
        return converted

    column_lookup: Dict[str, int] = {}
    by_page: Dict[int, List[Block]] = {}
    for block in main_candidates:
        by_page.setdefault(block.page, []).append(block)
    for page, page_blocks in by_page.items():
        page_width = _page_width(page_blocks)
        _, assignments = detect_columns(page_blocks, page_width)
        column_lookup.update(assignments)

    weights = {
        "style": 3.0,
        "indent": 2.0,
        "xalign": 2.0,
        "heading": 4.0,
        "gap": -1.0,
        "sent": 2.0,
    }
    tau_main = 1.5

    threads, unthreaded = build_threads(main_candidates, column_lookup, weights, tau_main)
    candidate_lookup = {block.block_id: block for block in main_candidates}
    sections, unused_main = assemble_sections(threads, candidate_lookup)

    if unused_main:
        start_idx = len(sections)
        sorted_unused = sorted(
            unused_main,
            key=lambda b: (b.page, b.bbox.get("y0", 0.0) if b.bbox else 0.0),
        )
        for offset, block in enumerate(sorted_unused):
            section_id = f"sec_{start_idx + offset:04d}"
            sections.append(
                {
                    "section_id": section_id,
                    "heading_path": list(block.heading_path),
                    "blocks": [block.block_id],
                    "thread_ids": [None],
                    "cohesion": 0.0,
                    "_resolved_blocks": [block],
                }
            )
            threads.append(
                {
                    "thread_id": f"thr_fallback_{start_idx + offset:04d}",
                    "section": list(block.heading_path),
                    "blocks": [block.block_id],
                    "cohesion": 0.0,
                    "tokens": int(block.est_tokens),
                }
            )

    used_ids = {block_id for section in sections for block_id in section["blocks"]}
    aux_index: Dict[str, Block] = {}
    for block in unthreaded:
        if block.block_id not in used_ids:
            aux_index.setdefault(block.block_id, block)
    for block in blocks:
        if block.block_id not in used_ids:
            aux_index.setdefault(block.block_id, block)

    for section in sections:
        if "_resolved_blocks" in section:
            continue
        resolved = [candidate_lookup[b_id] for b_id in section["blocks"] if b_id in candidate_lookup]
        section["_resolved_blocks"] = resolved

    sections = [section for section in sections if section["_resolved_blocks"]]
    if not sections:
        degraded_payloads = build_degraded_segment_chunks(doc_id, blocks, config)
        converted = _convert_chunks(doc_id, degraded_payloads)
        _record_flow_metrics(converted, telemetry)
        if hasattr(telemetry, "flag"):
            telemetry.flag("DEGRADED_CHUNKER")
        if hasattr(telemetry, "inc"):
            telemetry.inc("emitted_chunks", len(converted))
        return converted

    aux_pool = list(aux_index.values())
    chunks = _emit_sections(doc_id, sections, aux_pool, config)

    chunks = _apply_microfixes(chunks, telemetry)

    if hasattr(telemetry, "inc"):
        telemetry.inc("threads_total", len(threads))
        telemetry.inc("threads_main", len(sections))
        telemetry.inc("threads_aux", max(len(threads) - len(sections), 0))
        if sections:
            cohesion_sum = sum(section.get("cohesion", 0.0) for section in sections)
            avg = cohesion_sum / len(sections)
            if hasattr(telemetry, "avg_thread_cohesion"):
                telemetry.avg_thread_cohesion = avg
    if hasattr(telemetry, "inc"):
        telemetry.inc("emitted_chunks", len(chunks))

    _record_flow_metrics(chunks, telemetry)
    _assert_invariants(chunks, sections, threads, block_lookup, telemetry)
    return chunks


def _emit_sections(
    doc_id: str,
    sections: Sequence[Dict[str, object]],
    aux_pool: Sequence[Block],
    config: PipelineConfig,
) -> List[Chunk]:
    aux_map: Dict[Tuple[str, ...], List[Block]] = {}
    for block in aux_pool:
        heading = tuple(block.heading_path)
        aux_map.setdefault(heading, []).append(block)

    all_chunks: List[Chunk] = []
    for section in sections:
        section_id = section["section_id"]  # type: ignore[index]
        heading_path = section["heading_path"]  # type: ignore[index]
        thread_ids: List[str | None] = [
            tid if isinstance(tid, str) or tid is None else str(tid)
            for tid in section.get("thread_ids", [])  # type: ignore[call-arg]
        ]
        resolved_blocks: List[Block] = section["_resolved_blocks"]  # type: ignore[index]
        plans = build_flow_chunk_plan(resolved_blocks, config, count_tokens)
        seq = 0
        for plan in plans:
            thread_id = thread_ids[0] if thread_ids else None
            chunk = _chunk_from_plan(
                doc_id,
                section_id,
                heading_path,
                thread_id,
                plan,
                config,
            )
            chunk.segment_seq = seq
            all_chunks.append(chunk)
            seq += 1
        key = tuple(heading_path)
        aux_blocks = aux_map.pop(key, [])
        if aux_blocks:
            chunk = _build_aux_chunk(doc_id, section_id, heading_path, aux_blocks, config)
            chunk.segment_seq = seq
            all_chunks.append(chunk)

    # Emit aux-only chunks for leftover pools (e.g., missing heading_path)
    for heading, blocks in aux_map.items():
        if not blocks:
            continue
        section_id = f"aux_{len(all_chunks):04d}"
        chunk = _build_aux_chunk(doc_id, section_id, list(heading), blocks, config)
        all_chunks.append(chunk)
    return all_chunks


def _chunk_from_plan(
    doc_id: str,
    section_id: str,
    heading_path: Sequence[str],
    thread_id: str | None,
    plan: FlowChunkPlan,
    config: PipelineConfig,
) -> Chunk:
    combined = ""
    evidence_map: Dict[str, dict] = {}
    evidence_order: List[str] = []
    pages: set[int] = set()
    notes: List[str] = []
    last_block_id: str | None = None

    for fragment in plan.blocks:
        block = fragment.block
        text = (fragment.text or "").strip()
        if not text:
            continue
        if combined:
            if fragment.is_continuation or block.block_id == last_block_id:
                combined += " "
            else:
                combined += "\n\n"
        start = len(combined)
        combined += text
        end = len(combined)
        if block.block_id not in evidence_map:
            evidence_order.append(block.block_id)
            evidence_map[block.block_id] = {
                "para_block_id": block.block_id,
                "start": start,
                "end": end,
            }
        else:
            evidence_map[block.block_id]["end"] = end
        pages.add(block.page)
        last_block_id = block.block_id

    if plan.forced_split:
        notes.append("forced-split")

    token_count = count_tokens(combined)
    evidence_spans = [evidence_map[block_id] for block_id in evidence_order]
    ocr_pages = {
        fragment.block.page
        for fragment in plan.blocks
        if fragment.block.source.get("stage") == "ocr"
    }
    rescued = any(fragment.block.source.get("stage") == "layout" for fragment in plan.blocks)
    limits = {
        "target": config.flow.limits.target,
        "soft": config.flow.limits.soft,
        "hard": config.flow.limits.hard,
        "min": config.flow.limits.minimum,
    }

    chunk = Chunk(
        chunk_id=make_chunk_id(),
        doc_id=doc_id,
        page_span=[min(pages), max(pages)] if pages else [0, 0],
        heading_path=list(heading_path),
        text=combined,
        token_count=token_count,
        sidecars=[],
        evidence_spans=evidence_spans,
        quality={
            "ocr_pages": len(ocr_pages),
            "rescued": rescued,
            "notes": ",".join(notes) if notes else "",
        },
        aux_groups={"sidecars": [], "footnotes": [], "other": []},
        notes=",".join(notes) if notes else None,
        limits=limits,
        flow_overflow=max(0, token_count - config.flow.limits.target),
        closed_at_boundary=plan.closed_at,
        aux_in_followup=False,
        link_prev=None,
        link_next=None,
        segment_id=section_id,
        segment_seq=0,
        is_main_only=True,
        is_aux_only=False,
        aux_subtypes_present=[],
        aux_group_seq=None,
        section_id=section_id,
        thread_id=thread_id,
        debug_block_ids=list(evidence_order),
    )
    return chunk


def _build_aux_chunk(
    doc_id: str,
    section_id: str,
    heading_path: Sequence[str],
    blocks: Sequence[Block],
    config: PipelineConfig,
) -> Chunk:
    aux_groups, subtypes = group_auxiliary_blocks(blocks)
    pages = {block.page for block in blocks}
    limits = {
        "target": config.flow.limits.target,
        "soft": config.flow.limits.soft,
        "hard": config.flow.limits.hard,
        "min": config.flow.limits.minimum,
    }
    chunk = Chunk(
        chunk_id=make_chunk_id(),
        doc_id=doc_id,
        page_span=[min(pages), max(pages)] if pages else [0, 0],
        heading_path=list(heading_path),
        text="",
        token_count=0,
        sidecars=[],
        evidence_spans=[],
        quality={"ocr_pages": 0, "rescued": False, "notes": ""},
        aux_groups=aux_groups,
        notes=None,
        limits=limits,
        flow_overflow=0,
        closed_at_boundary="EOF",
        aux_in_followup=True,
        link_prev=None,
        link_next=None,
        segment_id=section_id,
        segment_seq=0,
        is_main_only=False,
        is_aux_only=True,
        aux_subtypes_present=subtypes,
        aux_group_seq=1,
        section_id=section_id,
        thread_id=None,
        debug_block_ids=[],
    )
    return chunk


def _convert_chunks(doc_id: str, payloads: Iterable[SegmentChunk]) -> List[Chunk]:
    chunks: List[Chunk] = []
    payload_list = list(payloads)
    for payload in payload_list:
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
            limits=dict(getattr(payload, "limits", {})),
            flow_overflow=getattr(payload, "flow_overflow", 0),
            closed_at_boundary=getattr(payload, "closed_at_boundary", "EOF"),
            aux_in_followup=getattr(payload, "aux_in_followup", False),
            link_prev=None,
            link_next=None,
            segment_id=getattr(payload, "segment_id", None),
            segment_seq=getattr(payload, "segment_seq", None),
            is_main_only=getattr(payload, "is_main_only", False),
            is_aux_only=getattr(payload, "is_aux_only", False),
            aux_subtypes_present=list(getattr(payload, "aux_subtypes_present", [])),
            aux_group_seq=getattr(payload, "aux_group_seq", None),
            section_id=getattr(payload, "segment_id", None),
            thread_id=None,
            debug_block_ids=[span.get("para_block_id") for span in payload.evidence_spans if span.get("para_block_id")],
        )
        chunks.append(chunk)

    id_lookup = {idx: chunk.chunk_id for idx, chunk in enumerate(chunks)}
    for idx, payload in enumerate(payload_list):
        prev_idx = getattr(payload, "link_prev_index", None)
        next_idx = getattr(payload, "link_next_index", None)
        if prev_idx is not None and prev_idx in id_lookup:
            chunks[idx].link_prev = id_lookup[prev_idx]
        if next_idx is not None and next_idx in id_lookup:
            chunks[idx].link_next = id_lookup[next_idx]
    return chunks


def _apply_microfixes(chunks: List[Chunk], telemetry) -> List[Chunk]:
    if not chunks:
        return []

    working = list(chunks)
    working, stitched = cross_page_stitch(working)
    if stitched and hasattr(telemetry, "inc"):
        telemetry.inc("stitches_used", stitched)

    working, delayed_aux = sentence_closure_gate(working)
    if delayed_aux and hasattr(telemetry, "inc"):
        telemetry.inc("aux_delays", delayed_aux)

    return working


def _record_flow_metrics(chunks: Sequence[Chunk], telemetry) -> None:
    if telemetry is None:
        return
    for chunk in chunks:
        if hasattr(telemetry, "flow_chunk_count"):
            telemetry.flow_chunk_count += 1
            telemetry.flow_chunk_tokens_sum += chunk.token_count
            telemetry.flow_overflow_sum += chunk.flow_overflow
            unique_blocks = {
                span.get("para_block_id")
                for span in chunk.evidence_spans
                if span.get("para_block_id") is not None
            }
            telemetry.flow_block_sum += len(unique_blocks)
            limits = chunk.limits or {}
            soft_limit = limits.get("soft") or 0
            hard_limit = limits.get("hard") or 0
            if soft_limit and chunk.token_count > soft_limit:
                telemetry.flow_soft_exceed_count += 1
            if hard_limit and chunk.token_count >= hard_limit:
                telemetry.flow_hard_count += 1
            if chunk.aux_in_followup and not chunk.text:
                telemetry.flow_aux_followup_count += 1
        telemetry.inc("flow_overflow_tokens", chunk.flow_overflow)
        if chunk.flow_overflow > 0:
            telemetry.inc("flow_overflow_chunks")
        telemetry.inc(f"closed_at_{chunk.closed_at_boundary.lower()}")
        telemetry.inc("aux_sidecars_count", len(chunk.aux_groups.get("sidecars", [])))
        telemetry.inc("aux_footnotes_count", len(chunk.aux_groups.get("footnotes", [])))
        telemetry.inc("aux_other_count", len(chunk.aux_groups.get("other", [])))
        if chunk.is_aux_only:
            telemetry.inc("aux_only_chunks")
            telemetry.inc("tokens_aux", chunk.token_count)
        elif chunk.is_main_only:
            telemetry.inc("main_chunks")
            telemetry.inc("tokens_main", chunk.token_count)


def _assert_invariants(
    chunks: Sequence[Chunk],
    sections: Sequence[Dict[str, object]],
    threads: Sequence[Dict[str, object]],
    block_lookup: Dict[str, Block],
    telemetry,
) -> None:
    thread_blocks: Dict[str, set[str]] = {}
    all_thread_blocks: set[str] = set()
    for thread in threads:
        block_ids = set(thread.get("blocks", []))
        thread_id = thread.get("thread_id")
        if isinstance(thread_id, str):
            thread_blocks[thread_id] = block_ids
        for block_id in block_ids:
            if block_id in all_thread_blocks:
                telemetry.inc("invariant_violations")
                telemetry.flag("I3_THREAD_DUPLICATE")
                break
            all_thread_blocks.add(block_id)
        for block_id in block_ids:
            block = block_lookup.get(block_id)
            if block and block.type in FLOAT_TYPES:
                telemetry.inc("invariant_violations")
                telemetry.flag("I4_FLOAT_IN_THREAD")
                break

    section_order: Dict[str, bool] = {}
    threaded_ids = set(all_thread_blocks)
    for chunk in chunks:
        section_id = chunk.section_id or chunk.segment_id or ""
        if chunk.is_aux_only:
            section_order[section_id] = True
        if chunk.is_main_only:
            seen_aux = section_order.get(section_id, False)
            if seen_aux:
                telemetry.inc("invariant_violations")
                telemetry.flag("I2_AUX_ORDER")
            for block_id in chunk.debug_block_ids:
                if block_id not in threaded_ids:
                    telemetry.inc("invariant_violations")
                    telemetry.flag("I1_NON_THREADED")
                    break


def _should_use_degraded(blocks: Sequence[Block]) -> bool:
    if not blocks:
        return False
    if any(block.type == "heading" for block in blocks):
        return False
    permitted_sources = {"extractor", "triage", "ocr"}
    return all(block.source.get("stage") in permitted_sources for block in blocks)


def _page_width(blocks: Sequence[Block]) -> float:
    widths = [block.bbox["x1"] for block in blocks if block.bbox]
    return max(widths) if widths else 1000.0
