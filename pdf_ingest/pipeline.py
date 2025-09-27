"""Deterministic streaming pipeline for PDF parsing and chunking."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .chunker import ChunkBuilder, Chunk, paragraphs_from_lines, estimate_tokens
from .column_order import reorder_page_lines
from .config import Config
from .logging import EventLogger
from .ocr import OCRDecision, classify_pages, pytesseract_page_image, run_ocrmypdf
from .outputs import append_jsonl, hash_file, write_json
from .pdf_io import Line, collect_page_lines, load_document_payload, open_document
from .progress import ProgressTracker
from .sidecars import split_sidecars
from .table_detect import detect_tables_for_page, write_table_csv


@dataclass
class PipelineResult:
    doc_id: str
    doc_name: str
    out_dir: Path
    stats: Dict[str, Any]
    config: Dict[str, Any]
    progress: Dict[str, Any]
    chunks: List[Dict[str, Any]]


def _chunk_record(doc_id: str, chunk: Chunk, provenance_hash: str) -> Dict[str, object]:
    return {
        "doc_id": doc_id,
        "chunk_id": chunk.chunk_id,
        "type": chunk.type,
        "text": chunk.text,
        "tokens_est": chunk.tokens,
        "page_spans": chunk.page_spans,
        "section_hints": [],
        "neighbors": chunk.neighbors,
        "table_csv": chunk.table_csv,
        "evidence_offsets": chunk.evidence_offsets,
        "provenance": {"hash": provenance_hash, "byte_range": None},
    }


def _sidecar_record(doc_id: str, *, chunk_id: str, chunk_type: str, text: str, page: int, line_index: int, provenance_hash: str) -> Dict[str, object]:
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "type": chunk_type,
        "text": text,
        "tokens_est": estimate_tokens(text),
        "page_spans": [[page + 1, [line_index + 1, line_index + 1]]],
        "section_hints": [],
        "neighbors": {"prev": None, "next": None},
        "table_csv": None,
        "evidence_offsets": [],
        "provenance": {"hash": provenance_hash, "byte_range": None},
    }


def run_pipeline(pdf_path: Path, out_dir: Path, config: Config) -> PipelineResult:
    start = time.perf_counter()
    pdf_path = pdf_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_hash = hash_file(pdf_path)

    with open_document(pdf_path) as doc:
        total_pages = min(len(doc), config.max_pages)

    logger = EventLogger(out_dir / "parse.log")
    logger.emit("parse_started", path=str(pdf_path), mode=config.mode, pages=total_pages)

    progress = ProgressTracker(
        out_dir / "progress.json",
        doc_id=pdf_path.stem,
        input_pdf=pdf_path,
        mode=config.mode,
        pages_total=total_pages,
    )

    _, signals = load_document_payload(pdf_path, max_pages=total_pages)
    decisions = classify_pages(signals, config)

    full_pages = [decision.page_index for decision in decisions if decision.mode == "full"]
    ocr_pdf_path: Optional[Path] = None
    if full_pages:
        ocr_pdf_path = out_dir / "_ocr" / "full.pdf"
        run_ocrmypdf(pdf_path, ocr_pdf_path)
        logger.emit("ocr_full", pages=full_pages, output=str(ocr_pdf_path))

    original_doc = open_document(pdf_path)
    ocr_doc = open_document(ocr_pdf_path) if ocr_pdf_path else None

    builder = ChunkBuilder(
        config,
        next_index=progress.state.next_chunk_index,
        last_chunk_id=progress.state.last_chunk_id,
        pending=progress.state.pending_paragraphs,
    )

    chunk_path = out_dir / "chunks.jsonl"
    if progress.state.next_chunk_index == 0:
        chunk_path.unlink(missing_ok=True)

    counters = progress.state.counters.copy()
    body_token_accumulator: List[int] = []
    last_record: Optional[Dict[str, object]] = None
    last_owner: Optional[int] = None
    pending_chunk_ids: Dict[int, List[str]] = {}
    written_chunk_ids: Dict[int, List[str]] = {}

    caption_counter = 0
    footnote_counter = 0
    table_skips = 0
    noise_dropped = counters.get("noise_dropped", 0)
    ocr_retries = {decision.page_index: 0 for decision in decisions}

    for page_index in range(total_pages):
        if progress.state.pages[page_index].status == "done":
            continue
        decision = decisions[page_index] if page_index < len(decisions) else OCRDecision(page_index, "none")
        progress.page_started(page_index, ocr_mode=decision.mode)

        page_doc = ocr_doc if decision.mode == "full" and ocr_doc is not None else original_doc
        page = page_doc.load_page(page_index)
        lines, glyphs = collect_page_lines(page)
        counters["lines_total"] = counters.get("lines_total", 0) + len(lines)

        notes: List[str] = []
        if decision.mode == "partial" and ocr_retries[page_index] < config.ocr_retry:
            additional_lines = pytesseract_page_image(page)
            if additional_lines:
                base_index = len(lines)
                for offset, text in enumerate(additional_lines):
                    lines.append(Line(page_index=page_index, line_index=base_index + offset, text=text, bbox=None, x_center=None))
                notes.append("partial_ocr")
            ocr_retries[page_index] += 1
        elif decision.mode == "partial" and ocr_retries[page_index] >= config.ocr_retry:
            notes.append("ocr_retry_exhausted")

        ordered_lines = reorder_page_lines(lines)
        sidecars = split_sidecars(ordered_lines)
        counters["captions_extracted"] = counters.get("captions_extracted", 0) + len(sidecars.captions)

        paragraphs, dropped = paragraphs_from_lines(sidecars.body, config=config)
        noise_dropped += dropped

        page_chunk_ids: List[str] = written_chunk_ids.pop(page_index, []) + list(pending_chunk_ids.get(page_index, []))
        page_tables: List[str] = []

        for paragraph in paragraphs:
            emitted = builder.add_paragraph(paragraph)
            for chunk in emitted:
                record = _chunk_record(pdf_path.stem, chunk, provenance_hash)
                if last_record is not None and last_owner is not None:
                    last_record["neighbors"]["next"] = record["chunk_id"]
                    append_jsonl(chunk_path, last_record)
                    body_token_accumulator.append(int(last_record["tokens_est"]))
                    written_chunk_ids.setdefault(last_owner, []).append(last_record["chunk_id"])
                    pending_ids = pending_chunk_ids.get(last_owner, [])
                    if last_record["chunk_id"] in pending_ids:
                        pending_ids.remove(last_record["chunk_id"])
                        if not pending_ids:
                            pending_chunk_ids.pop(last_owner, None)
                owner = chunk.page_spans[0][0] - 1 if chunk.page_spans else page_index
                pending_chunk_ids.setdefault(owner, []).append(record["chunk_id"])
                last_record = record
                last_owner = owner

        table_candidates = detect_tables_for_page(sidecars.body, page_index=page_index)
        for candidate in table_candidates:
            if candidate.confidence >= config.table_score_conf and candidate.digit_ratio >= config.table_digit_ratio:
                table_path = write_table_csv(candidate, out_dir)
                chunk_id = f"T{page_index:04d}_{len(page_tables)}"
                table_chunk = _chunk_record(
                    pdf_path.stem,
                    Chunk(
                        chunk_id=chunk_id,
                        text=f"Table with {len(candidate.rows)} rows",
                        page_spans=[[page_index + 1, None]],
                        tokens=0,
                        type="table",
                        table_csv=f"{table_path}#{len(candidate.rows)},{len(candidate.rows[0]) if candidate.rows else 0}",
                        evidence_offsets=[],
                        neighbors={"prev": None, "next": None},
                    ),
                    provenance_hash,
                )
                append_jsonl(chunk_path, table_chunk)
                page_chunk_ids.append(chunk_id)
                page_tables.append(table_path)
                counters["tables_emitted"] = counters.get("tables_emitted", 0) + 1
            else:
                table_skips += 1

        for caption in sidecars.captions:
            chunk_id = f"C{page_index:04d}_{caption_counter}"
            caption_counter += 1
            record = _sidecar_record(pdf_path.stem, chunk_id=chunk_id, chunk_type="caption", text=caption.text, page=caption.page_index, line_index=caption.line_index, provenance_hash=provenance_hash)
            append_jsonl(chunk_path, record)
            page_chunk_ids.append(chunk_id)

        for footnote in sidecars.footnotes:
            chunk_id = f"F{page_index:04d}_{footnote_counter}"
            footnote_counter += 1
            record = _sidecar_record(pdf_path.stem, chunk_id=chunk_id, chunk_type="footnote", text=footnote.text, page=footnote.page_index, line_index=footnote.line_index, provenance_hash=provenance_hash)
            append_jsonl(chunk_path, record)
            page_chunk_ids.append(chunk_id)

        builder_state = builder.state_payload()
        progress.page_completed(
            page_index,
            chunks=page_chunk_ids,
            tables=page_tables,
            notes=notes,
            counters={
                "lines_total": counters.get("lines_total", 0),
                "tables_emitted": counters.get("tables_emitted", 0),
                "captions_extracted": counters.get("captions_extracted", 0),
                "skipped_tables": table_skips,
                "noise_dropped": noise_dropped,
            },
            pending_paragraphs=builder_state["pending_paragraphs"],
            last_chunk_id=builder_state["last_chunk_id"],
        )
        progress.update_next_chunk_index(builder_state["next_chunk_index"])

    remaining = builder.finalize()
    for chunk in remaining:
        record = _chunk_record(pdf_path.stem, chunk, provenance_hash)
        if last_record is not None and last_owner is not None:
            last_record["neighbors"]["next"] = record["chunk_id"]
            append_jsonl(chunk_path, last_record)
            body_token_accumulator.append(int(last_record["tokens_est"]))
            written_chunk_ids.setdefault(last_owner, []).append(last_record["chunk_id"])
            pending_ids = pending_chunk_ids.get(last_owner, [])
            if last_record["chunk_id"] in pending_ids:
                pending_ids.remove(last_record["chunk_id"])
                if not pending_ids:
                    pending_chunk_ids.pop(last_owner, None)
        owner = chunk.page_spans[0][0] - 1 if chunk.page_spans else (last_owner if last_owner is not None else 0)
        pending_chunk_ids.setdefault(owner, []).append(record["chunk_id"])
        last_record = record
        last_owner = owner

    if last_record:
        append_jsonl(chunk_path, last_record)
        body_token_accumulator.append(int(last_record["tokens_est"]))
        if last_owner is not None:
            written_chunk_ids.setdefault(last_owner, []).append(last_record["chunk_id"])
            pending_ids = pending_chunk_ids.get(last_owner, [])
            if last_record["chunk_id"] in pending_ids:
                pending_ids.remove(last_record["chunk_id"])
                if not pending_ids:
                    pending_chunk_ids.pop(last_owner, None)

    elapsed = time.perf_counter() - start
    body_chunks = len(body_token_accumulator)
    avg_tokens = sum(body_token_accumulator) / body_chunks if body_chunks else 0
    stats = {
        "parse_time_s": round(elapsed, 3),
        "lines_total": counters.get("lines_total", 0),
        "tables_emitted": counters.get("tables_emitted", 0),
        "captions_extracted": counters.get("captions_extracted", 0),
        "chunks_n": progress.state.next_chunk_index
        + caption_counter
        + footnote_counter
        + counters.get("tables_emitted", 0),
        "avg_tokens": round(avg_tokens, 2) if avg_tokens else 0,
        "noise_ratio": round(noise_dropped / max(counters.get("lines_total", 1), 1), 4),
        "skipped_tables_n": table_skips,
    }
    write_json(out_dir / "stats.json", stats)
    config.write(out_dir / "config_used.json")
    progress.mark_completed()
    logger.emit("parse_completed", duration_s=stats["parse_time_s"], chunks=stats["chunks_n"])

    chunks: List[Dict[str, Any]] = []
    with chunk_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))

    return PipelineResult(
        doc_id=pdf_path.stem,
        doc_name=pdf_path.name,
        out_dir=out_dir,
        stats=stats,
        config=config.to_dict(),
        progress=progress.state.to_dict(),
        chunks=chunks,
    )

