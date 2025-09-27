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
from .ocr import (
    OCRDecision,
    classify_pages,
    render_page_to_image,
    run_tesseract_hocr,
    run_tesseract_tsv,
)
from .outputs import append_jsonl, hash_file, write_json
from .pdf_io import Line, collect_page_lines, load_document_payload, open_document
from .progress import ProgressTracker
from .sidecars import split_sidecars
from .table_detect import detect_tables_for_page, write_table_csv
from .tsv import hocr_to_lines, merge_lines, tsv_to_lines


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

    original_doc = open_document(pdf_path)

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
    counters.setdefault("tsv_empty_alerts", 0)
    body_token_accumulator: List[int] = []
    last_record: Optional[Dict[str, object]] = None
    last_owner: Optional[int] = None
    pending_chunk_ids: Dict[int, List[str]] = {}
    written_chunk_ids: Dict[int, List[str]] = {}

    caption_counter = 0
    footnote_counter = 0
    table_skips = 0
    noise_dropped = counters.get("noise_dropped", 0)

    for page_index in range(total_pages):
        if progress.state.pages[page_index].status == "done":
            continue
        decision = decisions[page_index] if page_index < len(decisions) else OCRDecision(page_index, "none")
        progress.page_started(page_index, ocr_mode=decision.mode)

        page = original_doc.load_page(page_index)
        native_lines, _ = collect_page_lines(page)

        ocr_lines: List[Line] = []
        dpi_used: Optional[int] = None
        alerts: List[str] = []
        tsv_line_count = 0

        if decision.mode in {"partial", "full"}:
            attempts = 0
            rasterisations = 0
            while attempts <= config.ocr_retry and rasterisations < config.rasterizations_per_page:
                dpi = config.dpi_full if attempts == 0 else config.dpi_retry
                rasterisations += 1
                logger.emit("render_start", page=page_index + 1, dpi=dpi, attempt=attempts + 1)
                image_path, width, height = render_page_to_image(page, dpi)
                logger.emit("render_stop", page=page_index + 1, dpi=dpi, width=width, height=height)
                tsv_path: Optional[Path] = None
                hocr_path: Optional[Path] = None
                try:
                    logger.emit("tesseract_start", page=page_index + 1, dpi=dpi, attempt=attempts + 1)
                    tsv_path = run_tesseract_tsv(image_path, dpi=dpi)
                    ocr_lines = tsv_to_lines(
                        page_index,
                        tsv_path,
                        pdf_width=page.rect.width,
                        pdf_height=page.rect.height,
                        image_width=width,
                        image_height=height,
                        dpi=dpi,
                    )
                    logger.emit("tesseract_stop", page=page_index + 1, dpi=dpi, lines=len(ocr_lines))
                    dpi_used = dpi
                    tsv_line_count = len(ocr_lines)
                    if ocr_lines:
                        break
                    alerts.append("tsv_empty")
                    counters["tsv_empty_alerts"] = counters.get("tsv_empty_alerts", 0) + 1
                    logger.emit("tesseract_empty", page=page_index + 1, dpi=dpi, attempt=attempts + 1)
                    hocr_path = run_tesseract_hocr(image_path, dpi=dpi)
                    hocr_lines = hocr_to_lines(
                        page_index,
                        hocr_path,
                        pdf_width=page.rect.width,
                        pdf_height=page.rect.height,
                        image_width=width,
                        image_height=height,
                    )
                    if hocr_lines:
                        alerts.append("hocr_fallback")
                        ocr_lines = hocr_lines
                        tsv_line_count = len(ocr_lines)
                        logger.emit(
                            "hocr_fallback",
                            page=page_index + 1,
                            dpi=dpi,
                            attempt=attempts + 1,
                            lines=len(hocr_lines),
                        )
                        break
                finally:
                    image_path.unlink(missing_ok=True)
                    if tsv_path is not None:
                        tsv_path.unlink(missing_ok=True)
                    if hocr_path is not None:
                        hocr_path.unlink(missing_ok=True)
                attempts += 1
            if not ocr_lines and decision.mode == "full":
                alerts.append("ocr_no_text")

        merged_lines = merge_lines(native_lines, ocr_lines)
        counters["lines_total"] = counters.get("lines_total", 0) + len(merged_lines)

        ordered_lines = reorder_page_lines(merged_lines)
        for idx, line in enumerate(ordered_lines):
            line.line_index = idx
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
                cols = len(candidate.rows[0]) if candidate.rows else 0
                summary = (
                    f"Table detected on page {page_index + 1} with {len(candidate.rows)} rows and {cols} columns. "
                    f"CSV stored at {table_path}."
                )
                table_chunk = _chunk_record(
                    pdf_path.stem,
                    Chunk(
                        chunk_id=chunk_id,
                        text=summary,
                        page_spans=[[page_index + 1, None]],
                        tokens=estimate_tokens(summary),
                        type="table",
                        table_csv=f"{table_path}#{len(candidate.rows)},{cols}",
                        evidence_offsets=[],
                        neighbors={"prev": None, "next": None},
                    ),
                    provenance_hash,
                )
                table_chunk["evidence_offsets"] = [
                    [round(coord, 2) for coord in bbox] if bbox else None for bbox in candidate.bbox_rows
                ]
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
            notes=None,
            counters={
                "lines_total": counters.get("lines_total", 0),
                "tables_emitted": counters.get("tables_emitted", 0),
                "captions_extracted": counters.get("captions_extracted", 0),
                "skipped_tables": table_skips,
                "noise_dropped": noise_dropped,
                "tsv_empty_alerts": counters.get("tsv_empty_alerts", 0),
            },
            pending_paragraphs=builder_state["pending_paragraphs"],
            last_chunk_id=builder_state["last_chunk_id"],
            dpi_used=dpi_used,
            tsv_lines=tsv_line_count,
            alerts=alerts,
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

    original_doc.close()
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
        "tsv_empty_alerts": counters.get("tsv_empty_alerts", 0),
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

