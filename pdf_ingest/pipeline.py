"""Shared ingestion helpers used by the CLI and the Gradio UI."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .column_order import reorder_page_lines
from .config import IngestConfig
from .ocr import ensure_ocr_layer
from .outputs import hash_file, write_json, write_jsonl
from .pdf_io import Line, compute_glyph_counts, extract_pages
from .chunker import build_chunks, build_paragraphs, estimate_tokens
from .sidecars import split_sidecars
from .table_detect import TableArtifact, detect_tables


@dataclass
class Budget:
    """Simple stopwatch tracking the configured time allowance."""

    limit: float
    start: float = field(default_factory=time.perf_counter)
    exhausted_reason: str | None = None

    def check(self, reason: str) -> bool:
        if self.elapsed > self.limit:
            if not self.exhausted_reason:
                self.exhausted_reason = reason
            return True
        return False

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.elapsed)


@dataclass
class PagePayload:
    index: int
    glyphs: int
    text: str


@dataclass
class IngestResult:
    doc_id: str
    doc_name: str
    mode: str
    pages: List[PagePayload]
    chunks: List[Dict[str, object]]
    captions: List[Dict[str, object]]
    footnotes: List[Dict[str, object]]
    tables: List[TableArtifact]
    stats: Dict[str, object]
    config_payload: Dict[str, object]
    events: List[Dict[str, object]]
    provenance_hash: str
    working_pdf: Path
    skipped_tables: int


def _log(events: List[Dict[str, object]], event: str, **payload: object) -> None:
    record = {"event": event, **payload}
    events.append(record)


def run_pipeline(
    pdf_path: Path,
    out_dir: Path,
    config: IngestConfig,
    *,
    mode: str = "fast",
) -> IngestResult:
    """Execute parsing, optional OCR, chunking, and table export."""

    pdf_path = pdf_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    events: List[Dict[str, object]] = []

    doc_id = pdf_path.stem
    doc_name = pdf_path.name

    resolved_mode = mode.lower()
    if resolved_mode not in {"fast", "thorough", "auto"}:
        raise ValueError(f"Unsupported mode: {mode}")

    try:
        import fitz  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("PyMuPDF (fitz) is required to parse PDFs") from exc

    with open(pdf_path, "rb") as handle:
        handle.seek(0, 2)
        file_size = handle.tell()
    _log(events, "ingest_start", doc_id=doc_id, mode=resolved_mode, bytes=file_size)

    with fitz.open(pdf_path) as doc:
        page_count = len(doc)

    if resolved_mode == "auto" and page_count <= 50:
        resolved_mode = "thorough"

    config.mode = resolved_mode
    budget = Budget(config.budget_for_mode(resolved_mode))

    glyph_counts = compute_glyph_counts(pdf_path, config.max_pages, budget.check)
    pages_needing_ocr = [idx for idx, count in enumerate(glyph_counts) if count < config.glyph_min_for_text_page]

    working_pdf = pdf_path
    if pages_needing_ocr and budget.remaining > 5:
        temp_dir = out_dir / "_ocr"
        temp_dir.mkdir(parents=True, exist_ok=True)
        candidate = ensure_ocr_layer(pdf_path, temp_dir, timeout=budget.remaining)
        if candidate:
            working_pdf = Path(candidate)
            _log(events, "ocr_applied", pages=len(pages_needing_ocr))

    pages = extract_pages(working_pdf, config.max_pages, budget.check)
    body_pages: List[List[Line]] = []
    page_payloads: List[PagePayload] = []
    for page in pages:
        ordered = reorder_page_lines(page.lines)
        body_pages.append(ordered)
        page_text = "\n".join(line.text for line in ordered)
        page_payloads.append(PagePayload(index=page.index, glyphs=page.glyphs, text=page_text))

    body_pages, captions_lines, footnote_lines = split_sidecars(body_pages)
    tables, skipped_tables = detect_tables(body_pages, out_dir, confidence_threshold=config.table_digit_ratio)
    for artifact in tables:
        for line in artifact.lines:
            try:
                body_pages[artifact.page_index].remove(line)
            except ValueError:
                continue

    paragraphs = build_paragraphs(body_pages)
    provenance_hash = hash_file(working_pdf)
    body_chunks, noise_ratio = build_chunks(doc_id, paragraphs, config, provenance_hash=provenance_hash)

    def _make_sidecar_chunk(line: Line, chunk_type: str, suffix: str) -> Dict[str, object]:
        page_span = [line.page_index + 1, [line.line_index + 1, line.line_index + 1]]
        return {
            "doc_id": doc_id,
            "chunk_id": suffix,
            "type": chunk_type,
            "text": line.text,
            "tokens_est": estimate_tokens(line.text),
            "page_spans": [page_span],
            "section_hints": [],
            "neighbors": {"prev": None, "next": None},
            "table_csv": None,
            "evidence_offsets": [],
            "provenance": {"hash": provenance_hash, "byte_range": None},
        }

    chunks_dicts = [chunk.to_dict() for chunk in body_chunks]
    caption_chunks = [_make_sidecar_chunk(line, "caption", f"C{idx:05d}") for idx, line in enumerate(captions_lines)]
    footnote_chunks = [_make_sidecar_chunk(line, "footnote", f"F{idx:05d}") for idx, line in enumerate(footnote_lines)]

    table_chunks: List[Dict[str, object]] = []
    for idx, artifact in enumerate(tables):
        csv_ref = None
        if artifact.csv_path:
            csv_ref = f"{artifact.csv_path}#{artifact.rows},{artifact.cols}"
        table_chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"T{idx:05d}",
                "type": "table",
                "text": "",
                "tokens_est": 0,
                "page_spans": [[artifact.page_index + 1, None]],
                "section_hints": [],
                "neighbors": {"prev": None, "next": None},
                "table_csv": csv_ref,
                "evidence_offsets": [],
                "provenance": {"hash": provenance_hash, "byte_range": None},
            }
        )

    all_chunks = chunks_dicts + caption_chunks + footnote_chunks + table_chunks

    lines_total = sum(len(page.lines) for page in pages)
    body_token_values = [chunk["tokens_est"] for chunk in chunks_dicts if chunk["type"] == "body"]
    avg_tokens = float(sum(body_token_values) / max(len(body_token_values), 1)) if body_token_values else 0.0

    stats = {
        "parse_time_s": round(budget.elapsed, 3),
        "lines_total": lines_total,
        "tables_emitted": len([t for t in tables if t.csv_path]),
        "captions_extracted": len(captions_lines),
        "chunks_n": len(all_chunks),
        "avg_tokens": round(avg_tokens, 2),
        "noise_ratio": round(noise_ratio, 3),
        "skipped_tables_n": skipped_tables,
    }
    if budget.exhausted_reason:
        stats["budget_exhausted"] = budget.exhausted_reason

    config_payload = config.to_serializable()
    config_payload["mode"] = resolved_mode

    _log(events, "ingest_complete", doc_id=doc_id, mode=resolved_mode, chunks=len(all_chunks))

    return IngestResult(
        doc_id=doc_id,
        doc_name=doc_name,
        mode=resolved_mode,
        pages=page_payloads,
        chunks=all_chunks,
        captions=caption_chunks,
        footnotes=footnote_chunks,
        tables=tables,
        stats=stats,
        config_payload=config_payload,
        events=events,
        provenance_hash=provenance_hash,
        working_pdf=working_pdf,
        skipped_tables=skipped_tables,
    )


def write_artifacts(result: IngestResult, out_dir: Path, *, config_source: Path | None = None) -> None:
    """Persist chunks, stats, and resolved configuration to disk."""

    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "chunks.jsonl"
    stats_path = out_dir / "stats.json"
    config_path = out_dir / "config_used.json"

    write_jsonl(chunks_path, result.chunks)
    payload = dict(result.stats)
    write_json(stats_path, payload)
    config_payload = dict(result.config_payload)
    if config_source is not None:
        config_payload["config_source"] = str(config_source)
    write_json(config_path, config_payload)
