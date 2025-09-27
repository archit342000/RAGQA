"""Command-line interface wiring together parsing and chunking."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List


from .column_order import reorder_page_lines
from .config import IngestConfig, load_config
from .ocr import ensure_ocr_layer
from .outputs import hash_file, write_json, write_jsonl
from .pdf_io import Line, compute_glyph_counts, extract_pages
from .chunker import build_chunks, build_paragraphs, estimate_tokens
from .sidecars import split_sidecars
from .table_detect import TableArtifact, detect_tables


class Budget:
    def __init__(self, seconds: float) -> None:
        self.limit = float(max(seconds, 0.0))
        self.start = time.perf_counter()
        self.exhausted_reason: str | None = None

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


def log_event(event: str, **payload: object) -> None:
    record = {"event": event, **payload}
    print(json.dumps(record, ensure_ascii=False))


def parse_and_chunk(
    pdf_path: Path,
    out_dir: Path,
    config: IngestConfig,
    *,
    mode: str = "fast",
    config_path: Path | None = None,
) -> Dict[str, object]:
    pdf_path = pdf_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_id = pdf_path.stem

    resolved_mode = mode.lower()
    config.mode = resolved_mode
    if resolved_mode not in {"fast", "thorough", "auto"}:
        raise ValueError(f"Unsupported mode: {mode}")

    with open(pdf_path, "rb") as handle:
        handle.seek(0, 2)
        file_size = handle.tell()
    log_event("ingest_start", doc_id=doc_id, mode=resolved_mode, bytes=file_size)

    try:
        import fitz  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("PyMuPDF (fitz) is required to parse PDFs") from exc

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
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = ensure_ocr_layer(pdf_path, Path(tmp_dir), timeout=budget.remaining)
            if tmp:
                working_pdf = Path(tmp)
                log_event("ocr_applied", pages=len(pages_needing_ocr))

    pages = extract_pages(working_pdf, config.max_pages, budget.check)
    if not pages:
        log_event("no_pages_extracted", doc_id=doc_id)

    body_pages: List[List[Line]] = []
    for page in pages:
        ordered = reorder_page_lines(page.lines)
        body_pages.append(ordered)

    body_pages, captions, footnotes = split_sidecars(body_pages)
    tables, skipped_tables = detect_tables(body_pages, out_dir, confidence_threshold=config.table_digit_ratio)
    for artifact in tables:
        for line in artifact.lines:
            try:
                body_pages[artifact.page_index].remove(line)
            except ValueError:
                continue

    paragraphs = build_paragraphs(body_pages)
    provenance_hash = hash_file(pdf_path)
    body_chunks, noise_ratio = build_chunks(doc_id, paragraphs, config, provenance_hash=provenance_hash)

    chunks_dicts = [chunk.to_dict() for chunk in body_chunks]

    def make_sidecar_chunk(line: Line, chunk_type: str, suffix: str) -> Dict[str, object]:  # type: ignore[name-defined]
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

    caption_chunks = [
        make_sidecar_chunk(line, "caption", f"C{idx:05d}")
        for idx, line in enumerate(captions)
    ]
    footnote_chunks = [
        make_sidecar_chunk(line, "footnote", f"F{idx:05d}")
        for idx, line in enumerate(footnotes)
    ]

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

    chunks_path = out_dir / "chunks.jsonl"
    stats_path = out_dir / "stats.json"
    config_path_out = out_dir / "config_used.json"

    write_jsonl(chunks_path, all_chunks)

    lines_total = sum(len(page.lines) for page in pages)
    body_token_values = [chunk["tokens_est"] for chunk in chunks_dicts if chunk["type"] == "body"]
    avg_tokens = float(sum(body_token_values) / max(len(body_token_values), 1)) if body_token_values else 0.0

    stats = {
        "parse_time_s": round(budget.elapsed, 3),
        "lines_total": lines_total,
        "tables_emitted": len([t for t in tables if t.csv_path]),
        "captions_extracted": len(captions),
        "chunks_n": len(all_chunks),
        "avg_tokens": round(avg_tokens, 2),
        "noise_ratio": round(noise_ratio, 3),
        "skipped_tables_n": skipped_tables,
    }
    if budget.exhausted_reason:
        stats["budget_exhausted"] = budget.exhausted_reason
    write_json(stats_path, stats)

    config_payload = config.to_serializable()
    config_payload["mode"] = resolved_mode
    if config_path:
        config_payload["config_source"] = str(config_path)
    write_json(config_path_out, config_payload)

    summary = {
        "doc_id": doc_id,
        "chunks": len(all_chunks),
        "tables": len(tables),
        "out_dir": str(out_dir),
    }
    log_event("ingest_complete", **summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse PDFs into retrieval-ready chunks")
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--mode", choices=["fast", "thorough", "auto"], default="fast")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(str(args.config) if args.config else None)
    summary = parse_and_chunk(args.input_pdf, args.outdir, config, mode=args.mode, config_path=args.config)
    log_event("cli_summary", **summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
