"""Command line interface for CPU-first parsing and chunking."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

try:
    import orjson
except ModuleNotFoundError:  # pragma: no cover
    orjson = None  # type: ignore

from chunking import Chunker
from parser import PDFParser, load_config
from parser.logging_utils import log_event


def _dump_json(data, path: Path) -> None:
    if orjson:
        path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(records: Iterable[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if orjson:
                handle.write(orjson.dumps(record).decode("utf-8"))
            else:
                handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse a PDF and emit retrieval-ready chunks.")
    parser.add_argument("input_pdf", type=str, help="Path to the PDF file")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for artifacts")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file")
    parser.add_argument(
        "--mode",
        type=str,
        default="fast",
        choices=["fast", "thorough", "auto"],
        help="Parsing mode controlling time budgets",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    pdf_parser = PDFParser(config)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"

    document = pdf_parser.parse(args.input_pdf, mode=args.mode, tables_out=tables_dir)

    chunker = Chunker(config)
    chunks = chunker.chunk_document(document)

    chunks_path = out_dir / "chunks.jsonl"
    stats_path = out_dir / "stats.json"
    config_path = out_dir / "config_used.json"

    chunk_dicts = [chunk.__dict__ for chunk in chunks]
    _write_jsonl(chunk_dicts, chunks_path)
    _dump_json(document.config_used, config_path)

    stats = _collect_stats(document, chunks)
    _dump_json(stats, stats_path)

    summary = {
        "doc_id": document.doc_id,
        "chunks": len(chunks),
        "tables": stats.get("tables_emitted"),
        "out_dir": str(out_dir),
    }
    log_event("parse_and_chunk_complete", **summary)
    return 0


def _collect_stats(document, chunks) -> dict:
    lines_total, captions_total = document.stats_summary()
    tables_emitted = sum(
        1
        for page in document.pages
        for table in page.tables
        if table.csv_path
    )
    skipped_tables = sum(
        1
        for page in document.pages
        for table in page.tables
        if not table.csv_path
    )
    avg_tokens = 0.0
    if chunks:
        avg_tokens = sum(chunk.tokens_est for chunk in chunks if chunk.type == "body") / max(
            1, sum(1 for chunk in chunks if chunk.type == "body")
        )
    noise_ratio = 0.0
    if document.pages:
        noise_ratio = sum(page.noise_ratio for page in document.pages) / len(document.pages)
    return {
        "parse_time_s": document.parse_time_s,
        "lines_total": lines_total,
        "tables_emitted": tables_emitted,
        "captions_extracted": captions_total,
        "chunks_n": len(chunks),
        "avg_tokens": avg_tokens,
        "noise_ratio": noise_ratio,
        "skipped_tables_n": skipped_tables,
    }


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
