"""Command-line entry point for the ingestion pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .config import IngestConfig, load_config
from .pipeline import IngestResult, run_pipeline, write_artifacts


def _print_events(result: IngestResult) -> None:
    for event in result.events:
        print(json.dumps(event, ensure_ascii=False))


def parse_and_chunk(
    pdf_path: Path,
    out_dir: Path,
    config: IngestConfig,
    *,
    mode: str = "fast",
    config_path: Path | None = None,
) -> dict:
    """Run the ingestion pipeline and persist JSONL/CSV artifacts."""

    result = run_pipeline(pdf_path, out_dir, config, mode=mode)
    _print_events(result)
    write_artifacts(result, out_dir, config_source=config_path)
    summary = {
        "doc_id": result.doc_id,
        "chunks": len(result.chunks),
        "tables": len(result.tables),
        "out_dir": str(out_dir),
        "stats": result.stats,
    }
    print(json.dumps({"event": "cli_summary", **summary}, ensure_ascii=False))
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
    parse_and_chunk(args.input_pdf, args.outdir, config, mode=args.mode, config_path=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
