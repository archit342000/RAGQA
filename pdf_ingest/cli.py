"""Command-line entry point for the deterministic parser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .config import Config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse PDFs into retrieval-ready chunks")
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Optional JSON config override")
    parser.add_argument("--mode", choices=["fast", "thorough"], default="fast")
    return parser


def parse_and_chunk(input_pdf: Path, outdir: Path, *, config: Config, mode: str) -> Dict[str, Any]:
    config.mode = mode
    result = run_pipeline(input_pdf, outdir, config)
    summary = {"event": "cli_summary", "path": str(input_pdf), "outdir": str(outdir), **result.stats}
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = str(args.config) if args.config else None
    config = Config.from_sources(json_path=config_path)
    parse_and_chunk(args.input_pdf, args.outdir, config=config, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
