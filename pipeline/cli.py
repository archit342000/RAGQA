from __future__ import annotations

from pathlib import Path

import typer

from .config import PipelineConfig
from .service import parse_to_chunks

app = typer.Typer(help="Parse PDFs into Blocks JSON and Chunks JSONL")


@app.command()
def parse_to_chunks_cmd(input_pdf: Path, out_dir: Path = typer.Argument(...)) -> None:
    """Run the pipeline on ``input_pdf`` and store outputs under ``out_dir``."""

    config = PipelineConfig.from_mapping({})
    result = parse_to_chunks(str(input_pdf), out_dir, config)
    typer.echo(f"Processed {result.doc_name} -> {out_dir}")


def main() -> None:  # pragma: no cover - CLI entrypoint
    app()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
