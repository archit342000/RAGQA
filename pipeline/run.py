"""Command-line driver for the hybrid PDF parsing pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Sequence

from pipeline.chunking.chunker import DocumentChunker
from pipeline.ingest.pdf_parser import parse_pdf_with_pymupdf
from pipeline.layout.lp_fuser import LayoutParserEngine, fuse_layout
from pipeline.layout.router import plan_layout_routing
from pipeline.layout.signals import compute_layout_signals
from pipeline.repair.repair_pass import run_repair_pass
from pipeline.threading.threader import Threader
from pipeline.telemetry.metrics import TelemetryCollector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:  # Optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


def _load_embedder(model_name: str | None) -> Callable[[Sequence[str]], Sequence[Sequence[float]]] | None:
    if SentenceTransformer is None:
        logger.warning("sentence-transformers unavailable; repair and semantic chunking degrade")
        return None
    name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    model = SentenceTransformer(name)

    def _embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors = model.encode(list(texts), normalize_embeddings=False)
        return [list(map(float, vector)) for vector in vectors]

    return _embed


def _write_chunks(chunks: Sequence, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "tokens": chunk.tokens,
                "metadata": chunk.metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid PDF parsing pipeline")
    parser.add_argument("--pdf", required=True, help="Path to the PDF to parse")
    parser.add_argument("--emit-chunks", help="Where to write chunk JSONL output")
    parser.add_argument("--embedding-model", help="Override embedding model name")
    parser.add_argument("--dpi", type=int, default=180, help="DPI used for LayoutParser rendering")
    args = parser.parse_args(argv)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return 1

    logger.info("Parsing %s", pdf_path.name)
    document = parse_pdf_with_pymupdf(str(pdf_path))
    signals = compute_layout_signals(document)

    telemetry = TelemetryCollector()
    threader = Threader()

    repair_failures: dict[int, int] = {}
    plan = plan_layout_routing(document, signals, repair_failures=repair_failures, dpi=args.dpi)
    telemetry.record_router(plan, signals)

    engine = LayoutParserEngine()
    start = time.perf_counter()
    fused = fuse_layout(document, signals, plan, engine=engine)
    elapsed = time.perf_counter() - start
    if plan.selected_pages:
        telemetry.record_lp_latency(elapsed / len(plan.selected_pages))

    embedder = _load_embedder(args.embedding_model)
    fused, repair_stats, repair_failures = run_repair_pass(fused, signals, embedder=embedder, previous_failures=repair_failures)
    total_blocks = sum(len(page.main_flow) for page in fused.pages)
    telemetry.record_repair(repair_stats, total_blocks)
    fused, threading_report = threader.thread_document(document, fused, signals)
    telemetry.record_threading(threading_report)

    if any(count >= 2 for count in repair_failures.values()):
        logger.info("Re-running routing after repair failures")
        plan = plan_layout_routing(document, signals, repair_failures=repair_failures, dpi=args.dpi)
        telemetry.record_router(plan, signals)
        fused = fuse_layout(document, signals, plan, engine=engine)
        fused, repair_stats, repair_failures = run_repair_pass(fused, signals, embedder=embedder, previous_failures=repair_failures)
        total_blocks = sum(len(page.main_flow) for page in fused.pages)
        telemetry.record_repair(repair_stats, total_blocks)
        fused, threading_report = threader.thread_document(document, fused, signals)
        telemetry.record_threading(threading_report)

    chunker = DocumentChunker(embedder=embedder)
    chunks = chunker.chunk_document(fused, signals)
    logger.info("Generated %d chunks", len(chunks))

    if args.emit_chunks:
        output_path = Path(args.emit_chunks)
        _write_chunks(chunks, output_path)
        logger.info("Chunks written to %s", output_path)

    telemetry.log_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
