"""Command-line entrypoint for the pdfchunks pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import fitz  # type: ignore

from .audit.guards import run_audits
from .chunking.chunker import Chunker
from .config import load_config
from .parsing.baselines import BaselineEstimator
from .parsing.block_extractor import BlockExtractor
from .parsing.classifier import BlockClassifier
from .parsing.ownership import assign_aux_ownership, assign_sections
from .serialize.serializer import Serializer
from .telemetry.metrics import compute_metrics
from .threading.threader import ThreadResult, Threader

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-fidelity PDF to chunk pipeline")
    parser.add_argument("pdf", type=Path, help="Path to the PDF document")
    parser.add_argument("--config", type=Path, default=None, help="Optional configuration YAML")
    parser.add_argument("--json", action="store_true", help="Emit JSON with metrics and chunk counts")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default: INFO)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    config = load_config(args.config) if args.config else load_config()

    with fitz.open(args.pdf) as document:
        doc_id = Path(args.pdf).stem
        extractor = BlockExtractor(config.block_extraction)
        layout = extractor.extract(document, doc_id=doc_id)
        baselines = BaselineEstimator(config.baselines).fit(layout)
        classifier = BlockClassifier(config.classifier)
        labels = classifier.classify(layout.blocks, baselines)
        assignments = assign_aux_ownership(assign_sections(labels))
        threader = Threader(config.threading)
        thread_result = threader.thread(assignments, doc_id=doc_id)
        serializer = Serializer()
        ordered_units = serializer.serialize(thread_result.units)
        serialized_result = ThreadResult(units=ordered_units, delayed_aux_counts=thread_result.delayed_aux_counts)
        chunker = Chunker(config.chunker)
        chunks = chunker.chunk(ordered_units)
        run_audits(serialized_result, chunks, config.audits)
        metrics = compute_metrics(serialized_result, chunks)

    if args.json:
        output: Dict[str, Any] = {
            "main_units": metrics.main_units,
            "aux_units": metrics.aux_units,
            "aux_sections": metrics.aux_sections,
            "chunk_counts": metrics.chunk_counts,
        }
        print(json.dumps(output, indent=2))
    else:
        LOGGER.info(
            "Processed doc_id=%s | MAIN units=%s | AUX units=%s | chunks=%s",
            doc_id,
            metrics.main_units,
            metrics.aux_units,
            metrics.chunk_counts,
        )


if __name__ == "__main__":  # pragma: no cover
    main()

