"""Minimal smoke suite runner for the chunking pipeline."""
from __future__ import annotations

from typing import Dict

from pipeline.cli import parse_to_chunks


def run_smoke(doc_map: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    """Run the parser over a mapping of name -> pdf path, returning telemetry metrics."""
    results: Dict[str, Dict[str, int]] = {}
    for name, path in doc_map.items():
        output = parse_to_chunks(path)
        telemetry = output.telemetry
        results[name] = {
            "emitted_chunks": telemetry.emitted_chunks,
            "gate5_denies": telemetry.gate5_denies,
            "invariant_violations": telemetry.invariant_violations,
        }
    return results


__all__ = ["run_smoke"]
