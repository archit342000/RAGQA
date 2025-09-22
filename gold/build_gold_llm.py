"""Orchestrator for LLM-driven gold set construction."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import orjson

from . import quota
from .llm_synthesize import load_config, run_synthesis

logger = logging.getLogger(__name__)



def main() -> None:
    parser = argparse.ArgumentParser(description="Build a gold set using LLM synthesis")
    parser.add_argument("--parsed", type=Path, required=True, help="Directory of parsed document JSON files")
    parser.add_argument("--out", type=Path, required=True, help="Final gold JSONL output path")
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration file")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel workers for synthesis")
    parser.add_argument("--resume", action="store_true", help="Reuse cached generations when available")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = load_config(args.config)

    candidate_path = args.out.parent / "candidates.jsonl" if config.get("write_candidates") else None
    candidates, synth_stats = run_synthesis(
        args.parsed,
        config,
        args.concurrency,
        args.resume,
        out_path=candidate_path,
    )

    logger.info("Synthesized %d candidates", len(candidates))

    targets = config.get("wh_targets", {})
    seed = config.get("seed", 0)
    balanced = quota.enforce_wh_targets(candidates, targets, seed)

    logger.info("Final gold count after quotas: %d", len(balanced))

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        for item in balanced:
            fh.write(orjson.dumps({k: v for k, v in item.items() if k != "window_text"}))
            fh.write(b"\n")

    stats_path = out_path.parent / "stats.json"
    _write_stats(stats_path, candidates, balanced, synth_stats, targets)


def _write_stats(
    stats_path: Path,
    candidates: List[dict],
    balanced: List[dict],
    synth_stats: Dict,
    targets: Dict[str, float],
) -> None:
    candidate_wh = Counter(item.get("wh") for item in candidates if item.get("wh"))
    final_wh = Counter(item.get("wh") for item in balanced if item.get("wh"))
    candidate_types = Counter(item.get("type") for item in candidates if item.get("type"))
    final_types = Counter(item.get("type") for item in balanced if item.get("type"))
    per_doc = Counter(item.get("doc_id") for item in balanced)

    def _evidence_counter(items: List[dict]) -> Counter:
        counts: Counter[str] = Counter()
        for item in items:
            for entry in item.get("evidence", []) or []:
                if isinstance(entry, str):
                    if entry.strip():
                        counts["text"] += 1
                    continue
                if isinstance(entry, dict):
                    ev_type = entry.get("type")
                    if isinstance(ev_type, str):
                        ev_value = ev_type.strip().lower()
                        if ev_value:
                            counts[ev_value] += 1
        return counts

    candidate_evidence = _evidence_counter(candidates)
    final_evidence = _evidence_counter(balanced)

    quota_removed = {wh: candidate_wh.get(wh, 0) - final_wh.get(wh, 0) for wh in candidate_wh}

    synthesis_stats = dict(synth_stats)
    if "per_doc" in synthesis_stats:
        synthesis_stats["per_doc"] = {
            doc_id: dict(payload) for doc_id, payload in synthesis_stats["per_doc"].items()
        }
    for key in ("drop_reasons", "wh_counts", "type_counts", "evidence_type_counts"):
        if key in synthesis_stats:
            synthesis_stats[key] = dict(synthesis_stats[key])

    stats = {
        "total_candidates": len(candidates),
        "total_final": len(balanced),
        "wh_counts_candidates": dict(candidate_wh),
        "wh_counts_final": dict(final_wh),
        "type_counts_candidates": dict(candidate_types),
        "type_counts_final": dict(final_types),
        "evidence_type_counts_candidates": dict(candidate_evidence),
        "evidence_type_counts_final": dict(final_evidence),
        "per_doc_final": dict(per_doc),
        "drop_reasons": synthesis_stats.get("drop_reasons", {}),
        "quota": {
            "targets": targets,
            "removed": quota_removed,
        },
        "synthesis": synthesis_stats,
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_bytes(orjson.dumps(stats, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    main()
