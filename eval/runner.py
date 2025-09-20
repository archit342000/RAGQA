"""Offline retrieval evaluation harness (Standard v1.1)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from retrieval.driver import RetrievalConfig, build_indexes, retrieve

from .metrics import (
    context_precision,
    hit_at_k,
    latency_percentiles,
    mrr_at_k,
    ndcg_at_k,
)
from .types import EvalConfig, GoldItem, ItemMetrics, RunResult
from .utils import (
    build_retrieval_config,
    checksum,
    ensure_dir,
    load_chunks,
    load_gold,
    load_yaml,
    set_seeds,
)


def _make_retrieval_config(base: EvalConfig, engine: str) -> RetrievalConfig:
    cfg = build_retrieval_config(base)
    cfg.default_engine = engine  # type: ignore[attr-defined]
    return cfg


def _response_candidates(payload: Dict[str, object], key: str) -> List[Dict[str, object]]:
    entries = payload.get(key, [])
    return [dict(entry) for entry in entries if isinstance(entry, dict)]


def run_single_engine(
    gold_items: List[GoldItem],
    chunks: List[Dict[str, object]],
    engine: str,
    base_config: EvalConfig,
    runs_dir: Path,
    gold_path: Path,
) -> Path:
    cfg = _make_retrieval_config(base_config, engine)
    indexes = build_indexes(list(chunks), cfg)
    per_item: List[ItemMetrics] = []
    timing_rows: List[Dict[str, float]] = []

    for gold in gold_items:
        response = retrieve(gold.question, engine, indexes, cfg)
        pre_candidates = _response_candidates(response, "pre_candidates")
        rerank_candidates = _response_candidates(response, "rerank_candidates")
        final_chunks = response.get("final_chunks", [])

        pre_hit = hit_at_k(pre_candidates, gold, base_config.pre_hit_k)
        post_hit = hit_at_k(rerank_candidates, gold, base_config.post_hit_k)
        mrr = mrr_at_k(rerank_candidates, gold, k=10)
        ndcg = ndcg_at_k(rerank_candidates, gold, k=10)
        ctx_prec = context_precision(final_chunks, gold)

        timings = {k: float(v) for k, v in (response.get("timings") or {}).items()}
        per_item.append(
            ItemMetrics(
                id=gold.id,
                tags=gold.tags,
                pre_hit_at_k=pre_hit,
                post_hit_at_k=post_hit,
                mrr_at_10=mrr,
                ndcg_at_10=ndcg,
                context_precision=ctx_prec,
                timings=timings,
            )
        )
        timing_rows.append(timings)

    aggregates = {
        f"pre_hit@{base_config.pre_hit_k}": float(np.mean([item.pre_hit_at_k for item in per_item])),
        f"post_hit@{base_config.post_hit_k}": float(np.mean([item.post_hit_at_k for item in per_item])),
        "mrr@10": float(np.mean([item.mrr_at_10 for item in per_item])),
        "ndcg@10": float(np.mean([item.ndcg_at_10 for item in per_item])),
        "context_precision": float(np.mean([item.context_precision for item in per_item])),
    }

    latency = latency_percentiles(timing_rows)
    run = RunResult(
        engine=engine,
        config=base_config,
        created_at=datetime.utcnow(),
        gold_checksum=checksum(gold_path),
        aggregates=aggregates,
        latency=latency,
        items=per_item,
    )

    run_dir = ensure_dir(runs_dir)
    out_path = Path(run_dir) / f"{engine}_run.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(run.to_dict(), fh, indent=2)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline retrieval evaluator")
    parser.add_argument("--gold", type=str, required=True, help="Path to gold JSONL/CSV file")
    parser.add_argument("--chunks", type=str, help="Path to chunk JSON/JSONL file")
    parser.add_argument("--engine", type=str, default="all", help="lexical|semantic|hybrid|all")
    parser.add_argument("--config", type=str, default="eval/config.yaml", help="YAML config override")
    parser.add_argument("--out", type=str, help="Optional explicit output path for single-engine runs")
    return parser.parse_args()


def load_eval_config(path: str | Path) -> EvalConfig:
    if Path(path).exists():
        cfg = EvalConfig.from_dict(load_yaml(path))
    else:
        cfg = EvalConfig()
    return cfg


def main() -> None:
    args = parse_args()
    config = load_eval_config(args.config)
    if args.chunks:
        config.chunks_path = args.chunks
    if not config.chunks_path:
        raise ValueError("chunks_path must be provided via config or --chunks")

    set_seeds(config.seed)
    gold_items = load_gold(args.gold)
    chunks = load_chunks(config.chunks_path)

    engines: Iterable[str]
    if args.engine == "all":
        engines = config.engines
    else:
        engines = [args.engine]

    runs_dir = ensure_dir(config.runs_dir)
    gold_path = Path(args.gold)

    outputs: List[Path] = []
    for engine in engines:
        outputs.append(run_single_engine(gold_items, chunks, engine, config, runs_dir, gold_path))

    if args.out and len(outputs) == 1:
        Path(args.out).write_text(Path(outputs[0]).read_text())


if __name__ == "__main__":  # pragma: no cover
    main()
