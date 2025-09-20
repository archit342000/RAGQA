"""Predefined ablation suites for retrieval evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import List

from .runner import load_eval_config, run_single_engine
from .types import EvalConfig
from .utils import ensure_dir, load_chunks, load_gold, set_seeds


SUITES = {
    "basic": {
        "top_k_values": [40, 80, 120],
        "rerank_top_n": None,
        "fusion_modes": None,
    },
    "rerank_off": {
        "top_k_values": [80],
        "rerank_top_n": 0,
        "fusion_modes": None,
    },
    "fusion_modes": {
        "top_k_values": [80],
        "rerank_top_n": None,
        "fusion_modes": ["rrf", "weighted"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation ablation suites")
    parser.add_argument("--suite", required=True, choices=SUITES.keys())
    parser.add_argument("--gold", required=True, help="Path to gold data")
    parser.add_argument("--chunks", help="Path to chunks file")
    parser.add_argument("--config", default="eval/config.yaml")
    return parser.parse_args()


def configure_run(base: EvalConfig, top_k: int, rerank_top_n: int | None, fusion: str | None) -> EvalConfig:
    cfg = deepcopy(base)
    cfg.top_k = top_k
    if rerank_top_n is not None:
        cfg.rerank_top_n = rerank_top_n
    if fusion is not None:
        cfg.hybrid_fusion = fusion
    return cfg


def main() -> None:
    args = parse_args()
    config = load_eval_config(args.config)
    if args.chunks:
        config.chunks_path = args.chunks
    if not config.chunks_path:
        raise ValueError("chunks_path must be provided via config or --chunks")

    suite = SUITES[args.suite]
    gold_items = load_gold(args.gold)
    chunks = load_chunks(config.chunks_path)
    set_seeds(config.seed)

    runs_root = ensure_dir(Path(config.runs_dir) / args.suite)
    gold_path = Path(args.gold)

    fusion_modes: List[str]
    if suite["fusion_modes"]:
        fusion_modes = suite["fusion_modes"]
    else:
        fusion_modes = [config.hybrid_fusion]

    for top_k in suite["top_k_values"]:
        for fusion in fusion_modes:
            cfg = configure_run(config, top_k, suite["rerank_top_n"], fusion)
            for engine in cfg.engines:
                run_single_engine(
                    gold_items,
                    chunks,
                    engine,
                    cfg,
                    runs_root / f"k{top_k}_{fusion}",
                    gold_path,
                )


if __name__ == "__main__":  # pragma: no cover
    main()
