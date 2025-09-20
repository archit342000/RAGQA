"""Utility helpers for the evaluation harness."""

from __future__ import annotations

import json
import os
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import yaml

from retrieval.driver import RetrievalConfig, build_indexes

from .types import EvalConfig, GoldItem


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(path: str | Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_gold(path: str | Path) -> List[GoldItem]:
    path = Path(path)
    records: List[Dict[str, object]] = []
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    elif path.suffix.lower() in {".json"}:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
            if isinstance(payload, list):
                records.extend(payload)
            else:
                raise ValueError("JSON gold file must contain a list of records.")
    elif path.suffix.lower() == ".csv":
        import csv  # lazy import

        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                records.append(row)
    else:
        raise ValueError(f"Unsupported gold file format: {path.suffix}")

    gold_items: List[GoldItem] = []
    for row in records:
        tags = row.get("tags") or []
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(";") if tag.strip()]
        hard_neg = row.get("hard_negative_ids") or []
        if isinstance(hard_neg, str):
            hard_neg = [item.strip() for item in hard_neg.split(";") if item.strip()]
        gold_items.append(
            GoldItem(
                id=str(row["id"]),
                question=str(row["question"]),
                doc_id=str(row["doc_id"]),
                page_start=int(row["page_start"]),
                page_end=int(row["page_end"]),
                answer_text=row.get("answer_text"),
                char_start=_safe_int(row.get("char_start")),
                char_end=_safe_int(row.get("char_end")),
                tags=list(tags),
                hard_negative_ids=list(hard_neg),
            )
        )
    return gold_items


def _safe_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def load_chunks(path: str | Path) -> List[Dict[str, object]]:
    """Load chunk dictionaries produced by the chunking pipeline."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    records: List[Dict[str, object]] = []
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
            if isinstance(payload, list):
                records.extend(payload)
            else:
                raise ValueError("Chunk JSON must contain a list of objects.")
    else:
        raise ValueError(f"Unsupported chunks format: {path.suffix}")
    return records


def checksum(path: str | Path) -> str:
    import hashlib

    hasher = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(65536)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_retrieval_config(base: EvalConfig) -> RetrievalConfig:
    return RetrievalConfig(
        default_engine=base.engines[0] if base.engines else "semantic",
        vector_top_k=base.top_k,
        lexical_top_k=base.top_k,
        rerank_top_n=base.rerank_top_n,
        hybrid_fusion=base.hybrid_fusion,
        hybrid_rrf_k=base.hybrid_rrf_k,
        hybrid_weight_vector=base.hybrid_weight_vector,
        hybrid_weight_lexical=base.hybrid_weight_lexical,
        hybrid_top_k_vector=base.hybrid_top_k_vector,
        hybrid_top_k_lexical=base.hybrid_top_k_lexical,
    )


def build_indexes_once(chunks: Sequence[Dict[str, object]], cfg: RetrievalConfig) -> dict:
    return build_indexes(list(chunks), cfg)


def timing_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    arr = np.asarray(values)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def now_iso() -> str:
    return datetime.utcnow().isoformat()


__all__ = [
    "build_indexes_once",
    "build_retrieval_config",
    "checksum",
    "ensure_dir",
    "load_chunks",
    "load_gold",
    "load_yaml",
    "now_iso",
    "set_seeds",
    "timing_summary",
]
