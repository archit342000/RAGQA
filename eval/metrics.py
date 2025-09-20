"""Evaluation metrics for retrieval experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .types import GoldItem


def _page_overlap(candidate: Dict[str, object], gold: GoldItem) -> bool:
    start = int(candidate.get("page_start", 0) or 0)
    end = int(candidate.get("page_end", start) or start)
    return not (end < gold.page_start or start > gold.page_end)


def _char_span(candidate: Dict[str, object]) -> Tuple[int, int] | None:
    meta = candidate.get("meta") or {}
    start = meta.get("char_start")
    end = meta.get("char_end")
    if start is None or end is None:
        return None
    return int(start), int(end)


def hit_at_k(candidates: Sequence[Dict[str, object]], gold: GoldItem, k: int) -> float:
    """Return 1.0 if any of the top-K candidates overlaps the gold page span."""

    for candidate in candidates[:k]:
        if _page_overlap(candidate, gold):
            return 1.0
    return 0.0


def mrr_at_k(candidates: Sequence[Dict[str, object]], gold: GoldItem, k: int = 10) -> float:
    """Compute Mean Reciprocal Rank at K for a single question."""

    for rank, candidate in enumerate(candidates[:k], start=1):
        if _page_overlap(candidate, gold):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(candidates: Sequence[Dict[str, object]], gold: GoldItem, k: int = 10) -> float:
    """Binary nDCG@K with relevance 1 on page overlap."""

    gains = [1.0 if _page_overlap(candidate, gold) else 0.0 for candidate in candidates[:k]]
    dcg = 0.0
    for idx, gain in enumerate(gains, start=1):
        if gain:
            dcg += gain / np.log2(idx + 1)
    ideal_gains = sorted(gains, reverse=True)
    idcg = 0.0
    for idx, gain in enumerate(ideal_gains, start=1):
        if gain:
            idcg += gain / np.log2(idx + 1)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _chunk_char_length(chunk: Dict[str, object]) -> int:
    meta = chunk.get("meta") or {}
    start = meta.get("char_start")
    end = meta.get("char_end")
    if start is not None and end is not None and end >= start:
        return int(end) - int(start)
    text = chunk.get("text")
    return len(text) if isinstance(text, str) else 0


def _span_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return max(0, end - start)


def context_precision(final_chunks: Sequence[Dict[str, object]], gold: GoldItem) -> float:
    """Estimate how much of the selected context overlaps the answer span."""

    if not final_chunks:
        return 0.0

    total_chars = 0
    matched_chars = 0
    gold_span = None
    if gold.char_start is not None and gold.char_end is not None and gold.char_end >= gold.char_start:
        gold_span = (int(gold.char_start), int(gold.char_end))

    for chunk in final_chunks:
        char_len = _chunk_char_length(chunk)
        total_chars += max(char_len, 1)  # avoid zero denominator
        chunk_span = _char_span(chunk)
        if gold_span and chunk_span:
            matched_chars += _span_overlap(chunk_span, gold_span)
        elif _page_overlap(chunk, gold):
            matched_chars += char_len or 1

    if total_chars == 0:
        return 0.0
    return min(1.0, matched_chars / total_chars)


def latency_percentiles(per_item_timings: Iterable[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute P50/P95 latency for each timing key."""

    buckets: Dict[str, List[float]] = defaultdict(list)
    for timings in per_item_timings:
        for key, value in timings.items():
            buckets[key].append(float(value))

    summary: Dict[str, Dict[str, float]] = {}
    for key, values in buckets.items():
        arr = np.asarray(values)
        summary[key] = {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(np.mean(arr)),
        }
    return summary


def paired_bootstrap_ci(
    metric_a: Sequence[float],
    metric_b: Sequence[float],
    iters: int = 1000,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired bootstrap confidence interval and p-value for mean difference."""

    if len(metric_a) != len(metric_b):
        raise ValueError("Metric vectors must be the same length for paired bootstrap.")
    if not metric_a:
        return {"diff_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}

    rng = np.random.default_rng()
    diffs = np.array(metric_b) - np.array(metric_a)
    boot_diffs = np.empty(iters)
    n = len(metric_a)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = diffs[idx].mean()
    low = np.percentile(boot_diffs, 100 * (alpha / 2))
    high = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    p_value = min(1.0, 2 * min((boot_diffs <= 0).mean(), (boot_diffs >= 0).mean()))
    return {
        "diff_mean": float(diffs.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(p_value),
    }


def win_loss_tie(metric_a: Sequence[float], metric_b: Sequence[float], tol: float = 1e-6) -> Dict[str, int]:
    """Count per-query wins/losses/ties between two engines."""

    wins = losses = ties = 0
    for val_a, val_b in zip(metric_a, metric_b):
        if val_b - val_a > tol:
            wins += 1
        elif val_a - val_b > tol:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties}


__all__ = [
    "context_precision",
    "hit_at_k",
    "latency_percentiles",
    "mrr_at_k",
    "ndcg_at_k",
    "paired_bootstrap_ci",
    "win_loss_tie",
]
