"""WH-distribution enforcement utilities."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import quality


def enforce_wh_targets(items: List[dict], targets: Dict[str, float], seed: int) -> List[dict]:
    """Subsample items so that WH categories do not exceed the requested share."""

    if not items:
        return []
    total = len(items)
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        wh = (item.get("wh") or "").strip().lower()
        if not wh:
            wh = quality.detect_wh(item.get("question", ""))
            item["wh"] = wh
        buckets[wh].append(idx)

    allowed: Dict[str, int] = {}
    for wh, indices in buckets.items():
        share = targets.get(wh)
        if share is None:
            allowed[wh] = len(indices)
            continue
        limit = int(math.ceil(total * max(share, 0.0)))
        if limit <= 0:
            allowed[wh] = 0
        else:
            allowed[wh] = min(len(indices), limit)

    question_texts = [item.get("question", "") for item in items]
    context_texts = [item.get("window_text", "") for item in items]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(question_texts + context_texts)
    question_matrix = matrix[:total]
    context_matrix = matrix[total:]
    relevance_scores = [
        float(cosine_similarity(question_matrix[i], context_matrix[i])[0, 0])
        if question_matrix[i].nnz and context_matrix[i].nnz
        else 0.0
        for i in range(total)
    ]

    rng = random.Random(seed)
    selected_indices: List[int] = []

    for wh, indices in buckets.items():
        limit = allowed.get(wh, len(indices))
        if len(indices) <= limit:
            selected_indices.extend(indices)
            continue
        chosen = _mmr_select(indices, limit, question_matrix, relevance_scores, rng)
        selected_indices.extend(chosen)

    selected_indices = sorted(set(selected_indices))
    return [items[idx] for idx in selected_indices]


def _mmr_select(
    candidate_indices: List[int],
    limit: int,
    question_matrix,
    relevance_scores: List[float],
    rng: random.Random,
    lambda_: float = 0.7,
) -> List[int]:
    """Select questions using Maximal Marginal Relevance within a bucket."""

    selected: List[int] = []
    candidates = candidate_indices[:]
    rng.shuffle(candidates)
    while candidates and len(selected) < limit:
        best_candidate = None
        best_score = -float("inf")
        for idx in candidates:
            relevance = relevance_scores[idx]
            redundancy = 0.0
            if selected:
                sims = cosine_similarity(
                    question_matrix[idx], question_matrix[selected]
                )
                redundancy = float(sims.max()) if sims.size else 0.0
            score = lambda_ * relevance - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_candidate = idx
        if best_candidate is None:
            break
        selected.append(best_candidate)
        candidates.remove(best_candidate)
    return selected
