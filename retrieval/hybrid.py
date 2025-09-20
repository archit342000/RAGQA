"""Hybrid retrieval fusion strategies combining lexical and semantic hits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import math


@dataclass(frozen=True)
class RankedHit:
    """Represent a retrieval hit along with its 1-based rank."""

    chunk_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class FusedCandidate:
    """Describe a fused hybrid candidate with provenance metadata."""

    chunk_id: str
    fused_score: float
    semantic_rank: Optional[int]
    semantic_score: Optional[float]
    lexical_rank: Optional[int]
    lexical_score: Optional[float]


def add_ranks(hits: Sequence[Sequence[float]]) -> List[RankedHit]:
    """Attach 1-based ranks to an iterable of ``(chunk_id, score)`` pairs."""

    ranked: List[RankedHit] = []
    for position, (chunk_id, score) in enumerate(hits, start=1):
        ranked.append(RankedHit(str(chunk_id), float(score), position))
    return ranked


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise the provided scores into the [0,1] range.

    When all scores are identical (or the mapping is empty) we default to 1.0 so
    the downstream combiner still receives a useful signal.
    """

    if not scores:
        return {}
    values = list(scores.values())
    minimum = min(values)
    maximum = max(values)
    if math.isclose(maximum, minimum):
        return {key: 1.0 for key in scores}
    scale = maximum - minimum
    return {key: (value - minimum) / scale for key, value in scores.items()}


def _sort_key(
    chunk_id: str,
    fused: float,
    semantic_scores: Dict[str, float],
    lexical_scores: Dict[str, float],
) -> tuple:
    """Produce a deterministic sort key favouring semantic/lexical strength."""

    sem = semantic_scores.get(chunk_id, float("-inf"))
    lex = lexical_scores.get(chunk_id, float("-inf"))
    return (-fused, -sem, -lex, chunk_id)


def _finalise(
    fused_scores: Dict[str, float],
    semantic_hits: Dict[str, RankedHit],
    lexical_hits: Dict[str, RankedHit],
    limit: int,
) -> List[FusedCandidate]:
    """Convert fused score dictionaries into a sorted candidate list."""

    semantic_scores = {cid: hit.score for cid, hit in semantic_hits.items()}
    lexical_scores = {cid: hit.score for cid, hit in lexical_hits.items()}

    ordered_ids = sorted(
        fused_scores,
        key=lambda cid: _sort_key(cid, fused_scores[cid], semantic_scores, lexical_scores),
    )

    candidates: List[FusedCandidate] = []
    for cid in ordered_ids[:limit]:
        semantic_hit = semantic_hits.get(cid)
        lexical_hit = lexical_hits.get(cid)
        candidates.append(
            FusedCandidate(
                chunk_id=cid,
                fused_score=fused_scores[cid],
                semantic_rank=semantic_hit.rank if semantic_hit else None,
                semantic_score=semantic_hit.score if semantic_hit else None,
                lexical_rank=lexical_hit.rank if lexical_hit else None,
                lexical_score=lexical_hit.score if lexical_hit else None,
            )
        )
    return candidates


def rrf_fuse(
    semantic_hits: Sequence[RankedHit],
    lexical_hits: Sequence[RankedHit],
    k_const: int,
    limit: int,
) -> List[FusedCandidate]:
    """Fuse semantic and lexical results using Reciprocal Rank Fusion."""

    k = max(k_const, 1)
    fused: Dict[str, float] = {}

    semantic_lookup = {hit.chunk_id: hit for hit in semantic_hits}
    lexical_lookup = {hit.chunk_id: hit for hit in lexical_hits}

    # Apply RRF; ranks are 1-based. Missing entries simply contribute zero.
    for chunk_id, hit in semantic_lookup.items():
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + hit.rank)
    for chunk_id, hit in lexical_lookup.items():
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + hit.rank)

    return _finalise(fused, semantic_lookup, lexical_lookup, limit)


def weighted_fuse(
    semantic_hits: Sequence[RankedHit],
    lexical_hits: Sequence[RankedHit],
    weight_semantic: float,
    weight_lexical: float,
    limit: int,
) -> List[FusedCandidate]:
    """Fuse results via weighted sum of min-max normalised scores."""

    semantic_lookup = {hit.chunk_id: hit for hit in semantic_hits}
    lexical_lookup = {hit.chunk_id: hit for hit in lexical_hits}

    semantic_norm = normalize_scores({hit.chunk_id: hit.score for hit in semantic_hits})
    lexical_norm = normalize_scores({hit.chunk_id: hit.score for hit in lexical_hits})

    fused: Dict[str, float] = {}
    all_ids = set(semantic_lookup) | set(lexical_lookup)
    for cid in all_ids:
        sem_component = semantic_norm.get(cid, 0.0)
        lex_component = lexical_norm.get(cid, 0.0)
        fused[cid] = weight_semantic * sem_component + weight_lexical * lex_component

    return _finalise(fused, semantic_lookup, lexical_lookup, limit)


__all__ = [
    "FusedCandidate",
    "RankedHit",
    "add_ranks",
    "normalize_scores",
    "rrf_fuse",
    "weighted_fuse",
]
