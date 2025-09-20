"""Utilities for selecting and packing reranked chunks into context."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .types import ChunkLike
else:  # pragma: no cover - runtime fallback
    ChunkLike = dict


def _has_embeddings(chunks_by_id: Dict[str, ChunkLike], ids: Sequence[str]) -> bool:
    """Return True when at least one chunk in ``ids`` carries an embedding."""

    for cid in ids:
        chunk = chunks_by_id.get(cid)
        if chunk and isinstance(chunk.get("embedding"), np.ndarray):
            return True
    return False


def select_for_context(
    chunks_by_id: Dict[str, ChunkLike],
    ordered_ids: Sequence[str],
    max_answer_ctx_tokens: int,
    mmr_lambda: float = 0.5,
    per_doc_cap: int = 2,
) -> List[ChunkLike]:
    """Pick a diversified subset of chunks to feed the answer generator."""

    if not ordered_ids:
        return []

    # Deduplicate the ordered candidate list while preserving order so reranked
    # priorities remain intact.
    seen: set[str] = set()
    candidate_ids: List[str] = []
    for cid in ordered_ids:
        if cid in seen:
            continue
        seen.add(cid)
        candidate_ids.append(cid)

    if not candidate_ids:
        return []

    embeddings_available = _has_embeddings(chunks_by_id, candidate_ids)

    selected: List[str] = []
    doc_counts: Dict[str, int] = defaultdict(int)
    used_tokens = 0

    # Greedily pick the best candidate while respecting MMR penalties,
    # per-document caps, and the overall token budget.
    while candidate_ids:
        best_id = None
        best_score = float("-inf")
        for cid in list(candidate_ids):
            chunk = chunks_by_id.get(cid)
            if not chunk:
                candidate_ids.remove(cid)
                continue
            doc_id = chunk.get("doc_id", "")
            if per_doc_cap > 0 and doc_counts[doc_id] >= per_doc_cap:
                continue
            base_score = float(chunk.get("score", 0.0))
            mmr_score = base_score
            if embeddings_available and selected and isinstance(chunk.get("embedding"), np.ndarray):
                candidate_emb = chunk["embedding"]
                sims: List[float] = []
                for sid in selected:
                    other = chunks_by_id.get(sid)
                    other_emb = other.get("embedding") if other else None
                    if isinstance(other_emb, np.ndarray):
                        sims.append(float(np.dot(candidate_emb, other_emb)))
                if sims:
                    penalty = (1.0 - mmr_lambda) * max(sims)
                    mmr_score = mmr_lambda * base_score - penalty
                else:
                    mmr_score = mmr_lambda * base_score
            if mmr_score > best_score:
                best_score = mmr_score
                best_id = cid

        if best_id is None:
            break

        chunk = chunks_by_id.get(best_id)
        if not chunk:
            candidate_ids.remove(best_id)
            continue
        token_len = int(chunk.get("token_len", 0) or 0)
        if token_len <= 0:
            token_len = 1
        # Respect the global token limit so the answer generator receives a
        # compact set of sources.
        if selected and max_answer_ctx_tokens > 0 and used_tokens + token_len > max_answer_ctx_tokens:
            candidate_ids.remove(best_id)
            continue

        selected.append(best_id)
        used_tokens += token_len
        doc_counts[chunk.get("doc_id", "")] += 1
        candidate_ids.remove(best_id)

    results: List[ChunkLike] = []
    for cid in selected:
        chunk = chunks_by_id.get(cid)
        if not chunk:
            continue
        payload: ChunkLike = {
            "id": cid,
            "text": chunk.get("text", ""),
            "doc_id": chunk.get("doc_id", ""),
            "doc_name": chunk.get("doc_name", ""),
            "page_start": chunk.get("page_start", 0),
            "page_end": chunk.get("page_end", 0),
            "token_len": int(chunk.get("token_len", 0) or 0),
            "meta": dict(chunk.get("meta") or {}),
            "score": float(chunk.get("score", 0.0)),
        }
        results.append(payload)

    # Sort by score so the consumer (UI or LLM) accesses the highest-confidence
    # chunks first.
    results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return results


__all__ = ["select_for_context"]
