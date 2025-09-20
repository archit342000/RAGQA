"""Cross-encoder based reranking for retrieval candidates."""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Cache cross-encoder instances so repeated queries do not re-download weights
# or reinitialise models, which would quickly exhaust Spaces cold-start budgets.
_RERANKER_CACHE: Dict[str, CrossEncoder] = {}


def get_reranker(model_name: str) -> CrossEncoder:
    """Return a cached cross-encoder instance."""

    if model_name in _RERANKER_CACHE:
        return _RERANKER_CACHE[model_name]
    logger.info("Loading cross-encoder model %s", model_name)
    model = CrossEncoder(model_name)
    model.eval()
    _RERANKER_CACHE[model_name] = model
    return model


def rerank_cross_encoder(
    query: str,
    candidates: Sequence[Tuple[str, str]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 5,
    batch_size: int = 32,
) -> List[tuple[str, float]]:
    """Rerank candidate chunks using a cross-encoder.

    Parameters
    ----------
    query:
        User query string.
    candidates:
        Sequence of ``(chunk_id, text)`` pairs produced by a first-pass retriever.
    model_name:
        Hugging Face model identifier to load.
    top_n:
        Maximum number of reranked entries to keep.
    batch_size:
        Batch size for model inference; adjust downwards on CPU/ZeroGPU.
    """

    if not query.strip() or not candidates:
        return []
    try:
        model = get_reranker(model_name)
    except Exception as exc:  # pragma: no cover - load failure
        logger.warning("Unable to load reranker %s: %s", model_name, exc)
        return []

    # The cross-encoder expects a list of (query, document) tuples so it can
    # jointly encode and score relevance per pair.
    paired_texts = [(query, text) for _, text in candidates]
    try:
        scores = model.predict(paired_texts, batch_size=batch_size, convert_to_numpy=True)
    except Exception as exc:  # pragma: no cover - inference failure
        logger.warning("Reranker inference failed: %s", exc)
        return []

    if not isinstance(scores, np.ndarray):
        scores = np.asarray(scores, dtype=np.float32)
    scores = scores.astype(np.float32)
    scored_pairs = list(zip((cid for cid, _ in candidates), scores.tolist()))
    # Sort descending so the most relevant chunks appear first in the final
    # candidate list.
    scored_pairs.sort(key=lambda item: float(item[1]), reverse=True)
    return [(cid, float(score)) for cid, score in scored_pairs[:top_n]]


__all__ = ["get_reranker", "rerank_cross_encoder"]
