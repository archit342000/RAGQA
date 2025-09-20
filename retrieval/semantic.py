"""Semantic retrieval helpers backed by FAISS and E5 embeddings."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guard for environments without FAISS
    import faiss
except ImportError:  # pragma: no cover - handled by caller
    faiss = None  # type: ignore

# Cache encoder instances so repeated calls avoid re-downloading weights and
# reinitialising heavy models.
_ENCODER_CACHE: Dict[str, SentenceTransformer] = {}


def get_encoder(model_name: str) -> SentenceTransformer:
    """Return a cached :class:`SentenceTransformer` instance."""

    if model_name in _ENCODER_CACHE:
        return _ENCODER_CACHE[model_name]
    logger.info("Loading sentence-transformer model %s", model_name)
    encoder = SentenceTransformer(model_name)
    encoder.eval()
    _ENCODER_CACHE[model_name] = encoder
    return encoder


def _format_passage(text: str) -> str:
    # E5 models expect prompts such as "passage:" to differentiate between
    # document embeddings and queries. See official guidance from the model
    # authors.
    return f"passage: {text.strip()}" if text else "passage:"


def _format_query(query: str) -> str:
    # Symmetric to ``_format_passage`` but using the "query:" prefix required by
    # E5 so the encoder knows it is processing a search query.
    return f"query: {query.strip()}" if query else "query:"


def embed_chunks(chunks: Sequence[Dict[str, object]], model_name: str) -> np.ndarray:
    """Encode chunks using the specified model and return L2-normalised vectors."""

    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    encoder = get_encoder(model_name)
    texts = [_format_passage(str(chunk.get("text", ""))) for chunk in chunks]
    # Dynamically size the batch to balance throughput and memory requirements.
    batch_size = min(64, max(8, len(texts)))
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)
    embeddings = embeddings.astype(np.float32)
    return embeddings


def build_faiss(embeddings: np.ndarray, id_map: Sequence[str]) -> dict:
    """Materialise a FAISS index around the provided (normalised) embeddings."""

    if faiss is None:  # pragma: no cover - runtime guard
        raise RuntimeError("FAISS is not available in this environment.")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings array must be 2D.")
    if embeddings.shape[0] != len(id_map):
        raise ValueError("Embeddings and id_map must align.")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return {"faiss": index, "embeddings": embeddings, "id_map": list(id_map)}


def search_faiss(
    query: str,
    encoder: SentenceTransformer,
    faiss_index: dict,
    k: int,
) -> List[tuple[str, float]]:
    """Encode ``query`` and search against the FAISS index."""

    if faiss is None:  # pragma: no cover - runtime guard
        return []
    if not query.strip():
        return []
    index = faiss_index.get("faiss")
    if index is None:
        return []
    id_map: List[str] = faiss_index.get("id_map", [])

    # Encode the query with L2 normalisation so cosine similarity reduces to
    # inner product (the metric used by the FAISS index).
    encode_start = time.perf_counter()
    encoded = encoder.encode(
        [_format_query(query)],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if not isinstance(encoded, np.ndarray):
        encoded = np.asarray(encoded, dtype=np.float32)
    encoded = encoded.astype(np.float32)
    encode_ms = (time.perf_counter() - encode_start) * 1000.0

    # Query FAISS for the top ``k`` matches and capture timing information for
    # observability in the UI.
    search_start = time.perf_counter()
    scores, indices = index.search(encoded, k)
    search_ms = (time.perf_counter() - search_start) * 1000.0

    faiss_index["_timings"] = {"encode_ms": encode_ms, "search_ms": search_ms}

    if scores.size == 0:
        return []
    top_scores = scores[0]
    top_indices = indices[0]
    results: List[tuple[str, float]] = []
    for idx, score in zip(top_indices, top_scores):
        if idx < 0 or idx >= len(id_map):
            continue
        results.append((id_map[idx], float(score)))
    return results


__all__ = ["build_faiss", "embed_chunks", "get_encoder", "search_faiss"]
