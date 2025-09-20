"""Orchestrates retrieval pipelines and shared utilities."""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import random
import time
from collections import OrderedDict
from dataclasses import asdict
from typing import Dict, List, Literal, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .hybrid import FusedCandidate, RankedHit, add_ranks, rrf_fuse, weighted_fuse
from .lexical import build_bm25, search_bm25
from .pack import select_for_context
from .rerank import rerank_cross_encoder
from .semantic import build_faiss, embed_chunks, get_encoder, search_faiss
from .types import ChunkLike

logger = logging.getLogger(__name__)


class QueryCache:
    """Simple LRU cache scoped to a Spaces session."""

    def __init__(self, max_size: int) -> None:
        # Use an OrderedDict so we can reorder entries on access and evict the
        # least-recently-used item with ``popitem(last=False)``.
        self._max_size = max_size
        self._entries: "OrderedDict[str, dict]" = OrderedDict()

    def get(self, key: str) -> Optional[dict]:
        if key not in self._entries:
            return None
        self._entries.move_to_end(key)
        return copy.deepcopy(self._entries[key])

    def set(self, key: str, value: dict) -> None:
        self._entries[key] = copy.deepcopy(value)
        self._entries.move_to_end(key)
        if len(self._entries) > self._max_size:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()


class RetrievalConfig:
    """Configuration bundle for retrieval engines."""

    def __init__(
        self,
        *,
        default_engine: Literal["semantic", "lexical", "hybrid"] | str = "semantic",
        embedding_model: str = "intfloat/e5-base-v2",
        embedding_model_fallback: str = "intfloat/e5-small-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        vector_top_k: int = 80,
        lexical_top_k: int = 80,
        rerank_top_n: int = 5,
        rerank_batch_size: int = 32,
        max_answer_ctx_tokens: int = 2800,
        mmr_lambda: float = 0.5,
        per_doc_cap: int = 2,
        cache_size: int = 64,
        hybrid_fusion: str = "rrf",
        hybrid_rrf_k: int = 60,
        hybrid_weight_vector: float = 0.6,
        hybrid_weight_lexical: float = 0.4,
        hybrid_top_k_vector: int = 80,
        hybrid_top_k_lexical: int = 80,
    ) -> None:
        engine = str(default_engine or "semantic").lower()
        if engine not in {"semantic", "lexical", "hybrid"}:
            engine = "semantic"
        self.default_engine: Literal["semantic", "lexical", "hybrid"] = engine  # type: ignore[assignment]
        self.embedding_model = embedding_model
        self.embedding_model_fallback = embedding_model_fallback
        self.reranker_model = reranker_model
        self.vector_top_k = int(vector_top_k)
        self.lexical_top_k = int(lexical_top_k)
        self.rerank_top_n = int(rerank_top_n)
        self.rerank_batch_size = int(rerank_batch_size)
        self.max_answer_ctx_tokens = int(max_answer_ctx_tokens)
        self.mmr_lambda = float(mmr_lambda)
        self.per_doc_cap = int(per_doc_cap)
        self.cache_size = int(cache_size)
        self.hybrid_fusion = str(hybrid_fusion or "rrf").lower()
        self.hybrid_rrf_k = max(int(hybrid_rrf_k), 1)
        self.hybrid_weight_vector = float(hybrid_weight_vector)
        self.hybrid_weight_lexical = float(hybrid_weight_lexical)
        self.hybrid_top_k_vector = max(int(hybrid_top_k_vector), 1)
        self.hybrid_top_k_lexical = max(int(hybrid_top_k_lexical), 1)

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """Build configuration overrides from environment variables."""

        def _get_env_int(name: str, fallback: int) -> int:
            try:
                return int(os.getenv(name, fallback))
            except ValueError:
                return fallback

        def _get_env_float(name: str, fallback: float) -> float:
            try:
                return float(os.getenv(name, fallback))
            except ValueError:
                return fallback

        return cls(
            default_engine=os.getenv("RETRIEVAL_DEFAULT_ENGINE", "semantic"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2"),
            embedding_model_fallback=os.getenv("EMBEDDING_MODEL_FALLBACK", "intfloat/e5-small-v2"),
            reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            vector_top_k=_get_env_int("VECTOR_TOP_K", 80),
            lexical_top_k=_get_env_int("LEXICAL_TOP_K", 80),
            rerank_top_n=_get_env_int("RERANK_TOP_N", 5),
            rerank_batch_size=_get_env_int("RERANK_BATCH_SIZE", 32),
            max_answer_ctx_tokens=_get_env_int("MAX_ANSWER_CTX_TOKENS", 2800),
            mmr_lambda=_get_env_float("MMR_LAMBDA", 0.5),
            per_doc_cap=_get_env_int("PER_DOC_CAP", 2),
            cache_size=_get_env_int("CACHE_SIZE", 64),
            hybrid_fusion=os.getenv("HYBRID_FUSION", "rrf"),
            hybrid_rrf_k=_get_env_int("HYBRID_RRF_K", 60),
            hybrid_weight_vector=_get_env_float("HYBRID_WEIGHT_VECTOR", 0.6),
            hybrid_weight_lexical=_get_env_float("HYBRID_WEIGHT_LEXICAL", 0.4),
            hybrid_top_k_vector=_get_env_int("HYBRID_TOP_K_VECTOR", 80),
            hybrid_top_k_lexical=_get_env_int("HYBRID_TOP_K_LEXICAL", 80),
        )

    def hybrid_signature(self) -> str:
        """Serialize hybrid-specific knobs for cache keys."""

        return "|".join(
            [
                self.hybrid_fusion,
                str(self.hybrid_rrf_k),
                f"{self.hybrid_weight_vector:.6f}",
                f"{self.hybrid_weight_lexical:.6f}",
                str(self.hybrid_top_k_vector),
                str(self.hybrid_top_k_lexical),
            ]
        )


def _seed_everything(seed: int = 17) -> None:
    """Seed RNGs for deterministic behaviour during index construction."""

    random.seed(seed)
    np.random.seed(seed)


def _chunk_from_raw(raw: MutableMapping[str, object], fallback_id: str) -> ChunkLike:
    """Normalise raw chunk dictionaries into ``ChunkLike`` structures."""

    chunk_id = str(raw.get("id") or raw.get("chunk_id") or raw.get("chunk_key") or fallback_id)
    chunk: ChunkLike = {
        "id": chunk_id,
        "text": str(raw.get("text", "")),
        "doc_id": str(raw.get("doc_id", "")),
        "doc_name": str(raw.get("doc_name", "")),
        "page_start": int(raw.get("page_start", 0) or 0),
        "page_end": int(raw.get("page_end", 0) or 0),
        "token_len": int(raw.get("token_len", 0) or 0),
        "meta": dict(raw.get("meta") or {}),
        "score": 0.0,
    }
    return chunk


def _hash_query(engine: str, query: str, cfg: Optional[RetrievalConfig] = None) -> str:
    """Derive a stable cache key factoring in engine-specific parameters."""

    digest = hashlib.sha1()
    digest.update(engine.encode("utf-8"))
    digest.update(b"::")
    digest.update(query.strip().lower().encode("utf-8"))
    if engine == "hybrid" and cfg is not None:
        digest.update(b"::hybrid::")
        digest.update(cfg.hybrid_signature().encode("utf-8"))
    return digest.hexdigest()


def build_indexes(
    chunks: Sequence[MutableMapping[str, object]],
    cfg: RetrievalConfig,
) -> dict:
    """Initialise lexical and semantic indexes for the provided chunks."""

    _seed_everything()

    chunk_records: List[ChunkLike] = []
    id2meta: Dict[str, dict] = {}
    id2text: Dict[str, str] = {}

    for idx, raw in enumerate(chunks):
        chunk = _chunk_from_raw(raw, f"chunk-{idx}")
        chunk_records.append(chunk)
        id2meta[chunk["id"]] = {
            "doc_id": chunk["doc_id"],
            "doc_name": chunk["doc_name"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
        }
        id2text[chunk["id"]] = chunk["text"]

    lexical_index = build_bm25(chunk_records) if chunk_records else None

    encoder = None
    semantic_index: Optional[dict] = None
    semantic_error: Optional[str] = None
    chosen_model: Optional[str] = None

    if chunk_records:
        model_candidates = [cfg.embedding_model]
        if cfg.embedding_model_fallback and cfg.embedding_model_fallback not in model_candidates:
            model_candidates.append(cfg.embedding_model_fallback)

        for model_name in model_candidates:
            try:
                encoder = get_encoder(model_name)
                embeddings = embed_chunks(chunk_records, model_name)
                semantic_index = build_faiss(embeddings, [chunk["id"] for chunk in chunk_records])
                for chunk, vector in zip(chunk_records, embeddings):
                    chunk["embedding"] = vector
                chosen_model = model_name
                semantic_error = None
                break
            except Exception as exc:  # pragma: no cover - import/runtime guard
                logger.warning("Failed to initialise semantic index with %s: %s", model_name, exc)
                semantic_error = str(exc)
                semantic_index = None
                encoder = None

    sidecar = {
        "chunks": {chunk["id"]: chunk for chunk in chunk_records},
        "id2meta": id2meta,
        "id2text": id2text,
        "encoder_model": chosen_model,
    }

    indexes = {
        "lexical": lexical_index,
        "semantic": semantic_index,
        "models": {
            "encoder": encoder,
            "encoder_model": chosen_model,
            "reranker_model": cfg.reranker_model,
        },
        "sidecar": sidecar,
        "cache": QueryCache(cfg.cache_size),
        "errors": {"semantic": semantic_error},
    }
    return indexes


def _apply_reranker(
    query: str,
    candidate_pairs: List[Tuple[str, str]],
    cfg: RetrievalConfig,
    timings: Dict[str, float],
    notices: List[str],
) -> List[Tuple[str, float]]:
    """Run the reranker and handle failure fallback."""

    if cfg.rerank_top_n <= 0:
        # Bypass the reranker and reuse first-pass ordering.
        return [(cid, 0.0) for cid, _ in candidate_pairs]

    rerank_start = time.perf_counter()
    reranked = rerank_cross_encoder(
        query,
        candidate_pairs,
        model_name=cfg.reranker_model,
        top_n=max(cfg.rerank_top_n, len(candidate_pairs)),
        batch_size=cfg.rerank_batch_size,
    )
    timings["rerank_ms"] = (time.perf_counter() - rerank_start) * 1000.0
    if not reranked:
        notices.append("reranker_unavailable")
        limit = cfg.rerank_top_n or len(candidate_pairs)
        reranked = [(cid, 0.0) for cid, _ in candidate_pairs][:limit]
    return reranked


def _prepare_fusion_debug(fused: Sequence[FusedCandidate], limit: int = 5) -> List[Dict[str, object]]:
    """Convert fused candidates into JSON-friendly debug structures."""

    snapshot: List[Dict[str, object]] = []
    for candidate in fused[:limit]:
        payload = asdict(candidate)
        snapshot.append(payload)
    return snapshot


def retrieve(
    query: str,
    engine: Literal["lexical", "semantic", "hybrid"],
    indexes: dict,
    cfg: RetrievalConfig,
) -> dict:
    """Execute retrieval using the requested engine and shared reranker."""

    query_stripped = query.strip()
    if not query_stripped:
        return {
            "final_chunks": [],
            "provenance": {
                "engine": engine,
                "requested_engine": engine,
                "first_pass": [],
                "rerank": [],
                "cache_hit": False,
                "notices": ["empty_query"],
            },
            "timings": {
                "encode_ms": 0.0,
                "search_ms": 0.0,
                "rerank_ms": 0.0,
                "pack_ms": 0.0,
                "bm25_ms": 0.0,
                "faiss_ms": 0.0,
                "fusion_ms": 0.0,
                "total_ms": 0.0,
            },
        }

    requested_engine = engine
    start_total = time.perf_counter()
    cache: Optional[QueryCache] = indexes.get("cache")
    cache_key = _hash_query(engine, query_stripped, cfg if engine == "hybrid" else None)
    if cache:
        cached = cache.get(cache_key)
        if cached:
            cached.setdefault("provenance", {})
            cached["provenance"]["cache_hit"] = True
            return cached

    sidecar = indexes.get("sidecar") or {}
    chunk_map: Dict[str, ChunkLike] = sidecar.get("chunks", {})
    if not chunk_map:
        raise RuntimeError("No chunks available; parse documents before querying.")

    notices: List[str] = []
    timings = {
        "encode_ms": 0.0,
        "search_ms": 0.0,
        "rerank_ms": 0.0,
        "pack_ms": 0.0,
        "bm25_ms": 0.0,
        "faiss_ms": 0.0,
        "fusion_ms": 0.0,
    }

    first_pass: List[Tuple[str, float]] = []
    reranked: List[Tuple[str, float]] = []
    engine_effective = engine
    pre_candidates_payload: List[Dict[str, object]] = []
    rerank_candidates_payload: List[Dict[str, object]] = []

    if engine == "hybrid":
        lexical_index = indexes.get("lexical")
        semantic_index = indexes.get("semantic")
        encoder = (indexes.get("models") or {}).get("encoder")

        lexical_ready = lexical_index is not None
        semantic_ready = semantic_index is not None and encoder is not None

        if not lexical_ready and not semantic_ready:
            raise RuntimeError("Hybrid retrieval requires lexical or semantic indexes; none available.")
        if not lexical_ready and semantic_ready:
            notices.append("hybrid_lexical_unavailable")
            engine = "semantic"
            engine_effective = "semantic"
        elif not semantic_ready and lexical_ready:
            notices.append("hybrid_semantic_unavailable")
            engine = "lexical"
            engine_effective = "lexical"
        else:
            lexical_limit = cfg.hybrid_top_k_lexical or cfg.lexical_top_k
            vector_limit = cfg.hybrid_top_k_vector or cfg.vector_top_k
            fused_limit = max(lexical_limit, vector_limit)

            bm25_start = time.perf_counter()
            lexical_hits_raw = search_bm25(query_stripped, lexical_index, lexical_limit)
            timings["bm25_ms"] = (time.perf_counter() - bm25_start) * 1000.0
            lexical_ranked: List[RankedHit] = add_ranks(lexical_hits_raw[:lexical_limit])

            semantic_hits_raw = search_faiss(query_stripped, encoder, semantic_index, vector_limit)
            timings_info = semantic_index.get("_timings", {}) if isinstance(semantic_index, dict) else {}
            timings["encode_ms"] = float(timings_info.get("encode_ms", 0.0))
            timings["faiss_ms"] = float(timings_info.get("search_ms", 0.0))
            semantic_ranked: List[RankedHit] = add_ranks(semantic_hits_raw[:vector_limit])

            fusion_start = time.perf_counter()
            fusion_method = cfg.hybrid_fusion
            if fusion_method == "weighted":
                fused_candidates = weighted_fuse(
                    semantic_ranked,
                    lexical_ranked,
                    cfg.hybrid_weight_vector,
                    cfg.hybrid_weight_lexical,
                    fused_limit,
                )
            else:
                fusion_method = "rrf"
                fused_candidates = rrf_fuse(semantic_ranked, lexical_ranked, cfg.hybrid_rrf_k, fused_limit)
            timings["fusion_ms"] = (time.perf_counter() - fusion_start) * 1000.0
            timings["search_ms"] = timings["bm25_ms"] + timings["faiss_ms"]

            if not fused_candidates:
                response = {
                    "final_chunks": [],
                    "pre_candidates": pre_candidates_payload,
                    "rerank_candidates": rerank_candidates_payload,
                    "provenance": {
                        "engine": "hybrid",
                        "requested_engine": requested_engine,
                        "first_pass": [],
                        "rerank": [],
                        "cache_hit": False,
                        "notices": notices + ["no_candidates"],
                        "fusion": {
                            "method": fusion_method,
                            "candidate_counts": {
                                "semantic": len(semantic_ranked),
                                "lexical": len(lexical_ranked),
                                "fused": 0,
                            },
                        },
                    },
                    "timings": {
                        **timings,
                        "total_ms": (time.perf_counter() - start_total) * 1000.0,
                    },
                }
                if cache:
                    cache.set(cache_key, response)
                return response

            candidate_limit = min(len(fused_candidates), fused_limit)
            candidate_pairs: List[Tuple[str, str]] = []
            for candidate in fused_candidates[:candidate_limit]:
                chunk = chunk_map.get(candidate.chunk_id)
                if not chunk:
                    continue
                chunk["score"] = float(candidate.fused_score)
                candidate_pairs.append((candidate.chunk_id, chunk["text"]))
                pre_candidates_payload.append({
                    "chunk_id": candidate.chunk_id,
                    "score": float(candidate.fused_score),
                    "rank": len(pre_candidates_payload) + 1,
                    "doc_id": chunk.get("doc_id"),
                    "doc_name": chunk.get("doc_name"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "source": {
                        "semantic_rank": candidate.semantic_rank,
                        "semantic_score": candidate.semantic_score,
                        "lexical_rank": candidate.lexical_rank,
                        "lexical_score": candidate.lexical_score,
                    },
                })

            reranked = _apply_reranker(query_stripped, candidate_pairs, cfg, timings, notices)

            for position, (cid, score) in enumerate(reranked, start=1):
                chunk = chunk_map.get(cid)
                if not chunk:
                    continue
                rerank_candidates_payload.append({
                    "chunk_id": cid,
                    "score": float(score),
                    "rank": position,
                    "doc_id": chunk.get("doc_id"),
                    "doc_name": chunk.get("doc_name"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                })

            score_map: Dict[str, float] = {}
            ordered_ids: List[str] = []
            for chunk_id, score in reranked:
                if chunk_id not in chunk_map:
                    continue
                score_map[chunk_id] = float(score)
                ordered_ids.append(chunk_id)

            for chunk_id, score in score_map.items():
                chunk_map[chunk_id]["score"] = score

            pack_start = time.perf_counter()
            final_chunks = select_for_context(
                chunk_map,
                ordered_ids,
                cfg.max_answer_ctx_tokens,
                mmr_lambda=cfg.mmr_lambda,
                per_doc_cap=cfg.per_doc_cap,
            )
            timings["pack_ms"] = (time.perf_counter() - pack_start) * 1000.0
            timings["total_ms"] = (time.perf_counter() - start_total) * 1000.0

            provenance = {
                "engine": "hybrid",
                "requested_engine": requested_engine,
                "first_pass": [(cand.chunk_id, cand.fused_score) for cand in fused_candidates[:10]],
                "rerank": reranked[:10],
                "reranker_model": cfg.reranker_model,
                "reranker_applied": "reranker_unavailable" not in notices,
                "cache_hit": False,
                "notices": notices,
                "fusion": {
                    "method": fusion_method,
                    "rrf_k": cfg.hybrid_rrf_k if fusion_method == "rrf" else None,
                    "weights": {
                        "semantic": cfg.hybrid_weight_vector,
                        "lexical": cfg.hybrid_weight_lexical,
                    }
                    if fusion_method == "weighted"
                    else None,
                    "candidate_counts": {
                        "semantic": len(semantic_ranked),
                        "lexical": len(lexical_ranked),
                        "fused": len(fused_candidates),
                    },
                    "top": _prepare_fusion_debug(fused_candidates),
                },
            }

            response = {
                "final_chunks": final_chunks,
                "pre_candidates": pre_candidates_payload,
                "rerank_candidates": rerank_candidates_payload,
                "provenance": provenance,
                "timings": timings,
            }

            if cache:
                cache.set(cache_key, response)

            return response

    sidecar_models = indexes.get("models") or {}
    encoder = sidecar_models.get("encoder")

    # Semantic first-pass handling (used for semantic-only or hybrid fallback).
    if engine == "semantic":
        semantic_index = indexes.get("semantic")
        if not semantic_index or encoder is None:
            notices.append("semantic_unavailable")
            engine = "lexical"
            engine_effective = "lexical"
        else:
            hits = search_faiss(query_stripped, encoder, semantic_index, cfg.vector_top_k)
            timings_info = semantic_index.get("_timings", {}) if isinstance(semantic_index, dict) else {}
            timings["encode_ms"] = float(timings_info.get("encode_ms", 0.0))
            timings["faiss_ms"] = float(timings_info.get("search_ms", 0.0))
            timings["search_ms"] = timings["faiss_ms"]
            first_pass = hits
            for position, (cid, score) in enumerate(first_pass, start=1):
                chunk = chunk_map.get(cid)
                if not chunk:
                    continue
                pre_candidates_payload.append({
                    "chunk_id": cid,
                    "score": float(score),
                    "rank": position,
                    "doc_id": chunk.get("doc_id"),
                    "doc_name": chunk.get("doc_name"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "source": {"engine": "semantic"},
                })

    # Lexical first-pass handling.
    if engine == "lexical":
        lexical_index = indexes.get("lexical")
        if not lexical_index:
            raise RuntimeError("Lexical index unavailable; cannot satisfy query.")
        search_start = time.perf_counter()
        hits = search_bm25(query_stripped, lexical_index, cfg.lexical_top_k)
        timings["bm25_ms"] = (time.perf_counter() - search_start) * 1000.0
        timings["search_ms"] = timings["bm25_ms"]
        first_pass = hits
        for position, (cid, score) in enumerate(first_pass, start=1):
            chunk = chunk_map.get(cid)
            if not chunk:
                continue
            pre_candidates_payload.append({
                "chunk_id": cid,
                "score": float(score),
                "rank": position,
                "doc_id": chunk.get("doc_id"),
                "doc_name": chunk.get("doc_name"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "source": {"engine": "lexical"},
            })

    if not first_pass:
        response = {
            "final_chunks": [],
            "pre_candidates": pre_candidates_payload,
            "rerank_candidates": rerank_candidates_payload,
            "provenance": {
                "engine": engine_effective,
                "requested_engine": requested_engine,
                "first_pass": [],
                "rerank": [],
                "reranker_model": cfg.reranker_model,
                "reranker_applied": False,
                "cache_hit": False,
                "notices": notices + ["no_candidates"],
            },
            "timings": {
                **timings,
                "total_ms": (time.perf_counter() - start_total) * 1000.0,
            },
        }
        if cache:
            cache.set(cache_key, response)
        return response

    candidate_pairs: List[Tuple[str, str]] = []
    ordered_first_pass: List[Tuple[str, float]] = []
    for chunk_id, score in first_pass:
        if chunk_id not in chunk_map:
            continue
        candidate_pairs.append((chunk_id, chunk_map[chunk_id]["text"]))
        ordered_first_pass.append((chunk_id, float(score)))

    reranked = _apply_reranker(query_stripped, candidate_pairs, cfg, timings, notices)

    for position, (cid, score) in enumerate(reranked, start=1):
        chunk = chunk_map.get(cid)
        if not chunk:
            continue
        rerank_candidates_payload.append({
            "chunk_id": cid,
            "score": float(score),
            "rank": position,
            "doc_id": chunk.get("doc_id"),
            "doc_name": chunk.get("doc_name"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
        })

    score_map: Dict[str, float] = {}
    ordered_ids: List[str] = []
    for chunk_id, score in reranked:
        if chunk_id not in chunk_map:
            continue
        score_map[chunk_id] = float(score)
        ordered_ids.append(chunk_id)

    for chunk_id, score in score_map.items():
        chunk_map[chunk_id]["score"] = score

    pack_start = time.perf_counter()
    final_chunks = select_for_context(
        chunk_map,
        ordered_ids,
        cfg.max_answer_ctx_tokens,
        mmr_lambda=cfg.mmr_lambda,
        per_doc_cap=cfg.per_doc_cap,
    )
    timings["pack_ms"] = (time.perf_counter() - pack_start) * 1000.0
    timings["total_ms"] = (time.perf_counter() - start_total) * 1000.0

    provenance = {
        "engine": engine_effective,
        "requested_engine": requested_engine,
        "first_pass": ordered_first_pass[:10],
        "rerank": reranked[:10],
        "reranker_model": cfg.reranker_model,
        "reranker_applied": "reranker_unavailable" not in notices,
        "cache_hit": False,
        "notices": notices,
    }

    response = {
        "final_chunks": final_chunks,
        "pre_candidates": pre_candidates_payload,
        "rerank_candidates": rerank_candidates_payload,
        "provenance": provenance,
        "timings": timings,
    }

    if cache:
        cache.set(cache_key, response)

    return response


__all__ = ["ChunkLike", "RetrievalConfig", "build_indexes", "retrieve"]
