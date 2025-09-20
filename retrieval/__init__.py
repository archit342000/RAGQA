"""Retrieval engines and helpers for the RAG demo."""

from __future__ import annotations

from .hybrid import FusedCandidate, RankedHit
from .driver import RetrievalConfig, build_indexes, retrieve
from .types import ChunkLike

__all__ = [
    "ChunkLike",
    "RetrievalConfig",
    "build_indexes",
    "retrieve",
]
