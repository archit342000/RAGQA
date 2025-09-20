"""Shared type definitions for retrieval components."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


class ChunkLike(TypedDict, total=False):
    """Chunk payload tracked by the retrieval system."""

    id: str
    text: str
    doc_id: str
    doc_name: str
    page_start: int
    page_end: int
    token_len: int
    meta: dict[str, str]
    score: float
    embedding: "np.ndarray"


__all__ = ["ChunkLike"]
