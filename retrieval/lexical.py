"""Lexical retrieval helpers based on BM25."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable, List, Sequence

from rank_bm25 import BM25Okapi as RankBM25Okapi

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .types import ChunkLike
else:  # pragma: no cover - runtime fallback
    ChunkLike = dict

# Precompile the token pattern so repeated calls remain fast even for large
# corpora. The pattern splits on alphanumeric/underscore boundaries while
# preserving hyphenated tokens (e.g. "zero-shot").
_TOKEN_PATTERN = re.compile(r"\b[\w-]+\b", flags=re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Tokenize text by lowercasing and splitting on simple word boundaries."""

    if not text:
        return []
    lowered = text.lower()
    # BM25 benefits from naive tokenisation so we avoid stemming to keep
    # behaviour transparent and reversible.
    return _TOKEN_PATTERN.findall(lowered)


def build_bm25(chunks: Sequence[ChunkLike]) -> dict:
    """Create a BM25 index for the provided chunks."""

    # Tokenise each chunk once so the BM25 constructor can compute document
    # frequencies efficiently.
    tokenized = [_tokenize(chunk.get("text", "")) for chunk in chunks]
    if not any(tokenized):
        # Ensure the BM25 implementation receives an iterable even when all
        # chunks are empty to avoid runtime errors.
        tokenized = [[] for _ in chunks]
    bm25 = RankBM25Okapi(tokenized)
    id_map = [chunk["id"] for chunk in chunks]
    return {"bm25": bm25, "tokenized": tokenized, "id_map": id_map}


def search_bm25(query: str, index: dict, k: int) -> List[tuple[str, float]]:
    """Run a BM25 search and return the top ``k`` candidates."""

    if not query.strip():
        return []
    if not index:
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    bm25: RankBM25Okapi = index.get("bm25")
    if bm25 is None:
        return []
    # Compute relevance scores for the query against the full chunk corpus.
    scores = bm25.get_scores(query_tokens)
    id_map: Iterable[str] = index.get("id_map", [])
    paired = list(zip(id_map, scores))
    # Sort in descending order so the most relevant chunks appear first.
    paired.sort(key=lambda item: float(item[1]), reverse=True)
    top = paired[:k]
    return [(chunk_id, float(score)) for chunk_id, score in top if chunk_id is not None]


__all__ = ["build_bm25", "search_bm25"]
