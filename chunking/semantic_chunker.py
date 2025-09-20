"""Semantic chunking helper built on top of LangChain's ``SemanticChunker``.

LangChain has moved the semantic chunking implementation between packages a
few times. To keep the HF Space lightweight while still benefiting from the
semantic splitter when available, we dynamically probe the known import
locations and fall back gracefully when the dependency is missing. Callers can
therefore attempt semantic chunking unconditionally and the helper returns an
empty list when the feature is unavailable.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import List

logger = logging.getLogger(__name__)

try:  # Optional heavy dependency; gracefully degrade when unavailable.
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        try:
            from langchain_experimental.text_splitters.semantic_chunker import SemanticChunker  # type: ignore
        except ImportError:
            try:
                from langchain_text_splitters import SemanticChunker
            except ImportError:
                from langchain_text_splitters.semantic_chunker import SemanticChunker  # type: ignore
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:  # compatibility with newer package layout
        from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings  # type: ignore
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document  # type: ignore
    _SEMANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency is optional in tests
    SemanticChunker = None  # type: ignore
    HuggingFaceEmbeddings = None  # type: ignore
    Document = None  # type: ignore
    _SEMANTIC_AVAILABLE = False

_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


def _to_sentences(text: str, max_sentences: int) -> List[str]:
    """Naively split text into sentences with a configurable cap."""
    sentences: List[str] = []
    cursor = 0
    for match in _SENTENCE_SPLIT_REGEX.finditer(text):
        segment = text[cursor:match.start()].strip()
        if segment:
            sentences.append(segment)
        cursor = match.end()
        if len(sentences) >= max_sentences:
            break
    tail = text[cursor:].strip()
    if tail and len(sentences) < max_sentences:
        sentences.append(tail)
    return sentences if sentences else [text.strip()]


@lru_cache(maxsize=2)
def _get_embeddings(model_name: str):
    """Instantiate the HuggingFace embedding model once per process."""
    if not _SEMANTIC_AVAILABLE:  # pragma: no cover - handled by caller
        raise RuntimeError("Semantic chunker dependencies are unavailable")
    return HuggingFaceEmbeddings(model_name=model_name)


@lru_cache(maxsize=2)
def _get_chunker(model_name: str) -> SemanticChunker:
    """Construct the LangChain ``SemanticChunker`` for the given model."""
    if not _SEMANTIC_AVAILABLE:  # pragma: no cover
        raise RuntimeError("Semantic chunker dependencies are unavailable")
    embeddings = _get_embeddings(model_name)
    return SemanticChunker(embeddings)


def semantic_segments(text: str, model_name: str, max_sentences: int = 300) -> List[str]:
    """Return semantic segments for a blob of prose text.

    When LangChain or its dependencies are unavailable, the function returns an
    empty list and the caller should fall back to a deterministic strategy.
    """

    if not _SEMANTIC_AVAILABLE:
        logger.warning("SemanticChunker unavailable; falling back to fixed windowing")
        return []

    sentences = _to_sentences(text, max_sentences=max_sentences)
    if not sentences:
        return []

    # ``SemanticChunker`` expects LangChain ``Document`` objects and performs
    # sentence-level grouping internally. We feed it the reconstructed sentence
    # list and let the driver decide how to pack the resulting segments.
    chunker = _get_chunker(model_name)
    docs = chunker.split_documents([Document(page_content=" ".join(sentences))])
    segments = [doc.page_content.strip() for doc in docs if doc.page_content.strip()]
    return segments
