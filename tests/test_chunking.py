"""Unit tests covering both semantic and fixed chunking pathways."""

from __future__ import annotations

from pathlib import Path

import pytest

from chunking.driver import chunk_documents
from parser.types import ParsedDoc, ParsedPage
import chunking.semantic_chunker as semantic_module


def _make_page(doc_id: str, page_num: int, text: str, file_name: str = "fixture.pdf") -> ParsedPage:
    """Helper for creating ``ParsedPage`` fixtures with minimal boilerplate."""
    metadata = {"file_name": file_name}
    return ParsedPage(
        doc_id=doc_id,
        page_num=page_num,
        text=text,
        char_range=(0, len(text)),
        metadata=metadata,
    )


def _make_doc(doc_id: str, pages: list[ParsedPage], parser_used: str = "pypdf") -> ParsedDoc:
    """Bundle pages into a ``ParsedDoc`` for the chunker to consume."""
    return ParsedDoc(
        doc_id=doc_id,
        pages=pages,
        total_chars=sum(len(p.text) for p in pages),
        parser_used=parser_used,
        stats={},
    )


@pytest.fixture(autouse=True)
def patch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENIZER_NAME", "hf-internal-testing/llama-tokenizer")
    monkeypatch.setenv("MAX_TOTAL_TOKENS_FOR_CHUNKING", "50000")


def test_semantic_chunking_produces_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Semantic mode should respect the semantic splitter when available."""
    calls: list[str] = []

    def fake_semantic_segments(text: str, model_name: str, max_sentences: int = 300):  # type: ignore[override]
        calls.append(text)
        return ["Paragraph one." , "Paragraph two continuing the discussion."]

    monkeypatch.setattr(semantic_module, "semantic_segments", fake_semantic_segments)

    doc = _make_doc(
        "doc-sem",
        [
            _make_page("doc-sem", 1, "Heading\n\nParagraph one. Paragraph two continuing the discussion."),
        ],
    )

    chunks, stats = chunk_documents([doc], mode="semantic")

    assert chunks, "Expected at least one semantic chunk"
    assert chunks[0].meta["strategy"].startswith("semantic")
    assert stats["doc-sem"]["chunks"] == len(chunks)
    assert 0 < chunks[0].token_len < 900


def test_fixed_chunking_window_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixed mode emits overlapping windows that respect token limits."""
    long_text = " ".join(["token" + str(i) for i in range(1000)])
    doc = _make_doc(
        "doc-fixed",
        [
            _make_page("doc-fixed", 1, long_text),
        ],
    )

    chunks, _ = chunk_documents([doc], mode="fixed")

    assert len(chunks) >= 2, "Fixed mode should create multiple windows for long text"
    lengths = [chunk.token_len for chunk in chunks]
    assert max(lengths) <= 900
    assert min(lengths) >= 200


def test_table_heavy_page_falls_back_to_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Table-heavy content should bypass the semantic chunker automatically."""
    monkeypatch.setattr(semantic_module, "semantic_segments", lambda *args, **kwargs: pytest.fail("Semantic chunker should not run"))

    table_text = "Row | Col\n1 | 2 | 3\n4 | 5 | 6"
    doc = _make_doc(
        "doc-table",
        [_make_page("doc-table", 1, table_text)],
    )

    chunks, stats = chunk_documents([doc], mode="semantic")

    assert chunks
    assert all(chunk.meta["strategy"].startswith("fixed") for chunk in chunks)
    assert stats["doc-table"]["strategies"]["fixed"] >= 1


def test_multi_doc_chunking_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-document runs should keep doc_ids intact across chunks."""
    doc_a = _make_doc(
        "doc-a",
        [
            _make_page("doc-a", 1, "Document A content sentence one. Sentence two."),
            _make_page("doc-a", 2, "Document A page two with more prose."),
        ],
    )
    doc_b = _make_doc(
        "doc-b",
        [_make_page("doc-b", 1, "Document B single page text.")],
    )

    chunks, stats = chunk_documents([doc_a, doc_b], mode="fixed")

    doc_ids = {chunk.doc_id for chunk in chunks}
    assert {"doc-a", "doc-b"}.issubset(doc_ids)
    assert set(stats.keys()) == {"doc-a", "doc-b"}


def test_chunk_metadata_preserves_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Page anchors and doc names must flow through to chunk metadata."""
    doc = _make_doc(
        "doc-pages",
        [
            _make_page("doc-pages", 1, "First page text."),
            _make_page("doc-pages", 2, "Second page text that extends."),
        ],
    )

    chunks, _ = chunk_documents([doc], mode="fixed")

    assert all(chunk.page_start <= chunk.page_end for chunk in chunks)
    assert all(chunk.doc_name for chunk in chunks)
