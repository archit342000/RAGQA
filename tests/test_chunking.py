"""Chunking tests covering paragraph assembly and boundary detection."""
from __future__ import annotations

from chunking import Chunker
from parser.config import ParserConfig
from parser.types import BBox, CaptionSidecar, LineSpan, PageParseResult, ParsedDocument


def _make_line(page: int, idx: int, text: str, caption: bool = False) -> LineSpan:
    return LineSpan(
        page_index=page,
        line_index=idx,
        text=text,
        bbox=BBox(0, idx * 10, 10, idx * 10 + 5),
        char_start=idx * 10,
        char_end=idx * 10 + len(text),
        is_caption=caption,
    )


def _make_document() -> ParsedDocument:
    config = ParserConfig()
    pages = []
    # Page 0 with heading and caption
    lines = [
        _make_line(0, 0, "INTRODUCTION", caption=False),
        _make_line(0, 1, "This section explains the goal of the paper."),
        _make_line(0, 2, "Figure 1. System diagram", caption=True),
    ]
    lines[0].is_heading = True
    captions = [CaptionSidecar(page_index=0, anchor_line=2, text=lines[2].text, bbox=lines[2].bbox)]
    pages.append(PageParseResult(page_index=0, glyph_count=300, had_text=True, ocr_performed=False, lines=lines, captions=captions))

    # Page 1 with content that should trigger a topic boundary
    lines_1 = [
        _make_line(1, 0, "METHODS"),
        _make_line(1, 1, "We run several controlled experiments."),
        _make_line(1, 2, "Results show strong improvements."),
    ]
    lines_1[0].is_heading = True
    pages.append(PageParseResult(page_index=1, glyph_count=320, had_text=True, ocr_performed=False, lines=lines_1, captions=[]))

    doc = ParsedDocument(
        doc_id="doc",
        file_path="doc.pdf",
        pages=pages,
        config_used=config.to_dict(),
        parse_time_s=1.0,
        content_hash="hash",
    )
    return doc


def test_chunker_produces_body_and_caption_chunks() -> None:
    config = ParserConfig(chunk_token_target_min=10, chunk_token_target_max=50)
    chunker = Chunker(config)
    doc = _make_document()
    chunks = chunker.chunk_document(doc)
    body_chunks = [chunk for chunk in chunks if chunk.type == "body"]
    caption_chunks = [chunk for chunk in chunks if chunk.type == "caption"]
    assert len(body_chunks) >= 1
    assert len(caption_chunks) == 1
    assert caption_chunks[0].text.startswith("Figure 1")
    # captions should not be part of body text
    assert all("Figure 1" not in chunk.text for chunk in body_chunks)


def test_topic_boundaries_split_chunks() -> None:
    config = ParserConfig(chunk_token_target_min=5, chunk_token_target_max=50)
    chunker = Chunker(config)
    doc = _make_document()
    chunks = [chunk for chunk in chunker.chunk_document(doc) if chunk.type == "body"]
    assert len(chunks) >= 1
    # ensure neighbors are wired
    for first, second in zip(chunks, chunks[1:]):
        assert first.neighbors["next"] == second.chunk_id
        assert second.neighbors["prev"] == first.chunk_id
