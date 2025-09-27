"""Chunking utilities for parsed documents.

The Gradio interface depends on a legacy ``chunk_documents`` helper that used
to live alongside the old semantic chunker.  The CPU-first rewrite keeps the
new :class:`~chunking.chunker.Chunker` class as the source of truth while this
module exposes a compatibility shim returning the lightweight chunk objects the
UI still expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from parser.config import ParserConfig
from parser.types import ParsedDocument

from .chunker import ChunkRecord, Chunker


@dataclass
class LegacyChunk:
    """Minimal chunk representation consumed by ``app.parse_batch``."""

    doc_id: str
    doc_name: str
    page_start: int
    page_end: int
    section_title: str | None
    text: str
    token_len: int
    meta: Dict[str, object]


def _page_span_bounds(spans: Sequence[Dict[str, object]]) -> Tuple[int, int]:
    pages = [span.get("page") for span in spans if span.get("page") is not None]
    if not pages:
        return 1, 1
    # Internal page indices are zero-based; convert to the one-based numbering
    # expected throughout the UI.
    return min(pages) + 1, max(pages) + 1


def _section_hint(record: ChunkRecord) -> str | None:
    return record.section_hints[0] if record.section_hints else None


def _chunk_meta(record: ChunkRecord) -> Dict[str, object]:
    return {
        "chunk_id": record.chunk_id,
        "type": record.type,
        "page_spans": record.page_spans,
        "section_hints": record.section_hints,
        "neighbors": record.neighbors,
        "table_csv": record.table_csv,
        "evidence_offsets": record.evidence_offsets,
        "provenance": record.provenance,
    }


def _chunker_for_doc(doc: ParsedDocument) -> Chunker:
    config_dict = dict(doc.config_used) if getattr(doc, "config_used", None) else {}
    config = ParserConfig(**config_dict) if config_dict else ParserConfig()
    return Chunker(config)


def chunk_documents(
    documents: Sequence[ParsedDocument],
    *,
    mode: str | None = None,
) -> Tuple[List[LegacyChunk], Dict[str, object]]:
    """Chunk parsed documents and emit legacy-friendly objects plus stats."""

    # Only the TF–IDF-driven chunker is currently implemented; ``mode`` is kept
    # for backwards compatibility with the previous public signature.
    _ = mode

    chunks: List[LegacyChunk] = []
    stats = {
        "documents": len(documents),
        "total_chunks": 0,
        "body_chunks": 0,
        "caption_chunks": 0,
        "footnote_chunks": 0,
        "per_doc": {},
    }

    for document in documents:
        chunker = _chunker_for_doc(document)
        records = chunker.chunk_document(document)
        doc_name = Path(getattr(document, "file_path", "")).name or document.doc_id
        doc_stats = {
            "chunks": len(records),
            "body": 0,
            "caption": 0,
            "footnote": 0,
        }
        for record in records:
            page_start, page_end = _page_span_bounds(record.page_spans)
            legacy = LegacyChunk(
                doc_id=document.doc_id,
                doc_name=doc_name,
                page_start=page_start,
                page_end=page_end,
                section_title=_section_hint(record),
                text=record.text,
                token_len=record.tokens_est,
                meta=_chunk_meta(record),
            )
            chunks.append(legacy)
            stats_key = record.type
            if stats_key in ("body", "caption", "footnote"):
                stats[f"{stats_key}_chunks"] += 1
                doc_stats[stats_key] += 1
        stats["total_chunks"] += len(records)
        stats["per_doc"][document.doc_id] = doc_stats

    return chunks, stats


__all__ = ["ChunkRecord", "Chunker", "LegacyChunk", "chunk_documents"]

