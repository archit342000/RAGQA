"""Hybrid chunking driver built on top of the fused layout pipeline."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

from env_loader import load_dotenv_once

from pipeline.chunking.chunker import Chunk as HybridChunk, DocumentChunker
from pipeline.layout.lp_fuser import FusedBlock, FusedDocument, FusedPage
from pipeline.layout.signals import PageLayoutSignals, PageSignalExtras
from pipeline.repair.repair_pass import EmbeddingFn
from parser.types import ParsedDoc

from chunking.types import Chunk

logger = logging.getLogger(__name__)

load_dotenv_once()

TOKENIZER_NAME = os.getenv("TOKENIZER_NAME")
MAX_TOTAL_TOKENS = int(os.getenv("MAX_TOTAL_TOKENS_FOR_CHUNKING", "300000"))
DEFAULT_CHUNK_MODE = os.getenv("CHUNK_MODE_DEFAULT", "semantic").lower()
DEFAULT_EMBEDDER_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_SIGNAL_NAMES = ("CIS", "OGR", "BXS", "DAS", "FVS", "ROJ", "TFI", "MSA", "FNL")

try:  # Optional dependency; unavailable in lightweight environments.
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - dependency missing in tests
    SentenceTransformer = None  # type: ignore[misc]


@lru_cache(maxsize=1)
def _load_embedder(model_name: str | None = None) -> EmbeddingFn | None:
    """Lazily instantiate the sentence-transformer embedder used by the chunker."""

    if os.getenv("PIPELINE_SKIP_EMBEDDER", "false").strip().lower() == "true":
        return None
    if SentenceTransformer is None:
        return None
    name = model_name or DEFAULT_EMBEDDER_MODEL
    try:
        model = SentenceTransformer(name)
    except Exception as exc:  # pragma: no cover - dependency/runtime issues
        logger.warning("Failed to load sentence-transformer %s: %s", name, exc)
        return None

    def _embed(texts: Sequence[str]):
        vectors = model.encode(list(texts), normalize_embeddings=False)
        return [list(map(float, vector)) for vector in vectors]

    return _embed


def _synthetic_pipeline_view(doc: ParsedDoc) -> Tuple[FusedDocument, List[PageLayoutSignals]]:
    """Build a minimal fused document/signals pair for non-PDF inputs."""

    if doc.fused_document is not None and doc.layout_signals is not None:
        return doc.fused_document, list(doc.layout_signals)

    pages: List[FusedPage] = []
    index: Dict[str, FusedBlock] = {}
    signals: List[PageLayoutSignals] = []
    for page in doc.pages:
        block_id = f"{doc.doc_id}_p{page.page_num}_b0"
        block = FusedBlock(
            block_id=block_id,
            page_number=page.page_num,
            text=page.text,
            bbox=(0.0, 0.0, 1.0, 1.0),
            block_type="main",
            region_label="text",
            aux_category=None,
            anchor=None,
            column=0,
            char_start=page.char_range[0],
            char_end=page.char_range[1],
            avg_font_size=12.0,
            metadata={"synthetic": True},
        )
        index[block_id] = block
        fused_page = FusedPage(
            page_number=page.page_num,
            width=1.0,
            height=1.0,
            main_flow=[block],
            auxiliaries=[],
        )
        pages.append(fused_page)

        extras = PageSignalExtras(
            column_count=1,
            column_assignments={block_id: 0},
            table_overlap_ratio=0.0,
            figure_overlap_ratio=0.0,
            dominant_font_size=12.0,
            footnote_block_ids=[],
            superscript_spans=0,
            total_line_count=1,
            has_normal_density=True,
            char_density=0.001,
        )
        raw = {name: 0.0 for name in _SIGNAL_NAMES}
        normalized = {name: 0.0 for name in _SIGNAL_NAMES}
        signals.append(
            PageLayoutSignals(
                page_number=page.page_num,
                raw=raw,
                normalized=normalized,
                page_score=0.0,
                extras=extras,
            )
        )

    return FusedDocument(doc_id=doc.doc_id, pages=pages, block_index=index), signals


def _convert_chunk(
    hybrid_chunk: HybridChunk,
    *,
    doc_id: str,
    doc_name: str,
    mode_label: str,
) -> Chunk:
    metadata = dict(hybrid_chunk.metadata)
    metadata["chunk_id"] = hybrid_chunk.chunk_id
    metadata["strategy"] = f"hybrid-{mode_label}"
    page_start = int(metadata.pop("page_start", 0) or 0)
    page_end = int(metadata.pop("page_end", page_start) or page_start)
    return Chunk(
        doc_id=doc_id,
        doc_name=doc_name,
        page_start=page_start,
        page_end=page_end,
        section_title=None,
        text=hybrid_chunk.text,
        token_len=int(hybrid_chunk.tokens),
        meta=metadata,
    )


def chunk_documents(
    docs: Sequence[ParsedDoc],
    *,
    mode: str | None = None,
    model_name: str | None = None,
) -> Tuple[List[Chunk], Dict[str, dict]]:
    """Chunk parsed documents into retrieval windows using the hybrid pipeline."""

    chunks: List[Chunk] = []
    chunk_stats: Dict[str, dict] = {}
    requested_mode = (mode or DEFAULT_CHUNK_MODE).lower()
    use_semantic = requested_mode != "fixed"
    embedder = _load_embedder(model_name if use_semantic else None) if use_semantic else None
    chunker = DocumentChunker(tokenizer_name=TOKENIZER_NAME, embedder=embedder)

    total_tokens_emitted = 0

    for doc in docs:
        doc_name = doc.pages[0].metadata.get("file_name") if doc.pages else doc.doc_id
        fused, signals = _synthetic_pipeline_view(doc)
        hybrid_chunks = chunker.chunk_document(fused, signals)
        doc_chunks: List[Chunk] = []
        attached_aux = 0
        table_chunks = 0

        for hybrid in hybrid_chunks:
            if total_tokens_emitted + hybrid.tokens > MAX_TOTAL_TOKENS:
                logger.warning("Token guardrail reached; stopping chunk generation for doc %s", doc.doc_id)
                break
            chunk = _convert_chunk(hybrid, doc_id=doc.doc_id, doc_name=doc_name or doc.doc_id, mode_label=requested_mode)
            meta_aux = chunk.meta.get("aux_attached")
            if isinstance(meta_aux, list):
                attached_aux += len(meta_aux)
            if chunk.meta.get("table_row_range"):
                table_chunks += 1
            chunks.append(chunk)
            doc_chunks.append(chunk)
            total_tokens_emitted += chunk.token_len

        if doc_chunks:
            avg_tokens = sum(ch.token_len for ch in doc_chunks) / len(doc_chunks)
        else:
            avg_tokens = 0.0

        stats_entry = {
            "doc_name": doc_name or doc.doc_id,
            "chunks": len(doc_chunks),
            "avg_tokens": round(avg_tokens, 2),
            "mode": requested_mode,
            "tables": table_chunks,
            "aux_attached": attached_aux,
        }
        if doc.routing_plan is not None:
            stats_entry["lp_ratio"] = float(doc.routing_plan.ratio)
            stats_entry["lp_pages"] = len(doc.routing_plan.selected_pages)
        chunk_stats[doc.doc_id] = stats_entry

    return chunks, chunk_stats
