"""High-level orchestration helpers for the pdfchunks pipeline."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import fitz  # type: ignore

from .audit.guards import run_audits
from .chunking.chunker import Chunk, Chunker
from .config import ParserConfig, load_config
from .parsing.baselines import BaselineEstimator
from .parsing.block_extractor import Block, BlockExtractor, DocumentLayout
from .parsing.classifier import BlockClassifier
from .parsing.ownership import assign_aux_ownership, assign_sections
from .serialize.serializer import Serializer
from .telemetry.metrics import compute_metrics
from .threading.threader import ThreadResult, ThreadUnit, Threader
from .types import ChunkPayload, ParsedDoc, ParsedPage, PipelineResult, RunReport

LOGGER = logging.getLogger(__name__)


def run_pipeline(
    file_paths: Sequence[str | Path],
    *,
    config: ParserConfig | None = None,
) -> PipelineResult:
    """Parse and chunk the provided documents in one pass."""

    if not file_paths:
        return PipelineResult(
            docs=[],
            chunks=[],
            chunk_stats={},
            report=RunReport(total_docs=0, parsed_docs=0, total_pages=0, skipped_docs=[]),
        )

    cfg = config or load_config()
    extractor = BlockExtractor(cfg.block_extraction)
    baseline_estimator = BaselineEstimator(cfg.baselines)
    classifier = BlockClassifier(cfg.classifier)
    threader = Threader(cfg.threading)
    chunker = Chunker(cfg.chunker)

    docs: List[ParsedDoc] = []
    all_chunks: List[ChunkPayload] = []
    chunk_stats: Dict[str, Dict[str, float]] = {}
    skipped: List[str] = []
    parsed_docs = 0
    total_pages = 0

    for raw_path in file_paths:
        pdf_path = Path(raw_path)
        display_name = pdf_path.name or str(pdf_path)
        if not pdf_path.exists():
            LOGGER.warning("File not found: %s", raw_path)
            skipped.append(display_name)
            continue
        try:
            with fitz.open(pdf_path) as document:
                total_pages += int(document.page_count or 0)
                doc_id = _stable_doc_id(pdf_path)
                doc_name = display_name
                layout = extractor.extract(document, doc_id=doc_id)
                if not layout.blocks:
                    skipped.append(display_name)
                    LOGGER.info("No blocks extracted for %s; skipping", display_name)
                    continue
                baselines = baseline_estimator.fit(layout)
                labels = classifier.classify(layout.blocks, baselines)
                assignments = assign_aux_ownership(assign_sections(labels))
                thread_result = threader.thread(assignments, doc_id=doc_id)
                serializer = Serializer()
                ordered_units = serializer.serialize(thread_result.units)
                serialized = ThreadResult(units=ordered_units, delayed_aux_counts=thread_result.delayed_aux_counts)
                chunks = chunker.chunk(ordered_units)
                run_audits(serialized, chunks, cfg.audits)
                metrics = compute_metrics(serialized, chunks)

                section_titles = _collect_section_titles(ordered_units)
                doc_chunks = _build_chunk_payloads(doc_id, doc_name, section_titles, chunks)
                doc_pages = _build_pages(doc_id, doc_name, layout)
                total_chars = sum(len(page.text) for page in doc_pages)
                doc_stats = {
                    "main_units": float(metrics.main_units),
                    "aux_units": float(metrics.aux_units),
                    "aux_sections": float(metrics.aux_sections),
                    "main_chunks": float(sum(1 for chunk in doc_chunks if chunk.role == "MAIN")),
                    "aux_chunks": float(sum(1 for chunk in doc_chunks if chunk.role == "AUX")),
                }
                parsed_doc = ParsedDoc(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    pages=doc_pages,
                    total_chars=total_chars,
                    parser_used="pdfchunks",
                    stats=doc_stats,
                    meta={"file_name": doc_name, "source_path": str(pdf_path)},
                )

                docs.append(parsed_doc)
                all_chunks.extend(doc_chunks)
                chunk_stats[doc_id] = {
                    "main_chunks": doc_stats["main_chunks"],
                    "aux_chunks": doc_stats["aux_chunks"],
                }
                parsed_docs += 1
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Failed to parse %s: %s", raw_path, exc)
            skipped.append(display_name)
            continue

    report = RunReport(
        total_docs=len(file_paths),
        parsed_docs=parsed_docs,
        total_pages=total_pages,
        skipped_docs=skipped,
        message="" if not skipped else "Some documents failed to parse.",
    )

    return PipelineResult(docs=docs, chunks=all_chunks, chunk_stats=chunk_stats, report=report)


def _stable_doc_id(path: Path) -> str:
    try:
        stat = path.stat()
        token = f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        token = path.name
    digest = hashlib.sha1(token.encode("utf-8"), usedforsecurity=False)
    return digest.hexdigest()[:12]


def _collect_section_titles(units: Iterable[ThreadUnit]) -> Dict[int, str]:
    titles: Dict[int, str] = {}
    for unit in units:
        if unit.role == "MAIN" and unit.subtype == "heading" and unit.text:
            titles[unit.section_seq] = unit.text
    return titles


def _build_chunk_payloads(
    doc_id: str,
    doc_name: str,
    section_titles: Dict[int, str],
    chunks: Sequence[Chunk],
) -> List[ChunkPayload]:
    payloads: List[ChunkPayload] = []
    for chunk in chunks:
        meta = {
            "role": chunk.role,
            "section_seq": str(chunk.section_seq),
        }
        if chunk.subtype:
            meta["subtype"] = chunk.subtype
        payloads.append(
            ChunkPayload(
                id=chunk.chunk_id,
                doc_id=doc_id,
                doc_name=doc_name,
                role=chunk.role,
                section_seq=chunk.section_seq,
                section_title=section_titles.get(chunk.section_seq),
                text=chunk.text,
                token_len=chunk.token_count,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                meta=meta,
            )
        )
    return payloads


def _build_pages(doc_id: str, doc_name: str, layout: DocumentLayout) -> List[ParsedPage]:
    pages: Dict[int, List[Block]] = defaultdict(list)
    for block in layout.blocks:
        pages[block.page_num].append(block)

    parsed_pages: List[ParsedPage] = []
    offset = 0
    for page_index in sorted(pages):
        text = _join_page_blocks(pages[page_index])
        if not text:
            continue
        start = offset
        end = start + len(text)
        metadata = {
            "file_name": doc_name,
            "page_label": str(page_index + 1),
        }
        parsed_pages.append(
            ParsedPage(
                doc_id=doc_id,
                page_num=page_index + 1,
                text=text,
                char_range=(start, end),
                metadata=metadata,
                offset_start=start,
                offset_end=end,
            )
        )
        offset = end
    return parsed_pages


def _join_page_blocks(blocks: Iterable[Block]) -> str:
    ordered = sorted(blocks, key=lambda block: (block.y0, block.x0))
    parts = [block.text.strip() for block in ordered if block.text.strip()]
    return "\n\n".join(parts).strip()

