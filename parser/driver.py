"""High-level parsing entrypoints backed by the hybrid layout pipeline.

This module exposes the public parsing API used by the UI and tests. PDF
inputs are routed through the PyMuPDF → layout signal scoring → LayoutParser
fusion pipeline and emit :class:`ParsedDoc` objects that retain both the plain
text view used by legacy components and the rich block/anchor metadata required
by the new chunking stack.

Text/Markdown helpers remain lightweight and operate on the existing cleaning
utilities so that non-PDF inputs still round-trip through the same public API.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from env_loader import load_dotenv_once

from pipeline.ingest.pdf_parser import parse_pdf_with_pymupdf
from pipeline.layout.lp_fuser import FusedDocument, FusedPage, LayoutParserEngine, fuse_layout
from pipeline.layout.router import LayoutRoutingPlan, PageRoutingDecision, plan_layout_routing
from pipeline.layout.signals import PageLayoutSignals, compute_layout_signals
from pipeline.repair.repair_pass import EmbeddingFn, RepairStats, run_repair_pass
from pipeline.threading.threader import Threader, ThreadingReport

from .cleaning import clean_text_block, remove_headers_footers
from .metrics import compute_document_metrics
from .types import ParsedDoc, ParsedPage, RunReport

logger = logging.getLogger(__name__)

_DEFAULT_STRATEGY = "fast"

# Ensure environment defaults are materialised before any helper consults them.
load_dotenv_once()

try:  # Optional heavy dependency used for semantic refinements.
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - dependency unavailable during tests
    SentenceTransformer = None  # type: ignore[misc]


def _get_env_int(name: str, default: int) -> int:
    """Fetch an integer from the environment, falling back safely."""

    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int for %s=%s; using default %d", name, value, default)
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    """Interpret common truthy strings from the environment."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _make_doc_id(file_path: Path) -> str:
    """Create a stable identifier for the document using file metadata."""

    try:
        stat = file_path.stat()
        token = f"{file_path.name}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        token = file_path.name
    digest = hashlib.sha1(token.encode("utf-8"), usedforsecurity=False)
    return digest.hexdigest()[:12]


@lru_cache(maxsize=1)
def _load_embedder(model_name: str | None = None) -> EmbeddingFn | None:
    """Lazily instantiate a sentence-transformer embedder when available."""

    if os.getenv("PIPELINE_SKIP_EMBEDDER", "false").strip().lower() == "true":
        return None
    if SentenceTransformer is None:
        return None
    name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    try:
        model = SentenceTransformer(name)
    except Exception as exc:  # pragma: no cover - dependency/runtime issues
        logger.warning("Failed to load sentence-transformer %s: %s", name, exc)
        return None

    def _embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors = model.encode(list(texts), normalize_embeddings=False)
        return [list(map(float, vector)) for vector in vectors]

    return _embed


def _compose_page_text(page: FusedPage) -> Tuple[str, Dict[str, object]]:
    """Return the human-readable page text plus auxiliary metadata."""

    parts: List[str] = []
    main_blocks_meta: List[Dict[str, object]] = []
    for block in page.main_flow:
        snippet = block.text.strip()
        if snippet:
            parts.append(snippet)
        main_blocks_meta.append(
            {
                "block_id": block.block_id,
                "region": block.region_label,
                "column": block.column,
                "char_start": block.char_start,
                "char_end": block.char_end,
                "has_anchor": bool(block.metadata.get("has_anchor_refs")),
            }
        )

    aux_blocks_meta: List[Dict[str, object]] = []
    aux_parts: List[str] = []
    for aux in page.auxiliaries:
        category = aux.aux_category or aux.region_label or "aux"
        prefix = f"[{category.upper()}] " if category else ""
        anchor = f"{aux.anchor} " if aux.anchor else ""
        snippet = aux.text.strip()
        if snippet:
            aux_parts.append(f"{anchor}{prefix}{snippet}")
        aux_blocks_meta.append(
            {
                "block_id": aux.block_id,
                "category": category,
                "anchor": aux.anchor,
                "char_start": aux.char_start,
                "char_end": aux.char_end,
            }
        )

    if aux_parts:
        parts.append("\n".join(aux_parts))

    text = "\n\n".join(part for part in parts if part)
    meta = {"main_blocks": main_blocks_meta, "auxiliary_blocks": aux_blocks_meta}
    return text, meta


def _page_metadata(
    *,
    page: FusedPage,
    signal: PageLayoutSignals,
    decision: PageRoutingDecision | None,
    file_name: str,
    truncated: bool,
    hi_res_blocked: bool,
) -> Dict[str, str]:
    metadata: Dict[str, str] = {
        "source": "pymupdf",
        "strategy": "hybrid",
        "file_name": file_name,
        "layout_score": f"{signal.page_score:.3f}",
    }
    if decision is not None:
        metadata["lp_used"] = "true" if decision.use_layout_parser else "false"
        if decision.triggers:
            metadata["lp_triggers"] = ",".join(decision.triggers)
        if decision.model:
            metadata["lp_model"] = decision.model
        if decision.neighbor:
            metadata["lp_neighbor"] = "true"
    if truncated:
        metadata["truncated"] = "true"
    if hi_res_blocked:
        metadata["hi_res_blocked"] = "true"
    return metadata


def _summarise_signals(signals: Sequence[PageLayoutSignals]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for signal in signals:
        ordered = sorted(signal.normalized.items(), key=lambda item: item[1], reverse=True)
        summary.append(
            {
                "page": signal.page_number,
                "score": signal.page_score,
                "top_signals": ordered[:2],
            }
        )
    return summary


def parse_pdf(
    file_path: str,
    *,
    strategy_env: str | None = None,
    doc_id: str | None = None,
    max_pages_override: int | None = None,
    file_name: str | None = None,
) -> ParsedDoc:
    """Parse a PDF into normalised pages with hybrid layout metadata."""

    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc_identifier = doc_id or _make_doc_id(pdf_path)

    max_pages_env = _get_env_int("MAX_PAGES", 800)
    min_chars_per_page = _get_env_int("MIN_CHARS_PER_PAGE", 200)

    effective_max_pages = max_pages_env
    if max_pages_override is not None:
        effective_max_pages = min(max_pages_env, max(max_pages_override, 0))
    if effective_max_pages <= 0:
        return ParsedDoc(
            doc_id=doc_identifier,
            pages=[],
            total_chars=0,
            parser_used="pymupdf-hybrid",
            stats={
                "total_pages": 0.0,
                "total_chars": 0.0,
                "fallback_triggered": 0.0,
                "fallback_reason": "none",
                "parse_duration_seconds": 0.0,
                "hi_res_blocked": 0.0,
                "layout_parser_pages": 0.0,
            },
            meta={"file_name": file_name or pdf_path.name, "truncated": False},
        )

    strategy_requested_raw = (strategy_env or os.getenv("UNSTRUCTURED_STRATEGY", _DEFAULT_STRATEGY)).lower()
    enable_hi_res = _get_env_bool("ENABLE_HI_RES", False)
    hi_res_requested = strategy_requested_raw == "hi_res"
    hi_res_blocked = hi_res_requested and not enable_hi_res

    overall_start = time.perf_counter()
    document = parse_pdf_with_pymupdf(
        str(pdf_path),
        doc_id=doc_identifier,
        max_pages=effective_max_pages,
    )

    truncated = max_pages_override is not None and len(document.pages) >= max_pages_override
    signals = compute_layout_signals(document)

    repair_failures: Dict[int, int] = {}
    threading_report = ThreadingReport()
    plan = plan_layout_routing(document, signals, repair_failures=repair_failures)
    threader = Threader()
    engine = LayoutParserEngine()
    fused = fuse_layout(document, signals, plan, engine=engine)

    embedder = _load_embedder()
    fused, repair_stats, repair_failures = run_repair_pass(
        fused,
        signals,
        embedder=embedder,
        previous_failures=repair_failures,
    )
    fused, threading_report = threader.thread_document(document, fused, signals)
    if any(count >= 2 for count in repair_failures.values()):
        plan = plan_layout_routing(document, signals, repair_failures=repair_failures)
        fused = fuse_layout(document, signals, plan, engine=engine)
        fused, repair_stats, repair_failures = run_repair_pass(
            fused,
            signals,
            embedder=embedder,
            previous_failures=repair_failures,
        )
        fused, threading_report = threader.thread_document(document, fused, signals)

    parse_duration = time.perf_counter() - overall_start

    parsed_pages: List[ParsedPage] = []
    char_offset = 0
    decisions = {decision.page_number: decision for decision in plan.decisions}
    file_label = file_name or pdf_path.name

    for fused_page, signal in zip(fused.pages, signals):
        text, meta = _compose_page_text(fused_page)
        char_range = (char_offset, char_offset + len(text))
        metadata = _page_metadata(
            page=fused_page,
            signal=signal,
            decision=decisions.get(fused_page.page_number),
            file_name=file_label,
            truncated=truncated,
            hi_res_blocked=hi_res_blocked,
        )
        page = ParsedPage(
            doc_id=doc_identifier,
            page_num=fused_page.page_number,
            text=text,
            char_range=char_range,
            metadata=metadata,
            meta={
                "layout": {
                    "score": signal.page_score,
                    "signals": signal.normalized,
                },
                "blocks": meta,
            },
        )
        parsed_pages.append(page)
        char_offset = char_range[1]

    total_chars = char_offset
    stats = compute_document_metrics([page.text for page in parsed_pages], min_chars_per_page=min_chars_per_page)
    stats.update(
        {
            "fallback_triggered": 0.0,
            "fallback_reason": "none",
            "parse_duration_seconds": float(parse_duration),
            "hi_res_blocked": 1.0 if hi_res_blocked else 0.0,
            "layout_parser_pages": float(len(plan.selected_pages)),
            "layout_parser_ratio": float(plan.ratio),
            "repair_merged_blocks": float(repair_stats.merged_blocks),
            "repair_split_blocks": float(repair_stats.split_blocks),
            "footnotes_linked": float(repair_stats.footnotes_linked),
            "thread_aux_queued": float(threading_report.queued_aux),
            "thread_aux_placed": float(threading_report.placed_aux),
            "thread_aux_carried": float(threading_report.carried_aux),
            "thread_dehyphenated_pairs": float(threading_report.dehyphenated_pairs),
            "thread_audit_fixes": float(threading_report.audit_fixes),
            "total_pages": float(len(parsed_pages)),
            "total_chars": float(total_chars),
        }
    )

    doc_meta: Dict[str, object] = {
        "file_name": file_label,
        "truncated": truncated,
        "layout_plan": {
            "budget": plan.budget,
            "selected_pages": plan.selected_pages,
            "overflow": plan.overflow,
        },
        "repair": {
            "merged_blocks": repair_stats.merged_blocks,
            "split_blocks": repair_stats.split_blocks,
            "footnotes_linked": repair_stats.footnotes_linked,
            "failure_counts": repair_stats.failure_counts,
        },
        "threading": {
            "queued": threading_report.queued_aux,
            "placed": threading_report.placed_aux,
            "carried": threading_report.carried_aux,
            "dehyphenated": threading_report.dehyphenated_pairs,
            "audit_fixes": threading_report.audit_fixes,
        },
        "signals_summary": _summarise_signals(signals),
    }

    logger.info(
        "Parsed PDF %s: pages=%d lp_ratio=%.2f duration=%.2fs",
        file_label,
        len(parsed_pages),
        plan.ratio,
        parse_duration,
    )

    return ParsedDoc(
        doc_id=doc_identifier,
        pages=parsed_pages,
        total_chars=total_chars,
        parser_used="pymupdf-hybrid",
        stats=stats,
        meta=doc_meta,
        fused_document=fused,
        layout_signals=list(signals),
        routing_plan=plan,
    )


def parse_text(
    file_path: str,
    *,
    doc_id: str | None = None,
    max_pages_override: int | None = None,
    file_name: str | None = None,
) -> ParsedDoc:
    """Parse simple UTF-8 text or Markdown documents."""

    text_path = Path(file_path)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    doc_identifier = doc_id or _make_doc_id(text_path)
    max_pages_env = _get_env_int("MAX_PAGES", 800)
    min_chars_per_page = _get_env_int("MIN_CHARS_PER_PAGE", 200)

    effective_max_pages = max_pages_env
    if max_pages_override is not None:
        effective_max_pages = min(max_pages_env, max(max_pages_override, 0))
    if effective_max_pages <= 0:
        return ParsedDoc(
            doc_id=doc_identifier,
            pages=[],
            total_chars=0,
            parser_used="text",
            stats={
                "total_pages": 0.0,
                "total_chars": 0.0,
                "fallback_triggered": 0.0,
                "fallback_reason": "none",
                "parse_duration_seconds": 0.0,
                "hi_res_blocked": 0.0,
            },
            meta={"file_name": file_name or text_path.name, "truncated": False},
        )

    raw_text = text_path.read_text(encoding="utf-8", errors="replace")
    raw_pages = raw_text.split("\f") if "\f" in raw_text else [raw_text]
    raw_pages = raw_pages[:effective_max_pages]

    cleaned = [clean_text_block(page) for page in raw_pages]
    without_headers = remove_headers_footers(cleaned)
    final_pages = [clean_text_block(page) for page in without_headers]

    parsed_pages: List[ParsedPage] = []
    offset = 0
    for idx, page_text in enumerate(final_pages, start=1):
        metadata = {
            "source": "text",
            "strategy": "n/a",
            "file_name": file_name or text_path.name,
        }
        char_range = (offset, offset + len(page_text))
        page = ParsedPage(
            doc_id=doc_identifier,
            page_num=idx,
            text=page_text,
            char_range=char_range,
            metadata=metadata,
            meta={},
        )
        parsed_pages.append(page)
        offset = char_range[1]

    total_chars = offset
    stats = compute_document_metrics(final_pages, min_chars_per_page=min_chars_per_page)
    stats.update(
        {
            "fallback_triggered": 0.0,
            "fallback_reason": "none",
            "parse_duration_seconds": 0.0,
            "hi_res_blocked": 0.0,
            "total_pages": float(len(parsed_pages)),
            "total_chars": float(total_chars),
        }
    )

    logger.info(
        "Parsed text file %s: pages=%d chars=%d",
        text_path.name,
        len(parsed_pages),
        total_chars,
    )

    return ParsedDoc(
        doc_id=doc_identifier,
        pages=parsed_pages,
        total_chars=total_chars,
        parser_used="text",
        stats=stats,
        meta={"file_name": file_name or text_path.name, "truncated": False},
    )


def parse_documents(
    file_paths: Sequence[str],
    *,
    strategy_env: str | None = None,
) -> Tuple[List[ParsedDoc], RunReport]:
    """Parse multiple documents sequentially, enforcing aggregate limits."""

    max_total_pages = _get_env_int("MAX_TOTAL_PAGES", 1200)
    max_pages_per_doc = _get_env_int("MAX_PAGES", 800)

    normalized: List[Tuple[Path, str]] = []
    for path_str in file_paths:
        path = Path(path_str)
        normalized.append((path, path.suffix.lower()))

    docs: List[ParsedDoc] = []
    skipped_docs: List[str] = []
    total_pages = 0
    truncated_total = False

    for path, suffix in normalized:
        file_label = path.name or str(path)
        if not path.exists():
            skipped_docs.append(f"{file_label}: missing")
            continue

        remaining = max_total_pages - total_pages
        if remaining <= 0:
            truncated_total = True
            skipped_docs.append(f"{file_label}: skipped (total page limit)")
            continue

        allowed = min(max_pages_per_doc, remaining)
        doc_id = _make_doc_id(path)
        try:
            if suffix == ".pdf":
                doc = parse_pdf(
                    str(path),
                    strategy_env=strategy_env,
                    doc_id=doc_id,
                    max_pages_override=allowed,
                    file_name=file_label,
                )
            elif suffix in {".txt", ".md"}:
                doc = parse_text(
                    str(path),
                    doc_id=doc_id,
                    max_pages_override=allowed,
                    file_name=file_label,
                )
            else:
                skipped_docs.append(f"{file_label}: unsupported type")
                continue
        except Exception as exc:  # pragma: no cover - logged for visibility
            logger.exception("Failed to parse %s", path)
            skipped_docs.append(f"{file_label}: {exc}")
            continue

        total_pages += len(doc.pages)
        if len(doc.pages) >= allowed and allowed < max_pages_per_doc:
            truncated_total = True

        docs.append(doc)

    truncated_effective = truncated_total
    message = ""
    if truncated_effective:
        message = (
            f"Parsed {total_pages} pages across {len(docs)} docs (limit {max_total_pages}); "
            "some pages were skipped."
        )

    report = RunReport(
        total_docs=len(docs),
        total_pages=total_pages,
        truncated=truncated_effective,
        skipped_docs=skipped_docs,
        message=message,
    )

    return docs, report
