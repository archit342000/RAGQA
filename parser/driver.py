"""High-level parsing entrypoints used by the UI and tests.

The driver orchestrates the entire parsing lifecycle:

* Reads environment flags that tweak heuristics.
* Invokes the lightweight ``pypdf`` extractor first and falls back to the more
  robust ``unstructured`` pipeline when necessary.
* Normalises, cleans, and annotates each page with metadata that downstream RAG
  components can rely on for citation anchors.
* Exposes batch helpers that make it easy for user interfaces to parse multiple
  documents while respecting total page budgets.

The functions herein are intentionally deterministic so unit tests and Spaces
deployments behave identically.
"""

from __future__ import annotations

import logging
import os
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .cleaning import clean_text_block, remove_headers_footers
from .metrics import compute_document_metrics, compute_density_metrics, compute_layout_features
from .pdf_pypdf import PDFEncryptedError, extract_pdf_with_pypdf
from .pdf_unstructured import extract_pdf_with_unstructured
from .types import ParsedDoc, ParsedPage, RunReport
from env_loader import load_dotenv_once

logger = logging.getLogger(__name__)

_DEFAULT_STRATEGY = "fast"

# Ensure environment defaults are materialised before any helper consults them.
load_dotenv_once()


def _get_env_int(name: str, default: int) -> int:
    """Fetch an integer from the environment, falling back safely.

    Parameters
    ----------
    name:
        Environment variable to inspect.
    default:
        Value returned when the variable is unset or unparsable.
    """
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int for %s=%s; using default %d", name, value, default)
        return default


def _get_env_float(name: str, default: float) -> float:
    """Fetch a float from the environment with defensive parsing."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%s; using default %.2f", name, value, default)
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


def _build_page_metadata(
    *,
    doc_id: str,
    page_num: int,
    text: str,
    offset: int,
    source: str,
    strategy: str,
    file_name: str,
    fallback_reason: str,
    warnings: List[str],
    truncated: bool,
    hi_res_blocked: bool,
) -> ParsedPage:
    """Attach standard metadata to a parsed page for downstream use."""
    metadata: Dict[str, str] = {
        "source": source,
        "strategy": strategy,
        "file_name": file_name,
    }
    if fallback_reason:
        metadata["fallback_reason"] = fallback_reason
    if warnings:
        metadata["warning"] = "; ".join(warnings)
    if truncated:
        metadata["truncated"] = "true"
    if hi_res_blocked:
        metadata["hi_res_blocked"] = "true"

    char_range = (offset, offset + len(text))
    return ParsedPage(doc_id=doc_id, page_num=page_num, text=text, char_range=char_range, metadata=metadata)


def parse_pdf(
    file_path: str,
    *,
    strategy_env: str | None = None,
    doc_id: str | None = None,
    max_pages_override: int | None = None,
    file_name: str | None = None,
) -> ParsedDoc:
    """Parse a PDF into normalised pages with metadata and metrics.

    Parameters mirror the low-level extractor but incorporate heuristics that
    decide when to fall back to ``unstructured``. All numeric thresholds are
    sourced from environment variables to keep the public API stable.
    """

    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc_identifier = doc_id or _make_doc_id(pdf_path)

    max_pages_env = _get_env_int("MAX_PAGES", 800)
    min_chars_per_page = _get_env_int("MIN_CHARS_PER_PAGE", 200)
    fallback_empty_ratio = _get_env_float("FALLBACK_EMPTY_PAGE_RATIO", 0.3)

    effective_max_pages = max_pages_env
    if max_pages_override is not None:
        effective_max_pages = min(max_pages_env, max(max_pages_override, 0))
    if effective_max_pages <= 0:
        return ParsedDoc(
            doc_id=doc_identifier,
            pages=[],
            total_chars=0,
            parser_used="pypdf",
            stats={
                "total_pages": 0.0,
                "total_chars": 0.0,
                "fallback_triggered": 0.0,
                "fallback_reason": "none",
                "parse_duration_seconds": 0.0,
                "hi_res_blocked": 0.0,
            },
        )

    strategy_requested_raw = (strategy_env or os.getenv("UNSTRUCTURED_STRATEGY", _DEFAULT_STRATEGY)).lower()
    enable_hi_res = _get_env_bool("ENABLE_HI_RES", False)
    hi_res_requested = strategy_requested_raw == "hi_res"
    hi_res_blocked = hi_res_requested and not enable_hi_res
    effective_strategy = "fast" if hi_res_blocked else strategy_requested_raw

    timings: Dict[str, float] = {}

    overall_start = time.perf_counter()
    try:
        pypdf_pages, pypdf_meta = extract_pdf_with_pypdf(str(pdf_path), max_pages=effective_max_pages)
    except PDFEncryptedError as exc:
        raise RuntimeError(str(exc)) from exc

    pypdf_elapsed = float(pypdf_meta.get("elapsed", "0") or 0)
    timings["pypdf"] = pypdf_elapsed
    truncated = pypdf_meta.get("truncated", "false").lower() == "true"

    density_metrics_raw = compute_density_metrics(pypdf_pages, min_chars_per_page=min_chars_per_page)
    layout_metrics_raw = compute_layout_features(pypdf_pages)

    total_pages_raw = int(density_metrics_raw.get("total_pages", 0.0))
    empty_ratio = density_metrics_raw.get("empty_or_low_ratio", 0.0)
    layout_complexity_score = layout_metrics_raw.get("layout_complexity_score", 0.0)

    fallback_reason = ""
    if total_pages_raw == 0:
        fallback_reason = "no_text_extracted"
    elif empty_ratio >= fallback_empty_ratio:
        fallback_reason = "empty_ratio"
    elif layout_complexity_score >= 0.6:
        fallback_reason = "layout_complexity"

    warnings: List[str] = []
    if hi_res_blocked:
        warnings.append("hi_res requested but unavailable in CPU build")

    parser_used = "pypdf"
    pages_source = pypdf_pages[:effective_max_pages]

    if fallback_reason:
        pages_source, unstructured_meta = extract_pdf_with_unstructured(
            str(pdf_path),
            effective_strategy,
            max_pages=effective_max_pages,
            enable_hi_res=enable_hi_res,
        )
        parser_used = f"unstructured-{unstructured_meta.get('strategy_used', 'fast')}"
        timings["unstructured"] = float(unstructured_meta.get("elapsed", "0") or 0)
        truncated = truncated or (len(pages_source) > effective_max_pages)
        warnings_meta = unstructured_meta.get("warnings", "")
        if warnings_meta:
            warnings.extend([w for w in warnings_meta.split("|") if w])
        if hi_res_requested and enable_hi_res and "hi_res" not in parser_used:
            warnings.append("hi_res requested but unavailable in CPU build")
        if len(pages_source) > effective_max_pages:
            pages_source = pages_source[:effective_max_pages]
    else:
        pages_source = pages_source[:effective_max_pages]

    if len(pages_source) >= effective_max_pages and effective_max_pages < max_pages_env:
        truncated = True

    warnings = [w.strip() for w in warnings if w.strip()]

    cleaned = [clean_text_block(page) for page in pages_source]
    without_headers = remove_headers_footers(cleaned)
    final_pages = [clean_text_block(page) for page in without_headers]

    parsed_pages: List[ParsedPage] = []
    offset = 0
    for idx, page_text in enumerate(final_pages, start=1):
        parsed_page = _build_page_metadata(
            doc_id=doc_identifier,
            page_num=idx,
            text=page_text,
            offset=offset,
            source="pypdf" if parser_used == "pypdf" else "unstructured",
            strategy=parser_used.split("-", 1)[-1] if "-" in parser_used else parser_used,
            file_name=file_name or pdf_path.name,
            fallback_reason=fallback_reason,
            warnings=warnings,
            truncated=truncated,
            hi_res_blocked=hi_res_blocked,
        )
        parsed_pages.append(parsed_page)
        offset = parsed_page.char_range[1]

    total_chars = offset
    stats = compute_document_metrics(final_pages, min_chars_per_page=min_chars_per_page)
    stats["fallback_triggered"] = 1.0 if fallback_reason else 0.0
    stats["fallback_reason"] = fallback_reason or "none"
    stats["pypdf_empty_ratio"] = float(empty_ratio)
    stats["layout_complexity_raw"] = float(layout_complexity_score)
    stats["elapsed_pypdf_seconds"] = float(timings.get("pypdf", 0.0))
    if "unstructured" in timings:
        stats["elapsed_unstructured_seconds"] = float(timings["unstructured"])
    stats["parse_duration_seconds"] = float(time.perf_counter() - overall_start)
    stats["hi_res_blocked"] = 1.0 if hi_res_blocked else 0.0
    stats["total_pages"] = float(len(final_pages))

    logger.info(
        "Parsed %s via %s: pages=%d chars=%d empty_ratio=%.2f layout=%.2f fallback=%s",
        pdf_path.name,
        parser_used,
        len(parsed_pages),
        total_chars,
        empty_ratio,
        layout_complexity_score,
        fallback_reason or "none",
    )

    return ParsedDoc(
        doc_id=doc_identifier,
        pages=parsed_pages,
        total_chars=total_chars,
        parser_used=parser_used,
        stats=stats,
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
        page = _build_page_metadata(
            doc_id=doc_identifier,
            page_num=idx,
            text=page_text,
            offset=offset,
            source="text",
            strategy="n/a",
            file_name=file_name or text_path.name,
            fallback_reason="",
            warnings=[],
            truncated=False,
            hi_res_blocked=False,
        )
        parsed_pages.append(page)
        offset = page.char_range[1]

    total_chars = offset
    stats = compute_document_metrics(final_pages, min_chars_per_page=min_chars_per_page)
    stats["fallback_triggered"] = 0.0
    stats["fallback_reason"] = "none"
    stats["parse_duration_seconds"] = float(0.0)
    stats["hi_res_blocked"] = 0.0
    stats["total_pages"] = float(len(final_pages))

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
    )


def parse_documents(
    file_paths: Sequence[str],
    *,
    strategy_env: str | None = None,
) -> Tuple[List[ParsedDoc], RunReport]:
    """Parse multiple documents sequentially, enforcing aggregate limits.

    The helper keeps a running tally of emitted pages so large uploads are
    truncated deterministically. It also reports summary data that the UI can
    surface to operators (e.g., which documents were skipped).
    """

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

        # Determine how many more pages we are allowed to process.
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
        # Note when we had to truncate a document midway.
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
