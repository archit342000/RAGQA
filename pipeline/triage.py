from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List

try:  # pragma: no cover - optional dependency guard
    import fitz  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - imported lazily at runtime
    fitz = None  # type: ignore

from .config import PipelineConfig
from .ids import compute_doc_id

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PageTriageResult:
    doc_id: str
    doc_name: str
    page_number: int
    char_count: int
    text_coverage: float
    docling_ok: bool
    ocr_used: bool
    layout_rescue: bool
    latency_ms: float
    text: str
    bbox_spans: List[tuple[float, float, float, float]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    stage_used: str = "triage"
    fallback_applied: bool = False
    error_codes: List[str] = field(default_factory=list)

    @property
    def docling_empty(self) -> bool:
        return not self.docling_ok or not self.text.strip()


@dataclass(slots=True)
class PageTriageSummary:
    doc_id: str
    file_name: str
    pdf_bytes: bytes
    pages: List[PageTriageResult]

    def to_csv_rows(self) -> List[List[str]]:
        rows = [[
            "doc_id",
            "page",
            "stage_used",
            "latency_ms",
            "text_len",
            "fallback_applied",
            "error_codes",
        ]]
        for page in self.pages:
            rows.append([
                page.doc_id,
                str(page.page_number),
                f"{page.latency_ms:.3f}",
                str(len(page.text)),
                "true" if page.fallback_applied else "false",
                ";".join(page.error_codes or page.errors),
            ])
        return rows


def _load_pdf_bytes(path: str | bytes) -> bytes:
    if isinstance(path, bytes):
        return path
    with open(path, "rb") as f:
        return f.read()


def triage_document(pdf_path: str, config: PipelineConfig) -> PageTriageSummary:
    start_time = time.perf_counter()
    try:
        pdf_bytes = _load_pdf_bytes(pdf_path)
    except OSError as exc:  # pragma: no cover - filesystem errors rare
        raise RuntimeError(f"Failed to read PDF: {exc}")

    doc_id = compute_doc_id(pdf_bytes)
    doc_name = pdf_path.split("/")[-1]
    logger.debug("Starting triage for %s (%s)", doc_name, doc_id)

    if fitz is None:
        raise RuntimeError("pymupdf (fitz) is required for triage but is not installed")

    pages: List[PageTriageResult] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for idx, page in enumerate(doc, start=1):
            page_start = time.perf_counter()
            errors: List[str] = []
            text = ""
            bbox_spans: List[tuple[float, float, float, float]] = []
            char_count = 0
            text_coverage = 0.0
            try:
                text = page.get_text("text")
                char_count = len(text)
                blocks = page.get_text("blocks") or []
                bbox_spans = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in blocks]
                page_area = float(page.rect.width * page.rect.height)
                if page_area > 0:
                    coverage_area = sum(
                        max(0.0, (b[2] - b[0]) * (b[3] - b[1])) for b in blocks
                    )
                    text_coverage = min(1.0, coverage_area / page_area)
            except Exception as exc:  # pragma: no cover - depends on PDF content
                errors.append(str(exc))
            elapsed_ms = (time.perf_counter() - page_start) * 1000
            pages.append(
                PageTriageResult(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page_number=idx,
                    char_count=char_count,
                    text_coverage=text_coverage,
                    docling_ok=bool(text),
                    ocr_used=False,
                    layout_rescue=False,
                    latency_ms=elapsed_ms,
                    text=text,
                    bbox_spans=bbox_spans,
                    errors=errors,
                )
            )
    finally:
        doc.close()
    logger.debug("Triage completed for %s in %.2f seconds", doc_name, time.perf_counter() - start_time)
    return PageTriageSummary(
        doc_id=doc_id, file_name=doc_name, pdf_bytes=pdf_bytes, pages=pages
    )
