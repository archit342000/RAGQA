from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

from .config import PipelineConfig
from .ids import compute_doc_id
from .multi_extractor import ExtractorPageVote, run_multi_extractor

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
    len_text_fitz: int = 0
    len_text_pdfium: int = 0
    len_text_pdfminer: int = 0
    has_type3: bool = False
    has_cid: bool = False
    has_tounicode: bool = True
    force_ocr: bool = False
    digital_text: bool = False
    path_used: str = "triage"
    filter_relaxed: bool = False
    extractor_text_map: Dict[str, str] = field(default_factory=dict)

    @property
    def docling_empty(self) -> bool:
        return not self.docling_ok or not self.text.strip()

    @property
    def no_text_layer(self) -> bool:
        return not self.digital_text

    def best_extractor_text(self) -> str:
        if self.extractor_text_map.get("pdfminer", "").strip():
            return self.extractor_text_map["pdfminer"]
        if self.extractor_text_map.get("fitz", "").strip():
            return self.extractor_text_map["fitz"]
        return self.extractor_text_map.get("pdfium", "")


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

    votes: List[ExtractorPageVote] = run_multi_extractor(
        pdf_bytes, config.extractor.char_threshold
    )
    pages: List[PageTriageResult] = []
    vote_map = {vote.page_number: vote for vote in votes}

    try:  # pragma: no cover - dependency heavy
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pymupdf (fitz) is required for triage but is not installed"
        ) from exc

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for idx, page in enumerate(doc, start=1):
            vote = vote_map.get(idx)
            if vote is None:
                continue
            page_start = time.perf_counter()
            errors: List[str] = []
            bbox_spans: List[tuple[float, float, float, float]] = []
            text_coverage = 0.0
            try:
                blocks = page.get_text("blocks") or []
                bbox_spans = [
                    (float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in blocks
                ]
                page_area = float(page.rect.width * page.rect.height)
                if page_area > 0:
                    coverage_area = sum(
                        max(0.0, (b[2] - b[0]) * (b[3] - b[1])) for b in blocks
                    )
                    text_coverage = min(1.0, coverage_area / page_area)
            except Exception as exc:  # pragma: no cover - depends on PDF content
                errors.append(str(exc))
            elapsed_ms = (time.perf_counter() - page_start) * 1000
            text = vote.text_fitz
            char_count = max(vote.len_text_fitz, vote.len_text_pdfium, vote.len_text_pdfminer)
            pages.append(
                PageTriageResult(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page_number=idx,
                    char_count=char_count,
                    text_coverage=text_coverage,
                    docling_ok=vote.digital_text,
                    ocr_used=False,
                    layout_rescue=False,
                    latency_ms=elapsed_ms,
                    text=text,
                    bbox_spans=bbox_spans,
                    errors=errors,
                    len_text_fitz=vote.len_text_fitz,
                    len_text_pdfium=vote.len_text_pdfium,
                    len_text_pdfminer=vote.len_text_pdfminer,
                    has_type3=vote.has_type3,
                    has_cid=vote.has_cid,
                    has_tounicode=vote.has_tounicode,
                    force_ocr=vote.force_ocr,
                    digital_text=vote.digital_text,
                    extractor_text_map=vote.extractor_texts(),
                )
            )
    finally:
        doc.close()
    logger.debug("Triage completed for %s in %.2f seconds", doc_name, time.perf_counter() - start_time)
    return PageTriageSummary(
        doc_id=doc_id, file_name=doc_name, pdf_bytes=pdf_bytes, pages=pages
    )
