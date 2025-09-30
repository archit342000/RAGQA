from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List

from .config import PipelineConfig
from .triage import PageTriageResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OCRDecision:
    page_number: int
    should_ocr: bool
    engine: str | None
    reason: str


MATH_TOKENS = {"\\", "∑", "∫", "√", "π", "≈", "≅", "≥", "≤"}


def math_density(text: str) -> float:
    if not text:
        return 0.0
    math_chars = 0
    for ch in text:
        if ch in MATH_TOKENS or ch in "±×÷∝∞≠":
            math_chars += 1
        elif ch.isnumeric() and ch not in {"0", "1"}:
            math_chars += 1
        elif "ℚℤℝℂℕ".find(ch) != -1:
            math_chars += 1
    return math_chars / max(len(text), 1)


def should_route_to_ocr(page: PageTriageResult, config: PipelineConfig) -> OCRDecision:
    gate = config.ocr.gate
    reasons: List[str] = []
    if page.force_ocr:
        reasons.append("force_ocr")
    if page.no_text_layer:
        reasons.append("no_text_layer")
    if (
        page.char_count < gate.char_count_threshold
        and page.text_coverage < gate.text_coverage_threshold
    ):
        reasons.append("low_text_density")
    if page.docling_empty:
        reasons.append("docling_empty")
    if page.errors:
        reasons.append("triage_error")
    should = bool(reasons)
    engine = None
    base_text = page.text or page.best_extractor_text()
    density = math_density(base_text)
    if should:
        engine = (
            "nougat" if density >= config.ocr.math_density_threshold else "tesseract"
        )
        reasons.append(f"math_density={density:.3f}")
    reason_str = ",".join(dict.fromkeys(reasons)) if reasons else "sufficient_text"
    return OCRDecision(
        page_number=page.page_number, should_ocr=should, engine=engine, reason=reason_str
    )


def plan_ocr(triage: Iterable[PageTriageResult], config: PipelineConfig) -> List[OCRDecision]:
    decisions: List[OCRDecision] = []
    for page in triage:
        decision = should_route_to_ocr(page, config)
        decisions.append(decision)
    return decisions


def apply_ocr(
    triage: List[PageTriageResult],
    decisions: List[OCRDecision],
) -> List[PageTriageResult]:
    page_by_number = {page.page_number: page for page in triage}
    for decision in decisions:
        if not decision.should_ocr:
            continue
        page = page_by_number.get(decision.page_number)
        if page is None:
            continue
        logger.info("OCR would run on page %s with %s due to %s", decision.page_number, decision.engine, decision.reason)
        page.ocr_used = True
        page.path_used = "ocr"
        # Placeholder for OCR output; actual integration handled elsewhere.
        page.text = page.best_extractor_text() or page.text
    return list(page_by_number.values())
