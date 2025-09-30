from __future__ import annotations

from pipeline.config import PipelineConfig
from pipeline.ocr import math_density, should_route_to_ocr
from pipeline.triage import PageTriageResult


def _page(**overrides):
    base = dict(
        doc_id="doc",
        doc_name="doc.pdf",
        page_number=1,
        char_count=500,
        text_coverage=0.5,
        docling_ok=True,
        ocr_used=False,
        layout_rescue=False,
        latency_ms=10.0,
        text="Plain text",  # default replaced per test
        bbox_spans=[],
        errors=[],
        len_text_fitz=500,
        len_text_pdfium=0,
        len_text_pdfminer=0,
        has_type3=False,
        has_cid=False,
        has_tounicode=True,
        force_ocr=False,
        digital_text=True,
    )
    base.update(overrides)
    return PageTriageResult(**base)


def test_gate_low_density_triggers_ocr_nougat() -> None:
    config = PipelineConfig()
    page = _page(
        char_count=100,
        text_coverage=0.01,
        text="This page contains ∑ and ∫ and sqrt symbols",
    )
    decision = should_route_to_ocr(page, config)
    assert decision.should_ocr is True
    assert decision.engine == "nougat"
    assert "low_text_density" in decision.reason
    assert "math_density" in decision.reason


def test_gate_docling_empty_triggers_ocr_tesseract() -> None:
    config = PipelineConfig()
    page = _page(docling_ok=False, text="")
    decision = should_route_to_ocr(page, config)
    assert decision.should_ocr is True
    assert decision.engine == "tesseract"
    assert "docling_empty" in decision.reason


def test_gate_error_triggers_ocr() -> None:
    config = PipelineConfig()
    page = _page(errors=["timeout"], text="Broken page")
    decision = should_route_to_ocr(page, config)
    assert decision.should_ocr is True
    assert "triage_error" in decision.reason


def test_math_density_thresholds() -> None:
    dense = math_density("A + B = C ∑ ∫")
    sparse = math_density("Plain text")
    assert dense >= 0.02
    assert sparse < 0.02
