import sys
from types import ModuleType

import pytest

from pipeline import docling_adapter
from pipeline.triage import PageTriageResult


class DummyDocumentInput:
    called_with: tuple | None = None

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str | None = None):
        cls.called_with = (data, mime_type)
        return {"data": data, "mime": mime_type}


@pytest.fixture(autouse=True)
def clear_docling_modules(monkeypatch):
    for name in list(sys.modules):
        if name.startswith("docling"):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_build_docling_input_prefers_models_document(monkeypatch):
    module = ModuleType("docling.models.document")
    module.DocumentInput = DummyDocumentInput  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "docling.models.document", module)

    result = docling_adapter._build_docling_input(b"pdf-bytes")

    assert result == {"data": b"pdf-bytes", "mime": "application/pdf"}
    assert DummyDocumentInput.called_with == (b"pdf-bytes", "application/pdf")


def test_extractor_fallback_prefers_pdfminer():
    page = PageTriageResult(
        doc_id="doc",
        doc_name="doc.pdf",
        page_number=1,
        char_count=0,
        text_coverage=0.0,
        docling_ok=False,
        ocr_used=False,
        layout_rescue=False,
        latency_ms=0.0,
        text="",
        extractor_text_map={"pdfminer": "Miner", "fitz": "Fitz", "pdfium": ""},
        len_text_fitz=4,
        len_text_pdfium=0,
        len_text_pdfminer=5,
        has_type3=False,
        has_cid=False,
        has_tounicode=True,
        force_ocr=False,
        digital_text=False,
    )

    block = docling_adapter._extractor_fallback_block(page)

    assert block is not None
    assert block.text == "Miner"
    assert block.source_stage == "extractor"
    assert block.source_tool == "pdfminer"
