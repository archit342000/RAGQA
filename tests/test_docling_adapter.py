from types import ModuleType

import sys

import pytest

from pipeline import docling_adapter


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
