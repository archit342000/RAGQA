"""Helper to import PyMuPDF with a clearer error message when missing."""
from __future__ import annotations

import importlib
import importlib.util
import types


def _load_fitz():
    spec = importlib.util.find_spec("fitz")
    if spec is not None and spec.loader is not None:
        return importlib.import_module("fitz")

    module = types.ModuleType("fitz")

    class DocumentStub:  # pragma: no cover - executed when PyMuPDF unavailable
        def __init__(self, *_, **__):
            raise RuntimeError(
                "PyMuPDF (fitz) is required at runtime but missing in this environment"
            )

    def open(*_, **__):  # pragma: no cover - executed when PyMuPDF unavailable
        raise RuntimeError("PyMuPDF (fitz) is required to parse PDFs. Install PyMuPDF==1.26.4.")

    module.Document = DocumentStub
    module.Page = object
    module.open = open
    return module


fitz = _load_fitz()

__all__ = ["fitz"]
