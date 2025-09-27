"""Lightweight stub of PyMuPDF for test environments without the dependency."""
from __future__ import annotations

class DocumentStub:
    def __init__(self, *_, **__):
        raise RuntimeError("PyMuPDF (fitz) is required at runtime but missing in this environment")


def open(*args, **kwargs):  # type: ignore
    raise RuntimeError("PyMuPDF (fitz) is required to parse PDFs. Install PyMuPDF==1.26.4.")
