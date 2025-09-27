"""FastAPI entrypoint exposing the ingestion pipeline."""
from __future__ import annotations

from pdf_ingest.http_app import app  # re-export for uvicorn

__all__ = ["app"]
