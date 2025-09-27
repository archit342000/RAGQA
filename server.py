"""FastAPI entrypoint exposing the ingestion pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pdf_ingest.http_app import app  # re-export for uvicorn

__all__ = ["app"]
