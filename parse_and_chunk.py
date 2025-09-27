"""Compatibility wrapper for the new ingestion CLI."""
from __future__ import annotations

from pdf_ingest.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
