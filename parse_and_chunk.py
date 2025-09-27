"""Compatibility wrapper for invoking :mod:`pdf_ingest.cli` directly."""
from __future__ import annotations

from pdf_ingest.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
