"""Compatibility wrapper for the new ingestion CLI.

Ensures the ``src`` directory is on ``sys.path`` so the ``pdf_ingest``
package can be imported without requiring an editable install.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pdf_ingest.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
