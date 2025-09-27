"""OCR helpers that favour OCRmyPDF but fail safely when unavailable."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


def run_ocrmypdf(
    src: Path,
    dst: Path,
    *,
    timeout: float | None = None,
) -> bool:
    """Invoke OCRmyPDF with conservative flags. Returns True on success."""

    cmd = [
        "ocrmypdf",
        "--optimize",
        "0",
        "--skip-text",
        "--force-ocr",
        str(src),
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def ensure_ocr_layer(src: Path, tmp_dir: Path, *, timeout: float | None = None) -> Path | None:
    """Return a path to an OCR-enhanced PDF, or None if OCR failed."""

    dst = tmp_dir / f"{src.stem}_ocr.pdf"
    if run_ocrmypdf(src, dst, timeout=timeout):
        return dst
    return None
