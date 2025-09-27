"""OCR heuristics and helpers."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pytesseract
from PIL import Image

from .config import Config
from .pdf_io import Line, PageSignals


@dataclass
class OCRDecision:
    page_index: int
    mode: str  # none | partial | full


def classify_pages(signals: Sequence[PageSignals], config: Config) -> List[OCRDecision]:
    decisions: List[OCRDecision] = []
    for signal in signals:
        mode = "none"
        low_text = signal.glyph_count < config.glyph_min_for_text_page
        bad_density = signal.text_density < 0.0005
        noisy_unicode = signal.unicode_ratio < 0.75
        suspect_fonts = not signal.has_fonts
        heavy_images = signal.image_coverage > 0.35
        dpi_poor = signal.dpi < config.bad_dpi
        if signal.hidden_text_layer:
            mode = "none"
        elif low_text and (heavy_images or dpi_poor):
            mode = "full"
        elif low_text or bad_density or noisy_unicode or suspect_fonts:
            mode = "partial"
        decisions.append(OCRDecision(page_index=signal.index, mode=mode))

    for idx, decision in enumerate(decisions):
        if decision.mode != "full":
            continue
        left = decisions[idx - 1].mode if idx > 0 else "none"
        right = decisions[idx + 1].mode if idx + 1 < len(decisions) else "none"
        if left != "full" and right != "full":
            decision.mode = "partial"
    return decisions


def run_ocrmypdf(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ocrmypdf",
        "--skip-text",
        "--optimize",
        "0",
        str(source),
        str(destination),
    ]
    subprocess.run(cmd, check=True)


def pytesseract_page_image(page) -> List[str]:  # type: ignore[no-untyped-def]
    pix = page.get_pixmap()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(pix.tobytes())
        temp_path = Path(handle.name)
    try:
        image = Image.open(temp_path)
        text = pytesseract.image_to_string(image)
    finally:
        temp_path.unlink(missing_ok=True)
    return [line.strip() for line in text.splitlines() if line.strip()]

