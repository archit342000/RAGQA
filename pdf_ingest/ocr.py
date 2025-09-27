"""OCR heuristics and Tesseract orchestration."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import fitz  # type: ignore
import pytesseract
import subprocess

from .config import Config
from .pdf_io import PageSignals


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
        if signal.hidden_text_layer:
            mode = "none"
        elif low_text and heavy_images:
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


def render_page_to_image(page: fitz.Page, dpi: int) -> tuple[Path, int, int]:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        image_path = Path(handle.name)
        pix.save(handle.name)
    return image_path, pix.width, pix.height


def _call_tesseract(
    image_path: Path,
    output_base: Path,
    *,
    extension: str,
    dpi: int,
    lang: str | None,
) -> None:
    """Invoke Tesseract with broad compatibility across pytesseract versions."""

    if hasattr(pytesseract, "run_tesseract"):
        pytesseract.run_tesseract(
            str(image_path),
            str(output_base),
            extension=extension,
            lang=lang,
            config=f"--dpi {dpi} --psm 6",
        )
        return

    module = getattr(pytesseract, "pytesseract", None)
    if module is not None and hasattr(module, "run_tesseract"):
        module.run_tesseract(
            str(image_path),
            str(output_base),
            extension=extension,
            lang=lang,
            config=f"--dpi {dpi} --psm 6",
        )
        return

    tesseract_cmd = getattr(pytesseract, "tesseract_cmd", "tesseract")
    cmd = [
        tesseract_cmd,
        str(image_path),
        str(output_base),
    ]
    if lang:
        cmd.extend(["-l", lang])
    cmd.extend(["--dpi", str(dpi), "--psm", "6", extension])
    subprocess.run(cmd, check=True)


def run_tesseract_tsv(image_path: Path, *, dpi: int, lang: str | None = None) -> Path:
    output_base = image_path.with_suffix("")
    _call_tesseract(
        image_path,
        output_base,
        extension="tsv",
        dpi=dpi,
        lang=lang,
    )
    return output_base.with_suffix(".tsv")


def run_tesseract_hocr(image_path: Path, *, dpi: int, lang: str | None = None) -> Path:
    output_base = image_path.with_suffix("")
    _call_tesseract(
        image_path,
        output_base,
        extension="hocr",
        dpi=dpi,
        lang=lang,
    )
    return output_base.with_suffix(".hocr")
