"""OCR heuristics, ROI management, and Tesseract orchestration."""

from __future__ import annotations

import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import fitz  # type: ignore
import pytesseract

from .config import Config
from .logging import EventLogger
from .pdf_io import Line, PageSignals


@dataclass
class OCRDecision:
    page_index: int
    mode: str  # none | partial | full
    promoted_reason: str | None = None


@dataclass
class ROI:
    bbox: fitz.Rect
    estimated_alnum: int
    band_index: int


@dataclass
class OCRResult:
    lines: List[Line]
    dpi_used: Optional[int]
    alerts: List[str]
    tsv_lines: int
    rois_considered: int
    rois_ocrd: int
    scaled_due_to_megapixels: bool
    surface_consumed_ratio: float


def classify_pages(signals: Sequence[PageSignals], config: Config) -> List[OCRDecision]:
    policy = config.ocr_policy
    decisions: List[OCRDecision] = []
    for signal in signals:
        native_score = 0.0
        if policy.promote_none_glyphs > 0:
            native_score = min(1.0, signal.glyph_count / policy.promote_none_glyphs)
        native_score *= signal.unicode_ratio
        promoted_reason: str | None = None

        low_glyphs = signal.glyph_count < policy.promote_none_glyphs
        low_quality = (
            signal.glyph_count < policy.promote_none_glyphs_with_quality
            and signal.unicode_ratio < policy.unicode_quality_threshold
        )
        heavy_images = signal.image_coverage > 0.35
        low_density = signal.text_density < 0.0003

        mode = "none"
        if signal.hidden_text_layer:
            mode = "none"
            promoted_reason = "hidden_text_layer"
        elif low_glyphs and heavy_images and native_score < policy.full_threshold_s_native:
            mode = "full"
            promoted_reason = "low_glyphs_heavy_images"
        elif low_glyphs or low_quality or low_density:
            mode = "partial"
            if low_quality:
                promoted_reason = "unicode_quality"
            elif low_density:
                promoted_reason = "low_density"
            else:
                promoted_reason = "low_glyphs"
        decisions.append(OCRDecision(page_index=signal.index, mode=mode, promoted_reason=promoted_reason))

    smoothing_window = max(1, policy.probe_selectable_bands)
    for idx, decision in enumerate(decisions):
        if decision.mode != "full":
            continue
        left_full = any(
            decisions[j].mode == "full"
            for j in range(max(0, idx - smoothing_window), idx)
        )
        right_full = any(
            decisions[j].mode == "full"
            for j in range(idx + 1, min(len(decisions), idx + 1 + smoothing_window))
        )
        if not left_full and not right_full:
            decision.mode = "partial"
            decision.promoted_reason = "neighbor_smoothing"
    return decisions


class OCREngine:
    """Coordinates rasterization, ROI selection, and Tesseract calls."""

    def __init__(self, config: Config, *, total_surface: float, logger: EventLogger) -> None:
        self.config = config
        self.logger = logger
        self.total_surface = total_surface
        self.surface_budget = total_surface * config.work_caps.max_doc_ocr_surface_multiplier
        self.surface_used = 0.0
        self.executor = ThreadPoolExecutor(max_workers=config.concurrency.ocr_workers)

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def process_page(
        self,
        page: fitz.Page,
        decision: OCRDecision,
        signals: PageSignals,
        native_lines: Sequence[Line],
    ) -> OCRResult:
        if decision.mode == "none":
            self.logger.emit(
                "ocr_skipped",
                page=page.number + 1,
                reason=decision.promoted_reason,
            )
            return OCRResult(
                lines=[],
                dpi_used=None,
                alerts=[],
                tsv_lines=0,
                rois_considered=0,
                rois_ocrd=0,
                scaled_due_to_megapixels=False,
                surface_consumed_ratio=self._surface_ratio(),
            )

        rois = self._select_rois(page, decision, signals)
        rois_considered = len(rois)
        if not rois:
            self.logger.emit(
                "ocr_no_roi",
                page=page.number + 1,
                decision=decision.mode,
            )
            return OCRResult(
                lines=[],
                dpi_used=None,
                alerts=["no_roi"],
                tsv_lines=0,
                rois_considered=0,
                rois_ocrd=0,
                scaled_due_to_megapixels=False,
                surface_consumed_ratio=self._surface_ratio(),
            )

        merged_lines: List[Line] = []
        dpi_used: Optional[int] = None
        alerts: List[str] = []
        total_tsv_lines = 0
        rois_ocrd = 0
        scaled_flag = False

        for roi_index, roi in enumerate(rois):
            if roi.estimated_alnum < self.config.fail_fast.roi_min_alnum_chars:
                alerts.append("roi_predicted_low_signal")
                self.logger.emit(
                    "ocr_roi_skipped_low_signal",
                    page=page.number + 1,
                    roi=roi_index,
                    estimated_alnum=roi.estimated_alnum,
                )
                continue
            if not self._can_consume_surface(roi.bbox):
                alerts.append("surface_budget_exhausted")
                self.logger.emit(
                    "ocr_surface_budget_hit",
                    page=page.number + 1,
                    roi=roi_index,
                )
                break
            ocr_lines, dpi_value, roi_alerts, tsv_lines, scaled = self._ocr_roi(page, roi)
            alerts.extend(roi_alerts)
            merged_lines.extend(ocr_lines)
            total_tsv_lines += tsv_lines
            scaled_flag = scaled_flag or scaled
            if ocr_lines:
                dpi_used = dpi_value
            rois_ocrd += 1

        return OCRResult(
            lines=merged_lines,
            dpi_used=dpi_used,
            alerts=alerts,
            tsv_lines=total_tsv_lines,
            rois_considered=rois_considered,
            rois_ocrd=rois_ocrd,
            scaled_due_to_megapixels=scaled_flag,
            surface_consumed_ratio=self._surface_ratio(),
        )

    def _surface_ratio(self) -> float:
        if self.surface_budget <= 0:
            return 1.0
        return min(1.0, self.surface_used / self.surface_budget)

    def current_surface_ratio(self) -> float:
        return self._surface_ratio()

    def _select_rois(self, page: fitz.Page, decision: OCRDecision, signals: PageSignals) -> List[ROI]:
        rect = page.rect
        work_caps = self.config.work_caps
        rois: List[ROI] = []

        if decision.mode == "full":
            rois.append(ROI(bbox=rect, estimated_alnum=max(signals.glyph_count, work_caps.max_rois_per_page), band_index=-1))
            self.logger.emit(
                "ocr_rois_selected",
                page=page.number + 1,
                decision=decision.mode,
                rois=[{
                    "band": -1,
                    "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                    "estimated_alnum": max(signals.glyph_count, work_caps.max_rois_per_page),
                }],
            )
            return rois

        bands = max(1, self.config.ocr_policy.probe_selectable_bands)
        band_height = rect.height / bands if bands else rect.height
        sorted_candidates: List[Tuple[float, int]] = []
        for idx, density in enumerate(signals.band_text_density):
            score = density
            sorted_candidates.append((score, idx))
        sorted_candidates.sort(key=lambda item: item[0])

        selected_meta: List[dict] = []
        for _, band_index in sorted_candidates[: work_caps.max_rois_per_page]:
            y0 = rect.y0 + band_index * band_height
            y1 = y0 + band_height
            bbox = fitz.Rect(rect.x0, y0, rect.x1, y1)
            estimated = max(1, int(signals.band_glyph_counts[band_index]))
            rois.append(ROI(bbox=bbox, estimated_alnum=estimated, band_index=band_index))
            selected_meta.append(
                {
                    "band": band_index,
                    "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                    "estimated_alnum": estimated,
                }
            )
        self.logger.emit(
            "ocr_rois_selected",
            page=page.number + 1,
            decision=decision.mode,
            rois=selected_meta,
        )
        return rois

    def _can_consume_surface(self, bbox: fitz.Rect) -> bool:
        roi_area = max(bbox.width * bbox.height, 0.0)
        projected = self.surface_used + roi_area
        if projected > self.surface_budget > 0:
            return False
        self.surface_used = projected
        return True

    def _ocr_roi(self, page: fitz.Page, roi: ROI) -> Tuple[List[Line], Optional[int], List[str], int, bool]:
        work_caps = self.config.work_caps
        alerts: List[str] = []
        dpi = work_caps.dpi
        scaled_due_to_megapixels = False
        attempts = 0
        lines: List[Line] = []
        tsv_total = 0
        retries = self.config.fail_fast.roi_max_retries
        while attempts <= retries:
            attempt_dpi = dpi if attempts == 0 else self.config.work_caps.dpi_retry_small_roi
            adjusted_dpi = attempt_dpi
            width = height = 0
            scaled_flag = False
            while True:
                image_path, width, height = render_page_to_image(page, adjusted_dpi, clip=roi.bbox)
                megapixels = (width * height) / 1_000_000
                if megapixels <= work_caps.max_page_megapixels:
                    break
                scaled_flag = True
                scaled_due_to_megapixels = True
                adjusted_dpi = max(72, int(adjusted_dpi * math.sqrt(work_caps.max_page_megapixels / max(megapixels, 1e-6))))
                image_path.unlink(missing_ok=True)
                if adjusted_dpi == attempt_dpi:
                    break
            self.logger.emit(
                "render_start",
                page=page.number + 1,
                dpi=adjusted_dpi,
                roi=roi.band_index,
                attempt=attempts + 1,
                scaled=scaled_flag,
            )
            try:
                self.logger.emit(
                    "tesseract_start",
                    page=page.number + 1,
                    roi=roi.band_index,
                    dpi=adjusted_dpi,
                    attempt=attempts + 1,
                    estimated_alnum=roi.estimated_alnum,
                )
                tsv_path = self._run_tesseract(image_path, adjusted_dpi, extension="tsv")
                ocr_lines = tsv_to_lines(
                    page.number,
                    tsv_path,
                    pdf_width=roi.bbox.width,
                    pdf_height=roi.bbox.height,
                    image_width=width,
                    image_height=height,
                    dpi=adjusted_dpi,
                    clip_bbox=(roi.bbox.x0, roi.bbox.y0, roi.bbox.x1, roi.bbox.y1),
                )
                lines.extend(ocr_lines)
                tsv_total += len(ocr_lines)
                self.logger.emit(
                    "tesseract_stop",
                    page=page.number + 1,
                    roi=roi.band_index,
                    dpi=adjusted_dpi,
                    lines=len(ocr_lines),
                )
                if ocr_lines:
                    return lines, adjusted_dpi, alerts, tsv_total, scaled_due_to_megapixels
                alerts.append("tsv_empty")
                self.logger.emit(
                    "tesseract_empty",
                    page=page.number + 1,
                    roi=roi.band_index,
                    dpi=adjusted_dpi,
                )
                hocr_path = self._run_tesseract(image_path, adjusted_dpi, extension="hocr")
                hocr_lines = hocr_to_lines(
                    page.number,
                    hocr_path,
                    pdf_width=roi.bbox.width,
                    pdf_height=roi.bbox.height,
                    image_width=width,
                    image_height=height,
                    clip_bbox=(roi.bbox.x0, roi.bbox.y0, roi.bbox.x1, roi.bbox.y1),
                )
                if hocr_lines:
                    lines.extend(hocr_lines)
                    tsv_total += len(hocr_lines)
                    alerts.append("hocr_fallback")
                    self.logger.emit(
                        "hocr_fallback",
                        page=page.number + 1,
                        roi=roi.band_index,
                        dpi=adjusted_dpi,
                        lines=len(hocr_lines),
                    )
                    return lines, adjusted_dpi, alerts, tsv_total, scaled_due_to_megapixels
            finally:
                self.logger.emit(
                    "render_stop",
                    page=page.number + 1,
                    roi=roi.band_index,
                    dpi=adjusted_dpi,
                    width=width,
                    height=height,
                )
                image_path.unlink(missing_ok=True)
                tsv_path = image_path.with_suffix(".tsv")
                tsv_path.unlink(missing_ok=True)
                hocr_file = image_path.with_suffix(".hocr")
                hocr_file.unlink(missing_ok=True)
            attempts += 1
        if not lines:
            alerts.append("ocr_failed")
            self.logger.emit(
                "ocr_roi_failed",
                page=page.number + 1,
                roi=roi.band_index,
                attempts=attempts,
            )
        return lines, None, alerts, tsv_total, scaled_due_to_megapixels

    def _run_tesseract(self, image_path: Path, dpi: int, *, extension: str = "tsv") -> Path:
        output_base = image_path.with_suffix("")
        args = [f"--oem {self.config.tesseract.oem}", f"--psm {self.config.tesseract.psm}", f"--dpi {dpi}"]
        if self.config.tesseract.disable_dawgs:
            args.extend(["-c", "load_system_dawg=F", "-c", "load_freq_dawg=F"])
        if self.config.tesseract.whitelist:
            args.extend(["-c", f"tessedit_char_whitelist={self.config.tesseract.whitelist}"])
        if self.config.tesseract.lang:
            lang = self.config.tesseract.lang
        else:
            lang = None

        cmd_config = " ".join(args)

        def _invoke() -> None:
            if hasattr(pytesseract, "run_tesseract"):
                pytesseract.run_tesseract(
                    str(image_path),
                    str(output_base),
                    extension=extension,
                    lang=lang,
                    config=cmd_config,
                )
                return
            module = getattr(pytesseract, "pytesseract", None)
            if module is not None and hasattr(module, "run_tesseract"):
                module.run_tesseract(
                    str(image_path),
                    str(output_base),
                    extension=extension,
                    lang=lang,
                    config=cmd_config,
                )
                return
            tesseract_cmd = getattr(pytesseract, "tesseract_cmd", "tesseract")
            cli_args = [
                tesseract_cmd,
                str(image_path),
                str(output_base),
                extension,
            ]
            if lang:
                cli_args.extend(["-l", lang])
            cli_args.extend(cmd_config.split())
            os.spawnvp(os.P_WAIT, cli_args[0], cli_args)

        future = self.executor.submit(_invoke)
        future.result()
        return output_base.with_suffix(f".{extension}")


def render_page_to_image(page: fitz.Page, dpi: int, *, clip: Optional[fitz.Rect] = None) -> Tuple[Path, int, int]:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix, alpha=False, clip=clip)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        pix.save(handle.name)
        image_path = Path(handle.name)
    return image_path, pix.width, pix.height


from .tsv import hocr_to_lines, tsv_to_lines  # late import to avoid cycle
