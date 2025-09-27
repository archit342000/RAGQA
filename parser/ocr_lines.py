"""OCR line detection utilities using PaddleOCR when available."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - keep typing optional
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from paddleocr import PaddleOCR  # type: ignore
except Exception:  # pragma: no cover - fallback when PaddleOCR missing
    PaddleOCR = None  # type: ignore


class OCRLineDetector:
    """Thin wrapper around :class:`PaddleOCR` with graceful degradation."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._ocr: Optional["PaddleOCR"] = None

    def _ensure_ocr(self) -> Optional["PaddleOCR"]:
        if self._ocr is not None:
            return self._ocr
        if PaddleOCR is None:
            return None
        common_kwargs: Dict[str, Any] = {
            "use_angle_cls": True,
            "lang": "en",
            "show_log": False,
        }
        # Try GPU first; if this fails fall back to CPU.
        try:  # pragma: no cover - heavy dependency path
            self._ocr = PaddleOCR(use_gpu=True, **common_kwargs)
            return self._ocr
        except Exception:
            self._ocr = None
        try:  # pragma: no cover - CPU fallback
            self._ocr = PaddleOCR(use_gpu=False, **common_kwargs)
        except Exception:
            self._ocr = None
        return self._ocr

    # ------------------------------------------------------------------
    def detect(self, image: "np.ndarray", scale: float) -> List[Dict[str, Any]]:
        if np is None or image is None or scale == 0:
            return []
        ocr = self._ensure_ocr()
        if ocr is None:
            return []
        try:  # pragma: no cover - OCR heavy path
            results = ocr.ocr(image, det=True, rec=True)
        except Exception:
            return []
        lines: List[Dict[str, Any]] = []
        for page_result in results or []:
            for entry in page_result or []:
                if not entry or len(entry) < 2:
                    continue
                box = entry[0]
                content = entry[1]
                if not box or not content:
                    continue
                text = str(content[0]) if isinstance(content, (list, tuple)) else str(content)
                conf = 0.0
                if isinstance(content, (list, tuple)) and len(content) > 1:
                    try:
                        conf = float(content[1])
                    except Exception:
                        conf = 0.0
                xs = [float(pt[0]) for pt in box]
                ys = [float(pt[1]) for pt in box]
                if not xs or not ys:
                    continue
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                width = max(0.0, x1 - x0)
                height = max(0.0, y1 - y0)
                alpha_chars = [c for c in text if c.isalpha()]
                allcaps = sum(1 for c in alpha_chars if c.isupper())
                allcaps_ratio = allcaps / len(alpha_chars) if alpha_chars else 0.0
                avg_char_width = width / max(len(text.strip()), 1)
                bbox_img = (x0, y0, x1, y1)
                bbox_pdf = (x0 / scale, y0 / scale, x1 / scale, y1 / scale)
                lines.append(
                    {
                        "text": text,
                        "confidence": conf,
                        "bbox_image": bbox_img,
                        "bbox_pdf": bbox_pdf,
                        "line_height": height,
                        "font_size": height,
                        "allcaps_ratio": allcaps_ratio,
                        "avg_char_width": avg_char_width,
                    }
                )
        return lines


_LINE_DETECTOR_SINGLETON: Optional[OCRLineDetector] = None


def detect_lines_and_ocr(
    image: "np.ndarray", scale: float, cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Convenience helper returning OCR line dictionaries."""

    if np is None or image is None or scale == 0:
        return []
    global _LINE_DETECTOR_SINGLETON
    if _LINE_DETECTOR_SINGLETON is None:
        _LINE_DETECTOR_SINGLETON = OCRLineDetector(cfg)
    return _LINE_DETECTOR_SINGLETON.detect(image, scale)
