"""CPU-only ONNXRuntime layout detector integration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency at runtime
    import fitz  # type: ignore
except Exception:  # pragma: no cover - guard for test environments
    fitz = None

try:  # pragma: no cover - optional dependency for inference
    import numpy as np
except Exception:  # pragma: no cover - keep typing happy without numpy
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency for inference
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - allow graceful degradation
    ort = None  # type: ignore


BBox = Tuple[float, float, float, float]


def _as_float_box(box: Sequence[float]) -> BBox:
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


def _scale_bbox(box: BBox, scale: float) -> BBox:
    if scale == 0:
        return box
    return (box[0] / scale, box[1] / scale, box[2] / scale, box[3] / scale)


@dataclass
class DetectorConfig:
    model_path: Path
    score_thresh: float
    nms_iou: float
    target_classes: List[str]
    dpi: int


class LayoutDetector:
    """Small wrapper around ONNXRuntime for document layout detection.

    The class is intentionally defensive – if the model or runtime are not
    available (which is the default in the unit tests), the detector simply
    returns an empty list and the downstream heuristics remain in control.
    """

    def __init__(self, cfg: Dict[str, Any]):
        detector_cfg = cfg.get("detector", {}) if isinstance(cfg, dict) else {}
        model_path = Path(detector_cfg.get("model_path", ""))
        target_classes = list(detector_cfg.get("target_classes", []))
        self.config = DetectorConfig(
            model_path=model_path,
            score_thresh=float(detector_cfg.get("score_thresh", 0.30)),
            nms_iou=float(detector_cfg.get("nms_iou", 0.50)),
            target_classes=target_classes,
            dpi=int(detector_cfg.get("dpi", 180)),
        )
        self._session: Optional["ort.InferenceSession"] = None
        self._input_name: Optional[str] = None
        self._output_names: Optional[List[str]] = None
        self._session_initialised = False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def _providers(self) -> List[str]:  # pragma: no cover - simple configuration hook
        return ["CPUExecutionProvider"]

    def _ensure_session(self) -> Optional["ort.InferenceSession"]:
        if self._session_initialised:
            return self._session
        self._session_initialised = True
        if ort is None:
            return None
        model_path = self.config.model_path
        if not model_path or not model_path.exists():
            return None
        providers = self._providers()
        try:  # pragma: no cover - heavy runtime path
            session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception:
            if providers != ["CPUExecutionProvider"]:
                try:  # pragma: no cover - fallback to CPU
                    session = ort.InferenceSession(
                        str(model_path), providers=["CPUExecutionProvider"]
                    )
                except Exception:
                    return None
            else:
                return None
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if not inputs or not outputs:
            return None
        self._session = session
        self._input_name = inputs[0].name
        self._output_names = [out.name for out in outputs]
        return self._session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, image: "np.ndarray") -> List[Dict[str, Any]]:
        """Run the detector on a numpy image array.

        The exact network head can vary between models, therefore the default
        implementation intentionally returns an empty list unless both the
        model and runtime are available and produce an expected tensor shape.
        Downstream tests can monkeypatch this method to return synthetic
        detections without touching the ONNX runtime.
        """

        if np is None or image is None:
            return []
        session = self._ensure_session()
        if session is None or self._input_name is None:
            return []
        try:  # pragma: no cover - inference path
            data = image
            if data.ndim == 2:
                data = np.expand_dims(data, axis=-1)
            if data.shape[-1] == 1:
                data = np.repeat(data, 3, axis=-1)
            data = data.astype("float32") / 255.0
            data = np.transpose(data, (2, 0, 1))
            data = np.expand_dims(data, axis=0)
            outputs = session.run(self._output_names, {self._input_name: data})
        except Exception:
            return []

        if not outputs:
            return []
        raw = outputs[0]
        detections: List[Dict[str, Any]] = []
        try:
            iterable: Iterable[Sequence[float]] = raw.reshape(-1, raw.shape[-1])  # type: ignore[arg-type]
        except Exception:
            return []
        classes = self.config.target_classes or []
        for det in iterable:
            if len(det) < 6:
                continue
            score = float(det[4])
            if score < self.config.score_thresh:
                continue
            cls_idx = int(det[5])
            if classes and (cls_idx < 0 or cls_idx >= len(classes)):
                continue
            cls_name = classes[cls_idx] if classes else str(cls_idx)
            x0, y0, x1, y1 = map(float, det[0:4])
            detections.append({
                "cls": cls_name,
                "score": score,
                "bbox": (x0, y0, x1, y1),
            })
        return detections

    # ------------------------------------------------------------------
    def detect_page(self, doc: "fitz.Document", page_number: int) -> List[Dict[str, Any]]:
        """Rasterise a page and run the detector; returns PDF-space bboxes."""

        if fitz is None or np is None:
            return []
        if doc is None or page_number < 0 or page_number >= len(doc):
            return []
        page = doc.load_page(page_number)
        scale = self.config.dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        try:  # pragma: no cover - heavy runtime path
            pix = page.get_pixmap(matrix=matrix, alpha=False)
        except Exception:
            return []
        samples = pix.samples
        if samples is None:
            return []
        if pix.n in (1, 3):
            array = np.frombuffer(samples, dtype=np.uint8)
            array = array.reshape(pix.height, pix.width, pix.n)
        else:  # pragma: no cover - convert RGBA to RGB
            array = np.frombuffer(samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            array = array[:, :, :3]
        return self.detect_from_image(array, scale)

    # ------------------------------------------------------------------
    def detect_from_image(
        self, image: "np.ndarray", scale: float
    ) -> List[Dict[str, Any]]:
        if np is None or image is None or scale == 0:
            return []
        detections = self.detect(image)
        results: List[Dict[str, Any]] = []
        for idx, det in enumerate(detections):
            bbox = det.get("bbox")
            if not bbox:
                continue
            pdf_box = _scale_bbox(_as_float_box(bbox), scale)
            results.append(
                {
                    "id": det.get("id", f"det_img_{idx}"),
                    "cls": det.get("cls", ""),
                    "score": float(det.get("score", 0.0)),
                    "bbox_pdf": pdf_box,
                    "bbox_image": _as_float_box(bbox),
                }
            )
        return results


def rasterize_page(pdf_path: Path, page_number: int, dpi: int) -> Tuple[Optional["np.ndarray"], Optional[float], Optional[Tuple[float, float]]]:
    """Utility for callers that do not hold an open PyMuPDF document."""

    if fitz is None or np is None:
        return None, None, None
    doc = fitz.open(pdf_path)  # pragma: no cover - heavy runtime path
    try:
        page = doc.load_page(page_number)
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        samples = np.frombuffer(pix.samples, dtype=np.uint8)
        array = samples.reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            array = array[:, :, :3]
        return array, scale, (page.rect.width, page.rect.height)
    finally:  # pragma: no cover - ensure document closes
        doc.close()

