"""GPU-enabled layout detection helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional GPU runtime dependency
    import numpy as np
except Exception:  # pragma: no cover - keep import optional for tests
    np = None  # type: ignore

from .detector import LayoutDetector


class GPULayoutDetector(LayoutDetector):
    """Attempt to run the ONNXRuntime session on a CUDA provider first."""

    def _providers(self) -> List[str]:  # pragma: no cover - simple configuration hook
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return providers

    def detect_from_image(
        self, image: "np.ndarray", scale: float
    ) -> List[Dict[str, Any]]:
        # Reuse the base implementation but guard against CUDA-only failures.
        try:
            return super().detect_from_image(image, scale)
        except Exception:  # pragma: no cover - fallback safety
            return []


_GPU_DETECTOR_SINGLETON: Optional[GPULayoutDetector] = None


def detect_layout_gpu(
    image: "np.ndarray", scale: float, cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Detect layout regions on ``image`` returning PDF-space bboxes.

    The helper caches a ``GPULayoutDetector`` instance to avoid repeatedly
    initialising ONNXRuntime sessions.  When CUDA is unavailable the
    detector gracefully falls back to returning an empty list so the caller
    can decide whether to run the CPU path.
    """

    if np is None or image is None or scale == 0:
        return []
    global _GPU_DETECTOR_SINGLETON
    if _GPU_DETECTOR_SINGLETON is None:
        _GPU_DETECTOR_SINGLETON = GPULayoutDetector(cfg)
    return _GPU_DETECTOR_SINGLETON.detect_from_image(image, scale)
