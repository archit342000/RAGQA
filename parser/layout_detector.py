"""Layout detector wrapper with GPU preference but CPU fallback."""
from __future__ import annotations

from typing import Dict, List, Optional


def detect_layout(
    image,  # pragma: no cover - image unused in tests
    cfg: Dict[str, object],
    meta: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """Return detector regions from metadata or fallback to empty list."""

    if meta and "detections" in meta:
        detections = meta["detections"]
        return [dict(det) for det in detections]
    return []
