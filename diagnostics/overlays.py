"""Diagnostic overlay helpers (no-op placeholders for tests)."""
from __future__ import annotations

from typing import Dict, List


def render_overlay(page, regions, blocks) -> Dict[str, object]:
    """Return a simple serialisable overlay description."""

    return {
        "page_id": getattr(page, "page_id", "unknown"),
        "region_count": len(regions),
        "block_count": len(blocks),
    }
