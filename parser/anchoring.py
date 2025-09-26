"""Auxiliary anchoring utilities."""
from __future__ import annotations

from typing import Dict, List, Optional


def _center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def anchor_captions(blocks: List[Dict[str, object]], regions, cfg: Dict[str, object]) -> List[Dict[str, object]]:
    """Anchor captions to nearest figure or table."""

    figures = [blk for blk in blocks if blk.get("subtype") in {"figure", "table"}]
    if not figures:
        return blocks
    for block in blocks:
        if block.get("subtype") != "caption":
            continue
        caption_center = _center(block["bbox"])
        best = None
        best_distance = None
        for fig in figures:
            fig_center = _center(fig["bbox"])
            dist = _distance(caption_center, fig_center)
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best = fig
        if best is not None:
            target_id = best.get("id")
            if target_id:
                block.setdefault("links", []).append({"kind": "anchor", "target_id": target_id})
    return blocks
