"""Export stitched blocks into DocBlock JSON structures."""
from __future__ import annotations

from typing import Dict, List, Optional

from .stitcher import StitchedBlock
from .utils import DocBlock, load_config, normalize_bbox


def _wrap_aux_text(kind: str, text: str) -> str:
    cleaned = (text or "").strip()
    if kind == "aux":
        if cleaned.startswith("<aux>") and cleaned.endswith("</aux>"):
            return cleaned
        return f"<aux>{cleaned}</aux>"
    return cleaned


def export_docblocks(
    stitched: List[StitchedBlock],
    config: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    cfg = config or load_config()
    sorted_blocks = sorted(stitched, key=lambda b: (b.page, b.bbox[1]))
    docblocks: List[Dict[str, object]] = []
    source_map: Dict[tuple[int, int], str] = {}
    for idx, block in enumerate(sorted_blocks):
        block_id = f"b_p{block.page:02}_{idx:04}"
        for source in block.sources:
            source_map[source] = block_id
        anchor_id = None
        if block.anchor_source:
            anchor_id = source_map.get(block.anchor_source)
        if anchor_id is None and block.kind == "aux":
            anchor_id = _last_main_id(docblocks)
        width = float(block.meta.get("page_width", 1.0) or 1.0)
        height = float(block.meta.get("page_height", 1.0) or 1.0)
        normalized = normalize_bbox(block.bbox, width, height)
        wrapped_text = _wrap_aux_text(block.kind, block.text)
        docblock = DocBlock(
            id=block_id,
            page=block.page,
            kind=block.kind,
            aux_type=block.aux_type,
            subtype=block.subtype,
            text=wrapped_text,
            bbox=normalized,
            region_tag=block.region_tag,
            flow=block.flow,
            ms=block.ms,
            hs=block.hs,
            reason=block.reason,
            anchor_to=anchor_id,
            attached_across_pages=block.attached_across_pages,
            confidence=block.confidence,
            quarantined=block.quarantined,
            meta={k: v for k, v in block.meta.items() if k != "sources"},
        )
        docblocks.append(docblock.to_dict())
    return docblocks


def _last_main_id(docblocks: List[Dict[str, object]]) -> Optional[str]:
    for block in reversed(docblocks):
        if block.get("kind") == "main":
            return block.get("id")
    return None
