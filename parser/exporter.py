"""Exporter for the extended DocBlock schema."""
from __future__ import annotations

from typing import Dict, List


def _wrap_aux_text(block_type: str, text: str) -> str:
    cleaned = (text or "").strip()
    if block_type == "aux":
        if cleaned.startswith("<aux>") and cleaned.endswith("</aux>"):
            return cleaned
        return f"<aux>{cleaned}</aux>"
    return cleaned


def export_doc(doc: Dict[str, object]) -> Dict[str, object]:
    """Normalise block payloads and wrap aux text."""

    pages_out: List[Dict[str, object]] = []
    for page in doc.get("pages", []):
        page_id = page.get("page_id", "p_0001")
        blocks_out: List[Dict[str, object]] = []
        for idx, block in enumerate(page.get("blocks", [])):
            block_id = block.get("id") or f"{page_id}_blk_{idx:03d}"
            text = _wrap_aux_text(block.get("type", "main"), block.get("text", ""))
            blocks_out.append(
                {
                    "id": block_id,
                    "type": block.get("type", "main"),
                    "subtype": block.get("subtype"),
                    "bbox": [float(v) for v in block.get("bbox", [0, 0, 0, 0])],
                    "text": text,
                    "links": block.get("links", []),
                    "page_references": block.get("page_references", []),
                    "ro_index": block.get("ro_index", 0),
                    "flow_id": block.get("flow_id"),
                    "region_tag": block.get("region_tag"),
                    "confidence": block.get("confidence", 0.0),
                    "aux_shadow": block.get("aux_shadow", False),
                    "inline_caption": block.get("inline_caption", False),
                    "quarantined": block.get("quarantined", False),
                    "meta": block.get("meta", {}),
                    "reason": block.get("reason", []),
                }
            )
        pages_out.append(
            {
                "page_id": page_id,
                "blocks": blocks_out,
                "aux_queue": page.get("aux_queue", []),
                "split_events": page.get("split_events", []),
            }
        )
    return {"pages": pages_out}
