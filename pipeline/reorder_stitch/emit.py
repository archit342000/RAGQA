from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..normalize import Block


def group_auxiliary_blocks(blocks: Sequence[Block]) -> Tuple[Dict[str, List[dict]], List[str]]:
    """Group auxiliary blocks by subtype for aux-only chunks."""

    sidecars: List[dict] = []
    footnotes: List[dict] = []
    other: List[dict] = []
    subtypes: set[str] = set()

    for block in blocks:
        subtype = (block.aux_subtype or "other").lower()
        text = (block.text or "").strip()
        if subtype == "caption":
            parent = block.parent_block_id
            figure_type = "figure"
            lower = text.lower()
            if "table" in lower:
                figure_type = "table"
            sidecars.append(
                {
                    "parent_block_id": parent,
                    "type": figure_type,
                    "page": block.page,
                    "text": text,
                }
            )
        elif subtype == "footnote":
            footnotes.append({"ref_id": block.block_id, "text": text})
        else:
            other.append({"aux_subtype": subtype, "text": text})
        subtypes.add(subtype)

    grouped = {"sidecars": sidecars, "footnotes": footnotes, "other": other}
    return grouped, sorted(subtypes)
