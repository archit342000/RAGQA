from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .docling_adapter import DoclingBlock
from .ids import make_block_id

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Block:
    doc_id: str
    block_id: str
    page: int
    order: int
    type: str
    text: str
    bbox: dict
    heading_level: int | None
    heading_path: List[str]
    source: dict
    aux: dict


def normalise_blocks(doc_id: str, blocks: Sequence[DoclingBlock]) -> List[Block]:
    normalised: List[Block] = []
    for order, block in enumerate(blocks):
        bbox = None
        if block.bbox:
            x0, y0, x1, y1 = block.bbox
            bbox = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        else:
            bbox = {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}
        block_type = block.block_type if block.block_type in {
            "heading",
            "paragraph",
            "list",
            "item",
            "table",
            "figure",
            "caption",
            "code",
            "footnote",
        } else "paragraph"
        normalised.append(
            Block(
                doc_id=doc_id,
                block_id=make_block_id(block.page_number, order + 1),
                page=block.page_number,
                order=order,
                type=block_type,
                text=block.text,
                bbox=bbox,
                heading_level=block.heading_level,
                heading_path=list(block.heading_path),
                source={
                    "stage": block.source_stage,
                    "tool": block.source_tool,
                    "version": block.source_version,
                },
                aux=dict(block.aux),
            )
        )
    return normalised
