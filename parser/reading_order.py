"""Reading order reconstruction helpers."""
from __future__ import annotations

from typing import List

from .grouping import Block


def order_reading(blocks: List[Block], cfg) -> List[Block]:
    """Assign reading order indices based on column and position."""

    sorted_blocks = sorted(
        blocks,
        key=lambda blk: (
            blk.meta.get("col_id", 0),
            blk.bbox[1],
            blk.bbox[0],
        ),
    )
    for idx, block in enumerate(sorted_blocks):
        block.ro_index = idx
    return sorted_blocks
