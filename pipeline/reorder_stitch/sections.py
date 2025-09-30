from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from ..normalize import Block


def assemble_sections(
    threads: Iterable[Dict[str, object]],
    block_lookup: Dict[str, Block],
) -> Tuple[List[Dict[str, object]], List[Block]]:
    """Assemble section narratives from threads; return (sections, aux_pool)."""

    by_heading: Dict[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for thread in threads:
        heading = tuple(thread.get("section", []))
        by_heading[heading].append(thread)

    sections: List[Dict[str, object]] = []
    used_blocks: set[str] = set()

    for idx, (heading, thread_list) in enumerate(sorted(by_heading.items(), key=lambda item: item[0])):
        sorted_threads = sorted(thread_list, key=lambda t: t.get("tokens", 0), reverse=True)
        if not sorted_threads:
            continue
        primary = sorted_threads[0]
        blocks_order = list(primary.get("blocks", []))
        used_blocks.update(blocks_order)
        section_thread_ids = [primary.get("thread_id")]

        for minor in sorted_threads[1:]:
            for block_id in minor.get("blocks", []):
                if block_id in used_blocks:
                    continue
                blocks_order.append(block_id)
                used_blocks.add(block_id)
            section_thread_ids.append(minor.get("thread_id"))

        sections.append(
            {
                "section_id": f"sec_{idx:04d}",
                "heading_path": list(heading),
                "blocks": blocks_order,
                "thread_ids": section_thread_ids,
                "cohesion": float(primary.get("cohesion", 0.0) or 0.0),
            }
        )

    aux_pool = [block for block_id, block in block_lookup.items() if block_id not in used_blocks]
    return sections, aux_pool
