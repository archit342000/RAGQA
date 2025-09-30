from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from .main_gate import main_gate
from .normalize import Block


def flow_fence_tail_sanitize(
    blocks: Iterable[Block | Dict[str, Any]], context: Dict[str, Any] | None = None
) -> Tuple[List[Block], List[Block], int]:
    """Re-validate trailing blocks before sealing a Flow chunk.

    Parameters
    ----------
    blocks:
        Iterable of canonical block dicts in chunk order.
    context:
        Optional dictionary carrying configuration/telemetry references.

    Returns
    -------
    tuple
        ``(kept_blocks, diverted_blocks, hits)`` where ``hits`` is the number of
        blocks reclassified as auxiliary. Diverted blocks should be appended to the
        segment's aux buffer for post-hoc emission.
    """

    kept: List[Block] = []
    diverted: List[Block] = []
    hits = 0
    for block in blocks:
        passed, reasons = main_gate(block, context)
        if passed:
            kept.append(block)  # type: ignore[arg-type]
        else:
            hits += 1
            if isinstance(block, dict):
                block.setdefault("rejection_reasons", reasons)
                block["main_gate_passed"] = False
            else:
                block.rejection_reasons = reasons
                block.main_gate_passed = False
            diverted.append(block)  # type: ignore[arg-type]
    return kept, diverted, hits
