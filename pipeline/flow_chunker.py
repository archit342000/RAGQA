from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .config import PipelineConfig
from .normalize import Block


@dataclass
class FlowChunkPlan:
    blocks: List[Block]
    closed_at: str


BOUNDARY_ORDER = ["H1", "H2", "H3", "Sub", "List", "Para", "Sent", "None"]


def build_flow_chunk_plan(
    blocks: Sequence[Block],
    config: PipelineConfig,
    token_counter: Callable[[str], int],
) -> List[FlowChunkPlan]:
    limits = config.flow.limits
    target = limits.target
    soft = limits.soft
    hard = limits.hard
    minimum = limits.minimum
    slack = config.flow.boundary_slack_tokens

    ordered = [block for block in blocks if (block.text or "").strip()]
    plans: List[FlowChunkPlan] = []
    current: List[Block] = []
    current_tokens = 0

    def _block_tokens(block: Block) -> int:
        tokens = getattr(block, "est_tokens", 0)
        if tokens:
            return tokens
        return token_counter(block.text or "")

    def _current_safe_index() -> Optional[int]:
        for idx in range(len(current) - 1, -1, -1):
            blk = current[idx]
            if getattr(blk, "safe_split_after", False):
                return idx
            if (blk.boundary_kind or "None") in {"Para", "List", "Sub", "Sent"}:
                return idx
        return None

    def _flush(reason: str) -> None:
        nonlocal current, current_tokens
        if not current:
            return
        plans.append(FlowChunkPlan(blocks=list(current), closed_at=reason or "Para"))
        current = []
        current_tokens = 0

    def _append(block: Block) -> None:
        nonlocal current_tokens
        current.append(block)
        current_tokens += _block_tokens(block)

    i = 0
    n = len(ordered)
    while i < n:
        block = ordered[i]
        block_tokens = _block_tokens(block)
        if not current:
            _append(block)
            if current_tokens >= hard:
                _flush(block.boundary_kind or "Sent")
            i += 1
            continue

        projected = current_tokens + block_tokens

        if current_tokens < minimum and block.boundary_kind != "H1":
            _append(block)
            i += 1
            continue

        if projected <= soft:
            _append(block)
            i += 1
            continue

        if current_tokens >= target:
            boundary_idx, ahead_tokens, boundary_kind = _scan_to_boundary(
                ordered, i, token_counter
            )
            if (
                boundary_idx is not None
                and ahead_tokens is not None
                and current_tokens + ahead_tokens <= hard
                and (ahead_tokens <= slack or (boundary_idx - i) <= 1)
            ):
                while i <= boundary_idx:
                    _append(ordered[i])
                    i += 1
                _flush(boundary_kind or "Para")
                continue

        safe_idx = _current_safe_index()
        if safe_idx is not None and safe_idx < len(current):
            reason = current[safe_idx].boundary_kind or "Para"
            keep = current[: safe_idx + 1]
            tail = current[safe_idx + 1 :]
            plans.append(FlowChunkPlan(blocks=list(keep), closed_at=reason))
            current = list(tail)
            current_tokens = sum(_block_tokens(b) for b in current)
            continue

        # As a last resort flush everything to avoid infinite loop
        reason = current[-1].boundary_kind or "Para"
        _flush(reason)
        # reprocess same block without incrementing i

    if current:
        _flush("EOF")

    return plans


def _scan_to_boundary(
    blocks: Sequence[Block],
    start: int,
    token_counter: Callable[[str], int],
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    best_idx: Optional[int] = None
    best_rank = float("inf")
    total_tokens = 0
    boundary_kind: Optional[str] = None

    for idx in range(start, len(blocks)):
        block = blocks[idx]
        block_tokens = getattr(block, "est_tokens", 0)
        if not block_tokens:
            block_tokens = token_counter(block.text or "")
        total_tokens += block_tokens
        kind = block.boundary_kind or "None"
        if getattr(block, "safe_split_after", False) or kind != "None":
            rank = BOUNDARY_ORDER.index(kind) if kind in BOUNDARY_ORDER else len(BOUNDARY_ORDER)
            if rank < best_rank:
                best_rank = rank
                best_idx = idx
                boundary_kind = kind
            if kind in {"H1", "H2", "H3", "Sub", "Para", "Sent", "List"}:
                return best_idx, total_tokens, boundary_kind
    if best_idx is not None:
        return best_idx, total_tokens, boundary_kind
    return None, None, None
