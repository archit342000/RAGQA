from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..normalize import Block
from .continuity import continuity_score

FLOAT_TYPES = {"table", "figure", "caption", "footnote"}


def build_threads(
    blocks: Sequence[Block],
    column_lookup: Dict[str, int],
    weights: Dict[str, float],
    tau_main: float,
) -> Tuple[List[Dict[str, object]], List[Block]]:
    """Build flow threads using greedy path cover with 1-page lookahead."""

    ordered = sorted(
        [b for b in blocks if _eligible_for_thread(b)],
        key=lambda b: (b.page, b.bbox["y0"] if b.bbox else 0.0, b.bbox["x0"] if b.bbox else 0.0),
    )
    used: set[str] = set()
    threads: List[Dict[str, object]] = []
    unthreaded: List[Block] = []

    for idx, block in enumerate(ordered):
        if block.block_id in used:
            continue
        thread_blocks: List[Block] = [block]
        scores: List[float] = []
        used.add(block.block_id)
        current_index = idx
        current_block = block
        while True:
            candidate = _best_successor(
                current_block,
                current_index,
                ordered,
                used,
                column_lookup,
                weights,
            )
            if candidate is None:
                break
            score, next_index, next_block = candidate
            if score < tau_main:
                break
            scores.append(score)
            thread_blocks.append(next_block)
            used.add(next_block.block_id)
            current_index = next_index
            current_block = next_block

        total_tokens = sum(max(b.est_tokens, 0) for b in thread_blocks)
        avg_score = (sum(scores) / len(scores)) if scores else tau_main
        if (len(thread_blocks) >= 2 or total_tokens >= 150) and avg_score >= tau_main:
            threads.append(
                {
                    "thread_id": f"thr_{len(threads):04d}",
                    "section": list(thread_blocks[0].heading_path),
                    "blocks": [b.block_id for b in thread_blocks],
                    "cohesion": float(avg_score),
                    "tokens": int(total_tokens),
                }
            )
        else:
            for b in thread_blocks:
                used.discard(b.block_id)
            unthreaded.extend(thread_blocks)

    for block in ordered:
        if block.block_id not in used and block not in unthreaded:
            unthreaded.append(block)

    return threads, unthreaded


def _eligible_for_thread(block: Block) -> bool:
    if block.role != "main":
        return False
    if block.type in FLOAT_TYPES:
        return False
    if not (block.text or "").strip():
        return False
    return True


def _best_successor(
    current: Block,
    current_index: int,
    ordered: Sequence[Block],
    used: set[str],
    column_lookup: Dict[str, int],
    weights: Dict[str, float],
) -> Tuple[float, int, Block] | None:
    best: Tuple[float, int, Block] | None = None
    current_col = column_lookup.get(current.block_id, 0)
    current_page = current.page
    for next_index in range(current_index + 1, len(ordered)):
        candidate = ordered[next_index]
        if candidate.block_id in used:
            continue
        page_delta = candidate.page - current_page
        if page_delta < 0:
            continue
        if page_delta > 1:
            break
        if tuple(candidate.heading_path) != tuple(current.heading_path):
            continue
        if page_delta == 0:
            if not candidate.bbox or not current.bbox:
                pass
            elif candidate.bbox["y0"] < current.bbox["y0"]:
                continue
            next_col = column_lookup.get(candidate.block_id, 0)
            if next_col != current_col:
                continue
        else:  # next page
            next_col = column_lookup.get(candidate.block_id, 0)
            if next_col != current_col:
                continue
        score = continuity_score(current, candidate, weights)
        if best is None or score > best[0]:
            best = (score, next_index, candidate)
    return best
