from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, List, Optional, Sequence, Tuple

from .config import PipelineConfig
from .normalize import Block


@dataclass(slots=True)
class FlowBlockFragment:
    block: Block
    text: str
    tokens: int
    boundary_kind: str
    safe_split_after: bool
    is_continuation: bool = False
    forced_split: bool = False


@dataclass
class FlowChunkPlan:
    blocks: List[FlowBlockFragment]
    closed_at: str
    forced_split: bool


BOUNDARY_ORDER = ["H1", "H2", "H3", "Sub", "List", "Para", "Sent", "None"]

ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "fig",
    "eq",
    "no",
    "st",
    "dept",
    "inc",
    "ltd",
}


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

    ordered = _expand_blocks_to_fragments(blocks, config, token_counter)
    plans: List[FlowChunkPlan] = []
    current: List[FlowBlockFragment] = []
    current_tokens = 0

    def _current_safe_index() -> Optional[int]:
        for idx in range(len(current) - 1, -1, -1):
            fragment = current[idx]
            if fragment.safe_split_after:
                return idx
            if (fragment.boundary_kind or "None") in {"Para", "List", "Sub", "Sent"}:
                return idx
        return None

    def _flush(reason: str) -> None:
        nonlocal current, current_tokens
        if not current:
            return
        forced = any(fragment.forced_split for fragment in current)
        plans.append(
            FlowChunkPlan(
                blocks=list(current),
                closed_at=reason or "Para",
                forced_split=forced,
            )
        )
        current = []
        current_tokens = 0

    def _append(fragment: FlowBlockFragment) -> None:
        nonlocal current_tokens
        current.append(fragment)
        current_tokens += fragment.tokens

    i = 0
    n = len(ordered)
    while i < n:
        fragment = ordered[i]
        block_tokens = fragment.tokens
        if not current:
            _append(fragment)
            if current_tokens >= hard:
                _flush(fragment.boundary_kind or "Sent")
            i += 1
            continue

        projected = current_tokens + block_tokens

        if current_tokens < minimum and fragment.boundary_kind != "H1":
            _append(fragment)
            i += 1
            continue

        if projected <= soft:
            _append(fragment)
            i += 1
            continue

        if current_tokens >= target:
            boundary_idx, ahead_tokens, boundary_kind = _scan_to_boundary(
                ordered, i
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
            plans.append(
                FlowChunkPlan(
                    blocks=list(keep),
                    closed_at=reason,
                    forced_split=any(fragment.forced_split for fragment in keep),
                )
            )
            current = list(tail)
            current_tokens = sum(fragment.tokens for fragment in current)
            continue

        # As a last resort flush everything to avoid infinite loop
        reason = current[-1].boundary_kind or "Para"
        _flush(reason)
        # reprocess same block without incrementing i

    if current:
        _flush("EOF")

    return plans


def _scan_to_boundary(
    blocks: Sequence[FlowBlockFragment],
    start: int,
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    best_idx: Optional[int] = None
    best_rank = float("inf")
    total_tokens = 0
    boundary_kind: Optional[str] = None

    for idx in range(start, len(blocks)):
        fragment = blocks[idx]
        total_tokens += fragment.tokens
        kind = fragment.boundary_kind or "None"
        if fragment.safe_split_after or kind != "None":
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


def _expand_blocks_to_fragments(
    blocks: Sequence[Block],
    config: PipelineConfig,
    token_counter: Callable[[str], int],
) -> List[FlowBlockFragment]:
    fragments: List[FlowBlockFragment] = []
    hard = config.flow.limits.hard
    for block in blocks:
        text = (block.text or "").strip()
        if not text:
            continue
        tokens = block.est_tokens or token_counter(text)
        if tokens <= hard:
            fragments.append(
                FlowBlockFragment(
                    block=block,
                    text=text,
                    tokens=tokens,
                    boundary_kind=block.boundary_kind or "None",
                    safe_split_after=block.safe_split_after,
                    is_continuation=False,
                )
            )
            continue
        fragments.extend(
            _split_block_into_fragments(block, text, tokens, hard, token_counter)
        )
    return fragments


def _split_block_into_fragments(
    block: Block,
    text: str,
    total_tokens: int,
    hard_limit: int,
    token_counter: Callable[[str], int],
) -> List[FlowBlockFragment]:
    sentences = _sentence_split(text)
    if not sentences:
        sentences = [text]

    fragments: List[FlowBlockFragment] = []
    buffer: List[str] = []
    buffer_tokens = 0

    def _flush_buffer(boundary_kind: str = "Sent", forced_split: bool = True) -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        snippet = " ".join(buffer).strip()
        if not snippet:
            buffer = []
            buffer_tokens = 0
            return
        safe_after = True if boundary_kind == "Sent" else block.safe_split_after
        fragments.append(
            FlowBlockFragment(
                block=block,
                text=snippet,
                tokens=buffer_tokens,
                boundary_kind=boundary_kind,
                safe_split_after=safe_after,
                is_continuation=len(fragments) > 0,
                forced_split=forced_split,
            )
        )
        buffer = []
        buffer_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_tokens = token_counter(sentence)
        if sentence_tokens > hard_limit:
            # sentence itself too long; fall back to word-based chunks
            word_fragments = _split_sentence_by_words(
                sentence, block, hard_limit, token_counter
            )
            for idx, frag in enumerate(word_fragments):
                frag.is_continuation = len(fragments) > 0 or bool(buffer)
                frag.safe_split_after = True
                frag.boundary_kind = "Sent"
                frag.forced_split = True
                if idx == len(word_fragments) - 1 and not buffer:
                    frag.boundary_kind = block.boundary_kind or "Para"
                    frag.safe_split_after = block.safe_split_after
                if buffer:
                    # flush existing buffer before appending word fragments
                    _flush_buffer()
                fragments.append(frag)
            continue
        projected = token_counter(" ".join(buffer + [sentence]).strip()) if buffer else sentence_tokens
        if projected > hard_limit and buffer:
            _flush_buffer()
            buffer.append(sentence)
            buffer_tokens = sentence_tokens
            continue
        buffer.append(sentence)
        buffer_tokens = projected

    if buffer:
        _flush_buffer(boundary_kind=block.boundary_kind or "Para")

    # if fragmentation still empty (e.g., whitespace), fall back to original text
    if not fragments:
        fragments.append(
            FlowBlockFragment(
                block=block,
                text=text,
                tokens=total_tokens,
                boundary_kind=block.boundary_kind or "None",
                safe_split_after=block.safe_split_after,
                is_continuation=False,
                forced_split=True,
            )
        )
    return fragments


def _sentence_split(text: str) -> List[str]:
    if not text:
        return []
    raw_segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?;])\s+", text)
        if segment.strip()
    ]
    sentences: List[str] = []
    buffer = ""
    for segment in raw_segments:
        candidate = f"{buffer} {segment}".strip() if buffer else segment
        words = segment.split()
        last_word = words[-1] if words else ""
        last_alpha = re.sub(r"[^A-Za-z]+$", "", last_word).lower()
        if last_alpha in ABBREVIATIONS and not segment.endswith("..."):
            buffer = candidate
            continue
        sentences.append(candidate)
        buffer = ""
    if buffer:
        sentences.append(buffer)
    return sentences


def _split_sentence_by_words(
    sentence: str,
    block: Block,
    hard_limit: int,
    token_counter: Callable[[str], int],
) -> List[FlowBlockFragment]:
    words = sentence.split()
    fragments: List[FlowBlockFragment] = []
    buffer: List[str] = []
    buffer_tokens = 0

    def _flush() -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        snippet = " ".join(buffer).strip()
        if not snippet:
            buffer = []
            buffer_tokens = 0
            return
        fragments.append(
            FlowBlockFragment(
                block=block,
                text=snippet,
                tokens=buffer_tokens,
                boundary_kind="Sent",
                safe_split_after=True,
                is_continuation=len(fragments) > 0,
                forced_split=True,
            )
        )
        buffer = []
        buffer_tokens = 0

    for word in words:
        candidate_list = buffer + [word] if buffer else [word]
        candidate_text = " ".join(candidate_list).strip()
        tokens = token_counter(candidate_text)
        if tokens > hard_limit and buffer:
            _flush()
            candidate_list = [word]
            candidate_text = word
            tokens = token_counter(candidate_text)
        buffer = candidate_list
        buffer_tokens = tokens
        if buffer_tokens >= hard_limit:
            _flush()

    if buffer:
        _flush()
    return fragments
