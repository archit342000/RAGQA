"""Utilities for cross-page dehyphenation while tracking character offsets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from pipeline.layout.lp_fuser import FusedBlock


@dataclass(slots=True)
class DehyphenationResult:
    prev_block_id: str
    next_block_id: str
    removed_suffix: str
    prev_delta: int
    next_delta: int


def dehyphenate_pair(prev_text: str, next_text: str) -> Optional[tuple[str, str, DehyphenationResult]]:
    """Return updated texts when ``prev_text`` ends with a hyphenated break."""

    if not prev_text or not next_text:
        return None
    match_prev = re.search(r"([A-Za-z]{3,})-\s*$", prev_text)
    if not match_prev:
        return None
    match_next = re.match(r"\s*([A-Za-z]{2,})", next_text)
    if not match_next:
        return None
    prefix = match_prev.group(1)
    suffix = match_next.group(1)
    start = match_prev.start(1)
    updated_prev = prev_text[:start] + prefix + suffix
    remainder_next = next_text[match_next.end():].lstrip()
    prev_delta = len(updated_prev) - len(prev_text)
    next_delta = len(remainder_next) - len(next_text)
    result = DehyphenationResult(
        prev_block_id="",
        next_block_id="",
        removed_suffix=suffix,
        prev_delta=prev_delta,
        next_delta=next_delta,
    )
    return updated_prev, remainder_next, result


def apply_dehyphenation(prev_block: FusedBlock, next_block: FusedBlock) -> Optional[DehyphenationResult]:
    """Mutate ``prev_block`` and ``next_block`` when they contain a hyphen break."""

    outcome = dehyphenate_pair(prev_block.text, next_block.text)
    if outcome is None:
        return None
    updated_prev, remainder_next, result = outcome
    prev_block.text = updated_prev
    next_block.text = remainder_next
    result.prev_block_id = prev_block.block_id
    result.next_block_id = next_block.block_id
    return result


__all__ = ["DehyphenationResult", "apply_dehyphenation", "dehyphenate_pair"]
