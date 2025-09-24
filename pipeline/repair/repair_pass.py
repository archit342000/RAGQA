"""Repair pass that stitches intrusions and links footnotes."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pipeline.layout.lp_fuser import FusedBlock, FusedDocument, FusedPage
from pipeline.layout.signals import PageLayoutSignals

EmbeddingFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]

_SUPERSCRIPT_MAP = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
}
_SUPERSCRIPT_REVERSE = {v: k for k, v in _SUPERSCRIPT_MAP.items()}


@dataclass(slots=True)
class RepairStats:
    merged_blocks: int = 0
    split_blocks: int = 0
    footnotes_linked: int = 0
    pages_with_failures: int = 0
    failure_counts: Dict[int, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.failure_counts is None:
            self.failure_counts = {}


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _should_guard_merge(block_a: FusedBlock, block_b: FusedBlock, page: FusedPage) -> bool:
    if block_a.column is None or block_b.column is None:
        return False
    if block_a.column == block_b.column:
        return False
    delta_y = abs(block_b.bbox[1] - block_a.bbox[3]) / max(page.height, 1.0)
    if delta_y < 0.12:
        return False
    if block_a.text.rstrip().endswith(('.', '?', '!', ';', ':')):
        return False
    return True


def _stitch_page(
    page: FusedPage,
    *,
    embedder: Optional[EmbeddingFn],
) -> int:
    if embedder is None:
        return 0
    merges = 0
    idx = 0
    while idx < len(page.main_flow) - 1:
        current_block = page.main_flow[idx]
        next_block = page.main_flow[idx + 1]
        if not current_block.text or not next_block.text:
            idx += 1
            continue
        vectors = embedder([current_block.text, next_block.text])
        if len(vectors) < 2:
            idx += 1
            continue
        similarity = _cosine_similarity(vectors[0], vectors[1])
        if similarity >= 0.80 and not _should_guard_merge(current_block, next_block, page):
            current_block.text = current_block.text.rstrip() + "\n" + next_block.text.lstrip()
            current_block.char_end = current_block.char_start + len(current_block.text)
            merged_from = current_block.metadata.setdefault("merged_from", [])
            if isinstance(merged_from, list):
                merged_from.append(next_block.block_id)
            del page.main_flow[idx + 1]
            merges += 1
            continue
        idx += 1
    return merges


def _split_overmerged(
    page: FusedPage,
    *,
    document: FusedDocument,
) -> int:
    splits = 0
    updated_blocks: List[FusedBlock] = []
    for block in page.main_flow:
        candidates = _split_block_if_needed(block, page)
        if not candidates:
            updated_blocks.append(block)
            continue
        splits += 1
        document.block_index.pop(block.block_id, None)
        first, second = candidates
        updated_blocks.extend([first, second])
        document.block_index[first.block_id] = first
        document.block_index[second.block_id] = second
    page.main_flow[:] = updated_blocks
    return splits


def _split_block_if_needed(block: FusedBlock, page: FusedPage) -> Optional[Tuple[FusedBlock, FusedBlock]]:
    width_ratio = (block.bbox[2] - block.bbox[0]) / max(page.width, 1.0)
    height_ratio = (block.bbox[3] - block.bbox[1]) / max(page.height, 1.0)
    punctuation = sum(block.text.count(char) for char in ".!?")
    if width_ratio < 0.75 or height_ratio < 0.25 or punctuation > 0 or len(block.text) < 200:
        return None
    midpoint = len(block.text) // 2
    split_idx = block.text.find(" ", midpoint)
    if split_idx == -1 or split_idx < 80 or len(block.text) - split_idx < 80:
        return None
    head = block.text[:split_idx].strip()
    tail = block.text[split_idx:].strip()
    if len(head) < 80 or len(tail) < 80:
        return None
    mid_y = (block.bbox[1] + block.bbox[3]) / 2.0
    first = replace(block)
    second = replace(block)
    first.text = head
    first.char_end = first.char_start + len(head)
    first.bbox = (block.bbox[0], block.bbox[1], block.bbox[2], mid_y)
    first.metadata = dict(block.metadata)
    first.metadata["split_from"] = block.block_id
    second.block_id = f"{block.block_id}_split"
    second.text = tail
    second.char_start = first.char_end
    second.char_end = second.char_start + len(tail)
    second.bbox = (block.bbox[0], mid_y, block.bbox[2], block.bbox[3])
    second.metadata = dict(block.metadata)
    second.metadata["split_from"] = block.block_id
    return first, second


def _extract_footnote_marker(text: str) -> Optional[str]:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped[0] in _SUPERSCRIPT_REVERSE:
        return _SUPERSCRIPT_REVERSE[stripped[0]]
    match = re.match(r"(\d+)[\).\s]", stripped)
    if match:
        return match.group(1)
    return None


def _link_footnotes(page: FusedPage, signal: PageLayoutSignals) -> int:
    linked = 0
    candidates = [
        block
        for block in page.auxiliaries
        if block.block_id in set(signal.extras.footnote_block_ids)
        or (
            block.avg_font_size > 0
            and block.avg_font_size < signal.extras.dominant_font_size * 0.75
            and block.bbox[1] > page.height * 0.75
        )
    ]
    if not candidates or not page.main_flow:
        return 0
    for footnote in candidates:
        marker = _extract_footnote_marker(footnote.text)
        if marker is None:
            continue
        super_marker = "".join(_SUPERSCRIPT_MAP.get(ch, ch) for ch in marker)
        matched_block: Optional[FusedBlock] = None
        for main_block in page.main_flow:
            text = main_block.text
            if super_marker and super_marker in text:
                matched_block = main_block
                break
            if f"[{marker}]" in text or f"^{marker}" in text:
                matched_block = main_block
                break
        if matched_block is None:
            continue
        matched_block.metadata.setdefault("footnotes", []).append(footnote.block_id)
        footnote.metadata["linked_to"] = matched_block.block_id
        footnote.metadata["reference_number"] = marker
        footnote.metadata["is_footnote"] = True
        linked += 1
    return linked


def run_repair_pass(
    document: FusedDocument,
    signals: Sequence[PageLayoutSignals],
    *,
    embedder: Optional[EmbeddingFn] = None,
    previous_failures: Optional[Dict[int, int]] = None,
) -> Tuple[FusedDocument, RepairStats, Dict[int, int]]:
    if len(document.pages) != len(signals):
        raise ValueError("Signals must align with document pages")

    failure_counts = dict(previous_failures or {})
    stats = RepairStats(failure_counts=failure_counts)

    for page, signal in zip(document.pages, signals):
        merges = _stitch_page(page, embedder=embedder)
        splits = _split_overmerged(page, document=document)
        linked = _link_footnotes(page, signal)
        stats.merged_blocks += merges
        stats.split_blocks += splits
        stats.footnotes_linked += linked
        if embedder is None:
            failure_counts[page.page_number] = failure_counts.get(page.page_number, 0) + 1
        elif merges == 0 and splits == 0:
            failure_counts[page.page_number] = failure_counts.get(page.page_number, 0)
        else:
            failure_counts[page.page_number] = 0
    stats.pages_with_failures = sum(1 for count in failure_counts.values() if count)
    return document, stats, failure_counts


__all__ = ["RepairStats", "run_repair_pass"]
