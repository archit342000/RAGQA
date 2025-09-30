from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

from ..token_utils import count_tokens

if TYPE_CHECKING:  # pragma: no cover
    from ..chunker import Chunk

_STITCH_BUDGET_TOKENS = 300
_ALLOW_CROSS_SECTION = False


def _is_sentence_closed(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    last_line = stripped.splitlines()[-1]
    return last_line.endswith(('.', '?', '!'))


def cross_page_stitch(chunks: List["Chunk"]) -> Tuple[List["Chunk"], int]:
    """Merge spill-over paragraphs from the next main chunk when safe."""

    stitched = 0
    idx = 0
    while idx < len(chunks) - 1:
        current = chunks[idx]
        if not current.is_main_only or _is_sentence_closed(current.text):
            idx += 1
            continue
        lookahead = idx + 1
        while lookahead < len(chunks) and not chunks[lookahead].is_main_only:
            lookahead += 1
        if lookahead >= len(chunks):
            break
        next_chunk = chunks[lookahead]
        same_section = (current.section_id or current.segment_id) == (
            next_chunk.section_id or next_chunk.segment_id
        )
        if not _ALLOW_CROSS_SECTION and not same_section:
            idx += 1
            continue
        if next_chunk.token_count > _STITCH_BUDGET_TOKENS:
            idx += 1
            continue
        merged_text = (current.text.rstrip() + "\n\n" + next_chunk.text.lstrip()).strip()
        current.text = merged_text
        current.token_count = count_tokens(merged_text)
        current.flow_overflow = max(0, current.token_count - current.limits.get("target", 1600))
        current.closed_at_boundary = next_chunk.closed_at_boundary
        current.page_span = [
            min(current.page_span[0], next_chunk.page_span[0]),
            max(current.page_span[1], next_chunk.page_span[1]),
        ]
        current.debug_block_ids.extend(next_chunk.debug_block_ids)
        current.evidence_spans.extend(next_chunk.evidence_spans)
        current.sidecars.extend(next_chunk.sidecars)
        chunks.pop(lookahead)
        stitched += 1
    return chunks, stitched
