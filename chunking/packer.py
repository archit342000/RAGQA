"""Utilities for packing segmented blocks into retrieval chunks.

The packer is responsible for turning a sequence of ``Block`` objects into
overlapping token windows. It keeps track of token counts using the same
tokenizer family that downstream generation models expect.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

from transformers import AutoTokenizer

from chunking.types import Block, Chunk

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER_NAME = "hf-internal-testing/llama-tokenizer"


class _WhitespaceTokenizer:
    """Minimal tokenizer used when pretrained models are unavailable."""

    def __init__(self) -> None:
        self.model_max_length = 16384
        self.padding_side = "left"

    def __call__(self, text: str, add_special_tokens: bool = False, return_attention_mask: bool = False):
        tokens = [tok for tok in text.split() if tok]
        return {"input_ids": tokens}

    def decode(self, token_ids: Sequence[str] | Sequence[int]) -> str:
        return " ".join(str(tok) for tok in token_ids)


@lru_cache(maxsize=2)
def get_tokenizer(model_name: str | None = None):
    """Load and cache the tokenizer used for token accounting."""

    name = model_name or DEFAULT_TOKENIZER_NAME
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
        except Exception as exc:  # pragma: no cover - exercised via tests
            logger.warning("Falling back to whitespace tokenizer for %s: %s", name, exc)
            return _WhitespaceTokenizer()
    tokenizer.model_max_length = 16384
    tokenizer.padding_side = "left"
    return tokenizer


def _encode(text: str, tokenizer) -> Tuple[List[int], int]:
    """Tokenise text and return both the ids and their length."""

    if not text.strip():
        return [], 0
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    token_ids = encoded["input_ids"]
    return token_ids, len(token_ids)


def pack_blocks(
    blocks: Sequence[Block],
    *,
    tokenizer,
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int,
    max_pages: int,
    strategy_label: str,
) -> List[Chunk]:
    """Pack ordered blocks into windowed chunks with optional overlap.

    Parameters are intentionally explicit so the driver can tune the packer for
    semantic vs. table-heavy content. All token accounting happens here so that
    the rest of the pipeline can reason in natural language units.
    """

    chunks: List[Chunk] = []
    buffer: List[Block] = []
    buffer_token_ids: List[int] = []
    buffer_token_len = 0
    seen_texts: set[str] = set()

    def flush_buffer() -> None:
        nonlocal buffer, buffer_token_ids, buffer_token_len
        if not buffer:
            return
        chunk_text = "\n\n".join(b.text for b in buffer).strip()
        if not chunk_text:
            buffer = []
            buffer_token_ids = []
            buffer_token_len = 0
            return
        full_ids, full_len = _encode(chunk_text, tokenizer)
        if full_len == 0:
            buffer = []
            buffer_token_ids = []
            buffer_token_len = 0
            return
        page_start = min(b.page_num for b in buffer)
        page_end = max(b.page_num for b in buffer)
        block_types = sorted({b.kind for b in buffer})
        heading = next((b.text for b in buffer if b.kind == "heading"), None)

        base_meta: Dict[str, str] = {
            "block_types": ",".join(block_types),
            "strategy": strategy_label,
        }

        def _emit(window_ids: Sequence[int] | Sequence[str], text: str, index: int | None = None) -> bool:
            if not text:
                return False
            key = text
            if key in seen_texts:
                return False
            seen_texts.add(key)
            meta = dict(base_meta)
            if index is not None:
                meta["window_index"] = str(index)
            chunks.append(
                Chunk(
                    doc_id=buffer[0].doc_id,
                    doc_name=buffer[0].doc_name,
                    page_start=page_start,
                    page_end=page_end,
                    section_title=heading,
                    text=text,
                    token_len=len(window_ids),
                    meta=meta,
                )
            )
            return True

        if full_len <= max_tokens:
            _emit(full_ids, chunk_text)
        else:
            stride = max(1, max_tokens - overlap_tokens)
            window_start = 0
            window_index = 0
            while window_start < full_len:
                window_end = min(full_len, window_start + max_tokens)
                window_ids = full_ids[window_start:window_end]
                window_text = tokenizer.decode(window_ids).strip()
                if _emit(window_ids, window_text, window_index):
                    window_index += 1
                if window_end >= full_len:
                    break
                window_start += stride

        if overlap_tokens > 0 and full_len > overlap_tokens:
            tail_ids = full_ids[-overlap_tokens:]
            overlap_text = tokenizer.decode(tail_ids).strip()
            if overlap_text:
                overlap_block = Block(
                    doc_id=buffer[-1].doc_id,
                    doc_name=buffer[-1].doc_name,
                    page_num=buffer[-1].page_num,
                    kind="overlap",
                    text=overlap_text,
                    char_start=buffer[-1].char_start,
                    char_end=buffer[-1].char_end,
                )
                buffer = [overlap_block]
                buffer_token_ids = list(tail_ids)
                buffer_token_len = len(tail_ids)
                return
        buffer = []
        buffer_token_ids = []
        buffer_token_len = 0

    for block in blocks:
        token_ids, token_len = _encode(block.text, tokenizer)
        if token_len == 0:
            continue

        pages_in_buffer = {b.page_num for b in buffer}
        if block.page_num not in pages_in_buffer:
            pages_in_buffer.add(block.page_num)
        would_exceed_pages = buffer and len(pages_in_buffer) > max_pages
        would_exceed_tokens = buffer and (buffer_token_len + token_len > max_tokens)

        if buffer and (would_exceed_tokens or would_exceed_pages) and buffer_token_len >= min_tokens:
            # Emit the current chunk before adding the block that would cause
            # us to exceed token or page constraints.
            flush_buffer()

        if not buffer:
            buffer = [block]
            buffer_token_ids = token_ids[:]
            buffer_token_len = token_len
        else:
            buffer.append(block)
            buffer_token_ids.extend(token_ids)
            buffer_token_len += token_len

    if buffer:
        if buffer_token_len < min_tokens and chunks:
            tail_text = "\n\n".join(b.text for b in buffer if b.text).strip()
            if tail_text:
                last = chunks[-1]
                merged_text = (last.text + "\n\n" + tail_text).strip()
                merged_ids, merged_len = _encode(merged_text, tokenizer)
                meta = dict(last.meta)
                meta.setdefault("merged_tail", "true")
                chunks[-1] = Chunk(
                    doc_id=last.doc_id,
                    doc_name=last.doc_name,
                    page_start=min(last.page_start, buffer[0].page_num),
                    page_end=max(last.page_end, buffer[-1].page_num),
                    section_title=last.section_title,
                    text=merged_text,
                    token_len=merged_len,
                    meta=meta,
                )
        else:
            flush_buffer()

    return chunks
