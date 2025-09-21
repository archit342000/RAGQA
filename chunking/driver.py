"""Public entrypoint for transforming parsed documents into retrieval chunks.

The driver stitches together segmentation analytics, semantic/fixed chunking
heuristics, and token packing. It operates on ``ParsedDoc`` objects produced by
the parser module and returns a list of ready-to-use ``Chunk`` instances along
with per-document statistics consumed by the UI.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from env_loader import load_dotenv_once

from chunking import semantic_chunker
from chunking.segmenter import segment_page
from chunking.types import Block, Chunk
from chunking.packer import get_tokenizer, pack_blocks
from parser.types import ParsedDoc

logger = logging.getLogger(__name__)

load_dotenv_once()

# Default configuration knobs pulled from environment at import time.
DEFAULT_SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL_NAME", "intfloat/e5-small-v2")
DEFAULT_CHUNK_MODE = os.getenv("CHUNK_MODE_DEFAULT", "semantic").lower()
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME")
MAX_TOTAL_TOKENS = int(os.getenv("MAX_TOTAL_TOKENS_FOR_CHUNKING", "300000"))


def _page_blocks(doc_id: str, doc_name: str, page_num: int, text: str) -> Tuple[List[Block], dict]:
    """Segment a page and drop empty blocks.

    Returns both the filtered block list and the layout profile so callers can
    make routing decisions without reprocessing the page.
    """

    blocks, profile = segment_page(doc_id, doc_name, page_num, text)
    return [blk for blk in blocks if blk.text], profile


def _semantic_or_fixed_blocks(
    *,
    doc_id: str,
    doc_name: str,
    page_num: int,
    text: str,
    base_blocks: List[Block],
    profile: dict,
    semantic_enabled: bool,
    model_name: str,
) -> Tuple[List[Block], str]:
    """Return blocks for the page using semantic splitting when appropriate."""

    table_heavy = bool(profile.get("table_heavy"))
    if not semantic_enabled or table_heavy:
        return base_blocks, "fixed"

    segments = semantic_chunker.semantic_segments(text, model_name=model_name)
    if not segments:
        return base_blocks, "fixed"

    semantic_blocks: List[Block] = []
    cursor = 0
    lowered = text
    for segment in segments:
        snippet = segment.strip()
        if not snippet:
            continue
        idx = lowered.find(snippet, cursor)
        if idx == -1:
            idx = cursor
        char_start = idx
        char_end = idx + len(snippet)
        cursor = char_end
        semantic_blocks.append(
            Block(
                doc_id=doc_id,
                doc_name=doc_name,
                page_num=page_num,
                kind="semantic",
                text=snippet,
                char_start=char_start,
                char_end=char_end,
            )
        )

    return (semantic_blocks or base_blocks), "semantic"


def chunk_documents(
    docs: Sequence[ParsedDoc],
    *,
    mode: str | None = None,
    model_name: str | None = None,
) -> Tuple[List[Chunk], Dict[str, dict]]:
    """Chunk cleaned documents into retrieval-friendly windows.

    Parameters
    ----------
    docs:
        Sequence of parsed documents produced by :mod:`parser.driver`.
    mode:
        Requested chunking mode (``"semantic"`` or ``"fixed"``). When ``None``
        we consult ``CHUNK_MODE_DEFAULT`` from the environment.
    model_name:
        Optional embedding model override used by the semantic chunker.

    Returns
    -------
    tuple[list[Chunk], dict]
        The first element contains the fully materialised retrieval chunks. The
        second maps ``doc_id`` to diagnostic statistics that the UI can surface
        in the debug panel.
    """

    chunks: List[Chunk] = []
    chunk_stats: Dict[str, dict] = {}
    tokenizer = get_tokenizer(TOKENIZER_NAME)

    requested_mode = (mode or DEFAULT_CHUNK_MODE).lower()
    semantic_enabled = requested_mode == "semantic"
    semantic_model = model_name or DEFAULT_SEMANTIC_MODEL

    total_tokens_emitted = 0

    for doc in docs:
        doc_name = doc.pages[0].metadata.get("file_name") if doc.pages else doc.doc_id
        # Each sequence groups contiguous blocks that share the same packing
        # configuration (semantic vs. fixed/table). We process sequences
        # sequentially, emitting chunks until the global token guardrail is hit.
        doc_configured_sequences: List[Tuple[List[Block], Dict[str, int | str]]] = []

        current_config_key = None
        current_sequence: List[Block] = []
        # Track how many pages fell into each strategy so the UI can surface
        # meaningful diagnostics to the operator.
        per_strategy_counts = defaultdict(int)

        for page in doc.pages:
            page_num = page.page_num
            base_blocks, profile = _page_blocks(doc.doc_id, doc_name or doc.doc_id, page.page_num, page.text)
            blocks, strategy_used = _semantic_or_fixed_blocks(
                doc_id=doc.doc_id,
                doc_name=doc_name or doc.doc_id,
                page_num=page_num,
                text=page.text,
                base_blocks=base_blocks,
                profile=profile,
                semantic_enabled=semantic_enabled,
                model_name=semantic_model,
            )
            per_strategy_counts[strategy_used] += 1

            # Tune packer parameters per strategy; tables prefer smaller windows.
            if strategy_used == "fixed" and profile.get("table_heavy"):
                config = {
                    "max_tokens": 400,
                    "min_tokens": 200,
                    "overlap_tokens": 40,
                    "max_pages": 3,
                    "strategy_label": "fixed-table",
                }
            elif strategy_used == "fixed":
                config = {
                    "max_tokens": 700,
                    "min_tokens": 200,
                    "overlap_tokens": 100,
                    "max_pages": 3,
                    "strategy_label": "fixed",
                }
            else:  # semantic
                config = {
                    "max_tokens": 700,
                    "min_tokens": 200,
                    "overlap_tokens": 100,
                    "max_pages": 3,
                    "strategy_label": "semantic",
                }

            config_key = tuple(sorted(config.items()))
            if current_config_key is None:
                current_config_key = config_key
                current_sequence = blocks[:]
            elif config_key == current_config_key:
                current_sequence.extend(blocks)
            else:
                if current_sequence:
                    config_dict = dict(current_config_key)
                    doc_configured_sequences.append((current_sequence, config_dict))
                current_config_key = config_key
                current_sequence = blocks[:]

        if current_sequence:
            config_dict = dict(current_config_key) if current_config_key else {
                "max_tokens": 700,
                "min_tokens": 200,
                "overlap_tokens": 100,
                "max_pages": 3,
                "strategy_label": "fixed",
            }
            doc_configured_sequences.append((current_sequence, config_dict))

        # Pack each configured sequence and accumulate resulting chunks.
        doc_chunk_list: List[Chunk] = []
        for sequence_blocks, cfg in doc_configured_sequences:
            if not sequence_blocks:
                continue
            # ``pack_blocks`` honours min/max tokens and applies overlap rules.
            generated = pack_blocks(
                sequence_blocks,
                tokenizer=tokenizer,
                max_tokens=int(cfg["max_tokens"]),
                min_tokens=int(cfg["min_tokens"]),
                overlap_tokens=int(cfg["overlap_tokens"]),
                max_pages=int(cfg["max_pages"]),
                strategy_label=str(cfg["strategy_label"]),
            )
            for chunk in generated:
                if total_tokens_emitted + chunk.token_len > MAX_TOTAL_TOKENS:
                    logger.warning("Token guardrail reached; stopping chunk generation for doc %s", doc.doc_id)
                    break
                doc_chunk_list.append(chunk)
                total_tokens_emitted += chunk.token_len
            else:
                continue
            break  # guardrail triggered: stop generating further sequences

        chunks.extend(doc_chunk_list)

        if doc_chunk_list:
            avg_tokens = sum(ch.token_len for ch in doc_chunk_list) / len(doc_chunk_list)
        else:
            avg_tokens = 0.0
        chunk_stats[doc.doc_id] = {
            "doc_name": doc_name or doc.doc_id,
            "chunks": len(doc_chunk_list),
            "avg_tokens": round(avg_tokens, 2),
            "strategies": {k: int(v) for k, v in per_strategy_counts.items()},
        }

    return chunks, chunk_stats
