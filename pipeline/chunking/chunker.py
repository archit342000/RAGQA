"""Chunking pipeline operating on fused layout documents."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from pipeline.layout.lp_fuser import FusedBlock, FusedDocument, FusedPage
from pipeline.layout.signals import PageLayoutSignals
from pipeline.repair.repair_pass import EmbeddingFn

try:  # Optional fast tokeniser.
    import tiktoken
except Exception:  # pragma: no cover - dependency optional
    tiktoken = None  # type: ignore


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    tokens: int
    metadata: Dict[str, object]


@dataclass(slots=True)
class BlockSegment:
    block: FusedBlock
    page_number: int
    text: str
    tokens: List[int] | List[str]
    token_count: int
    char_start_offset: int
    char_end_offset: int
    segment_type: str  # "prose" or "procedure"


class _TokenizerWrapper:
    def __init__(self, name: Optional[str] = None) -> None:
        self._encoder = None
        self._uses_int = False
        if tiktoken is not None:
            try:
                if name:
                    self._encoder = tiktoken.get_encoding(name)
                else:
                    self._encoder = tiktoken.get_encoding("cl100k_base")
                self._uses_int = True
            except Exception:  # pragma: no cover - fallback path
                self._encoder = None
                self._uses_int = False

    def encode(self, text: str) -> List[int] | List[str]:
        if self._encoder is not None:
            return list(self._encoder.encode(text))
        return text.split()

    def decode(self, tokens: List[int] | List[str]) -> str:
        if self._encoder is not None:
            return self._encoder.decode(tokens)  # type: ignore[arg-type]
        return " ".join(tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))


def _split_sentences(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    matches = list(re.finditer(r"[^.!?]+[.!?]?", text, re.MULTILINE))
    segments: List[Tuple[str, Tuple[int, int]]] = []
    for match in matches:
        span_text = match.group().strip()
        if not span_text:
            continue
        start, end = match.span()
        segments.append((span_text, (start, end)))
    if not segments:
        return [(text.strip(), (0, len(text)))] if text.strip() else []
    return segments


def _is_procedure_block(block: FusedBlock) -> bool:
    lines = [line.strip() for line in block.text.splitlines() if line.strip()]
    if not lines:
        return False
    procedure_lines = sum(1 for line in lines if re.match(r"^(\d+\.|[A-Z]\.|[-*•])\s", line))
    return procedure_lines >= max(2, len(lines) // 2)


def _segment_block(
    block: FusedBlock,
    page_number: int,
    tokenizer: _TokenizerWrapper,
    embedder: Optional[EmbeddingFn],
) -> List[BlockSegment]:
    segment_type = "procedure" if _is_procedure_block(block) else "prose"
    tokens = tokenizer.encode(block.text)
    token_count = len(tokens)
    if segment_type == "prose" and token_count > 600 and embedder is not None:
        sentences = _split_sentences(block.text)
        if len(sentences) > 1:
            embeddings = embedder([sentence for sentence, _ in sentences])
            if len(embeddings) == len(sentences):
                splits: List[int] = []
                accum_tokens = 0
                current_tokens: List[int] | List[str] = []
                for idx in range(1, len(sentences)):
                    prev_vec = embeddings[idx - 1]
                    curr_vec = embeddings[idx]
                    cos = 0.0
                    dot = sum(a * b for a, b in zip(prev_vec, curr_vec))
                    norm_prev = math.sqrt(sum(a * a for a in prev_vec))
                    norm_curr = math.sqrt(sum(a * a for a in curr_vec))
                    if norm_prev > 0 and norm_curr > 0:
                        cos = dot / (norm_prev * norm_curr)
                    delta = 1.0 - max(0.0, min(1.0, cos))
                    sentence_tokens = tokenizer.encode(sentences[idx - 1][0])
                    accum_tokens += len(sentence_tokens)
                    if delta > 0.15 and accum_tokens >= 160:
                        splits.append(idx)
                        accum_tokens = 0
                if splits:
                    segments: List[BlockSegment] = []
                    char_cursor = 0
                    for start_idx, end_idx in zip([0] + splits, splits + [len(sentences)]):
                        segment_text = " ".join(sentence for sentence, _ in sentences[start_idx:end_idx]).strip()
                        if not segment_text:
                            continue
                        start_char = sentences[start_idx][1][0]
                        end_char = sentences[end_idx - 1][1][1]
                        segment_tokens = tokenizer.encode(segment_text)
                        segments.append(
                            BlockSegment(
                                block=block,
                                page_number=page_number,
                                text=segment_text,
                                tokens=segment_tokens,
                                token_count=len(segment_tokens),
                                char_start_offset=start_char,
                                char_end_offset=end_char,
                                segment_type=segment_type,
                            )
                        )
                    if segments:
                        return segments
    return [
        BlockSegment(
            block=block,
            page_number=page_number,
            text=block.text,
            tokens=tokens,
            token_count=token_count,
            char_start_offset=0,
            char_end_offset=len(block.text),
            segment_type=segment_type,
        )
    ]


def _flatten_table_row(header_cols: List[str], row: str) -> str:
    if not header_cols:
        return row.strip()
    splitter = re.compile(r"\s{2,}|\t|\|")
    row_cols = [col.strip() for col in splitter.split(row) if col.strip()]
    if not row_cols:
        return row.strip()
    padded = row_cols + [""] * (len(header_cols) - len(row_cols))
    pairs = [f"{col}: {val}".strip() for col, val in zip(header_cols, padded)]
    return "; ".join(pair for pair in pairs if pair)


class DocumentChunker:
    def __init__(
        self,
        *,
        tokenizer_name: Optional[str] = None,
        embedder: Optional[EmbeddingFn] = None,
    ) -> None:
        self.tokenizer = _TokenizerWrapper(tokenizer_name)
        self.embedder = embedder
        self._chunk_counter = 0

    def _next_chunk_id(self, doc_id: str) -> str:
        self._chunk_counter += 1
        return f"{doc_id}_chunk_{self._chunk_counter:04d}"

    def _build_prose_segments(
        self,
        page: FusedPage,
    ) -> List[BlockSegment]:
        segments: List[BlockSegment] = []
        for block in page.main_flow:
            segments.extend(
                _segment_block(
                    block,
                    page_number=page.page_number,
                    tokenizer=self.tokenizer,
                    embedder=self.embedder,
                )
            )
        return segments

    def _table_chunks(self, block: FusedBlock, page: FusedPage, doc_id: str) -> List[Chunk]:
        lines = [line.strip() for line in block.text.splitlines() if line.strip()]
        if not lines:
            return []
        header = lines[0]
        rows = lines[1:] if len(lines) > 1 else []
        splitter = re.compile(r"\s{2,}|\t|\|")
        header_cols = [col.strip() for col in splitter.split(header) if col.strip()]
        num_rows = len(rows)
        if num_rows == 0:
            shard_sizes = [0]
        else:
            num_chunks = max(1, round(num_rows / 10))
            chunk_size = max(6, min(12, math.ceil(num_rows / num_chunks)))
            while chunk_size > 12:
                num_chunks += 1
                chunk_size = max(6, min(12, math.ceil(num_rows / num_chunks)))
            shard_sizes = [chunk_size] * (num_rows // chunk_size)
            remainder = num_rows % chunk_size
            if remainder:
                shard_sizes.append(remainder)
        chunks: List[Chunk] = []
        row_index = 0
        for size in shard_sizes:
            if size == 0 and num_rows == 0:
                rows_slice: List[str] = []
            else:
                rows_slice = rows[row_index : row_index + size]
            flattened_rows = [_flatten_table_row(header_cols, row) for row in rows_slice]
            start_row = row_index + 1 if rows_slice else 0
            end_row = row_index + len(rows_slice)
            parts = [f"Table rows {start_row}-{end_row}" if rows_slice else "Table (no rows)"]
            if header_cols:
                parts.append("Header: " + "; ".join(header_cols))
            for flat in flattened_rows:
                parts.append(flat)
            text = "\n".join(parts).strip()
            token_count = self.tokenizer.count(text)
            metadata = {
                "doc_id": doc_id,
                "page_start": block.page_number,
                "page_end": block.page_number,
                "char_start": block.char_start,
                "char_end": block.char_end,
                "block_ids": [block.block_id],
                "block_types": ["aux"],
                "region_type_counts": {block.region_label: 1},
                "table_row_range": (start_row, end_row) if rows_slice else None,
                "has_anchor_refs": bool(block.metadata.get("has_anchor_refs")),
                "aux_attached": [],
                "section_id": block.metadata.get("owner_section_id") or block.metadata.get("section_id"),
                "para_ids": [],
                "aux_kind": block.aux_category or "table",
                "owner_section_id": block.metadata.get("owner_section_id"),
                "linked_figure_id": block.metadata.get("linked_figure_id")
                or block.metadata.get("caption_target_bbox"),
                "references": block.metadata.get("references", []),
            }
            chunk_id = self._next_chunk_id(doc_id)
            chunks.append(Chunk(chunk_id=chunk_id, text=text, tokens=token_count, metadata=metadata))
            row_index += size
        return chunks

    def _finalise_chunk(
        self,
        doc_id: str,
        buffer: List[BlockSegment],
        *,
        section_id: str,
        para_ids: List[str],
    ) -> Optional[Chunk]:
        if not buffer:
            return None
        text_parts: List[str] = [segment.text for segment in buffer]
        text = "\n".join(part for part in text_parts if part)
        token_count = self.tokenizer.count(text)
        pages = [segment.page_number for segment in buffer]
        block_ids: List[str] = []
        block_types: List[str] = []
        region_counts: Dict[str, int] = {}
        char_start = min(segment.block.char_start + segment.char_start_offset for segment in buffer)
        char_end = max(segment.block.char_start + segment.char_start_offset + len(segment.text) for segment in buffer)
        has_anchor = any(segment.block.metadata.get("has_anchor_refs") for segment in buffer)
        for segment in buffer:
            if segment.block.block_id not in block_ids:
                block_ids.append(segment.block.block_id)
                block_types.append(segment.block.block_type)
            region_counts[segment.block.region_label] = region_counts.get(segment.block.region_label, 0) + 1
        metadata: Dict[str, object] = {
            "doc_id": doc_id,
            "page_start": min(pages),
            "page_end": max(pages),
            "char_start": char_start,
            "char_end": char_end,
            "block_ids": block_ids,
            "block_types": block_types,
            "region_type_counts": region_counts,
            "table_row_range": None,
            "has_anchor_refs": has_anchor,
            "aux_attached": [],
            "section_id": section_id,
            "para_ids": list(dict.fromkeys(para_ids)),
        }
        chunk_id = self._next_chunk_id(doc_id)
        return Chunk(chunk_id=chunk_id, text=text, tokens=token_count, metadata=metadata)

    def _build_aux_chunk(self, doc_id: str, block: FusedBlock) -> Optional[Chunk]:
        text = block.text.strip()
        if not text:
            return None
        token_count = self.tokenizer.count(text)
        references = block.metadata.get("references")
        if not isinstance(references, list):
            references = []
        metadata: Dict[str, object] = {
            "doc_id": doc_id,
            "page_start": block.page_number,
            "page_end": block.page_number,
            "char_start": block.char_start,
            "char_end": block.char_end,
            "block_ids": [block.block_id],
            "block_types": [block.block_type],
            "region_type_counts": {block.region_label: 1},
            "table_row_range": None,
            "has_anchor_refs": bool(block.metadata.get("has_anchor_refs")),
            "aux_kind": block.aux_category or block.region_label,
            "owner_section_id": block.metadata.get("owner_section_id"),
            "linked_figure_id": block.metadata.get("linked_figure_id")
            or block.metadata.get("caption_target_bbox"),
            "references": references,
            "section_id": block.metadata.get("owner_section_id") or block.metadata.get("section_id"),
            "para_ids": [],
            "aux_attached": [],
        }
        chunk_id = self._next_chunk_id(doc_id)
        return Chunk(chunk_id=chunk_id, text=text, tokens=token_count, metadata=metadata)

    def chunk_document(
        self,
        document: FusedDocument,
        signals: Sequence[PageLayoutSignals],
    ) -> List[Chunk]:
        if len(document.pages) != len(signals):
            raise ValueError("Signals must align with document pages")

        chunks: List[Chunk] = []
        page_segments: Dict[int, List[BlockSegment]] = {
            page.page_number: self._build_prose_segments(page) for page in document.pages
        }
        page_lookup: Dict[int, FusedPage] = {page.page_number: page for page in document.pages}

        section_order: List[str] = []
        aux_by_section: Dict[str, List[FusedBlock]] = {}

        for page in document.pages:
            for block in page.main_flow:
                section_id = str(block.metadata.get("section_id") or "0")
                if section_id not in section_order:
                    section_order.append(section_id)
            for aux in page.auxiliaries:
                owner = str(aux.metadata.get("owner_section_id") or aux.metadata.get("section_id") or "0")
                aux_by_section.setdefault(owner, []).append(aux)

        for section_id in aux_by_section:
            if section_id not in section_order:
                section_order.append(section_id)

        for section_id in section_order:
            buffer: List[BlockSegment] = []
            buffer_tokens = 0
            current_type: Optional[str] = None
            para_ids: List[str] = []
            for page in document.pages:
                segments = page_segments.get(page.page_number, [])
                for segment in segments:
                    seg_section = str(segment.block.metadata.get("section_id") or "0")
                    if seg_section != section_id:
                        continue
                    if not segment.text.strip():
                        continue
                    if current_type and segment.segment_type != current_type and buffer:
                        chunk = self._finalise_chunk(
                            document.doc_id,
                            buffer,
                            section_id=section_id,
                            para_ids=para_ids,
                        )
                        if chunk:
                            chunks.append(chunk)
                        buffer = []
                        buffer_tokens = 0
                        para_ids = []
                    current_type = segment.segment_type
                    buffer.append(segment)
                    buffer_tokens += segment.token_count
                    para_id = str(segment.block.metadata.get("paragraph_id") or segment.block.block_id)
                    if para_id not in para_ids:
                        para_ids.append(para_id)
                    target = 300 if current_type == "procedure" else 500
                    upper = 350 if current_type == "procedure" else 700
                    if buffer_tokens >= target or buffer_tokens >= upper:
                        chunk = self._finalise_chunk(
                            document.doc_id,
                            buffer,
                            section_id=section_id,
                            para_ids=para_ids,
                        )
                        if chunk:
                            chunks.append(chunk)
                        if buffer:
                            overlap = 40 if current_type == "procedure" else 80
                            tail_segment = buffer[-1]
                            if tail_segment.token_count > overlap:
                                tail_tokens = tail_segment.tokens[-overlap:]
                                tail_text = self.tokenizer.decode(tail_tokens)
                                buffer = [
                                    BlockSegment(
                                        block=tail_segment.block,
                                        page_number=tail_segment.page_number,
                                        text=tail_text,
                                        tokens=tail_tokens,
                                        token_count=len(tail_tokens),
                                        char_start_offset=max(
                                            0,
                                            tail_segment.char_end_offset - len(tail_text),
                                        ),
                                        char_end_offset=tail_segment.char_end_offset,
                                        segment_type=current_type,
                                    )
                                ]
                                buffer_tokens = len(tail_tokens)
                                para_ids = [
                                    str(
                                        buffer[0].block.metadata.get("paragraph_id")
                                        or buffer[0].block.block_id
                                    )
                                ]
                            else:
                                buffer = [tail_segment]
                                buffer_tokens = tail_segment.token_count
                                para_ids = [
                                    str(
                                        tail_segment.block.metadata.get("paragraph_id")
                                        or tail_segment.block.block_id
                                    )
                                ]
                        else:
                            buffer_tokens = 0
                            para_ids = []
            if buffer:
                chunk = self._finalise_chunk(
                    document.doc_id,
                    buffer,
                    section_id=section_id,
                    para_ids=para_ids,
                )
                if chunk:
                    chunks.append(chunk)

            for aux in aux_by_section.get(section_id, []):
                if aux.aux_category == "table" or aux.region_label == "table":
                    page = page_lookup.get(aux.page_number)
                    if page:
                        chunks.extend(self._table_chunks(aux, page, document.doc_id))
                else:
                    aux_chunk = self._build_aux_chunk(document.doc_id, aux)
                    if aux_chunk:
                        chunks.append(aux_chunk)
        return chunks


__all__ = ["Chunk", "DocumentChunker"]
