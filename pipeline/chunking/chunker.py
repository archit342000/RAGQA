"""Chunking pipeline operating on section-first serialized units."""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from pipeline.audit.order_guards import run_order_guards
from pipeline.layout.lp_fuser import FusedBlock, FusedDocument
from pipeline.layout.signals import PageLayoutSignals
from pipeline.repair.repair_pass import EmbeddingFn
from pipeline.serialize.serializer import SerializedUnit, Serializer

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
class UnitSegment:
    unit: SerializedUnit
    tokens: List[int] | List[str]
    token_count: int


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


def _compose_paragraphs(segments: Sequence[UnitSegment]) -> str:
    if not segments:
        return ""
    paragraphs: List[List[str]] = []
    current: List[str] = []
    for segment in segments:
        text = segment.unit.text.strip()
        if not text:
            continue
        if segment.unit.is_paragraph_start and current:
            paragraphs.append(current)
            current = []
        current.append(text)
    if current:
        paragraphs.append(current)
    return "\n".join(" ".join(parts).strip() for parts in paragraphs if parts)


def _has_anchor(segments: Sequence[UnitSegment]) -> bool:
    for segment in segments:
        if segment.unit.block.metadata.get("has_anchor_refs"):
            return True
    return False


class DocumentChunker:
    def __init__(
        self,
        *,
        tokenizer_name: Optional[str] = None,
        embedder: Optional[EmbeddingFn] = None,
        serializer: Optional[Serializer] = None,
    ) -> None:
        self.tokenizer = _TokenizerWrapper(tokenizer_name)
        self.embedder = embedder
        self.serializer = serializer or Serializer()
        self._chunk_counter = 0

    def _next_chunk_id(self, doc_id: str) -> str:
        self._chunk_counter += 1
        return f"{doc_id}_chunk_{self._chunk_counter:04d}"

    def _make_chunk(self, doc_id: str, segments: Sequence[UnitSegment]) -> Optional[Chunk]:
        text = _compose_paragraphs(segments)
        if not text.strip():
            return None
        token_count = self.tokenizer.count(text)
        pages = [segment.unit.page_number for segment in segments]
        char_start = min(segment.unit.char_start for segment in segments)
        char_end = max(segment.unit.char_end for segment in segments)
        block_ids: List[str] = []
        block_types: List[str] = []
        region_counts: Dict[str, int] = {}
        para_ids: List[str] = []
        for segment in segments:
            block = segment.unit.block
            if block.block_id not in block_ids:
                block_ids.append(block.block_id)
                block_types.append(block.block_type)
            region_counts[block.region_label] = region_counts.get(block.region_label, 0) + 1
            if segment.unit.paragraph_id not in para_ids:
                para_ids.append(segment.unit.paragraph_id)
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
            "has_anchor_refs": _has_anchor(segments),
            "aux_attached": [],
            "section_id": segments[0].unit.section_id,
            "section_seq": segments[0].unit.section_seq,
            "para_ids": para_ids,
        }
        chunk_id = self._next_chunk_id(doc_id)
        return Chunk(chunk_id=chunk_id, text=text, tokens=token_count, metadata=metadata)

    def _overlap_tail(self, segments: Sequence[UnitSegment], segment_type: str) -> Tuple[List[UnitSegment], int]:
        if not segments:
            return [], 0
        overlap_tokens = 40 if segment_type == "procedure" else 80
        tokens = 0
        overlap: List[UnitSegment] = []
        for segment in reversed(segments):
            overlap.insert(0, segment)
            tokens += segment.token_count
            if tokens >= overlap_tokens:
                break
        return overlap, tokens

    def _wrap_aux_text(self, text: str) -> str:
        """Wrap auxiliary chunk text in <aux> tags if not already wrapped."""

        trimmed = text.strip()
        if not trimmed:
            return "<aux></aux>"
        if trimmed.startswith("<aux>") and trimmed.endswith("</aux>"):
            return trimmed
        if "\n" in trimmed:
            return f"<aux>\n{trimmed}\n</aux>"
        return f"<aux>{trimmed}</aux>"

    def _table_chunks(self, block: FusedBlock, doc_id: str, *, section_seq: Optional[int]) -> List[Chunk]:
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
            inner_text = "\n".join(parts).strip()
            if not inner_text:
                row_index += size
                continue
            text = self._wrap_aux_text(inner_text)
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
                "section_seq": section_seq,
                "para_ids": [],
                "aux_kind": block.aux_category or "table",
                "owner_section_id": block.metadata.get("owner_section_id"),
                "owner_section_seq": section_seq,
                "linked_figure_id": block.metadata.get("linked_figure_id")
                or block.metadata.get("caption_target_bbox"),
                "references": block.metadata.get("references", []),
            }
            chunk_id = self._next_chunk_id(doc_id)
            chunks.append(Chunk(chunk_id=chunk_id, text=text, tokens=token_count, metadata=metadata))
            row_index += size
        return chunks

    def _build_aux_chunk(
        self,
        doc_id: str,
        block: FusedBlock,
        *,
        section_seq: Optional[int],
    ) -> Optional[Chunk]:
        text = block.text.strip()
        if not text:
            return None
        text = self._wrap_aux_text(text)
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
            "aux_attached": [],
            "aux_kind": block.aux_category or block.region_label,
            "owner_section_id": block.metadata.get("owner_section_id"),
            "owner_section_seq": section_seq,
            "linked_figure_id": block.metadata.get("linked_figure_id")
            or block.metadata.get("caption_target_bbox"),
            "references": references,
            "section_id": block.metadata.get("owner_section_id") or block.metadata.get("section_id"),
            "section_seq": section_seq,
            "para_ids": [],
        }
        chunk_id = self._next_chunk_id(doc_id)
        return Chunk(chunk_id=chunk_id, text=text, tokens=token_count, metadata=metadata)

    def _main_chunks(self, doc_id: str, units: Sequence[SerializedUnit]) -> List[Chunk]:
        chunks: List[Chunk] = []
        buffer: List[UnitSegment] = []
        buffer_tokens = 0
        current_type: Optional[str] = None
        for unit in units:
            text = unit.text.strip()
            if not text:
                continue
            tokens = self.tokenizer.encode(text)
            segment = UnitSegment(unit=unit, tokens=tokens, token_count=len(tokens))
            if current_type and unit.segment_type != current_type and buffer:
                chunk = self._make_chunk(doc_id, buffer)
                if chunk:
                    chunks.append(chunk)
                buffer = []
                buffer_tokens = 0
                current_type = None
            current_type = current_type or unit.segment_type
            buffer.append(segment)
            buffer_tokens += segment.token_count
            target = 300 if current_type == "procedure" else 500
            upper = 350 if current_type == "procedure" else 700
            if buffer_tokens >= target or buffer_tokens >= upper:
                chunk = self._make_chunk(doc_id, buffer)
                if chunk:
                    chunks.append(chunk)
                buffer, buffer_tokens = self._overlap_tail(buffer, current_type)
                if not buffer:
                    current_type = None
        if buffer:
            chunk = self._make_chunk(doc_id, buffer)
            if chunk:
                chunks.append(chunk)
        return chunks

    def _aux_chunks(self, doc_id: str, units: Sequence[SerializedUnit]) -> List[Chunk]:
        if not units:
            return []
        grouped: "OrderedDict[int, List[SerializedUnit]]" = OrderedDict()
        for unit in units:
            owner_seq = unit.owner_section_seq if unit.owner_section_seq is not None else unit.section_seq
            if owner_seq not in grouped:
                grouped[owner_seq] = []
            grouped[owner_seq].append(unit)
        chunks: List[Chunk] = []
        for section_seq, section_units in grouped.items():
            for unit in section_units:
                block = unit.block
                if block.aux_category == "table" or block.region_label == "table":
                    chunks.extend(self._table_chunks(block, doc_id, section_seq=section_seq))
                else:
                    aux_chunk = self._build_aux_chunk(doc_id, block, section_seq=section_seq)
                    if aux_chunk:
                        chunks.append(aux_chunk)
        return chunks

    def chunk_document(
        self,
        document: FusedDocument,
        signals: Sequence[PageLayoutSignals],
    ) -> List[Chunk]:
        if len(document.pages) != len(signals):
            raise ValueError("Signals must align with document pages")
        serialization = self.serializer.serialize(document)
        run_order_guards(serialization.units)
        main_units = [unit for unit in serialization.units if unit.emit_phase == 0]
        aux_units = [unit for unit in serialization.units if unit.emit_phase == 1]
        chunks: List[Chunk] = []
        chunks.extend(self._main_chunks(document.doc_id, main_units))
        chunks.extend(self._aux_chunks(document.doc_id, aux_units))
        return chunks


__all__ = ["Chunk", "DocumentChunker"]
