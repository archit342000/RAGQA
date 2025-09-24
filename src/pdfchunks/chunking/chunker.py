"""Chunk packing for MAIN and AUX threads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ..config import ChunkerConfig
from ..threading.threader import ThreadUnit


@dataclass(slots=True)
class Chunk:
    """Represents a retrieval-ready chunk."""

    doc_id: str
    chunk_id: str
    role: str
    section_seq: int
    text: str
    token_count: int
    para_range: Tuple[int, int]
    sent_range: Tuple[int, int]
    subtype: str | None = None
    page_start: int = 0
    page_end: int = 0


class Chunker:
    """Pure packing chunker obeying sentence boundaries and AUX isolation."""

    def __init__(self, config: ChunkerConfig):
        self.config = config

    def chunk(self, units: Sequence[ThreadUnit]) -> List[Chunk]:
        main_units = [unit for unit in units if unit.emit_phase == 0 and unit.role == "MAIN"]
        aux_units = [unit for unit in units if unit.emit_phase == 1]
        doc_id = units[0].doc_id if units else "doc"

        chunks: List[Chunk] = []
        chunks.extend(self._pack_main(doc_id, main_units))
        chunks.extend(self._pack_aux(doc_id, aux_units))
        return chunks

    def _pack_main(self, doc_id: str, units: Sequence[ThreadUnit]) -> List[Chunk]:
        if not units:
            return []
        chunks: List[Chunk] = []
        token_counts = [self._token_len(unit.text) for unit in units]
        start = 0
        chunk_index = 0
        while start < len(units):
            tokens = 0
            end = start
            while end < len(units):
                candidate = tokens + token_counts[end]
                if candidate > self.config.main_max_tokens and end > start:
                    break
                tokens = candidate
                end += 1
                if (
                    tokens >= self.config.main_target_tokens
                    and (end == len(units) or tokens + token_counts[end] > self.config.main_max_tokens)
                ):
                    break
            if end == start:
                end = min(len(units), start + 1)
                tokens = sum(token_counts[start:end])
            selected = units[start:end]
            page_start, page_end = self._page_span(selected)
            chunk_index += 1
            text = " ".join(unit.text for unit in selected)
            chunk = Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}-main-{chunk_index}",
                role="MAIN",
                section_seq=selected[0].section_seq,
                text=text,
                token_count=self._token_len(text),
                para_range=(selected[0].para_seq, selected[-1].para_seq),
                sent_range=(selected[0].sent_seq, selected[-1].sent_seq),
                page_start=page_start,
                page_end=page_end,
            )
            chunks.append(chunk)
            if end >= len(units):
                break
            overlap_target = (
                self.config.main_overlap_tokens
                if chunk.token_count >= self.config.main_target_tokens
                else self.config.main_small_overlap_tokens
            )
            overlap_tokens = 0
            overlap_index = end
            steps = 0
            while overlap_index > start and overlap_tokens < overlap_target:
                overlap_index -= 1
                overlap_tokens += token_counts[overlap_index]
                steps += 1
            if overlap_index <= start or steps >= end - start:
                start = end
            else:
                start = overlap_index
        return chunks

    def _pack_aux(self, doc_id: str, units: Sequence[ThreadUnit]) -> List[Chunk]:
        if not units:
            return []
        chunks: List[Chunk] = []
        current_key: Tuple[int, str | None] | None = None
        buffer: List[ThreadUnit] = []
        group_index: Dict[Tuple[int, str | None], int] = {}

        def flush_buffer() -> None:
            nonlocal buffer, current_key
            if not buffer or current_key is None:
                return
            section_seq, subtype = current_key
            group_index[current_key] = group_index.get(current_key, 0) + 1
            text = "\n".join(unit.text for unit in buffer)
            page_start, page_end = self._page_span(buffer)
            chunk = Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}-aux-{section_seq}-{group_index[current_key]}",
                role="AUX",
                section_seq=section_seq,
                text=text,
                token_count=self._token_len(text),
                para_range=(buffer[0].para_seq, buffer[-1].para_seq),
                sent_range=(buffer[0].sent_seq, buffer[-1].sent_seq),
                subtype=subtype,
                page_start=page_start,
                page_end=page_end,
            )
            chunks.append(chunk)
            buffer = []
            current_key = None

        for unit in units:
            key = (unit.section_seq, unit.subtype)
            if current_key is None:
                current_key = key
            if key != current_key:
                flush_buffer()
                current_key = key
            buffer.append(unit)
        flush_buffer()
        return chunks

    @staticmethod
    def _token_len(text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    @staticmethod
    def _page_span(units: Sequence[ThreadUnit]) -> Tuple[int, int]:
        pages: List[int] = []
        for unit in units:
            for block_id in unit.block_ids:
                try:
                    page_str = block_id.split("-", 1)[0]
                    page = int(page_str) + 1
                except (ValueError, IndexError):
                    continue
                pages.append(page)
        if not pages:
            return 0, 0
        return min(pages), max(pages)

