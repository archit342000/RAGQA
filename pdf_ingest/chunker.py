"""Paragraph assembly and chunk emission."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from statistics import median

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Config
from .pdf_io import Line


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def junk_ratio(text: str) -> float:
    if not text:
        return 0.0
    junk = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or ch in "-_,.;:()/%"))
    return junk / max(len(text), 1)


def is_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(":"):
        return True
    if stripped.isupper() and len(stripped.split()) <= 10:
        return True
    return False


def is_list_item(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] in {"-", "*", "•"}:
        return True
    prefix = stripped[:2]
    if prefix.isdigit() or (len(prefix) == 2 and prefix[0].isdigit() and prefix[1] == "."):
        return True
    return stripped[:2].lower() in {"a)", "b)", "c)"}


@dataclass
class Paragraph:
    text: str
    page_index: int
    line_start: int
    line_end: int
    tokens: int
    heading: bool = False
    list_item: bool = False

    def to_state(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "page_index": self.page_index,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "tokens": self.tokens,
            "heading": self.heading,
            "list_item": self.list_item,
        }

    @classmethod
    def from_state(cls, payload: Dict[str, object]) -> "Paragraph":
        return cls(
            text=str(payload["text"]),
            page_index=int(payload["page_index"]),
            line_start=int(payload["line_start"]),
            line_end=int(payload["line_end"]),
            tokens=int(payload["tokens"]),
            heading=bool(payload.get("heading", False)),
            list_item=bool(payload.get("list_item", False)),
        )


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_spans: List[List[int | List[int]]]
    tokens: int
    type: str = "body"
    table_csv: Optional[str] = None
    evidence_offsets: List[int] = field(default_factory=list)
    neighbors: Dict[str, Optional[str]] = field(default_factory=dict)


class ChunkBuilder:
    def __init__(self, config: Config, *, next_index: int = 0, last_chunk_id: Optional[str] = None, pending: Optional[List[Dict[str, object]]] = None) -> None:
        self.config = config
        self.next_index = next_index
        self.last_chunk_id = last_chunk_id
        self.pending: List[Paragraph] = [Paragraph.from_state(p) for p in pending or []]
        self.vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix: Optional[csr_matrix] = None
        self._tfidf_dirty = True

    def _new_chunk_id(self) -> str:
        chunk_id = f"{self.next_index:06d}"
        self.next_index += 1
        return chunk_id

    def _invalidate_vectors(self) -> None:
        self._tfidf_dirty = True
        self._tfidf_matrix = None
        if not self.pending:
            self.vectorizer = None

    def _build_vectors(self) -> Optional[csr_matrix]:
        if not self.pending:
            self._tfidf_matrix = None
            return None

        if not self._tfidf_dirty and self._tfidf_matrix is not None and self._tfidf_matrix.shape[0] == len(self.pending):
            return self._tfidf_matrix

        texts = [p.text for p in self.pending]
        try:
            self.vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, norm="l2")
            matrix = self.vectorizer.fit_transform(texts)
            self._tfidf_matrix = matrix.tocsr()
            self._tfidf_dirty = False
            return self._tfidf_matrix
        except ValueError:
            self._tfidf_matrix = None
            return None

    def _boundary_indices(self) -> List[int]:
        matrix = self._build_vectors()
        if matrix is None or matrix.shape[0] < 2:
            return []
        matrix = matrix.tocsr()
        sims: List[float] = []
        for row_index in range(matrix.shape[0] - 1):
            sim = matrix[row_index].multiply(matrix[row_index + 1]).sum()
            sims.append(float(sim))
        deltas = [1.0 - sim for sim in sims]
        if not deltas:
            return []
        med = median(deltas)
        mad = median([abs(delta - med) for delta in deltas])
        threshold = med + 0.5 * mad
        return [idx + 1 for idx, delta in enumerate(deltas) if delta >= threshold]

    def add_paragraph(self, paragraph: Paragraph) -> List[Chunk]:
        self.pending.append(paragraph)
        self._tfidf_dirty = True
        return self._emit_ready_chunks()

    def _emit_ready_chunks(self) -> List[Chunk]:
        emitted: List[Chunk] = []
        boundaries = set(self._boundary_indices())
        cursor = 0
        while cursor < len(self.pending):
            tokens = 0
            end = cursor
            while end < len(self.pending) and tokens < self.config.chunk_tokens_min:
                tokens += self.pending[end].tokens
                end += 1
            if end < len(self.pending):
                tokens_candidate = sum(p.tokens for p in self.pending[cursor:end])
                if tokens_candidate > self.config.chunk_tokens_max and cursor + 1 < len(self.pending):
                    end = cursor + 1
            if end >= len(self.pending):
                break
            if end not in boundaries and tokens < self.config.chunk_tokens_max:
                break
            emitted.append(self._make_chunk(self.pending[cursor:end]))
            cursor = self._overlap_cursor(cursor, end)
        self.pending = self.pending[cursor:]
        self._invalidate_vectors()
        return emitted

    def _make_chunk(self, paragraphs: Sequence[Paragraph]) -> Chunk:
        text = "\n".join(p.text for p in paragraphs)
        chunk_id = self._new_chunk_id()
        spans: List[List[int | List[int]]] = []
        for para in paragraphs:
            spans.append([para.page_index + 1, [para.line_start + 1, para.line_end + 1]])
        chunk = Chunk(
            chunk_id=chunk_id,
            text=text,
            page_spans=spans,
            tokens=estimate_tokens(text),
            neighbors={"prev": self.last_chunk_id, "next": None},
        )
        if self.last_chunk_id is not None:
            pass
        self.last_chunk_id = chunk_id
        return chunk

    def _overlap_cursor(self, start: int, end: int) -> int:
        overlap_ratio = (self.config.overlap_min + self.config.overlap_max) / 2
        tokens = 0
        cursor = end
        while cursor > start and tokens < overlap_ratio * self.config.chunk_tokens_min:
            cursor -= 1
            tokens += self.pending[cursor].tokens
        return cursor

    def finalize(self) -> List[Chunk]:
        emitted: List[Chunk] = []
        if self.pending:
            emitted.append(self._make_chunk(self.pending))
            self.pending = []
        self._invalidate_vectors()
        return emitted

    def state_payload(self) -> Dict[str, object]:
        return {
            "next_chunk_index": self.next_index,
            "last_chunk_id": self.last_chunk_id,
            "pending_paragraphs": [p.to_state() for p in self.pending],
        }


def paragraphs_from_lines(lines: Sequence[Line], *, config: Config) -> Tuple[List[Paragraph], int]:
    paragraphs: List[Paragraph] = []
    buffer: List[Line] = []
    for line in lines:
        if not line.text.strip():
            if buffer:
                paragraphs.append(_build_paragraph(buffer, config))
                buffer = []
            continue
        buffer.append(line)
    if buffer:
        paragraphs.append(_build_paragraph(buffer, config))
    kept: List[Paragraph] = []
    dropped = 0
    for paragraph in paragraphs:
        if _keep_paragraph(paragraph, config):
            kept.append(paragraph)
        else:
            dropped += 1
    return kept, dropped


def _build_paragraph(lines: Sequence[Line], config: Config) -> Paragraph:
    text = " ".join(line.text.strip() for line in lines)
    heading = any(is_heading(line.text) for line in lines)
    list_item = any(is_list_item(line.text) for line in lines)
    tokens = estimate_tokens(text)
    return Paragraph(
        text=text,
        page_index=lines[0].page_index,
        line_start=lines[0].line_index,
        line_end=lines[-1].line_index,
        tokens=tokens,
        heading=heading,
        list_item=list_item,
    )


def _keep_paragraph(paragraph: Paragraph, config: Config) -> bool:
    ratio = junk_ratio(paragraph.text)
    if ratio <= config.junk_char_ratio:
        return True
    return paragraph.heading or paragraph.list_item

