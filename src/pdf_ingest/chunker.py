"""Balanced chunking utilities with caption and footnote sidecars."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import IngestConfig
from .pdf_io import Line


@dataclass
class Paragraph:
    lines: List[Line]
    text: str
    page_spans: List[Tuple[int, Tuple[int, int]]]
    is_heading: bool
    junk_ratio: float

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    type: str
    text: str
    tokens_est: int
    page_spans: List[Tuple[int, Tuple[int, int] | None]]
    section_hints: List[str] = field(default_factory=list)
    neighbors: Dict[str, str | None] = field(default_factory=dict)
    table_csv: str | None = None
    evidence_offsets: List[Tuple[int, int]] = field(default_factory=list)
    provenance: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "type": self.type,
            "text": self.text,
            "tokens_est": self.tokens_est,
            "page_spans": [[page, span] for page, span in self.page_spans],
            "section_hints": list(self.section_hints),
            "neighbors": dict(self.neighbors),
            "table_csv": self.table_csv,
            "evidence_offsets": list(self.evidence_offsets),
            "provenance": dict(self.provenance),
        }


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def junk_ratio(text: str) -> float:
    if not text:
        return 0.0
    junk = sum(1 for ch in text if ch.isascii() and not ch.isalnum() and ch not in " .,;:-_()[]{}@#$%&*/\"'`+")
    return junk / max(len(text), 1)


def build_paragraphs(pages: Sequence[Sequence[Line]]) -> List[Paragraph]:
    paragraphs: List[Paragraph] = []
    for page_lines in pages:
        current: List[Line] = []
        for line in page_lines:
            if _starts_new_paragraph(line, current):
                if current:
                    paragraphs.append(_make_paragraph(current))
                    current = []
            if line.text.strip():
                current.append(line)
        if current:
            paragraphs.append(_make_paragraph(current))
    return paragraphs


def select_boundaries(paragraphs: Sequence[Paragraph]) -> set[int]:
    if len(paragraphs) < 2:
        return set()
    texts = [p.text for p in paragraphs]
    vectorizer = TfidfVectorizer(min_df=1)
    matrix = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(matrix[:-1], matrix[1:])
    drops = 1 - similarities.diagonal()
    median = float(np.median(drops)) if len(drops) else 0.0
    mad = float(np.median(np.abs(drops - median))) if len(drops) else 0.0
    threshold = median + 0.5 * mad
    strong = {idx for idx, drop in enumerate(drops) if drop >= threshold}
    for idx, paragraph in enumerate(paragraphs[:-1]):
        if paragraph.is_heading:
            strong.add(idx)
    return strong


def build_chunks(
    doc_id: str,
    paragraphs: Sequence[Paragraph],
    config: IngestConfig,
    *,
    provenance_hash: str,
) -> Tuple[List[Chunk], float]:
    chunks: List[Chunk] = []
    total_paras = len(paragraphs)
    dropped = 0
    strong_boundaries = select_boundaries(paragraphs)
    target_min, target_max = config.chunk_target_range
    overlap_ratio = config.overlap_avg

    chunk_lines: List[Line] = []
    chunk_text_parts: List[str] = []
    chunk_paragraphs: List[Paragraph] = []
    carryover_lines: List[Line] = []
    carryover_text: str = ""

    def flush_chunk(strong_split: bool = False) -> None:
        nonlocal chunk_lines, chunk_text_parts, chunk_paragraphs, carryover_lines, carryover_text
        if not chunk_text_parts:
            return
        combined_text = " ".join(chunk_text_parts).strip()
        tokens_est = estimate_tokens(combined_text)
        chunk_id = f"{len(chunks):06d}"
        page_spans = _merge_spans(chunk_lines)
        hints = [para.lines[0].text.strip() for para in chunk_paragraphs if para.is_heading][:3]
        chunk = Chunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            type="body",
            text=combined_text,
            tokens_est=tokens_est,
            page_spans=page_spans,
            section_hints=hints,
            neighbors={"prev": None, "next": None},
            provenance={"hash": provenance_hash, "byte_range": None},
        )
        chunks.append(chunk)
        carryover_lines = []
        carryover_text = ""
        if strong_split:
            overlap_line_count = max(1, int(len(chunk_lines) * overlap_ratio))
            carryover_lines = chunk_lines[-overlap_line_count:]
            overlap_tokens = max(1, int(tokens_est * overlap_ratio))
            words = combined_text.split()
            carryover_text = " ".join(words[-overlap_tokens:])
        chunk_lines = []
        chunk_text_parts = []
        chunk_paragraphs = []

    for idx, paragraph in enumerate(paragraphs):
        if paragraph.junk_ratio > config.noise_drop_ratio and not paragraph.is_heading:
            dropped += 1
            continue
        if not chunk_text_parts and carryover_text:
            chunk_text_parts.append(carryover_text)
            chunk_lines.extend(carryover_lines)
            chunk_paragraphs.extend([])
            carryover_lines = []
            carryover_text = ""
        chunk_lines.extend(paragraph.lines)
        chunk_text_parts.append(paragraph.text)
        chunk_paragraphs.append(paragraph)

        combined_text = " ".join(chunk_text_parts).strip()
        current_tokens = estimate_tokens(combined_text)
        strong_break = idx in strong_boundaries
        last_para = idx == len(paragraphs) - 1
        if current_tokens >= target_max or (strong_break and current_tokens >= target_min) or (last_para and chunk_text_parts):
            flush_chunk(strong_split=strong_break)

    flush_chunk()
    noise_ratio = dropped / max(total_paras, 1)
    _link_neighbors(chunks)
    return chunks, noise_ratio


def _merge_spans(lines: Sequence[Line]) -> List[Tuple[int, Tuple[int, int] | None]]:
    spans: Dict[int, Tuple[int, int]] = {}
    for line in lines:
        start, end = spans.get(line.page_index, (None, None))
        start_idx = line.line_index + 1
        if start is None or start_idx < start:
            start = start_idx
        end_idx = line.line_index + 1
        if end is None or end_idx > end:
            end = end_idx
        spans[line.page_index] = (start, end)
    return [(page + 1, spans[page]) for page in sorted(spans)]


def _starts_new_paragraph(line: Line, current: Sequence[Line]) -> bool:
    if not current:
        return False
    if not current[-1].text.strip():
        return True
    if _is_heading(line.text):
        return True
    if line.text.strip().startswith(("- ", "•", "* ", "●")):
        return True
    return False


def _is_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(":"):
        return True
    if stripped.isupper() and len(stripped) > 3:
        return True
    return False


def _make_paragraph(lines: Sequence[Line]) -> Paragraph:
    text = " ".join(line.text.strip() for line in lines if line.text.strip())
    spans = _merge_spans(lines)
    return Paragraph(
        lines=list(lines),
        text=text,
        page_spans=spans,
        is_heading=_is_heading(lines[0].text),
        junk_ratio=junk_ratio(text),
    )


def _link_neighbors(chunks: Sequence[Chunk]) -> None:
    for idx, chunk in enumerate(chunks):
        prev_id = chunks[idx - 1].chunk_id if idx > 0 else None
        next_id = chunks[idx + 1].chunk_id if idx < len(chunks) - 1 else None
        chunk.neighbors = {"prev": prev_id, "next": next_id}
