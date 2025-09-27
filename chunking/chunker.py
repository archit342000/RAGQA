"""Balanced chunking strategy using TF-IDF topic boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from parser.config import ParserConfig
from parser.types import CaptionSidecar, LineSpan, ParsedDocument
from parser.utils.text import estimate_tokens, junk_ratio, merge_whitespace


@dataclass
class Paragraph:
    lines: List[LineSpan]
    text: str
    tokens: int
    page_spans: List[Dict[str, object]]
    evidence_offsets: List[List[int]]
    heading_hint: Optional[str]


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: str
    type: str
    text: str
    tokens_est: int
    page_spans: List[Dict[str, object]]
    section_hints: List[str]
    neighbors: Dict[str, Optional[str]]
    table_csv: Optional[str]
    evidence_offsets: List[List[int]]
    provenance: Dict[str, Optional[str]]


class Chunker:
    def __init__(self, config: ParserConfig) -> None:
        self.config = config

    def chunk_document(self, doc: ParsedDocument) -> List[ChunkRecord]:
        paragraphs = self._build_paragraphs(doc)
        body_chunks = self._chunk_body(doc, paragraphs)
        caption_chunks = self._caption_sidecars(doc)
        footnote_chunks = self._footnote_sidecars(doc)
        all_chunks = body_chunks + caption_chunks + footnote_chunks
        self._link_neighbors(all_chunks)
        return all_chunks

    def _build_paragraphs(self, doc: ParsedDocument) -> List[Paragraph]:
        paragraphs: List[Paragraph] = []
        current_lines: List[LineSpan] = []
        for line in doc.all_lines():
            if line.is_caption:
                continue
            text = merge_whitespace(line.text)
            if not text:
                if current_lines:
                    paragraphs.append(self._create_paragraph(current_lines))
                    current_lines = []
                continue
            if line.is_heading and current_lines:
                paragraphs.append(self._create_paragraph(current_lines))
                current_lines = []
            current_lines.append(line)
        if current_lines:
            paragraphs.append(self._create_paragraph(current_lines))
        filtered: List[Paragraph] = []
        for paragraph in paragraphs:
            if not paragraph.text:
                continue
            if (
                junk_ratio(paragraph.text) > self.config.noise_drop_ratio
                and not any(line.is_heading or line.is_list_item for line in paragraph.lines)
            ):
                continue
            filtered.append(paragraph)
        return filtered

    def _create_paragraph(self, lines: Sequence[LineSpan]) -> Paragraph:
        merged_text = merge_whitespace(" ".join(line.cleaned_text() for line in lines))
        tokens = estimate_tokens(merged_text)
        spans: List[Dict[str, object]] = []
        offsets: List[List[int]] = []
        for line in lines:
            span_repr = {
                "page": line.page_index,
                "span": [line.line_index],
                "bbox": line.bbox.as_list() if line.bbox else None,
            }
            spans.append(span_repr)
            offsets.append([line.char_start, line.char_end])
        heading_hint = next((line.text.strip() for line in lines if line.is_heading), None)
        return Paragraph(
            lines=list(lines),
            text=merged_text,
            tokens=tokens,
            page_spans=spans,
            evidence_offsets=offsets,
            heading_hint=heading_hint,
        )

    def _chunk_body(self, doc: ParsedDocument, paragraphs: List[Paragraph]) -> List[ChunkRecord]:
        if not paragraphs:
            return []

        boundaries = self._topic_boundaries(paragraphs)
        chunks: List[ChunkRecord] = []
        current: List[Paragraph] = []
        chunk_index = 0
        for idx, paragraph in enumerate(paragraphs):
            current.append(paragraph)
            tokens = sum(p.tokens for p in current)
            target_max = self.config.chunk_token_target_max
            target_min = self.config.chunk_token_target_min
            should_split = tokens >= target_max or (
                tokens >= target_min and boundaries[idx]
            )
            if should_split or idx == len(paragraphs) - 1:
                chunk_index += 1
                chunks.append(
                    self._paragraphs_to_chunk(
                        doc,
                        current,
                        chunk_index,
                    )
                )
                overlap_ratio = (self.config.overlap_ratio_min + self.config.overlap_ratio_max) / 2.0
                overlap_tokens = int(sum(p.tokens for p in current) * overlap_ratio)
                retained: List[Paragraph] = []
                token_acc = 0
                for paragraph in reversed(current):
                    token_acc += paragraph.tokens
                    if token_acc <= overlap_tokens and boundaries[idx]:
                        retained.insert(0, paragraph)
                    else:
                        break
                current = retained
        return chunks

    def _paragraphs_to_chunk(
        self,
        doc: ParsedDocument,
        paragraphs: Sequence[Paragraph],
        chunk_index: int,
    ) -> ChunkRecord:
        text = "\n".join(paragraph.text for paragraph in paragraphs).strip()
        tokens = sum(paragraph.tokens for paragraph in paragraphs)
        page_spans: List[Dict[str, object]] = []
        evidence: List[List[int]] = []
        section_hints = []
        for paragraph in paragraphs:
            page_spans.extend(paragraph.page_spans)
            evidence.extend(paragraph.evidence_offsets)
            if paragraph.heading_hint:
                section_hints.append(paragraph.heading_hint)
        chunk_id = f"{doc.doc_id}::chunk_{chunk_index:04d}"
        return ChunkRecord(
            doc_id=doc.doc_id,
            chunk_id=chunk_id,
            type="body",
            text=text,
            tokens_est=tokens,
            page_spans=page_spans,
            section_hints=section_hints,
            neighbors={"prev": None, "next": None},
            table_csv=None,
            evidence_offsets=evidence,
            provenance={"byte_range": None, "hash": doc.content_hash},
        )

    def _topic_boundaries(self, paragraphs: List[Paragraph]) -> List[bool]:
        if len(paragraphs) <= 1:
            return [True] * len(paragraphs)
        texts = [paragraph.text for paragraph in paragraphs]
        vectorizer = TfidfVectorizer(min_df=1)
        tfidf = vectorizer.fit_transform(texts)
        window = 3
        deltas: List[float] = []
        for idx in range(len(paragraphs) - 1):
            left_start = max(0, idx - window + 1)
            left_vec = np.asarray(tfidf[left_start : idx + 1].mean(axis=0))
            right_end = min(len(paragraphs), idx + window + 1)
            right_vec = np.asarray(tfidf[idx + 1 : right_end].mean(axis=0))
            if not right_vec.any() or not left_vec.any():
                deltas.append(1.0)
                continue
            cos = cosine_similarity(left_vec, right_vec)[0][0]
            deltas.append(float(1.0 - cos))
        median = float(np.median(deltas))
        mad = float(np.median(np.abs(np.array(deltas) - median)))
        threshold = median + 0.5 * mad
        boundaries = [False] * len(paragraphs)
        for idx, delta in enumerate(deltas):
            if delta >= threshold:
                boundaries[idx] = True
        boundaries[-1] = True
        return boundaries

    def _caption_sidecars(self, doc: ParsedDocument) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        for idx, caption in enumerate(doc.caption_lines()):
            chunk_id = f"{doc.doc_id}::caption_{idx:04d}"
            text = caption.text.strip()
            tokens = estimate_tokens(text)
            page_span = {
                "page": caption.page_index,
                "span": [caption.anchor_line],
                "bbox": caption.bbox.as_list() if caption.bbox else None,
            }
            chunks.append(
                ChunkRecord(
                    doc_id=doc.doc_id,
                    chunk_id=chunk_id,
                    type="caption",
                    text=text,
                    tokens_est=tokens,
                    page_spans=[page_span],
                    section_hints=[],
                    neighbors={"prev": None, "next": None},
                    table_csv=None,
                    evidence_offsets=[[0, len(text)]],
                    provenance={"byte_range": None, "hash": doc.content_hash},
                )
            )
        return chunks

    def _footnote_sidecars(self, doc: ParsedDocument) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        footnote_lines = [line for line in doc.all_lines() if line.is_footnote]
        for idx, line in enumerate(footnote_lines):
            chunk_id = f"{doc.doc_id}::footnote_{idx:04d}"
            text = line.text.strip()
            tokens = estimate_tokens(text)
            page_span = {
                "page": line.page_index,
                "span": [line.line_index],
                "bbox": line.bbox.as_list() if line.bbox else None,
            }
            chunks.append(
                ChunkRecord(
                    doc_id=doc.doc_id,
                    chunk_id=chunk_id,
                    type="footnote",
                    text=text,
                    tokens_est=tokens,
                    page_spans=[page_span],
                    section_hints=[],
                    neighbors={"prev": None, "next": None},
                    table_csv=None,
                    evidence_offsets=[[line.char_start, line.char_end]],
                    provenance={"byte_range": None, "hash": doc.content_hash},
                )
            )
        return chunks

    @staticmethod
    def _link_neighbors(chunks: List[ChunkRecord]) -> None:
        for idx, chunk in enumerate(chunks):
            prev_chunk = chunks[idx - 1] if idx > 0 else None
            next_chunk = chunks[idx + 1] if idx + 1 < len(chunks) else None
            chunk.neighbors["prev"] = prev_chunk.chunk_id if prev_chunk else None
            chunk.neighbors["next"] = next_chunk.chunk_id if next_chunk else None
