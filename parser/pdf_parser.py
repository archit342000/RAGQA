"""Main PDF parsing pipeline implementation."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import fitz
import numpy as np
from sklearn.cluster import KMeans

from .config import ParserConfig
from .logging_utils import log_event
from .ocr import OCRTimeout, perform_page_ocr
from .table_extractor import detect_table_candidates, export_table
from .types import BBox, CaptionSidecar, LineSpan, PageParseResult, ParsedDocument
from .utils.text import (
    detect_caption,
    is_heading,
    is_list_item,
    junk_ratio,
    merge_whitespace,
)


@dataclass
class _PageContext:
    page: fitz.Page
    index: int


class PDFParser:
    """CPU-first PDF parsing pipeline with lightweight heuristics."""

    def __init__(self, config: ParserConfig) -> None:
        self.config = config

    def parse(
        self,
        pdf_path: str | Path,
        doc_id: Optional[str] = None,
        mode: str = "fast",
        tables_out: Optional[Path] = None,
    ) -> ParsedDocument:
        pdf_path = str(pdf_path)
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_hash = self._hash_file(path_obj)
        log_event("parse_started", path=pdf_path, mode=mode)
        start = time.time()
        pdf_doc = fitz.open(pdf_path)
        try:
            max_pages = min(len(pdf_doc), self.config.max_pages)
            budget_s = self._select_budget(mode, max_pages)
            pages: List[PageParseResult] = []
            ocr_budget_remaining = budget_s
            for page_idx in range(max_pages):
                now = time.time()
                elapsed = now - start
                if elapsed >= budget_s:
                    log_event(
                        "budget_exhausted",
                        at_page=page_idx,
                        elapsed=elapsed,
                        budget=budget_s,
                    )
                    break

                ctx = _PageContext(page=pdf_doc.load_page(page_idx), index=page_idx)
                page_result = self._parse_page(
                    ctx,
                    pdf_path,
                    start,
                    budget_s,
                    tables_out,
                )
                pages.append(page_result)

                if page_result.ocr_performed:
                    ocr_budget_remaining -= (time.time() - now)
                if ocr_budget_remaining <= 0:
                    log_event(
                        "ocr_budget_exhausted",
                        at_page=page_idx,
                        elapsed=time.time() - start,
                        budget=budget_s,
                    )
                    break

            parse_time = time.time() - start
        finally:
            pdf_doc.close()

        if doc_id is None:
            doc_id = path_obj.stem

        parsed = ParsedDocument(
            doc_id=doc_id,
            file_path=pdf_path,
            pages=pages,
            config_used=self.config.to_dict(),
            parse_time_s=parse_time,
            content_hash=doc_hash,
        )
        log_event("parse_completed", doc_id=doc_id, pages=len(pages), duration=parse_time)
        return parsed

    def _select_budget(self, mode: str, page_count: int) -> float:
        if mode == "thorough":
            return self.config.thorough_budget
        if mode == "auto" and page_count <= self.config.time_budget_mode_threshold:
            return self.config.thorough_budget
        return self.config.fast_budget

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _parse_page(
        self,
        ctx: _PageContext,
        pdf_path: str,
        start_time: float,
        budget_s: float,
        tables_out: Optional[Path],
    ) -> PageParseResult:
        page = ctx.page
        page_dict = page.get_text("rawdict")
        glyph_count = sum(
            len(span.get("text") or "")
            for block in page_dict.get("blocks", [])
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        )
        has_text = glyph_count >= self.config.glyph_min_for_text_page

        lines: List[LineSpan] = []
        char_cursor = 0
        for block in page_dict.get("blocks", []):
            for line_idx, line in enumerate(block.get("lines", [])):
                spans = line.get("spans", [])
                text = merge_whitespace("".join(span.get("text", "") for span in spans))
                if not text:
                    continue
                bbox_vals = line.get("bbox")
                bbox = BBox(*bbox_vals) if bbox_vals else None
                span = LineSpan(
                    page_index=ctx.index,
                    line_index=len(lines),
                    text=text,
                    bbox=bbox,
                    char_start=char_cursor,
                    char_end=char_cursor + len(text),
                )
                char_cursor += len(text) + 1
                lines.append(span)

        ocr_performed = False
        if not has_text:
            try:
                ocr_text = perform_page_ocr(
                    pdf_path=pdf_path,
                    page_index=ctx.index,
                    time_budget_s=budget_s,
                    start_time=start_time,
                )
            except OCRTimeout:
                ocr_text = None
            if ocr_text:
                ocr_lines = self._text_to_lines(ocr_text, ctx.index, char_cursor)
                lines.extend(ocr_lines)
                glyph_count = sum(len(line.text) for line in lines)
                has_text = True
                ocr_performed = True

        lines = self._repair_multicolumn(lines)
        self._apply_line_roles(lines)
        captions = self._extract_captions(lines)
        noise = self._compute_noise_ratio(lines)

        tables: List[TableExtraction] = []
        if tables_out:
            table_candidates = detect_table_candidates(lines)
            for candidate in table_candidates:
                tables.append(export_table(candidate, tables_out, self.config.table_digit_ratio))

        return PageParseResult(
            page_index=ctx.index,
            glyph_count=glyph_count,
            had_text=has_text,
            ocr_performed=ocr_performed,
            lines=lines,
            captions=captions,
            tables=tables,
            noise_ratio=noise,
        )

    def _text_to_lines(self, text: str, page_index: int, start_cursor: int) -> List[LineSpan]:
        lines: List[LineSpan] = []
        for offset, raw_line in enumerate(text.splitlines()):
            clean = merge_whitespace(raw_line)
            if not clean:
                continue
            span = LineSpan(
                page_index=page_index,
                line_index=offset,
                text=clean,
                bbox=None,
                char_start=start_cursor,
                char_end=start_cursor + len(clean),
            )
            start_cursor += len(clean) + 1
            lines.append(span)
        return lines

    def _repair_multicolumn(self, lines: List[LineSpan]) -> List[LineSpan]:
        with_bbox = [line for line in lines if line.bbox is not None]
        if len(with_bbox) < 4:
            return lines

        x_centers = np.array(
            [
                [(line.bbox.x0 + line.bbox.x1) / 2.0]
                for line in with_bbox
            ]
        )
        best_order: Optional[List[int]] = None
        for n_clusters in range(1, min(self.config.kmeans_max_clusters, len(with_bbox)) + 1):
            kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
            labels = kmeans.fit_predict(x_centers)
            cluster_order = np.argsort(kmeans.cluster_centers_.ravel())
            label_rank = {label: int(rank) for rank, label in enumerate(cluster_order)}
            order = sorted(
                range(len(with_bbox)),
                key=lambda i: (
                    label_rank[int(labels[i])],
                    with_bbox[i].bbox.y0 if with_bbox[i].bbox else 0.0,
                    i,
                ),
            )
            indices = [lines.index(with_bbox[idx]) for idx in order]
            if best_order is None or self._order_score(indices) > self._order_score(best_order):
                best_order = indices

        if best_order is None:
            return lines

        ordered_lines = [lines[idx] for idx in best_order]
        others = [line for line in lines if line not in with_bbox]
        merged = ordered_lines + others
        merged.sort(key=lambda line: (line.page_index, line.bbox.y0 if line.bbox else float("inf")))
        for new_idx, line in enumerate(merged):
            line.line_index = new_idx
        return merged

    @staticmethod
    def _order_score(indices: Sequence[int]) -> float:
        diffs = [j - i for i, j in zip(indices, indices[1:])]
        if not diffs:
            return 0.0
        inversions = sum(1 for diff in diffs if diff < 0)
        return -float(inversions)

    def _extract_captions(self, lines: List[LineSpan]) -> List[CaptionSidecar]:
        captions: List[CaptionSidecar] = []
        for line in lines:
            if detect_caption(line.text, self.config.caption_pattern):
                line.is_caption = True
                captions.append(
                    CaptionSidecar(
                        page_index=line.page_index,
                        anchor_line=line.line_index,
                        text=line.text,
                        bbox=line.bbox,
                    )
                )
            elif line.text.lower().startswith("footnote"):
                line.is_footnote = True
        return captions

    def _apply_line_roles(self, lines: List[LineSpan]) -> None:
        for line in lines:
            if line.is_caption:
                continue
            if is_heading(line.text, self.config.heading_max_line_length):
                line.is_heading = True
            if is_list_item(line.text, self.config.list_markers):
                line.is_list_item = True

    def _compute_noise_ratio(self, lines: List[LineSpan]) -> float:
        if not lines:
            return 0.0
        noisy = sum(1 for line in lines if junk_ratio(line.text) > self.config.junk_char_threshold)
        return noisy / len(lines)
