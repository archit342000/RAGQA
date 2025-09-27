"""Lightweight table detection and CSV emission."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .logging_utils import log_event
from .types import LineSpan, TableExtraction


@dataclass
class TableCandidate:
    page_index: int
    lines: List[LineSpan]


def _delimiter_score(text_lines: Iterable[str]) -> float:
    delimiter_counts = []
    delimiters = [",", ";", "\t", "|"]
    for line in text_lines:
        counts = [line.count(delim) for delim in delimiters]
        if not any(counts):
            continue
        delimiter_counts.append(max(counts))
    if not delimiter_counts:
        return 0.0
    avg = sum(delimiter_counts) / len(delimiter_counts)
    variance = sum((count - avg) ** 2 for count in delimiter_counts) / len(delimiter_counts)
    return avg / (1 + variance)


def detect_table_candidates(lines: List[LineSpan]) -> List[TableCandidate]:
    """Group contiguous lines with regular delimiters as table candidates."""

    candidates: List[TableCandidate] = []
    current: List[LineSpan] = []
    for line in lines:
        if line.text.count(",") >= 2 or "\t" in line.text or "|" in line.text:
            current.append(line)
        else:
            if current:
                candidates.append(TableCandidate(page_index=current[0].page_index, lines=current))
                current = []
    if current:
        candidates.append(TableCandidate(page_index=current[0].page_index, lines=current))
    return candidates


def export_table(candidate: TableCandidate, out_dir: Path, confidence_threshold: float) -> TableExtraction:
    confidence = _delimiter_score(line.text for line in candidate.lines)
    csv_path: str | None
    if confidence < confidence_threshold:
        csv_path = None
        log_event(
            "table_skipped",
            page=candidate.page_index,
            reason="low_confidence",
            confidence=confidence,
        )
        return TableExtraction(
            page_index=candidate.page_index,
            csv_path=None,
            rows=len(candidate.lines),
            cols=0,
            confidence=confidence,
            skipped_reason="low_confidence",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / f"page_{candidate.page_index:04d}_table.csv"
    rows = 0
    cols = 0
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_page", "row_idx", "col_idx", "cell_bbox", "text"])
        for row_idx, line in enumerate(candidate.lines):
            cells = [cell.strip() for cell in line.text.replace("\t", ",").split(",")]
            cols = max(cols, len(cells))
            for col_idx, cell in enumerate(cells):
                bbox_str = ",".join(str(v) for v in line.bbox.as_list()) if line.bbox else ""
                writer.writerow([
                    line.page_index,
                    row_idx,
                    col_idx,
                    bbox_str,
                    cell,
                ])
                rows += 1
    log_event(
        "table_emitted",
        page=candidate.page_index,
        csv=str(csv_file),
        rows=rows,
        cols=cols,
        confidence=confidence,
    )
    return TableExtraction(
        page_index=candidate.page_index,
        csv_path=str(csv_file),
        rows=rows,
        cols=cols,
        confidence=confidence,
    )
