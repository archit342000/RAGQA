"""Table detection and CSV export utilities."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .pdf_io import Line


@dataclass
class TableResult:
    page_index: int
    rows: List[List[str]]
    confidence: float
    digit_ratio: float
    bbox_rows: List[Tuple[float, float, float, float] | None]


DELIMITERS = [",", ";", "|", "\t"]


def _split_line(text: str) -> Tuple[str, List[str]]:
    for delim in DELIMITERS:
        parts = [part.strip() for part in text.split(delim)]
        if len(parts) > 1:
            return delim, parts
    parts = re.split(r"\s{2,}", text.strip())
    if len(parts) > 1:
        return " ", [part.strip() for part in parts]
    return ",", [text.strip()]


def delimiter_confidence(lines: Sequence[Line]) -> Tuple[float, float]:
    if not lines:
        return 0.0, 0.0
    counts: List[int] = []
    digit_total = 0
    char_total = 0
    for line in lines:
        _, parts = _split_line(line.text)
        counts.append(len(parts))
        digit_total += sum(ch.isdigit() for ch in line.text)
        char_total += len(line.text)
    if len(set(counts)) > 1:
        return 0.0, digit_total / max(char_total, 1)
    return float(sum(counts) / (len(counts) * max(counts[0], 1))), digit_total / max(char_total, 1)


def detect_tables_for_page(lines: Sequence[Line], *, page_index: int) -> List[TableResult]:
    clusters: List[List[Line]] = []
    current: List[Line] = []
    for line in lines:
        if any(delim in line.text for delim in DELIMITERS) or re.search(r"\d", line.text):
            current.append(line)
        else:
            if len(current) >= 2:
                clusters.append(current)
            current = []
    if len(current) >= 2:
        clusters.append(current)

    results: List[TableResult] = []
    for cluster in clusters:
        conf, digit_ratio = delimiter_confidence(cluster)
        rows: List[List[str]] = []
        bbox_rows: List[Tuple[float, float, float, float] | None] = []
        for line in cluster:
            delim, cells = _split_line(line.text)
            rows.append(cells)
            bbox_rows.append(line.bbox)
        results.append(TableResult(page_index=page_index, rows=rows, confidence=conf, digit_ratio=digit_ratio, bbox_rows=bbox_rows))
    return results


def write_table_csv(result: TableResult, out_dir: Path) -> str:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / f"page_{result.page_index:04d}_table.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_page", "row_idx", "col_idx", "cell_bbox", "text"])
        for row_idx, (cells, bbox) in enumerate(zip(result.rows, result.bbox_rows)):
            width = (bbox[2] - bbox[0]) if bbox else None
            cell_width = (width / max(len(cells), 1)) if width is not None else None
            for col_idx, cell in enumerate(cells):
                if bbox and cell_width is not None:
                    x0 = bbox[0] + col_idx * cell_width
                    x1 = x0 + cell_width
                    cell_bbox = [round(x0, 2), round(bbox[1], 2), round(x1, 2), round(bbox[3], 2)]
                else:
                    cell_bbox = None
                writer.writerow([
                    result.page_index + 1,
                    row_idx,
                    col_idx,
                    json.dumps(cell_bbox) if cell_bbox is not None else "null",
                    cell,
                ])
    return str(path)

