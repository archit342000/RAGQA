"""Lightweight table detection and CSV emission."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .pdf_io import Line

DELIMITERS = [",", "\t", ";", "|"]


@dataclass
class TableArtifact:
    page_index: int
    lines: List[Line]
    confidence: float
    delimiter: str
    csv_path: Path | None = None
    rows: int = 0
    cols: int = 0


def _line_confidence(text: str) -> Tuple[float, str, int]:
    best_conf = 0.0
    best_delim = ""
    best_cols = 0
    if not text.strip():
        return best_conf, best_delim, best_cols
    for delim in DELIMITERS:
        parts = [part.strip() for part in text.split(delim)]
        cols = len(parts)
        if cols <= 1:
            continue
        digits = sum(ch.isdigit() for ch in text)
        conf = digits / max(len(text), 1)
        if conf > best_conf:
            best_conf = conf
            best_delim = delim
            best_cols = cols
    return best_conf, best_delim, best_cols


def detect_tables(
    pages: Sequence[Sequence[Line]],
    out_dir: Path,
    *,
    confidence_threshold: float,
) -> Tuple[List[TableArtifact], int]:
    """Detect delimiter-regular tables and emit CSV artifacts."""

    artifacts: List[TableArtifact] = []
    skipped = 0
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for page_index, lines in enumerate(pages):
        current_group: List[Line] = []
        current_delim = ";"
        confidences: List[float] = []
        table_counter = 0
        for line in lines + [Line(page_index, -1, "", None, None, None)]:  # sentinel
            conf, delim, cols = _line_confidence(line.text)
            if conf >= confidence_threshold and cols >= 2:
                if current_group and delim != current_delim:
                    # flush current group
                    artifacts.append(
                        _emit_table(
                            page_index,
                            current_group,
                            current_delim,
                            confidences,
                            tables_dir,
                            table_counter,
                        )
                    )
                    table_counter += 1
                    current_group = []
                    confidences = []
                current_group.append(line)
                current_delim = delim or current_delim
                confidences.append(conf)
            else:
                if current_group:
                    artifact = _emit_table(
                        page_index,
                        current_group,
                        current_delim,
                        confidences,
                        tables_dir,
                        table_counter,
                    )
                    if artifact.confidence >= confidence_threshold:
                        artifacts.append(artifact)
                        table_counter += 1
                    else:
                        skipped += 1
                    current_group = []
                    confidences = []
        # handle trailing group already processed by sentinel
    return artifacts, skipped


def _emit_table(
    page_index: int,
    lines: List[Line],
    delim: str,
    confidences: List[float],
    tables_dir: Path,
    table_index: int,
) -> TableArtifact:
    mean_conf = sum(confidences) / max(len(confidences), 1)
    artifact = TableArtifact(page_index=page_index, lines=list(lines), confidence=mean_conf, delimiter=delim)
    rows = len(lines)
    cols = 0
    path = tables_dir / f"page_{page_index + 1:04d}_table{table_index:02d}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_page", "row_idx", "col_idx", "cell_bbox", "text"],
        )
        writer.writeheader()
        for row_idx, line in enumerate(lines):
            cells = [cell.strip() for cell in line.text.split(delim)]
            cols = max(cols, len(cells))
            bbox = line.bbox
            cell_width = 0.0
            if bbox and len(cells) > 0:
                x0, y0, x1, y1 = bbox
                cell_width = (x1 - x0) / len(cells) if len(cells) else 0.0
            for col_idx, cell in enumerate(cells):
                if bbox and cell_width:
                    cell_bbox = [
                        round(bbox[0] + col_idx * cell_width, 3),
                        round(bbox[1], 3),
                        round(bbox[0] + (col_idx + 1) * cell_width, 3),
                        round(bbox[3], 3),
                    ]
                else:
                    cell_bbox = None
                writer.writerow(
                    {
                        "source_page": page_index + 1,
                        "row_idx": row_idx,
                        "col_idx": col_idx,
                        "cell_bbox": json.dumps(cell_bbox) if cell_bbox else None,
                        "text": cell,
                    }
                )
    artifact.csv_path = path
    artifact.rows = rows
    artifact.cols = cols
    return artifact
