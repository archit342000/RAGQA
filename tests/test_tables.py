"""Tests for table detection heuristics."""
from __future__ import annotations

from pathlib import Path

from parser.table_extractor import detect_table_candidates, export_table
from parser.types import BBox, LineSpan


def _line(text: str, idx: int) -> LineSpan:
    return LineSpan(
        page_index=0,
        line_index=idx,
        text=text,
        bbox=BBox(0, idx * 10, 50, idx * 10 + 5),
        char_start=idx * 10,
        char_end=idx * 10 + len(text),
    )


def test_table_confidence_exports(tmp_path: Path) -> None:
    lines = [_line("A,B,C", 0), _line("1,2,3", 1)]
    candidates = detect_table_candidates(lines)
    assert len(candidates) == 1
    table = export_table(candidates[0], tmp_path, confidence_threshold=0.1)
    assert table.csv_path is not None
    csv_file = Path(table.csv_path)
    assert csv_file.exists()


def test_table_low_confidence_skips(tmp_path: Path) -> None:
    lines = [_line("No table here", 0)]
    candidates = detect_table_candidates(lines)
    assert candidates == []
    # artificially create a candidate but with low delimiter score
    bad_candidate = detect_table_candidates([_line("1 2 3", 0), _line("4 5 6", 1)])
    if bad_candidate:
        table = export_table(bad_candidate[0], tmp_path, confidence_threshold=10.0)
        assert table.csv_path is None
        assert table.skipped_reason == "low_confidence"
