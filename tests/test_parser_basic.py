"""Unit tests for the CPU-first parser configuration and helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from parser import PDFParser, load_config
from parser.driver import parse_documents
from parser.config import ParserConfig
from parser.types import BBox, LineSpan


def test_load_config_merges_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text('{"fast_budget_s": 15, "list_markers": ["-", "+"]}', encoding="utf-8")
    config = load_config(config_file)
    assert isinstance(config, ParserConfig)
    assert config.fast_budget_s == 15
    assert config.list_markers == ["-", "+"]
    # ensure untouched defaults remain intact
    assert config.glyph_min_for_text_page == 200


def _make_line(idx: int, text: str, bbox: tuple[float, float, float, float]) -> LineSpan:
    return LineSpan(
        page_index=0,
        line_index=idx,
        text=text,
        bbox=BBox(*bbox),
        char_start=idx * 10,
        char_end=idx * 10 + len(text),
    )


def test_multicolumn_repair_orders_by_x(tmp_path: Path) -> None:
    config = ParserConfig()
    parser = PDFParser(config)
    left_top = _make_line(0, "Left column top", (10, 10, 100, 20))
    right_top = _make_line(1, "Right column top", (300, 15, 390, 25))
    left_bottom = _make_line(2, "Left column bottom", (12, 120, 102, 130))
    right_bottom = _make_line(3, "Right column bottom", (305, 125, 395, 135))
    lines = [right_top, left_bottom, right_bottom, left_top]
    reordered = parser._repair_multicolumn(lines)
    ordered_texts = [line.text for line in reordered]
    assert ordered_texts[0].startswith("Left column")
    assert ordered_texts[1].startswith("Right column")


def test_caption_detection_marks_sidecars() -> None:
    config = ParserConfig()
    parser = PDFParser(config)
    lines = [
        _make_line(0, "Figure 1. Example architecture.", (0, 0, 10, 10)),
        _make_line(1, "Main body text", (0, 20, 10, 30)),
    ]
    parser._apply_line_roles(lines)
    captions = parser._extract_captions(lines)
    assert len(captions) == 1
    assert captions[0].text.startswith("Figure 1")
    assert lines[0].is_caption is True
    assert lines[1].is_caption is False


def test_noise_ratio_ignores_clean_lines() -> None:
    config = ParserConfig(junk_char_threshold=0.2)
    parser = PDFParser(config)
    noisy = _make_line(0, "@@@@", (0, 0, 10, 10))
    clean = _make_line(1, "Valid text", (0, 20, 10, 30))
    ratio = parser._compute_noise_ratio([noisy, clean])
    assert ratio == pytest.approx(0.5)


def test_parse_documents_reports_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    docs, report = parse_documents([str(missing)], mode="fast")
    assert docs == []
    assert report.skipped_docs == [str(missing)]
    assert "Failed to parse" in report.message
