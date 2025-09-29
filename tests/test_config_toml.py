from __future__ import annotations

import json
from pathlib import Path

from pdf_ingest.config import Config


def test_toml_overrides(tmp_path: Path) -> None:
    toml_path = tmp_path / "config.toml"
    toml_path.write_text("[ocr_policy]\npromote_none_glyphs = 123\n")
    config = Config.from_sources(toml_path=str(toml_path))
    assert config.ocr_policy.promote_none_glyphs == 123


def test_precedence_cli_overrides(tmp_path: Path) -> None:
    toml_path = tmp_path / "config.toml"
    toml_path.write_text("[work_caps]\nmax_rois_per_page = 1\n")
    json_path = tmp_path / "config.json"
    with json_path.open("w") as handle:
        json.dump({"glyph_min_for_text_page": 50}, handle)
    overrides = {"glyph_min_for_text_page": 75}
    config = Config.from_sources(
        json_path=str(json_path),
        toml_path=str(toml_path),
        overrides=overrides,
    )
    assert config.glyph_min_for_text_page == 75
    assert config.work_caps.max_rois_per_page == 1
