"""Configuration helpers for the PDF ingestion pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

DEFAULTS_VERBATIM = """\
glyph_min_for_text_page=200
fast_budget_s=20
thorough_budget_s=40
max_pages=500
table_digit_ratio>=0.4
chunk_token_target=350..600
overlap=0.1..0.15
noise_drop_ratio>0.3
"""


@dataclass
class IngestConfig:
    """Runtime configuration for parsing and chunking."""

    glyph_min_for_text_page: int = 200
    fast_budget_s: float = 20.0
    thorough_budget_s: float = 40.0
    max_pages: int = 500
    table_digit_ratio: float = 0.4
    chunk_target_min: int = 350
    chunk_target_max: int = 600
    overlap_min: float = 0.10
    overlap_max: float = 0.15
    noise_drop_ratio: float = 0.30
    bbox_mode: str = "line"
    persist_line_bboxes: bool = True
    mode: str = "fast"

    @property
    def overlap_avg(self) -> float:
        return (self.overlap_min + self.overlap_max) / 2.0

    @property
    def chunk_target_range(self) -> tuple[int, int]:
        return (self.chunk_target_min, self.chunk_target_max)

    def budget_for_mode(self, mode: str | None = None) -> float:
        selected = (mode or self.mode).lower()
        if selected == "thorough":
            return self.thorough_budget_s
        if selected == "auto":
            return self.thorough_budget_s
        return self.fast_budget_s

    def to_serializable(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "DEFAULTS_VERBATIM": DEFAULTS_VERBATIM.strip().splitlines(),
                "chunk_token_target": f"{self.chunk_target_min}..{self.chunk_target_max}",
                "overlap": f"{self.overlap_min:.2f}..{self.overlap_max:.2f}",
                "noise_drop_ratio": f">{self.noise_drop_ratio:.2f}",
                "table_digit_ratio": f">={self.table_digit_ratio:.2f}",
            }
        )
        return data


def _read_json(path: Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def load_config(config_path: str | None = None) -> IngestConfig:
    """Load an ingest configuration from disk, merging with defaults."""

    data = _read_json(Path(config_path) if config_path else None)
    config = IngestConfig()
    for key, value in data.items():
        if not hasattr(config, key):
            continue
        setattr(config, key, value)
    return config
