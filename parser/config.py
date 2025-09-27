"""Configuration utilities for the CPU-first parser pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict
import json

CONFIG_DEFAULTS = {
    "glyph_min_for_text_page": 200,
    "fast_budget_s": 20,
    "thorough_budget_s": 40,
    "max_pages": 500,
    "table_digit_ratio": 0.4,
    "chunk_token_target_min": 350,
    "chunk_token_target_max": 600,
    "overlap_ratio_min": 0.10,
    "overlap_ratio_max": 0.15,
    "noise_drop_ratio": 0.30,
    "bbox_mode": "line",
    "kmeans_max_clusters": 2,
    "junk_char_threshold": 0.30,
    "caption_anchor_window": 3,
    "caption_pattern": r"^(Fig(ure)?\.?|Table)\b",
    "heading_max_line_length": 90,
    "list_markers": ["-", "*", "•", "▪", "◦"],
    "time_budget_mode_threshold": 50,
}


@dataclass
class ParserConfig:
    """Runtime configuration for parsing and chunking."""

    glyph_min_for_text_page: int = CONFIG_DEFAULTS["glyph_min_for_text_page"]
    fast_budget_s: float = CONFIG_DEFAULTS["fast_budget_s"]
    thorough_budget_s: float = CONFIG_DEFAULTS["thorough_budget_s"]
    max_pages: int = CONFIG_DEFAULTS["max_pages"]
    table_digit_ratio: float = CONFIG_DEFAULTS["table_digit_ratio"]
    chunk_token_target_min: int = CONFIG_DEFAULTS["chunk_token_target_min"]
    chunk_token_target_max: int = CONFIG_DEFAULTS["chunk_token_target_max"]
    overlap_ratio_min: float = CONFIG_DEFAULTS["overlap_ratio_min"]
    overlap_ratio_max: float = CONFIG_DEFAULTS["overlap_ratio_max"]
    noise_drop_ratio: float = CONFIG_DEFAULTS["noise_drop_ratio"]
    bbox_mode: str = CONFIG_DEFAULTS["bbox_mode"]
    kmeans_max_clusters: int = CONFIG_DEFAULTS["kmeans_max_clusters"]
    junk_char_threshold: float = CONFIG_DEFAULTS["junk_char_threshold"]
    caption_anchor_window: int = CONFIG_DEFAULTS["caption_anchor_window"]
    caption_pattern: str = CONFIG_DEFAULTS["caption_pattern"]
    heading_max_line_length: int = CONFIG_DEFAULTS["heading_max_line_length"]
    list_markers: list[str] = field(default_factory=lambda: CONFIG_DEFAULTS["list_markers"].copy())
    time_budget_mode_threshold: int = CONFIG_DEFAULTS["time_budget_mode_threshold"]

    @property
    def fast_budget(self) -> float:
        return max(1.0, float(self.fast_budget_s))

    @property
    def thorough_budget(self) -> float:
        return max(self.fast_budget, float(self.thorough_budget_s))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = ParserConfig()


def _normalize_config(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in CONFIG_DEFAULTS:
            continue
        default = CONFIG_DEFAULTS[key]
        if isinstance(default, (int, float)):
            try:
                normalized[key] = type(default)(value)
            except (TypeError, ValueError):
                continue
        elif isinstance(default, list):
            if isinstance(value, list):
                normalized[key] = value
        elif isinstance(default, str):
            if isinstance(value, str):
                normalized[key] = value
    return normalized


def load_config(path: str | Path | None, overrides: Dict[str, Any] | None = None) -> ParserConfig:
    """Load configuration merging defaults with optional file and overrides."""

    config_data: Dict[str, Any] = {}
    if path:
        config_path = Path(path)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                try:
                    loaded = json.load(handle)
                    if isinstance(loaded, dict):
                        config_data.update(_normalize_config(loaded))
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON config at {config_path}") from None
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    if overrides:
        config_data.update(_normalize_config(overrides))

    merged = DEFAULT_CONFIG.to_dict()
    merged.update(config_data)
    return ParserConfig(**merged)
