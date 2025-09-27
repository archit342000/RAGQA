"""Configuration helpers for the deterministic PDF ingestion pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


_DEFAULTS: Dict[str, str] = {
    "glyph_min_for_text_page": "200",
    "table_digit_ratio": ">=0.4",
    "table_score_conf": ">=0.6",
    "chunk_tokens": "350..600",
    "overlap": "0.1..0.15",
    "ocr_retry": "1",
    "rasterizations_per_page": "<=2",
    "junk_char_ratio": ">0.3",
    "dpi_full": "300",
    "dpi_retry": "400",
}


@dataclass
class Config:
    """Runtime configuration for parsing and chunking."""

    glyph_min_for_text_page: int = int(_DEFAULTS["glyph_min_for_text_page"])
    table_digit_ratio: float = float(_DEFAULTS["table_digit_ratio"][2:])
    table_score_conf: float = float(_DEFAULTS["table_score_conf"][2:])
    chunk_tokens_min: int = int(_DEFAULTS["chunk_tokens"].split("..", 1)[0])
    chunk_tokens_max: int = int(_DEFAULTS["chunk_tokens"].split("..", 1)[1])
    overlap_min: float = float(_DEFAULTS["overlap"].split("..", 1)[0])
    overlap_max: float = float(_DEFAULTS["overlap"].split("..", 1)[1])
    ocr_retry: int = int(_DEFAULTS["ocr_retry"])
    rasterizations_per_page: int = int(_DEFAULTS["rasterizations_per_page"].split("<=", 1)[1])
    junk_char_ratio: float = float(_DEFAULTS["junk_char_ratio"][1:])
    dpi_full: int = int(_DEFAULTS["dpi_full"])
    dpi_retry: int = int(_DEFAULTS["dpi_retry"])
    bbox_mode: str = "line"
    mode: str = "fast"
    max_pages: int = 500
    progress_flush_interval: int = 1

    @classmethod
    def from_sources(
        cls,
        *,
        json_path: str | None = None,
        overrides: Dict[str, Any] | None = None,
    ) -> "Config":
        """Create a configuration, merging defaults with optional payloads."""

        base: Dict[str, Any] = asdict(cls())
        payload: Dict[str, Any] = {}
        if json_path:
            data = json.loads(Path(json_path).read_text())
            if not isinstance(data, dict):
                raise ValueError("Config JSON must be an object")
            payload.update(data)
        if overrides:
            payload.update(overrides)
        for key, value in payload.items():
            if key not in base:
                raise KeyError(f"Unknown config key: {key}")
            base[key] = value
        config = cls(**base)  # type: ignore[arg-type]
        config._validate()
        return config

    def _validate(self) -> None:
        if self.chunk_tokens_min <= 0 or self.chunk_tokens_max <= 0:
            raise ValueError("chunk token bounds must be positive")
        if self.chunk_tokens_min >= self.chunk_tokens_max:
            raise ValueError("chunk_tokens_min must be < chunk_tokens_max")
        if not (0 < self.overlap_min <= self.overlap_max < 1):
            raise ValueError("overlap range must be between 0 and 1")
        if self.bbox_mode not in {"line", "band"}:
            raise ValueError("bbox_mode must be 'line' or 'band'")
        if self.mode not in {"fast", "thorough"}:
            raise ValueError("mode must be fast or thorough")
        if self.dpi_full <= 0 or self.dpi_retry <= 0:
            raise ValueError("dpi values must be positive")
        if self.dpi_retry < self.dpi_full:
            raise ValueError("dpi_retry must be >= dpi_full")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

