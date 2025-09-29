"""Configuration helpers for the deterministic PDF ingestion pipeline."""

from __future__ import annotations

import configparser
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    import tomli as tomllib  # type: ignore[import]


def _coerce(value: Any) -> Any:
    """Best-effort conversion of string literals into native Python types."""

    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"true", "false"}:
            return text.lower() == "true"
        try:
            if "." in text:
                return float(text)
            return int(text)
        except ValueError:
            return text
    return value


def _merge_dict(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _merge_dict(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


@dataclass
class OcrPolicyConfig:
    promote_none_glyphs: int = 800
    promote_none_glyphs_with_quality: int = 400
    unicode_quality_threshold: float = 0.90
    full_threshold_s_native: float = 0.25
    probe_selectable_bands: int = 2

    def validate(self) -> None:
        if self.promote_none_glyphs < 0 or self.promote_none_glyphs_with_quality < 0:
            raise ValueError("OCR promotion thresholds must be non-negative")
        if not (0.0 <= self.unicode_quality_threshold <= 1.0):
            raise ValueError("unicode_quality_threshold must be between 0 and 1")
        if self.full_threshold_s_native < 0:
            raise ValueError("full_threshold_s_native must be non-negative")
        if self.probe_selectable_bands < 1:
            raise ValueError("probe_selectable_bands must be >= 1")


@dataclass
class WorkCapsConfig:
    max_page_megapixels: float = 10.0
    max_rois_per_page: int = 3
    max_doc_ocr_surface_multiplier: float = 1.0
    dpi: int = 300
    dpi_retry_small_roi: int = 400
    retry_small_roi_height_frac: float = 0.25

    def validate(self) -> None:
        if self.max_page_megapixels <= 0:
            raise ValueError("max_page_megapixels must be positive")
        if self.max_rois_per_page <= 0:
            raise ValueError("max_rois_per_page must be positive")
        if self.max_doc_ocr_surface_multiplier <= 0:
            raise ValueError("max_doc_ocr_surface_multiplier must be positive")
        if self.dpi <= 0 or self.dpi_retry_small_roi <= 0:
            raise ValueError("dpi values must be positive")
        if self.dpi_retry_small_roi < self.dpi:
            raise ValueError("dpi_retry_small_roi must be >= dpi")
        if not (0 < self.retry_small_roi_height_frac <= 1):
            raise ValueError("retry_small_roi_height_frac must be within (0, 1]")


@dataclass
class TesseractConfig:
    oem: int = 1
    psm: int = 6
    lang: str = "eng"
    disable_dawgs: bool = True
    whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:%()-"

    def validate(self) -> None:
        if self.oem not in {0, 1, 2, 3}:
            raise ValueError("tesseract.oem must be between 0 and 3")
        if not (0 <= self.psm <= 13):
            raise ValueError("tesseract.psm must be between 0 and 13")


@dataclass
class ConcurrencyConfig:
    ocr_workers: int = 2

    def validate(self) -> None:
        if self.ocr_workers <= 0:
            raise ValueError("ocr_workers must be positive")


@dataclass
class FailFastConfig:
    roi_min_alnum_chars: int = 20
    roi_max_retries: int = 1
    emit_stub_on_fail: bool = True

    def validate(self) -> None:
        if self.roi_min_alnum_chars < 0:
            raise ValueError("roi_min_alnum_chars must be non-negative")
        if self.roi_max_retries < 0:
            raise ValueError("roi_max_retries must be non-negative")


@dataclass
class Config:
    """Runtime configuration for parsing and chunking."""

    glyph_min_for_text_page: int = 200
    table_digit_ratio: float = 0.4
    table_score_conf: float = 0.6
    chunk_tokens_min: int = 350
    chunk_tokens_max: int = 600
    overlap_min: float = 0.1
    overlap_max: float = 0.15
    ocr_retry: int = 1
    rasterizations_per_page: int = 2
    junk_char_ratio: float = 0.3
    dpi_full: int = 300
    dpi_retry: int = 400
    bbox_mode: str = "line"
    mode: str = "fast"
    max_pages: int = 500
    progress_flush_interval: int = 1
    ocr_policy: OcrPolicyConfig = field(default_factory=OcrPolicyConfig)
    work_caps: WorkCapsConfig = field(default_factory=WorkCapsConfig)
    tesseract: TesseractConfig = field(default_factory=TesseractConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    fail_fast: FailFastConfig = field(default_factory=FailFastConfig)

    @classmethod
    def from_sources(
        cls,
        *,
        json_path: str | None = None,
        toml_path: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> "Config":
        base_dict: Dict[str, Any] = cls().to_dict()
        if json_path:
            payload = json.loads(Path(json_path).read_text())
            if not isinstance(payload, Mapping):
                raise ValueError("Config JSON must be an object")
            _merge_dict(base_dict, payload)
        if toml_path:
            toml_payload = _load_toml_or_ini(Path(toml_path))
            _merge_dict(base_dict, toml_payload)
        if overrides:
            _merge_dict(base_dict, overrides)
        config = cls.from_dict(base_dict)
        config._validate()
        return config

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Config":
        def _section(section: str, factory: type) -> Mapping[str, Any]:
            raw = payload.get(section, {})
            return raw if isinstance(raw, Mapping) else {}

        defaults = cls()
        kwargs: Dict[str, Any] = {}
        for field_name in [
            "glyph_min_for_text_page",
            "table_digit_ratio",
            "table_score_conf",
            "chunk_tokens_min",
            "chunk_tokens_max",
            "overlap_min",
            "overlap_max",
            "ocr_retry",
            "rasterizations_per_page",
            "junk_char_ratio",
            "dpi_full",
            "dpi_retry",
            "bbox_mode",
            "mode",
            "max_pages",
            "progress_flush_interval",
        ]:
            if field_name in payload:
                kwargs[field_name] = _coerce(payload[field_name])
        config = cls(**kwargs)  # type: ignore[arg-type]
        policy_payload = {k: _coerce(payload.get("ocr_policy", {}).get(k, getattr(defaults.ocr_policy, k))) for k in asdict(defaults.ocr_policy).keys()}
        work_payload = {k: _coerce(payload.get("work_caps", {}).get(k, getattr(defaults.work_caps, k))) for k in asdict(defaults.work_caps).keys()}
        tess_payload = {k: _coerce(payload.get("tesseract", {}).get(k, getattr(defaults.tesseract, k))) for k in asdict(defaults.tesseract).keys()}
        concurrency_payload = {k: _coerce(payload.get("concurrency", {}).get(k, getattr(defaults.concurrency, k))) for k in asdict(defaults.concurrency).keys()}
        fail_fast_payload = {k: _coerce(payload.get("fail_fast", {}).get(k, getattr(defaults.fail_fast, k))) for k in asdict(defaults.fail_fast).keys()}
        config.ocr_policy = OcrPolicyConfig(**policy_payload)
        config.work_caps = WorkCapsConfig(**work_payload)
        config.tesseract = TesseractConfig(**tess_payload)
        config.concurrency = ConcurrencyConfig(**concurrency_payload)
        config.fail_fast = FailFastConfig(**fail_fast_payload)
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
        self.ocr_policy.validate()
        self.work_caps.validate()
        self.tesseract.validate()
        self.concurrency.validate()
        self.fail_fast.validate()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))


def _load_toml_or_ini(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".toml" and tomllib is not None:
        return json.loads(json.dumps(tomllib.loads(path.read_text())))
    parser = configparser.ConfigParser()
    parser.read(path)
    result: Dict[str, Any] = {}
    for section in parser.sections():
        section_payload: Dict[str, Any] = {}
        for key, value in parser.items(section):
            section_payload[key] = _coerce(value)
        result[section] = section_payload
    return result
