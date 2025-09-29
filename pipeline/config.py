from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping


CONFIG_DEFAULTS: Dict[str, Any] = {
    "gpu": {"enable": False, "acquire_max_wait_seconds": 20},
    "ocr": {
        "gate": {
            "char_count_threshold": 200,
            "text_coverage_threshold": 0.05,
        },
        "math_density_threshold": 0.02,
    },
    "raster": {
        "dpi": {"default": 200, "math": 300},
        "max_megapixels": 12,
    },
    "chunk": {
        "tokens": {"target": 1000, "min": 500, "max": 1400},
    },
    "aux": {
        "header_footer": {"repetition_threshold": 0.50},
        "y_band": {"pct": 0.03},
        "segment0": {"min_chars": 150, "font_percentile": 0.80},
        "superscript": {"y_offset_xheight": 0.20},
        "soft_boundary": {"max_deferred_pages": 5},
    },
    "timeouts": {
        "triage": {"seconds": 5},
        "docling": {"seconds": 20},
        "ocr": {"seconds": 12},
        "layout": {"seconds": 8},
        "doc": {"cap_seconds": 240},
    },
    "concurrency": {"pages": 2},
    "parallel": {"workers": 3},
    "cache": {"dir": "/cache/{doc_sha}/{page_no}/{dpi}"},
}


@dataclass
class OCRGateConfig:
    char_count_threshold: int = 200
    text_coverage_threshold: float = 0.05


@dataclass
class OCRConfig:
    gate: OCRGateConfig = field(default_factory=OCRGateConfig)
    math_density_threshold: float = 0.02


@dataclass
class RasterConfig:
    default_dpi: int = 200
    math_dpi: int = 300
    max_megapixels: int = 12


@dataclass
class ChunkTokensConfig:
    target: int = 1000
    minimum: int = 500
    maximum: int = 1400


@dataclass
class ChunkConfig:
    tokens: ChunkTokensConfig = field(default_factory=ChunkTokensConfig)


@dataclass
class AuxHeaderFooterConfig:
    repetition_threshold: float = 0.50


@dataclass
class AuxConfig:
    header_footer: AuxHeaderFooterConfig = field(default_factory=AuxHeaderFooterConfig)
    y_band_pct: float = 0.03
    segment0_min_chars: int = 150
    segment0_font_percentile: float = 0.80
    superscript_y_offset_xheight: float = 0.20
    soft_boundary_max_deferred_pages: int = 5


@dataclass
class TimeoutConfig:
    triage_seconds: int = 5
    docling_seconds: int = 20
    ocr_seconds: int = 12
    layout_seconds: int = 8
    doc_cap_seconds: int = 240


@dataclass
class ParallelConfig:
    workers: int = 3


@dataclass
class CacheConfig:
    dir: str = "/cache/{doc_sha}/{page_no}/{dpi}"


@dataclass
class GPUConfig:
    enable: bool = False
    acquire_max_wait_seconds: int = 20


@dataclass
class ConcurrencyConfig:
    pages: int = 2


@dataclass
class PipelineConfig:
    gpu: GPUConfig = field(default_factory=GPUConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    raster: RasterConfig = field(default_factory=RasterConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    aux: AuxConfig = field(default_factory=AuxConfig)
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "PipelineConfig":
        if not data:
            return cls()

        def _merge(default: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
            merged = dict(default)
            for key, value in override.items():
                if isinstance(value, Mapping) and isinstance(default.get(key), Mapping):
                    merged[key] = _merge(default[key], value)  # type: ignore[index]
                else:
                    merged[key] = value
            return merged

        merged = _merge(CONFIG_DEFAULTS, data)
        return cls(
            gpu=GPUConfig(**merged.get("gpu", {})),
            ocr=OCRConfig(
                gate=OCRGateConfig(**merged.get("ocr", {}).get("gate", {})),
                math_density_threshold=float(merged.get("ocr", {}).get("math_density_threshold", 0.02)),
            ),
            raster=RasterConfig(
                default_dpi=int(merged.get("raster", {}).get("dpi", {}).get("default", 200)),
                math_dpi=int(merged.get("raster", {}).get("dpi", {}).get("math", 300)),
                max_megapixels=int(merged.get("raster", {}).get("max_megapixels", 12)),
            ),
            chunk=ChunkConfig(
                tokens=ChunkTokensConfig(
                    target=int(merged.get("chunk", {}).get("tokens", {}).get("target", 1000)),
                    minimum=int(merged.get("chunk", {}).get("tokens", {}).get("min", 500)),
                    maximum=int(merged.get("chunk", {}).get("tokens", {}).get("max", 1400)),
                )
            ),
            aux=AuxConfig(
                header_footer=AuxHeaderFooterConfig(
                    repetition_threshold=float(
                        merged.get("aux", {})
                        .get("header_footer", {})
                        .get("repetition_threshold", 0.50)
                    )
                ),
                y_band_pct=float(merged.get("aux", {}).get("y_band", {}).get("pct", 0.03)),
                segment0_min_chars=int(
                    merged.get("aux", {}).get("segment0", {}).get("min_chars", 150)
                ),
                segment0_font_percentile=float(
                    merged.get("aux", {}).get("segment0", {}).get("font_percentile", 0.80)
                ),
                superscript_y_offset_xheight=float(
                    merged.get("aux", {})
                    .get("superscript", {})
                    .get("y_offset_xheight", 0.20)
                ),
                soft_boundary_max_deferred_pages=int(
                    merged.get("aux", {})
                    .get("soft_boundary", {})
                    .get("max_deferred_pages", 5)
                ),
            ),
            timeouts=TimeoutConfig(
                triage_seconds=int(merged.get("timeouts", {}).get("triage", {}).get("seconds", 5)),
                docling_seconds=int(merged.get("timeouts", {}).get("docling", {}).get("seconds", 20)),
                ocr_seconds=int(merged.get("timeouts", {}).get("ocr", {}).get("seconds", 12)),
                layout_seconds=int(merged.get("timeouts", {}).get("layout", {}).get("seconds", 8)),
                doc_cap_seconds=int(merged.get("timeouts", {}).get("doc", {}).get("cap_seconds", 240)),
            ),
            concurrency=ConcurrencyConfig(
                pages=int(merged.get("concurrency", {}).get("pages", 2))
            ),
            parallel=ParallelConfig(workers=int(merged.get("parallel", {}).get("workers", 3))),
            cache=CacheConfig(dir=str(merged.get("cache", {}).get("dir", "/cache/{doc_sha}/{page_no}/{dpi}"))),
        )


DEFAULT_CONFIG = PipelineConfig()


def resolve_cache_path(config: PipelineConfig, doc_sha: str, page_no: int, dpi: int) -> Path:
    template = config.cache.dir
    return Path(template.format(doc_sha=doc_sha, page_no=page_no, dpi=dpi)).expanduser().resolve()
