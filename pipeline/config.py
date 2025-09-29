from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping


CONFIG_DEFAULTS: Mapping[str, Any] = {
    "gpu": {"enable": False},
    "ocr": {
        "gate": {
            "char_count_threshold": 200,
            "text_coverage_threshold": 0.05,
        },
        "math_density_threshold": 0.02,
    },
    "raster": {
        "dpi": {"default": 240, "math": 300},
    },
    "chunk": {
        "tokens": {"target": 1000, "min": 600, "max": 1400},
    },
    "timeouts": {"page_soft_seconds": 10},
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
    default_dpi: int = 240
    math_dpi: int = 300


@dataclass
class ChunkTokensConfig:
    target: int = 1000
    minimum: int = 600
    maximum: int = 1400


@dataclass
class ChunkConfig:
    tokens: ChunkTokensConfig = field(default_factory=ChunkTokensConfig)


@dataclass
class TimeoutConfig:
    page_soft_seconds: int = 10


@dataclass
class ParallelConfig:
    workers: int = 3


@dataclass
class CacheConfig:
    dir: str = "/cache/{doc_sha}/{page_no}/{dpi}"


@dataclass
class GPUConfig:
    enable: bool = False


@dataclass
class PipelineConfig:
    gpu: GPUConfig = field(default_factory=GPUConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    raster: RasterConfig = field(default_factory=RasterConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
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
                default_dpi=int(merged.get("raster", {}).get("dpi", {}).get("default", 240)),
                math_dpi=int(merged.get("raster", {}).get("dpi", {}).get("math", 300)),
            ),
            chunk=ChunkConfig(
                tokens=ChunkTokensConfig(
                    target=int(merged.get("chunk", {}).get("tokens", {}).get("target", 1000)),
                    minimum=int(merged.get("chunk", {}).get("tokens", {}).get("min", 600)),
                    maximum=int(merged.get("chunk", {}).get("tokens", {}).get("max", 1400)),
                )
            ),
            timeouts=TimeoutConfig(page_soft_seconds=int(merged.get("timeouts", {}).get("page_soft_seconds", 10))),
            parallel=ParallelConfig(workers=int(merged.get("parallel", {}).get("workers", 3))),
            cache=CacheConfig(dir=str(merged.get("cache", {}).get("dir", "/cache/{doc_sha}/{page_no}/{dpi}"))),
        )


DEFAULT_CONFIG = PipelineConfig()


def resolve_cache_path(config: PipelineConfig, doc_sha: str, page_no: int, dpi: int) -> Path:
    template = config.cache.dir
    return Path(
        template.format(doc_sha=doc_sha, page_no=page_no, dpi=dpi)
    ).expanduser().resolve()
