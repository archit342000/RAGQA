from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, List


CONFIG_DEFAULTS: Dict[str, Any] = {
    "gpu": {"enable": False, "acquire_max_wait_seconds": 20},
    "ocr": {
        "gate": {
            "char_count_threshold": 200,
            "text_coverage_threshold": 0.05,
        },
        "math_density_threshold": 0.02,
        "force": {
            "dpi": {"default": 200, "math": 300},
        },
        "psm": [6, 4],
        "oem": 1,
    },
    "extractor": {
        "vote": {"char_threshold": 150},
    },
    "raster": {
        "dpi": {"default": 200, "math": 300},
        "max_megapixels": 12,
    },
    "chunk": {
        "tokens": {"target": 1000, "min": 500, "max": 1400},
        "degraded": {"target": 1000, "min": 900, "max": 1100},
    },
    "flow": {
        "limits": {
            "target": 1600,
            "soft": 2000,
            "hard": 2400,
            "min": 900,
        },
        "boundary_slack_tokens": 200,
    },
    "gate5": {
        "header_footer": {"y_band_pct": 0.07, "repetition_threshold": 0.40},
        "caption_zone": {"lineheight_multiplier": 1.5},
        "sidebar": {"min_column_width_fraction": 0.60},
    },
    "paragraph_only": {"min_blocks_across_pages": 1, "window_pages": 2},
    "diagnostics": {"enable": True, "overlay_max_pages": 2},
    "aux": {
        "header_footer": {
            "repetition_threshold": 0.40,
            "dropcap_max_fraction": 0.30,
            "y_band_pct": 0.07,
        },
        "segment0": {"min_chars": 150, "font_percentile": 0.80},
        "superscript": {"y_offset_xheight": 0.20},
        "soft_boundary": {"max_deferred_pages": 5},
        "callout": {"column_width_fraction_max": 0.60},
        "font_band": {"small_quantile": 0.20},
    },
    "segments": {"soft_boundary_pages": 5},
    "anchor": {"lookahead_pages": 1},
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
    force_default_dpi: int = 200
    force_math_dpi: int = 300
    psm: List[int] = field(default_factory=lambda: [6, 4])
    oem: int = 1


@dataclass
class ExtractorVoteConfig:
    char_threshold: int = 150


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
    degraded_target: int = 1000
    degraded_minimum: int = 900
    degraded_maximum: int = 1100


@dataclass
class FlowLimitsConfig:
    target: int = 1600
    soft: int = 2000
    hard: int = 2400
    minimum: int = 900


@dataclass
class FlowConfig:
    limits: FlowLimitsConfig = field(default_factory=FlowLimitsConfig)
    boundary_slack_tokens: int = 200


@dataclass
class Gate5HeaderFooterConfig:
    y_band_pct: float = 0.07
    repetition_threshold: float = 0.40


@dataclass
class Gate5SidebarConfig:
    min_column_width_fraction: float = 0.60


@dataclass
class Gate5CaptionConfig:
    lineheight_multiplier: float = 1.5


@dataclass
class Gate5Config:
    header_footer: Gate5HeaderFooterConfig = field(default_factory=Gate5HeaderFooterConfig)
    sidebar: Gate5SidebarConfig = field(default_factory=Gate5SidebarConfig)
    caption_zone: Gate5CaptionConfig = field(default_factory=Gate5CaptionConfig)


@dataclass
class ParagraphOnlyConfig:
    min_blocks_across_pages: int = 1
    window_pages: int = 2


@dataclass
class DiagnosticsConfig:
    enable: bool = True
    overlay_max_pages: int = 2


@dataclass
class AuxHeaderFooterConfig:
    repetition_threshold: float = 0.40
    dropcap_max_fraction: float = 0.30
    y_band_pct: float = 0.07


@dataclass
class AuxConfig:
    header_footer: AuxHeaderFooterConfig = field(default_factory=AuxHeaderFooterConfig)
    y_band_pct: float = 0.07
    segment0_min_chars: int = 150
    segment0_font_percentile: float = 0.80
    superscript_y_offset_xheight: float = 0.20
    soft_boundary_max_deferred_pages: int = 5
    callout_column_width_fraction_max: float = 0.60
    font_band_small_quantile: float = 0.20


@dataclass
class SegmentConfig:
    soft_boundary_pages: int = 5


@dataclass
class AnchorConfig:
    lookahead_pages: int = 1


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
    extractor: ExtractorVoteConfig = field(default_factory=ExtractorVoteConfig)
    raster: RasterConfig = field(default_factory=RasterConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    aux: AuxConfig = field(default_factory=AuxConfig)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    gate5: Gate5Config = field(default_factory=Gate5Config)
    paragraph_only: ParagraphOnlyConfig = field(default_factory=ParagraphOnlyConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

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
                force_default_dpi=int(
                    merged.get("ocr", {})
                    .get("force", {})
                    .get("dpi", {})
                    .get("default", 200)
                ),
                force_math_dpi=int(
                    merged.get("ocr", {})
                    .get("force", {})
                    .get("dpi", {})
                    .get("math", 300)
                ),
                psm=list(merged.get("ocr", {}).get("psm", [6, 4])),
                oem=int(merged.get("ocr", {}).get("oem", 1)),
            ),
            extractor=ExtractorVoteConfig(
                char_threshold=int(
                    merged.get("extractor", {})
                    .get("vote", {})
                    .get("char_threshold", 150)
                )
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
                ),
                degraded_target=int(
                    merged.get("chunk", {}).get("degraded", {}).get("target", 1000)
                ),
                degraded_minimum=int(
                    merged.get("chunk", {}).get("degraded", {}).get("min", 900)
                ),
                degraded_maximum=int(
                    merged.get("chunk", {}).get("degraded", {}).get("max", 1100)
                ),
            ),
            flow=FlowConfig(
                limits=FlowLimitsConfig(
                    target=int(
                        merged.get("flow", {})
                        .get("limits", {})
                        .get("target", CONFIG_DEFAULTS["flow"]["limits"]["target"])
                    ),
                    soft=int(
                        merged.get("flow", {})
                        .get("limits", {})
                        .get("soft", CONFIG_DEFAULTS["flow"]["limits"]["soft"])
                    ),
                    hard=int(
                        merged.get("flow", {})
                        .get("limits", {})
                        .get("hard", CONFIG_DEFAULTS["flow"]["limits"]["hard"])
                    ),
                    minimum=int(
                        merged.get("flow", {})
                        .get("limits", {})
                        .get("min", CONFIG_DEFAULTS["flow"]["limits"]["min"])
                    ),
                ),
                boundary_slack_tokens=int(
                    merged.get("flow", {})
                    .get(
                        "boundary_slack_tokens",
                        CONFIG_DEFAULTS["flow"]["boundary_slack_tokens"],
                    )
                ),
            ),
            aux=AuxConfig(
                header_footer=AuxHeaderFooterConfig(
                    repetition_threshold=float(
                        merged.get("aux", {})
                        .get("header_footer", {})
                        .get(
                            "repetition_threshold",
                            CONFIG_DEFAULTS["aux"]["header_footer"]["repetition_threshold"],
                        )
                    ),
                    dropcap_max_fraction=float(
                        merged.get("aux", {})
                        .get("header_footer", {})
                        .get(
                            "dropcap_max_fraction",
                            CONFIG_DEFAULTS["aux"]["header_footer"]["dropcap_max_fraction"],
                        )
                    ),
                    y_band_pct=float(
                        merged.get("aux", {})
                        .get("header_footer", {})
                        .get(
                            "y_band_pct",
                            CONFIG_DEFAULTS["aux"]["header_footer"].get("y_band_pct", 0.07),
                        )
                    ),
                ),
                y_band_pct=float(
                    merged.get("aux", {})
                    .get("header_footer", {})
                    .get(
                        "y_band_pct",
                        CONFIG_DEFAULTS["aux"]["header_footer"].get("y_band_pct", 0.07),
                    )
                ),
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
                callout_column_width_fraction_max=float(
                    merged.get("aux", {})
                    .get("callout", {})
                    .get(
                        "column_width_fraction_max",
                        CONFIG_DEFAULTS["aux"]["callout"]["column_width_fraction_max"],
                    )
                ),
                font_band_small_quantile=float(
                    merged.get("aux", {})
                    .get("font_band", {})
                    .get(
                        "small_quantile",
                        CONFIG_DEFAULTS["aux"]["font_band"]["small_quantile"],
                    )
                ),
            ),
            segments=SegmentConfig(
                soft_boundary_pages=int(
                    merged.get("segments", {}).get(
                        "soft_boundary_pages", CONFIG_DEFAULTS["segments"]["soft_boundary_pages"]
                    )
                )
            ),
            anchor=AnchorConfig(
                lookahead_pages=int(
                    merged.get("anchor", {}).get(
                        "lookahead_pages", CONFIG_DEFAULTS["anchor"]["lookahead_pages"]
                    )
                )
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
            gate5=Gate5Config(
                header_footer=Gate5HeaderFooterConfig(
                    y_band_pct=float(
                        merged.get("gate5", {})
                        .get("header_footer", {})
                        .get(
                            "y_band_pct",
                            CONFIG_DEFAULTS["gate5"]["header_footer"]["y_band_pct"],
                        )
                    ),
                    repetition_threshold=float(
                        merged.get("gate5", {})
                        .get("header_footer", {})
                        .get(
                            "repetition_threshold",
                            CONFIG_DEFAULTS["gate5"]["header_footer"]["repetition_threshold"],
                        )
                    ),
                ),
                sidebar=Gate5SidebarConfig(
                    min_column_width_fraction=float(
                        merged.get("gate5", {})
                        .get("sidebar", {})
                        .get(
                            "min_column_width_fraction",
                            CONFIG_DEFAULTS["gate5"]["sidebar"]["min_column_width_fraction"],
                        )
                    )
                ),
                caption_zone=Gate5CaptionConfig(
                    lineheight_multiplier=float(
                        merged.get("gate5", {})
                        .get("caption_zone", {})
                        .get(
                            "lineheight_multiplier",
                            CONFIG_DEFAULTS["gate5"]["caption_zone"]["lineheight_multiplier"],
                        )
                    )
                ),
            ),
            paragraph_only=ParagraphOnlyConfig(
                min_blocks_across_pages=int(
                    merged.get("paragraph_only", {}).get(
                        "min_blocks_across_pages",
                        CONFIG_DEFAULTS["paragraph_only"]["min_blocks_across_pages"],
                    )
                ),
                window_pages=int(
                    merged.get("paragraph_only", {}).get(
                        "window_pages", CONFIG_DEFAULTS["paragraph_only"]["window_pages"]
                    )
                ),
            ),
            diagnostics=DiagnosticsConfig(
                enable=bool(
                    merged.get("diagnostics", {}).get(
                        "enable", CONFIG_DEFAULTS["diagnostics"]["enable"]
                    )
                ),
                overlay_max_pages=int(
                    merged.get("diagnostics", {}).get(
                        "overlay_max_pages",
                        CONFIG_DEFAULTS["diagnostics"]["overlay_max_pages"],
                    )
                ),
            ),
        )


DEFAULT_CONFIG = PipelineConfig()


def resolve_cache_path(config: PipelineConfig, doc_sha: str, page_no: int, dpi: int) -> Path:
    template = config.cache.dir
    return Path(template.format(doc_sha=doc_sha, page_no=page_no, dpi=dpi)).expanduser().resolve()
