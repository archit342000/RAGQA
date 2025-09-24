"""Configuration primitives for the pdfchunks pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


@dataclass(slots=True)
class BlockExtractionConfig:
    """Configuration for block extraction."""

    header_footer_repeat_threshold: int = 5
    top_bottom_exclusion_ratio: float = 0.1
    outer_margin_ratio: float = 0.1
    shading_threshold: float = 0.7
    border_threshold: float = 0.7
    max_repeat_candidates: int = 20


@dataclass(slots=True)
class BaselineConfig:
    """Configuration used to compute layout baselines."""

    column_gap_threshold: float = 36.0
    max_columns: int = 3
    column_x_tolerance: float = 8.0
    column_min_overlap: float = 0.9
    column_width_ratio: float = 0.8


@dataclass(slots=True)
class ClassifierConfig:
    """Configuration for block classification."""

    font_tolerance: float = 0.1
    line_height_tolerance: float = 0.12
    density_quantile_low: float = 0.2
    density_quantile_high: float = 0.8
    heading_scale: float = 1.25
    heading_regex: str = r"^(?:[A-Z][A-Z0-9\s\-.]{2,}|\d+(?:\.\d+)*\s+.+)"
    column_min_overlap: float = 0.9
    column_width_ratio: float = 0.8
    column_x_tolerance: float = 8.0
    lexical_cues: Iterable[str] = (
        r"^Fig\.",
        r"^Figure",
        r"^Table",
        r"^Source",
        r"^Activity",
        r"^Discuss",
        r"^Think",
        r"^Imagine",
        r"^Let’s",
        r"^Lets",
        r"^Keywords",
        r"^Answer",
        r"^Exercise",
        r"^Try",
        r"^Project",
        r"^Case Study",
    )


@dataclass(slots=True)
class ThreadingConfig:
    """Configuration driving paragraph threading."""

    lookahead_pages: int = 2
    dehyphenate: bool = True
    sentence_regex: str = r"(?<=[.!?])\s+(?=[A-Z0-9])"
    paragraph_end_regex: str = r"[.!?]\s*$"


@dataclass(slots=True)
class ChunkerConfig:
    """Configuration for chunk packing."""

    main_target_tokens: int = 500
    main_min_tokens: int = 180
    main_max_tokens: int = 700
    main_overlap_tokens: int = 80
    main_small_overlap_tokens: int = 40
    aux_max_tokens: int = 400


@dataclass(slots=True)
class AuditConfig:
    """Configuration for guardrails and logging."""

    log_order_key_limit: int = 20


@dataclass(slots=True)
class ParserConfig:
    """Top-level configuration object."""

    block_extraction: BlockExtractionConfig = field(default_factory=BlockExtractionConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    threading: ThreadingConfig = field(default_factory=ThreadingConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    audits: AuditConfig = field(default_factory=AuditConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParserConfig":
        """Build a :class:`ParserConfig` from a nested mapping."""

        def build(name: str, typ: Any) -> Any:
            section = data.get(name, {})
            if isinstance(section, dict):
                return typ(**section)
            return typ(**dict(section))

        return cls(
            block_extraction=build("block_extraction", BlockExtractionConfig),
            baselines=build("baselines", BaselineConfig),
            classifier=build("classifier", ClassifierConfig),
            threading=build("threading", ThreadingConfig),
            chunker=build("chunker", ChunkerConfig),
            audits=build("audits", AuditConfig),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration into a serialisable mapping."""

        return {
            "block_extraction": self.block_extraction.__dict__,
            "baselines": self.baselines.__dict__,
            "classifier": {
                **{k: v for k, v in self.classifier.__dict__.items() if k != "lexical_cues"},
                "lexical_cues": list(self.classifier.lexical_cues),
            },
            "threading": self.threading.__dict__,
            "chunker": self.chunker.__dict__,
            "audits": self.audits.__dict__,
        }


def load_config(path: Optional[Path | str] = None) -> ParserConfig:
    """Load configuration from YAML, defaulting to packaged defaults."""

    if path is None:
        base = Path(__file__).resolve()
        candidates = [
            base.parent.parent.parent / "configs" / "parser.yaml",
            base.parent.parent / "configs" / "parser.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        else:  # pragma: no cover - configuration missing is a deployment error.
            raise FileNotFoundError("Default configuration file could not be located.")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration YAML must produce a mapping")
    return ParserConfig.from_dict(data)

