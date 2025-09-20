"""Dataclasses and type helpers for the evaluation harness."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class GoldItem:
    """Representation of a single gold question entry."""

    id: str
    question: str
    doc_id: str
    page_start: int
    page_end: int
    answer_text: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    hard_negative_ids: List[str] = field(default_factory=list)


@dataclass
class EvalConfig:
    """Configuration snapshot driving a retrieval evaluation run."""

    engines: List[str] = field(default_factory=lambda: ["lexical", "semantic", "hybrid"])
    top_k: int = 80
    rerank_top_n: int = 5
    pre_hit_k: int = 10
    post_hit_k: int = 5
    bootstrap_iters: int = 1000
    bootstrap_alpha: float = 0.05
    seed: int = 42
    hybrid_fusion: str = "rrf"
    hybrid_rrf_k: int = 60
    hybrid_weight_vector: float = 0.6
    hybrid_weight_lexical: float = 0.4
    hybrid_top_k_vector: int = 80
    hybrid_top_k_lexical: int = 80
    chunks_path: Optional[str] = None
    runs_dir: str = "runs"
    report_path: str = "report.html"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalConfig":
        allowed = {field.name for field in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ItemMetrics:
    """Per-question evaluation record."""

    id: str
    tags: List[str]
    pre_hit_at_k: float
    post_hit_at_k: float
    mrr_at_10: float
    ndcg_at_10: float
    context_precision: float
    timings: Dict[str, float]


@dataclass
class RunResult:
    """Serialized output of a single evaluation run."""

    engine: str
    config: EvalConfig
    created_at: datetime
    gold_checksum: str
    aggregates: Dict[str, float]
    latency: Dict[str, Dict[str, float]]
    items: List[ItemMetrics]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "config": self.config.to_dict(),
            "created_at": self.created_at.isoformat(),
            "gold_checksum": self.gold_checksum,
            "aggregates": self.aggregates,
            "latency": self.latency,
            "items": [dataclasses.asdict(item) for item in self.items],
        }


__all__ = [
    "EvalConfig",
    "GoldItem",
    "ItemMetrics",
    "RunResult",
]
