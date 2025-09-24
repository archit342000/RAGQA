"""Telemetry collection utilities for the hybrid parsing pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

from pipeline.layout.router import LayoutRoutingPlan
from pipeline.layout.signals import PageLayoutSignals
from pipeline.repair.repair_pass import RepairStats

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TelemetryCollector:
    lp_pages: int = 0
    total_pages: int = 0
    lp_latencies: List[float] = field(default_factory=list)
    page_scores: List[float] = field(default_factory=list)
    lp_signal_top2: Dict[int, List[str]] = field(default_factory=dict)
    interleave_errors: int = 0
    interleave_total: int = 0
    repair_merges: int = 0
    repair_splits: int = 0
    repair_total_blocks: int = 0
    retrieval_baseline: Dict[str, float] = field(default_factory=dict)
    retrieval_candidate: Dict[str, float] = field(default_factory=dict)

    def record_router(self, plan: LayoutRoutingPlan, signals: Sequence[PageLayoutSignals]) -> None:
        if len(signals) != plan.total_pages:
            logger.debug("Router telemetry mismatch: %s signals vs %s pages", len(signals), plan.total_pages)
        self.total_pages += plan.total_pages
        lp_set = {page for page in plan.selected_pages}
        self.lp_pages += len(lp_set)
        for signal in signals:
            self.page_scores.append(signal.page_score)
            if signal.page_number in lp_set:
                sorted_signals = sorted(signal.normalized.items(), key=lambda item: item[1], reverse=True)
                top = [name for name, _ in sorted_signals[:2]]
                self.lp_signal_top2[signal.page_number] = top

    def record_lp_latency(self, latency_seconds: float) -> None:
        if latency_seconds < 0:
            return
        self.lp_latencies.append(latency_seconds)

    def record_interleave(self, errors: int, total_checks: int) -> None:
        self.interleave_errors += max(0, errors)
        self.interleave_total += max(0, total_checks)

    def record_repair(self, stats: RepairStats, total_blocks: int) -> None:
        self.repair_merges += max(0, stats.merged_blocks)
        self.repair_splits += max(0, stats.split_blocks)
        self.repair_total_blocks += max(0, total_blocks)

    def record_retrieval_metrics(
        self,
        *,
        baseline: Mapping[str, float],
        candidate: Mapping[str, float],
    ) -> None:
        self.retrieval_baseline = dict(baseline)
        self.retrieval_candidate = dict(candidate)

    def summary(self) -> Dict[str, object]:
        lp_ratio = (self.lp_pages / self.total_pages) if self.total_pages else 0.0
        avg_latency = sum(self.lp_latencies) / len(self.lp_latencies) if self.lp_latencies else 0.0
        interleave_rate = (self.interleave_errors / self.interleave_total) if self.interleave_total else 0.0
        repair_merge_pct = (self.repair_merges / self.repair_total_blocks) if self.repair_total_blocks else 0.0
        repair_split_pct = (self.repair_splits / self.repair_total_blocks) if self.repair_total_blocks else 0.0
        retrieval_deltas: Dict[str, float] = {}
        for key, candidate_value in self.retrieval_candidate.items():
            retrieval_deltas[key] = candidate_value - self.retrieval_baseline.get(key, 0.0)
        return {
            "lp_page_ratio": lp_ratio,
            "avg_lp_latency_per_page": avg_latency,
            "score_distribution": list(self.page_scores),
            "lp_signal_top2": dict(self.lp_signal_top2),
            "interleave_error_rate": interleave_rate,
            "repair_merge_pct": repair_merge_pct,
            "repair_split_pct": repair_split_pct,
            "retrieval_deltas": retrieval_deltas,
        }

    def log_summary(self) -> None:
        summary = self.summary()
        logger.info(
            "LP ratio %.2f, avg latency %.2fs, interleave rate %.2f%%",
            summary["lp_page_ratio"],
            summary["avg_lp_latency_per_page"],
            summary["interleave_error_rate"] * 100,
        )
        logger.debug("Score distribution: min=%.3f max=%.3f", min(self.page_scores or [0.0]), max(self.page_scores or [0.0]))
        if summary["retrieval_deltas"]:
            logger.info("Retrieval deltas: %s", summary["retrieval_deltas"])


__all__ = ["TelemetryCollector"]
