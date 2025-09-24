"""Routing heuristics that decide when to invoke LayoutParser."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

from pipeline.ingest.pdf_parser import DocumentGraph
from pipeline.layout.signals import PageLayoutSignals


@dataclass(slots=True)
class PageRoutingDecision:
    page_number: int
    score: float
    triggers: List[str]
    neighbor: bool = False
    use_layout_parser: bool = False
    model: str = "publaynet"
    dpi: int = 180


@dataclass(slots=True)
class LayoutRoutingPlan:
    doc_id: str
    total_pages: int
    budget: int
    selected_pages: List[int]
    decisions: List[PageRoutingDecision]
    overflow: int

    @property
    def ratio(self) -> float:
        return len(self.selected_pages) / max(self.total_pages, 1)


def _detect_triggers(
    idx: int,
    signals: Sequence[PageLayoutSignals],
    repair_failures: Dict[int, int],
) -> List[str]:
    signal = signals[idx]
    triggers: List[str] = []
    raw = signal.raw
    extras = signal.extras
    if raw["OGR"] >= 0.35 and raw["BXS"] >= 0.25:
        triggers.append("T1")
    if extras.table_overlap_ratio > 0.20:
        triggers.append("T2")
    if raw["CIS"] >= 0.45:
        prev_columns = signals[idx - 1].extras.column_count if idx > 0 else extras.column_count
        next_columns = signals[idx + 1].extras.column_count if idx + 1 < len(signals) else extras.column_count
        if prev_columns != extras.column_count or next_columns != extras.column_count:
            triggers.append("T3")
    if repair_failures.get(signal.page_number, 0) >= 2:
        triggers.append("T4")
    return triggers


def _select_model(signal: PageLayoutSignals) -> str:
    extras = signal.extras
    normalized = signal.normalized
    if extras.column_count >= 2 and normalized["CIS"] >= 0.55:
        return "docbank"
    if normalized["FVS"] >= 0.6 or normalized["MSA"] >= 0.55 or normalized["OGR"] >= 0.65:
        return "prima"
    return "publaynet"


def plan_layout_routing(
    document: DocumentGraph,
    signals: Sequence[PageLayoutSignals],
    *,
    repair_failures: Dict[int, int] | None = None,
    dpi: int = 180,
) -> LayoutRoutingPlan:
    """Plan which pages require LayoutParser based on signals and triggers."""

    if len(document.pages) != len(signals):
        raise ValueError("Signal/page length mismatch")

    failure_map = repair_failures or {}
    total_pages = len(signals)
    budget = max(1, math.ceil(total_pages * 0.3))

    decisions: List[PageRoutingDecision] = []
    base_candidates: List[int] = []

    for idx, signal in enumerate(signals):
        triggers = _detect_triggers(idx, signals, failure_map)
        decision = PageRoutingDecision(
            page_number=signal.page_number,
            score=signal.page_score,
            triggers=triggers,
            neighbor=False,
            use_layout_parser=False,
            model=_select_model(signal),
            dpi=dpi,
        )
        run_lp = signal.page_score >= 0.55 or bool(triggers)
        if run_lp:
            decision.use_layout_parser = True
            base_candidates.append(idx)
        decisions.append(decision)

    # Neighbor inclusion when score is moderately high.
    for idx in base_candidates:
        for neighbor_idx in (idx - 1, idx + 1):
            if 0 <= neighbor_idx < len(signals):
                neighbor_decision = decisions[neighbor_idx]
                if neighbor_decision.use_layout_parser:
                    continue
                if signals[neighbor_idx].page_score >= 0.50:
                    neighbor_decision.use_layout_parser = True
                    neighbor_decision.neighbor = True
                    neighbor_decision.triggers.append("neighbor")

    selected = [d for d in decisions if d.use_layout_parser]
    triggered = [d for d in selected if any(t.startswith("T") for t in d.triggers)]
    baseline_cap = max(len(triggered), budget)
    max_allowed = baseline_cap + 5
    if len(selected) > max_allowed:
        # Drop neighbor-only pages with lowest scores first.
        removable = [d for d in selected if d.neighbor and not any(t.startswith("T") for t in d.triggers if t != "neighbor")]
        removable.sort(key=lambda d: (d.score, d.page_number))
        for decision in removable:
            if len(selected) <= max_allowed:
                break
            decision.use_layout_parser = False
            decision.neighbor = False
            if "neighbor" in decision.triggers:
                decision.triggers.remove("neighbor")
            selected.remove(decision)
        # If still above the cap, remove lowest-score non-triggered pages.
        if len(selected) > max_allowed:
            non_triggered = [d for d in selected if not any(t.startswith("T") for t in d.triggers)]
            non_triggered.sort(key=lambda d: (d.score, d.page_number))
            for decision in non_triggered:
                if len(selected) <= max_allowed:
                    break
                decision.use_layout_parser = False
                if "neighbor" in decision.triggers:
                    decision.triggers.remove("neighbor")
                selected.remove(decision)

    overflow = max(0, len([d for d in decisions if d.use_layout_parser]) - budget)
    selected_pages = [d.page_number for d in decisions if d.use_layout_parser]

    return LayoutRoutingPlan(
        doc_id=document.doc_id,
        total_pages=total_pages,
        budget=budget,
        selected_pages=selected_pages,
        decisions=decisions,
        overflow=overflow,
    )


__all__ = ["LayoutRoutingPlan", "PageRoutingDecision", "plan_layout_routing"]
