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
    normalized = signal.normalized
    extras = signal.extras
    if normalized["OGR"] >= 0.45 and normalized["BXS"] >= 0.35:
        triggers.append("T1")
    if extras.intrusion_ratio >= 0.30:
        triggers.append("T2")
    if normalized["CIS"] >= 0.55:
        prev_columns = signals[idx - 1].extras.column_count if idx > 0 else extras.column_count
        next_columns = signals[idx + 1].extras.column_count if idx + 1 < len(signals) else extras.column_count
        if prev_columns != extras.column_count or next_columns != extras.column_count:
            triggers.append("T3")
    if repair_failures.get(signal.page_number, 0) >= 2:
        top_two = sorted(normalized.items(), key=lambda item: item[1], reverse=True)[:2]
        if any(name in {"CIS", "ROJ"} for name, _ in top_two):
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


def _simple_page_override(signal: PageLayoutSignals) -> bool:
    extras = signal.extras
    normalized = signal.normalized
    if extras.total_line_count < 25 or not extras.has_normal_density:
        return False
    if extras.intrusion_ratio > 0.05:
        return False
    return (
        normalized["CIS"] <= 0.20
        and normalized["OGR"] <= 0.10
        and normalized["TFI"] <= 0.05
        and normalized["ROJ"] <= 0.10
    )


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
    budget = math.floor(total_pages * 0.2)
    if total_pages > 0 and budget == 0:
        budget = 1

    decisions: List[PageRoutingDecision] = []
    base_candidates: List[int] = []

    for idx, signal in enumerate(signals):
        triggers = _detect_triggers(idx, signals, failure_map)
        score = signal.page_score
        structural = signal.extras.structural_score or max(
            signal.normalized["CIS"], signal.normalized["ROJ"], signal.normalized["TFI"]
        )
        decision = PageRoutingDecision(
            page_number=signal.page_number,
            score=score,
            triggers=triggers,
            neighbor=False,
            use_layout_parser=False,
            model=_select_model(signal),
            dpi=dpi,
        )
        run_lp = (score >= 0.62 and structural >= 0.50) or bool(triggers)
        if run_lp and _simple_page_override(signal):
            run_lp = False
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
                neighbor_signal = signals[neighbor_idx]
                neighbor_structural = neighbor_signal.extras.structural_score or max(
                    neighbor_signal.normalized["CIS"],
                    neighbor_signal.normalized["ROJ"],
                    neighbor_signal.normalized["TFI"],
                )
                if (
                    neighbor_signal.page_score >= 0.58
                    and neighbor_structural >= 0.55
                    and not _simple_page_override(neighbor_signal)
                ):
                    neighbor_decision.use_layout_parser = True
                    neighbor_decision.neighbor = True
                    if "neighbor" not in neighbor_decision.triggers:
                        neighbor_decision.triggers.append("neighbor")

    selected = [d for d in decisions if d.use_layout_parser]
    triggered = [d for d in selected if any(t.startswith("T") for t in d.triggers)]
    baseline_cap = max(len(triggered), budget)
    max_allowed = baseline_cap + 3
    if len(selected) > max_allowed:
        # Drop neighbor-only pages with lowest scores first.
        removable = [d for d in selected if d.neighbor and not any(t.startswith("T") for t in d.triggers)]
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
