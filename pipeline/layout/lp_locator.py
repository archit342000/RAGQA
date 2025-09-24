"""Selective LayoutParser localisation helpers for auxiliary-aware threading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pipeline.ingest.pdf_parser import DocumentGraph
from pipeline.layout.lp_fuser import LPRegion, LayoutParserEngine
from pipeline.layout.router import LayoutRoutingPlan

BBox = Tuple[float, float, float, float]


@dataclass(slots=True)
class LocatedRegion:
    page_number: int
    bbox: BBox
    label: str
    score: float
    order: int


def locate_regions(
    document: DocumentGraph,
    page_numbers: Sequence[int],
    *,
    model: str = "publaynet",
    dpi: int = 180,
    engine: Optional[LayoutParserEngine] = None,
) -> Dict[int, List[LocatedRegion]]:
    """Run LayoutParser for the given ``page_numbers`` and return structured regions."""

    if not page_numbers:
        return {}
    engine = engine or LayoutParserEngine()
    outputs = engine.detect(document, page_numbers, model=model, dpi=dpi)
    located: Dict[int, List[LocatedRegion]] = {}
    for page_number, regions in outputs.items():
        located[page_number] = [
            LocatedRegion(
                page_number=page_number,
                bbox=region.bbox,
                label=region.label,
                score=region.score,
                order=region.order,
            )
            for region in regions
        ]
    return located


def locate_for_plan(
    document: DocumentGraph,
    plan: LayoutRoutingPlan,
    *,
    engine: Optional[LayoutParserEngine] = None,
) -> Dict[int, List[LocatedRegion]]:
    """Convenience wrapper that respects the routing plan's model and DPI groups."""

    engine = engine or LayoutParserEngine()
    by_model: Dict[Tuple[str, int], List[int]] = {}
    for decision in plan.decisions:
        if not decision.use_layout_parser:
            continue
        by_model.setdefault((decision.model, decision.dpi), []).append(decision.page_number)

    located: Dict[int, List[LocatedRegion]] = {}
    for (model, dpi), page_numbers in by_model.items():
        batches = [page_numbers[i : i + engine.max_batch] for i in range(0, len(page_numbers), engine.max_batch)]
        for batch in batches:
            located.update(locate_regions(document, batch, model=model, dpi=dpi, engine=engine))
    return located


__all__ = ["LocatedRegion", "locate_regions", "locate_for_plan"]
