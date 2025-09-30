from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List

from .docling_adapter import DoclingBlock
from .triage import PageTriageResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RescueDecision:
    page_number: int
    should_rescue: bool
    reason: str


def plan_layout_rescue(pages: Iterable[PageTriageResult]) -> List[RescueDecision]:
    decisions: List[RescueDecision] = []
    for page in pages:
        should = bool(page.errors) or page.char_count == 0
        reason = "errors" if page.errors else "empty" if page.char_count == 0 else "ok"
        decisions.append(RescueDecision(page_number=page.page_number, should_rescue=should, reason=reason))
    return decisions


def apply_layout_rescue(blocks: List[DoclingBlock], decisions: List[RescueDecision]) -> List[DoclingBlock]:
    # Placeholder: The implementation would invoke layoutparser/detectron2.
    rescue_map = {d.page_number: d for d in decisions if d.should_rescue}
    if not rescue_map:
        return blocks
    logger.info("Layout rescue would run on pages: %s", sorted(rescue_map))
    for block in blocks:
        if block.page_number in rescue_map:
            block.source_stage = "layout"
            block.source_tool = "layoutparser"
            block.source_version = "stub"
    return blocks
