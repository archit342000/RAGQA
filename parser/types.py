"""Typed structures for parsed documents."""

# These NamedTuples act as the public contract for parser outputs
# and are shared between the core parsing logic, tests, and the UI layer.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover - import hints only when type checking
    from pipeline.layout.lp_fuser import FusedDocument
    from pipeline.layout.router import LayoutRoutingPlan
    from pipeline.layout.signals import PageLayoutSignals


@dataclass(slots=True)
class ParsedPage:
    """Represents one page extracted from a document."""

    doc_id: str
    page_num: int
    text: str
    char_range: tuple[int, int]
    metadata: Dict[str, str]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDoc:
    """Represents an entire parsed document."""

    doc_id: str
    pages: List[ParsedPage]
    total_chars: int
    parser_used: str
    stats: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)
    fused_document: Optional["FusedDocument"] = None
    layout_signals: Optional[List["PageLayoutSignals"]] = None
    routing_plan: Optional["LayoutRoutingPlan"] = None


class RunReport(NamedTuple):
    """Aggregate metadata for a multi-document parsing run.

    ``RunReport`` powers the UI banner that informs users when documents were
    skipped or truncated and helps developers understand how much work the
    parser performed.
    """

    total_docs: int
    total_pages: int
    truncated: bool
    skipped_docs: List[str]
    message: str
