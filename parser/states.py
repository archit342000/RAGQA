"""State containers for flow-safe classification hotfixes."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


@dataclass
class OpenParagraphState:
    """Metadata for an open paragraph that continues across page breaks."""

    source_block_id: str
    last_token: str
    needs_hyphen_repair: bool = False


@dataclass
class FlowState:
    """Document-level classification state with aux buffering."""

    section_state: str = "OUT_OF_SECTION"
    continuing_paragraph: bool = False
    last_main_meta: Optional[Dict[str, Any]] = None
    last_main_record: Optional[Dict[str, Any]] = None
    open_paragraph: Optional[OpenParagraphState] = None
    aux_buffer: Deque[Dict[str, Any]] = field(default_factory=deque)
    running_header_patterns: Dict[str, float] = field(default_factory=dict)
    running_footer_patterns: Dict[str, float] = field(default_factory=dict)

    def park_aux(self, block: Dict[str, Any]) -> None:
        """Queue an auxiliary block until a legal flush point is reached."""

        block.setdefault("reason", []).append("AuxParked")
        self.aux_buffer.append(block)

    def flush_aux(self, sink: List[Dict[str, Any]]) -> None:
        """Flush buffered auxiliary blocks into the output sink."""

        while self.aux_buffer:
            sink.append(self.aux_buffer.popleft())

    def flush_aux_if_allowed(self, sink: List[Dict[str, Any]], allow_when_open: bool = False) -> None:
        if self.open_paragraph and not allow_when_open:
            return
        self.flush_aux(sink)

    def clear_open_paragraph(self) -> None:
        self.open_paragraph = None
        self.continuing_paragraph = False

    # Dict-like helpers -------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)


__all__ = ["FlowState", "OpenParagraphState"]
