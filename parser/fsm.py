"""Main flow finite state machine for stitching and aux buffering."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FlowState:
    flow_index: int = 0
    open_flow: Optional[str] = None
    aux_buffer: List[Dict[str, object]] = field(default_factory=list)
    aux_queue_events: List[Dict[str, object]] = field(default_factory=list)
    split_events: List[Dict[str, object]] = field(default_factory=list)
    previous_block_id: Optional[str] = None


TERMINATORS = (".", "?", "!", "—")


def _new_flow_id(state: FlowState) -> str:
    state.flow_index += 1
    return f"flow_main_{state.flow_index:03d}"


def _is_continuation(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return stripped[0].islower()


def fsm_stitch_and_buffer(
    page_id: str,
    blocks: List[Dict[str, object]],
    state: FlowState,
    cfg: Dict[str, object],
) -> Dict[str, object]:
    """Process classified blocks for a single page."""

    page_blocks: List[Dict[str, object]] = []
    page_events_start = len(state.aux_queue_events)
    split_events_start = len(state.split_events)
    for block in blocks:
        if block["type"] == "aux":
            state.aux_buffer.append(block)
            continue
        if block["type"] == "control":
            if state.aux_buffer:
                state.aux_queue_events.append({"when": "before_control", "items": [b["id"] for b in state.aux_buffer]})
                page_blocks.extend(state.aux_buffer)
                state.aux_buffer.clear()
            page_blocks.append(block)
            state.open_flow = None
            state.previous_block_id = block["id"]
            continue

        flow_id = state.open_flow or _new_flow_id(state)
        block["flow_id"] = flow_id

        if state.aux_buffer:
            state.aux_queue_events.append({"when": "before_main", "items": [b["id"] for b in state.aux_buffer]})
            for aux in state.aux_buffer:
                aux.setdefault("flow_id", flow_id)
            page_blocks.extend(state.aux_buffer)
            state.aux_buffer.clear()

        if state.previous_block_id and block.get("continuation_from") == state.previous_block_id:
            state.split_events.append(
                {
                    "type": "pagebreak_join",
                    "from_block": state.previous_block_id,
                    "to_block": block["id"],
                    "hyphen_repaired": block.get("hyphen_repaired", False),
                }
            )
        page_blocks.append(block)
        state.previous_block_id = block["id"]

        if block["text"].rstrip().endswith(TERMINATORS):
            state.open_flow = None
        else:
            if _is_continuation(block["text"]):
                state.split_events.append(
                    {
                        "type": "continuation", "to_block": block["id"], "page": page_id
                    }
                )
            state.open_flow = flow_id

    page_payload = {
        "page_id": page_id,
        "blocks": page_blocks + state.aux_buffer,
        "aux_queue": state.aux_queue_events[page_events_start:],
        "split_events": state.split_events[split_events_start:],
    }
    state.aux_queue_events = state.aux_queue_events[:page_events_start]
    state.split_events = state.split_events[:split_events_start]
    return page_payload
