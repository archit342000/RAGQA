"""Guards ensuring section-first serialized units maintain monotonic order."""

from __future__ import annotations

import logging
from typing import Sequence

from pipeline.serialize.serializer import SerializedUnit

logger = logging.getLogger(__name__)


def ensure_monotonic_order(units: Sequence[SerializedUnit]) -> None:
    last_key: tuple[str, int, int, int, int] | None = None
    for unit in units:
        if last_key is not None and unit.order_key < last_key:
            raise AssertionError(
                f"order regression: {unit.order_key} after {last_key}"
            )
        last_key = unit.order_key


def ensure_emit_phase_monotonic(units: Sequence[SerializedUnit]) -> None:
    emit_state: dict[int, int] = {}
    for unit in units:
        prev = emit_state.get(unit.section_seq, 0)
        if unit.emit_phase < prev:
            raise AssertionError(
                f"emit_phase regression in section {unit.section_seq}: {unit.emit_phase} < {prev}"
            )
        emit_state[unit.section_seq] = max(prev, unit.emit_phase)


def ensure_aux_after_main(units: Sequence[SerializedUnit]) -> None:
    last_main_sent: dict[int, int] = {}
    for unit in units:
        if unit.emit_phase == 0:
            last_main_sent[unit.section_seq] = max(
                last_main_sent.get(unit.section_seq, 0), unit.sent_seq
            )
            continue
        baseline = last_main_sent.get(unit.section_seq)
        if baseline is None:
            continue
        if unit.sent_seq <= baseline:
            raise AssertionError(
                f"aux before section seal for section {unit.section_seq}:"
                f" sent_seq={unit.sent_seq} <= baseline={baseline}"
            )


def run_order_guards(units: Sequence[SerializedUnit]) -> None:
    ensure_monotonic_order(units)
    ensure_emit_phase_monotonic(units)
    ensure_aux_after_main(units)
    logger.debug("order guards passed for %d units", len(units))


__all__ = [
    "ensure_monotonic_order",
    "ensure_emit_phase_monotonic",
    "ensure_aux_after_main",
    "run_order_guards",
]

