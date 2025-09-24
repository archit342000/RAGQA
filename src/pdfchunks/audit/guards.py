"""Guardrail assertions for the pipeline."""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import Dict, Sequence

from ..config import AuditConfig
from ..chunking.chunker import Chunk
from ..threading.threader import ThreadResult, ThreadUnit

LOGGER = logging.getLogger(__name__)
_AUX_PATTERN = re.compile(r"^<aux>.*</aux>$", re.DOTALL)


def run_audits(result: ThreadResult, chunks: Sequence[Chunk], config: AuditConfig) -> None:
    units = result.units
    _assert_monotonic(units)
    _assert_no_aux_in_main(units)
    _assert_aux_after_lead(units)
    _assert_aux_after_section(units)
    _assert_aux_wrapped(units, chunks)
    _log_order_keys(units, config.log_order_key_limit)
    _log_aux_flushes(result.delayed_aux_counts)
    _log_section_flush_sizes(chunks)


def _assert_monotonic(units: Sequence[ThreadUnit]) -> None:
    last_key = None
    for unit in units:
        if last_key is not None and unit.order_key < last_key:
            raise AssertionError("Non-monotonic order key detected")
        last_key = unit.order_key


def _assert_no_aux_in_main(units: Sequence[ThreadUnit]) -> None:
    for unit in units:
        if unit.emit_phase == 0 and "<aux>" in unit.text:
            raise AssertionError("Aux tag leaked into MAIN sentence")


def _assert_aux_after_lead(units: Sequence[ThreadUnit]) -> None:
    lead_seen: Dict[int, bool] = defaultdict(bool)
    for unit in units:
        if unit.emit_phase == 0 and unit.para_seq > 0:
            lead_seen[unit.section_seq] = True
        if unit.emit_phase == 1 and not lead_seen[unit.section_seq]:
            raise AssertionError("AUX emitted before section lead paragraph")


def _assert_aux_after_section(units: Sequence[ThreadUnit]) -> None:
    aux_seen: Dict[int, bool] = defaultdict(bool)
    for unit in units:
        if unit.emit_phase == 1:
            aux_seen[unit.section_seq] = True
        if unit.emit_phase == 0 and aux_seen.get(unit.section_seq):
            raise AssertionError("MAIN appeared after AUX within a section")


def _assert_aux_wrapped(units: Sequence[ThreadUnit], chunks: Sequence[Chunk]) -> None:
    for unit in units:
        if unit.emit_phase == 1 and not _AUX_PATTERN.match(unit.text):
            raise AssertionError("AUX unit missing <aux> wrapper")
    for chunk in chunks:
        if chunk.role == "AUX" and not _AUX_PATTERN.match(chunk.text.replace("\n", "")):
            raise AssertionError("AUX chunk missing <aux> wrapper")


def _log_order_keys(units: Sequence[ThreadUnit], limit: int) -> None:
    preview = [unit.order_key for unit in units[:limit]]
    LOGGER.info("order_key preview: %s", preview)


def _log_aux_flushes(delayed: Dict[int, int]) -> None:
    LOGGER.info("delayed AUX counts: %s", dict(delayed))


def _log_section_flush_sizes(chunks: Sequence[Chunk]) -> None:
    counter: Counter[tuple[int, str]] = Counter()
    for chunk in chunks:
        if chunk.role == "AUX":
            counter[(chunk.section_seq, chunk.subtype or "aux")] += 1
    LOGGER.info("per-section AUX chunk counts: %s", dict(counter))

