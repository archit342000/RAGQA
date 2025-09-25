"""Lightweight telemetry helpers for the pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Sequence

from ..chunking.chunker import Chunk
from ..threading.threader import ThreadResult


@dataclass(slots=True)
class PipelineMetrics:
    """Simple container for pipeline metrics."""

    main_units: int
    aux_units: int
    aux_sections: int
    chunk_counts: Dict[str, int]


def compute_metrics(result: ThreadResult, chunks: Sequence[Chunk]) -> PipelineMetrics:
    main_units = sum(1 for unit in result.units if unit.emit_phase == 0)
    aux_units = sum(1 for unit in result.units if unit.emit_phase == 1)
    aux_sections = len({unit.section_seq for unit in result.units if unit.emit_phase == 1})
    chunk_counter: Counter[str] = Counter(chunk.role for chunk in chunks)
    return PipelineMetrics(
        main_units=main_units,
        aux_units=aux_units,
        aux_sections=aux_sections,
        chunk_counts=dict(chunk_counter),
    )

