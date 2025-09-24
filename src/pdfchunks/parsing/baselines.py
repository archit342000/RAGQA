"""Layout baseline utilities."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .block_extractor import Block, DocumentLayout
from ..config import BaselineConfig


@dataclass(slots=True)
class ColumnBand:
    """Represents an inferred column on the page."""

    index: int
    x0: float
    x1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    def overlap_ratio(self, block: Block) -> float:
        overlap = max(0.0, min(self.x1, block.x1) - max(self.x0, block.x0))
        if block.width == 0:
            return 0.0
        return overlap / block.width


@dataclass(slots=True)
class BaselineStats:
    """Aggregate statistics required for classification."""

    body_font_size: float
    body_line_height: float
    density_p20: float
    density_p80: float
    columns: List[ColumnBand]


class BaselineEstimator:
    """Derive baseline statistics for a document layout."""

    def __init__(self, config: BaselineConfig):
        self.config = config

    def fit(self, layout: DocumentLayout) -> BaselineStats:
        blocks = [b for b in layout.iter_blocks() if b.text.strip()]
        if not blocks:
            return BaselineStats(0.0, 0.0, 0.0, 0.0, [])

        font_sizes = [b.font_size for b in blocks]
        line_heights = [b.line_height for b in blocks if b.line_height > 0]
        densities = [b.density for b in blocks if b.density > 0]
        body_font = statistics.median(font_sizes) if font_sizes else 0.0
        body_line_height = statistics.median(line_heights) if line_heights else 0.0
        density_p20, density_p80 = _quantile_pair(densities, self.config)

        page_width = max((b.page_width for b in blocks), default=0.0)
        columns = infer_column_bands(blocks, page_width, self.config)
        return BaselineStats(
            body_font_size=body_font,
            body_line_height=body_line_height,
            density_p20=density_p20,
            density_p80=density_p80,
            columns=columns,
        )


def infer_column_bands(blocks: Sequence[Block], page_width: float, config: BaselineConfig) -> List[ColumnBand]:
    """Infer 1–3 column bands using simple gap analysis."""

    if not blocks:
        return [ColumnBand(index=0, x0=0.0, x1=page_width or 1.0)]

    spans = sorted((block.x0, block.x1) for block in blocks if block.width > 0)
    if not spans:
        return [ColumnBand(index=0, x0=0.0, x1=page_width or 1.0)]

    bands: List[ColumnBand] = []
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        if start - current_end > config.column_gap_threshold and len(bands) + 1 < config.max_columns:
            bands.append(ColumnBand(index=len(bands), x0=current_start, x1=current_end))
            current_start, current_end = start, end
        else:
            current_start = min(current_start, start)
            current_end = max(current_end, end)
    bands.append(ColumnBand(index=len(bands), x0=current_start, x1=current_end))

    if not bands:
        bands = [ColumnBand(index=0, x0=0.0, x1=page_width or 1.0)]
    else:
        # Normalise single narrow bands to full width when appropriate.
        if len(bands) == 1 and bands[0].width < page_width * 0.5 and page_width > 0:
            bands[0] = ColumnBand(index=0, x0=0.0, x1=page_width)

    return bands


def _quantile_pair(values: Iterable[float], config: BaselineConfig) -> Tuple[float, float]:
    data = sorted(values)
    _ = config  # retained for backwards compatibility / signature stability
    if not data:
        return 0.0, 0.0
    # Fixed percentiles (P20 and P80) keep density estimates consistent.
    return _quantile(data, 0.2, 0.8)


def _quantile(data: Sequence[float], low: float, high: float) -> Tuple[float, float]:
    def pick(q: float) -> float:
        if not data:
            return 0.0
        if q <= 0:
            return data[0]
        if q >= 1:
            return data[-1]
        index = (len(data) - 1) * q
        lower = int(index)
        upper = min(lower + 1, len(data) - 1)
        fraction = index - lower
        return data[lower] * (1 - fraction) + data[upper] * fraction

    return pick(low), pick(high)

