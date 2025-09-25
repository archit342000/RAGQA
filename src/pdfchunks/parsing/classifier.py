"""Block classification into MAIN and AUX roles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from rapidfuzz import fuzz

from ..config import ClassifierConfig
from .baselines import BaselineStats, ColumnBand
from .block_extractor import Block


_MAIN = "MAIN"
_AUX = "AUX"


@dataclass(slots=True)
class BlockLabel:
    """Represents the classification of a block."""

    block: Block
    role: str
    subtype: Optional[str] = None
    is_heading: bool = False
    section_level: int = 0

    @property
    def text(self) -> str:
        return self.block.text

    def __getitem__(self, name: str):  # convenience for tests
        return getattr(self, name)


class BlockClassifier:
    """Classify blocks based on allow-list constraints."""

    def __init__(self, config: ClassifierConfig, lexical_override: Optional[Iterable[str]] = None):
        self.config = config
        lexical_cues = list(lexical_override or config.lexical_cues)
        self._lexical_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in lexical_cues]
        self._heading_pattern = re.compile(config.heading_regex)

    def classify(self, blocks: Iterable[Block], baselines: BaselineStats) -> List[BlockLabel]:
        labels: List[BlockLabel] = []
        for block in blocks:
            is_heading = self._is_heading(block, baselines)
            if is_heading:
                labels.append(BlockLabel(block=block, role=_MAIN, is_heading=True, section_level=self._heading_level(block, baselines)))
                continue

            if self._is_main(block, baselines):
                labels.append(BlockLabel(block=block, role=_MAIN))
            else:
                labels.append(BlockLabel(block=block, role=_AUX, subtype=self._infer_aux_subtype(block)))
        return labels

    # --- heuristics --------------------------------------------------------------------

    def _is_heading(self, block: Block, baselines: BaselineStats) -> bool:
        if not block.text.strip():
            return False
        if baselines.body_font_size <= 0:
            return False
        if block.font_size >= baselines.body_font_size * self.config.heading_scale:
            return True
        return bool(self._heading_pattern.match(block.text.strip()))

    def _heading_level(self, block: Block, baselines: BaselineStats) -> int:
        ratio = baselines.body_font_size and block.font_size / baselines.body_font_size or 0.0
        if ratio >= 1.7:
            return 1
        if ratio >= 1.35:
            return 2
        if ratio >= 1.1:
            return 3
        match = re.match(r"^(\d+(?:\.\d+)*)", block.text.strip())
        if match:
            return match.group(1).count(".") + 1
        return 4

    def _is_main(self, block: Block, baselines: BaselineStats) -> bool:
        if baselines.body_font_size <= 0 or baselines.body_line_height <= 0:
            return False
        if block.has_border or block.has_shading:
            return False
        if block.metadata.get("figure_iou", 0.0) >= 0.2:
            return False
        if not self._within_body_metrics(block, baselines):
            return False
        if self._violates_margins(block):
            return False
        if self._in_exclusion_band(block):
            return False
        if self._density_out_of_range(block, baselines):
            return False
        if self._has_lexical_cue(block.text):
            return False
        if not self._column_alignment(block, baselines.columns):
            return False
        return True

    def _within_body_metrics(self, block: Block, baselines: BaselineStats) -> bool:
        font_match = abs(block.font_size - baselines.body_font_size) <= baselines.body_font_size * self.config.font_tolerance
        line_match = abs(block.line_height - baselines.body_line_height) <= baselines.body_line_height * self.config.line_height_tolerance
        return font_match and line_match

    def _violates_margins(self, block: Block) -> bool:
        outer_margin = block.page_width * 0.1
        return block.x0 <= outer_margin or block.x1 >= block.page_width - outer_margin

    def _in_exclusion_band(self, block: Block) -> bool:
        top_cut = block.page_height * 0.1
        bottom_cut = block.page_height * 0.9
        return block.y0 <= top_cut or block.y1 >= bottom_cut

    def _density_out_of_range(self, block: Block, baselines: BaselineStats) -> bool:
        if baselines.density_p20 == 0 and baselines.density_p80 == 0:
            return False
        return not (baselines.density_p20 <= block.density <= baselines.density_p80)

    def _has_lexical_cue(self, text: str) -> bool:
        stripped = text.strip()
        return any(pattern.search(stripped) for pattern in self._lexical_patterns)

    def _column_alignment(self, block: Block, columns: List[ColumnBand]) -> bool:
        if not columns:
            return True
        for band in columns:
            if band.width <= 0:
                continue
            if band.overlap_ratio(block) >= self.config.column_min_overlap:
                band_width = band.width
                if abs(block.x0 - band.x0) <= self.config.column_x_tolerance:
                    if block.width >= band_width * self.config.column_width_ratio:
                        return True
        return False

    def _infer_aux_subtype(self, block: Block) -> str:
        text = block.text.strip()
        lowered = text.lower()
        if re.match(r"^(fig\.|figure|table|source)\b", lowered, re.IGNORECASE):
            return "caption"
        if re.match(r"^(activity|discuss|think|imagine|let’s|lets|try|project)\b", lowered, re.IGNORECASE):
            return "activity"
        if block.metadata.get("footnote", False):
            return "footnote"
        if block.metadata.get("sidebar", False):
            return "callout"
        if block.metadata.get("header", False):
            return "header"
        # Fuzzy match fallback for keywords/answer/exercise cues.
        cues = ["keywords", "answer", "exercise", "case study"]
        for cue in cues:
            if lowered.startswith(cue) or fuzz.partial_ratio(lowered[:20], cue) > 80:
                return "callout"
        return "auxiliary"

