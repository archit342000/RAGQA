"""Block extraction utilities built on top of PyMuPDF."""

from __future__ import annotations

import itertools
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - PyMuPDF is optional during unit tests.
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None  # type: ignore

from ..config import BlockExtractionConfig

LOGGER = logging.getLogger(__name__)


BBox = Tuple[float, float, float, float]


def _bbox_width(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def _bbox_height(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])


def _bbox_area(bbox: BBox) -> float:
    return max(0.0, _bbox_width(bbox) * _bbox_height(bbox))


@dataclass(slots=True)
class Block:
    """A text block extracted from a PDF page."""

    block_id: str
    text: str
    font_name: str
    font_size: float
    line_height: float
    bbox: BBox
    page_num: int
    page_width: float
    page_height: float
    has_border: bool = False
    has_shading: bool = False
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return _bbox_width(self.bbox)

    @property
    def height(self) -> float:
        return _bbox_height(self.bbox)

    @property
    def area(self) -> float:
        return _bbox_area(self.bbox)

    @property
    def density(self) -> float:
        if self.area == 0:
            return 0.0
        return max(0.0, len(self.text.strip()) / self.area)


@dataclass(slots=True)
class DocumentLayout:
    """Holds per-document block information."""

    doc_id: str
    blocks: List[Block]
    page_sizes: Dict[int, Tuple[float, float]]

    def iter_blocks(self) -> Iterable[Block]:
        return iter(self.blocks)


class BlockExtractor:
    """Convert a PyMuPDF document into :class:`Block` objects."""

    def __init__(self, config: Optional[BlockExtractionConfig] = None):
        self.config = config or BlockExtractionConfig()

    def extract(self, document: "fitz.Document", doc_id: str) -> DocumentLayout:
        if fitz is None:  # pragma: no cover - handled in CLI/runtime.
            raise RuntimeError("PyMuPDF is required for block extraction.")

        page_sizes: Dict[int, Tuple[float, float]] = {}
        page_blocks: Dict[int, List[Block]] = defaultdict(list)
        header_counters: Counter[str] = Counter()
        footer_counters: Counter[str] = Counter()
        header_tracker: Dict[int, List[str]] = defaultdict(list)
        footer_tracker: Dict[int, List[str]] = defaultdict(list)

        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            width, height = page.rect.width, page.rect.height
            page_sizes[page_index] = (width, height)
            dict_page = page.get_text("dict")
            text_blocks = [b for b in dict_page.get("blocks", []) if b.get("type") == 0]

            for block_index, block in enumerate(text_blocks):
                lines = block.get("lines", [])
                spans = list(itertools.chain.from_iterable(l.get("spans", []) for l in lines))
                if not spans:
                    continue
                text = _normalise_text("\n".join(span.get("text", "") for span in spans))
                if not text.strip():
                    continue

                fonts = [span.get("font", "") for span in spans]
                font_sizes = [float(span.get("size", 0.0)) for span in spans]
                bbox = _compute_block_bbox(spans)
                line_heights = [span.get("bbox", [0, 0, 0, 0])[3] - span.get("bbox", [0, 0, 0, 0])[1] for span in spans]
                line_height = float(sum(line_heights) / max(len(line_heights), 1))
                font_name = _most_common(fonts)
                font_size = float(sum(font_sizes) / max(len(font_sizes), 1))

                block_id = f"{page_index}-{block_index}"
                has_border = bool(block.get("flags", 0) & 2)
                has_shading = bool(block.get("flags", 0) & 4)

                layout_block = Block(
                    block_id=block_id,
                    text=text,
                    font_name=font_name,
                    font_size=font_size,
                    line_height=line_height,
                    bbox=bbox,
                    page_num=page_index,
                    page_width=width,
                    page_height=height,
                    has_border=has_border,
                    has_shading=has_shading,
                )

                page_blocks[page_index].append(layout_block)

                if _is_top_band(layout_block, height, self.config.top_bottom_exclusion_ratio):
                    header_tracker[page_index].append(text)
                    header_counters[text] += 1
                if _is_bottom_band(layout_block, height, self.config.top_bottom_exclusion_ratio):
                    footer_tracker[page_index].append(text)
                    footer_counters[text] += 1

        headers_to_suppress = _collect_repeats(
            header_counters, self.config.header_footer_repeat_threshold, self.config.max_repeat_candidates
        )
        footers_to_suppress = _collect_repeats(
            footer_counters, self.config.header_footer_repeat_threshold, self.config.max_repeat_candidates
        )

        kept_blocks: List[Block] = []
        for page_index, blocks in page_blocks.items():
            for block in blocks:
                text = block.text.strip()
                if text in headers_to_suppress and text in header_tracker.get(page_index, []):
                    continue
                if text in footers_to_suppress and text in footer_tracker.get(page_index, []):
                    continue
                kept_blocks.append(block)

        kept_blocks.sort(key=lambda b: (b.page_num, b.y0, b.x0))
        LOGGER.debug("Extracted %s blocks for doc_id=%s", len(kept_blocks), doc_id)
        return DocumentLayout(doc_id=doc_id, blocks=kept_blocks, page_sizes=page_sizes)


def _collect_repeats(counter: Counter[str], threshold: int, limit: int) -> set[str]:
    """Return strings that appear on at least ``threshold`` pages."""

    return {text for text, count in counter.most_common(limit) if count >= threshold}


def _most_common(values: Sequence[str]) -> str:
    if not values:
        return ""
    counts = Counter(values)
    return counts.most_common(1)[0][0]


def _compute_block_bbox(spans: Iterable[Dict[str, Iterable[float]]]) -> BBox:
    x0, y0, x1, y1 = None, None, None, None
    for span in spans:
        bbox = span.get("bbox", [0, 0, 0, 0])
        if x0 is None or bbox[0] < x0:
            x0 = bbox[0]
        if y0 is None or bbox[1] < y0:
            y0 = bbox[1]
        if x1 is None or bbox[2] > x1:
            x1 = bbox[2]
        if y1 is None or bbox[3] > y1:
            y1 = bbox[3]
    return (float(x0 or 0.0), float(y0 or 0.0), float(x1 or 0.0), float(y1 or 0.0))


def _normalise_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\u00ad", "").splitlines())


def _is_top_band(block: Block, height: float, ratio: float) -> bool:
    return block.y0 <= height * ratio


def _is_bottom_band(block: Block, height: float, ratio: float) -> bool:
    return block.y1 >= height * (1 - ratio)

