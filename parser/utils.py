"""Utility classes and helpers for the layout aware parser.

The module provides light‑weight dataclasses used throughout the
pipeline.  They are intentionally Python only so the tests can inject
synthetic layouts without having to talk to the actual PDF
implementations.  The real extractors can populate the same structures
when PyMuPDF or a strong extractor is available at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import copy
import statistics

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without PyYAML
    yaml = None

BBox = Tuple[float, float, float, float]


DEFAULT_CONFIG: Dict[str, object] = {
    "thresholds": {
        "tau_main": 0.60,
        "tau_main_page_confident": 0.55,
        "tau_heading_h12": 0.75,
        "tau_heading_h3": 0.60,
    },
    "bands": {
        "margin_x_pct": [0.0, 0.12, 0.88, 1.0],
        "footer_y_pct": 0.88,
        "header_y_pct": 0.12,
    },
    "page_turn": {
        "nextpage_top_window_pct": {"caption": 0.15, "sidenote": 0.20, "callout": 0.25},
        "footnote_deadline": "same_page",
    },
    "wrapping": {
        "min_image_width_pct": 0.25,
        "wrap_text_width_max_pct_of_col": 0.60,
        "symmetric_band_tolerance_pct": 0.10,
    },
    "limits": {"max_aux_per_anchor_per_type": 2},
    "escalation": {
        "bootstrap_pages": 2,
        "per_section_bootstrap": 1,
        "lcs_tau_page": 0.38,
        "lcs_tau_doc": 0.34,
        "soft_signal_quorum": 2,
        "strong_signal_quorum": 1,
        "force_escalate_if_main_pct_below": 0.60,
    },
    "ocr": {
        "enable_on_low_yield": True,
        "enable_on_font_anomaly": True,
        "dpi": 250,
    },
}


@dataclass
class Span:
    """Low level span as produced by PyMuPDF like extractors."""

    text: str
    font: str
    size: float
    bbox: BBox
    flags: Dict[str, bool] = field(default_factory=dict)

    @property
    def bold(self) -> bool:
        return bool(self.flags.get("bold"))

    @property
    def smallcaps(self) -> bool:
        return bool(self.flags.get("smallcaps"))


@dataclass
class Line:
    spans: List[Span]
    bbox: BBox

    @property
    def text(self) -> str:
        return "".join(span.text for span in self.spans).strip()

    @property
    def font_sizes(self) -> List[float]:
        return [span.size for span in self.spans]

    @property
    def fonts(self) -> List[str]:
        return [span.font for span in self.spans]


@dataclass
class Block:
    lines: List[Line]
    bbox: BBox
    block_type: str = "text"
    id_hint: Optional[str] = None
    attrs: Dict[str, float | str | int | bool] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text).strip()

    @property
    def font_sizes(self) -> List[float]:
        sizes: List[float] = []
        for line in self.lines:
            sizes.extend(line.font_sizes)
        return sizes

    @property
    def fonts(self) -> List[str]:
        fonts: List[str] = []
        for line in self.lines:
            fonts.extend(line.fonts)
        return fonts

    @property
    def width(self) -> float:
        x0, _, x1, _ = self.bbox
        return max(0.0, x1 - x0)

    @property
    def height(self) -> float:
        _, y0, _, y1 = self.bbox
        return max(0.0, y1 - y0)

    @property
    def top(self) -> float:
        _, y0, _, _ = self.bbox
        return y0

    @property
    def bottom(self) -> float:
        _, _, _, y1 = self.bbox
        return y1

    @property
    def left(self) -> float:
        x0, _, _, _ = self.bbox
        return x0

    @property
    def right(self) -> float:
        _, _, x1, _ = self.bbox
        return x1

    def copy(self) -> "Block":
        return replace(self, lines=list(self.lines), attrs=dict(self.attrs))


@dataclass
class PageLayout:
    page_number: int
    width: float
    height: float
    blocks: List[Block]
    meta: Dict[str, float | str | int | bool] = field(default_factory=dict)

    def iter_blocks(self) -> Iterator[Block]:
        for block in self.blocks:
            yield block


@dataclass
class DocumentLayout:
    pages: List[PageLayout]
    meta: Dict[str, float | str | int | bool] = field(default_factory=dict)

    def iter_blocks(self) -> Iterator[Tuple[int, Block]]:
        for page in self.pages:
            for block in page.blocks:
                yield page.page_number, block


@dataclass
class DocBlock:
    id: str
    page: int
    kind: str
    aux_type: Optional[str]
    text: str
    bbox: List[float]
    flow: str
    ms: float
    hs: Optional[float]
    reason: List[str]
    anchor_to: Optional[str]
    attached_across_pages: bool
    confidence: float
    meta: Dict[str, float | str | int | bool]

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "id": self.id,
            "page": self.page,
            "kind": self.kind,
            "aux_type": self.aux_type,
            "text": self.text,
            "bbox": self.bbox,
            "flow": self.flow,
            "ms": self.ms,
            "hs": self.hs,
            "reason": self.reason,
            "anchor_to": self.anchor_to,
            "attached_across_pages": self.attached_across_pages,
            "confidence": self.confidence,
            "meta": self.meta,
        }
        return payload


def load_config(path: Optional[Path] = None) -> Dict[str, object]:
    """Load configuration from YAML."""

    if path is None:
        path = Path(__file__).with_name("config.yaml")
    if yaml is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def bbox_area(bbox: BBox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_union(b1: BBox, b2: BBox) -> BBox:
    x0 = min(b1[0], b2[0])
    y0 = min(b1[1], b2[1])
    x1 = max(b1[2], b2[2])
    y1 = max(b1[3], b2[3])
    return (x0, y0, x1, y1)


def normalize_bbox(bbox: BBox, width: float, height: float) -> List[float]:
    x0, y0, x1, y1 = bbox
    return [x0 / width, y0 / height, x1 / width, y1 / height]


def text_density(block: Block) -> float:
    area = bbox_area(block.bbox)
    if area <= 0:
        return 0.0
    chars = sum(len(span.text.strip()) for line in block.lines for span in line.spans)
    return chars / area


def line_spacing(block: Block) -> float:
    if len(block.lines) < 2:
        return 0.0
    spacings = []
    prev_bottom = block.lines[0].bbox[3]
    for line in block.lines[1:]:
        top = line.bbox[1]
        spacings.append(max(0.0, top - prev_bottom))
        prev_bottom = line.bbox[3]
    if not spacings:
        return 0.0
    return statistics.mean(spacings)


def font_stats(block: Block) -> Tuple[float, float]:
    sizes = block.font_sizes
    if not sizes:
        return 0.0, 0.0
    return statistics.mean(sizes), statistics.pstdev(sizes) if len(sizes) > 1 else 0.0


def estimate_indent(block: Block) -> float:
    if not block.lines:
        return 0.0
    x0, _, _, _ = block.lines[0].bbox
    return x0 - block.left


def is_all_caps(text: str) -> bool:
    candidate = [c for c in text if c.isalpha()]
    if not candidate:
        return False
    return all(c.isupper() for c in candidate)


def ends_with_terminal_punct(text: str) -> bool:
    return text.rstrip().endswith((".", "?", "!"))


def first_token(text: str) -> str:
    stripped = text.lstrip()
    if not stripped:
        return ""
    return stripped.split()[0]


def detect_footnote_marker(text: str) -> bool:
    token = first_token(text)
    if not token:
        return False
    if token[0].isdigit():
        return True
    if token[0] in {"*", "†", "‡", "§"}:
        return True
    return False


def whitespace_halo(page: PageLayout, block: Block) -> Tuple[float, float]:
    """Approximate white space before/after a block."""

    above = 0.0
    below = 0.0
    sorted_blocks = sorted(page.blocks, key=lambda b: b.top)
    idx = sorted_blocks.index(block)
    if idx > 0:
        above = max(0.0, block.top - sorted_blocks[idx - 1].bottom)
    else:
        above = block.top
    if idx < len(sorted_blocks) - 1:
        below = max(0.0, sorted_blocks[idx + 1].top - block.bottom)
    else:
        below = page.height - block.bottom
    return above, below


class LayoutComplexityAnalyzer:
    """Compute layout complexity score and escalation hints."""

    def __init__(self, config: Dict[str, object]):
        self.config = config
        escalation = config.get("escalation", {})
        self.page_tau = float(escalation.get("lcs_tau_page", 0.38))
        self.doc_tau = float(escalation.get("lcs_tau_doc", 0.34))
        self.soft_quorum = int(escalation.get("soft_signal_quorum", 2))
        self.strong_quorum = int(escalation.get("strong_signal_quorum", 1))
        self.bootstrap_pages = int(escalation.get("bootstrap_pages", 2))
        self.force_main_pct = float(
            escalation.get("force_escalate_if_main_pct_below", 0.60)
        )

    def analyze(self, doc: DocumentLayout) -> Dict[str, object]:
        page_scores: Dict[int, float] = {}
        page_soft: Dict[int, List[str]] = {}
        page_strong: Dict[int, List[str]] = {}
        main_ratios: Dict[int, float] = {}
        for page in doc.pages:
            soft, strong = self._signals(page)
            score = self._score(page, soft, strong)
            page_scores[page.page_number] = score
            page_soft[page.page_number] = soft
            page_strong[page.page_number] = strong
            main_ratio = self._approx_main_ratio(page)
            main_ratios[page.page_number] = main_ratio
        doc_score = statistics.mean(page_scores.values()) if page_scores else 0.0
        escalate_pages = set()
        for page in doc.pages:
            page_no = page.page_number
            if page_no < self.bootstrap_pages:
                escalate_pages.add(page_no)
                continue
            soft = page_soft[page_no]
            strong = page_strong[page_no]
            if len(strong) >= self.strong_quorum:
                escalate_pages.add(page_no)
                continue
            if len(soft) >= self.soft_quorum:
                escalate_pages.add(page_no)
                continue
            if page_scores[page_no] >= self.page_tau:
                escalate_pages.add(page_no)
                continue
            if main_ratios[page_no] < self.force_main_pct:
                escalate_pages.add(page_no)
                continue
        if doc_score >= self.doc_tau:
            escalate_pages.update(page_scores.keys())
        return {
            "page_scores": page_scores,
            "page_soft": page_soft,
            "page_strong": page_strong,
            "doc_score": doc_score,
            "escalate_pages": sorted(escalate_pages),
        }

    def _approx_main_ratio(self, page: PageLayout) -> float:
        if not page.blocks:
            return 0.0
        text_blocks = [b for b in page.blocks if b.block_type == "text"]
        return len(text_blocks) / len(page.blocks)

    def _score(self, page: PageLayout, soft: Sequence[str], strong: Sequence[str]) -> float:
        multi_column = 0.0
        col_ids = {
            int(block.attrs.get("col_id", 0))
            for block in page.blocks
            if "col_id" in block.attrs
        }
        if len(col_ids) >= 2:
            multi_column = 0.15 * (len(col_ids) - 1)
        tables = 0.2 * sum(1 for b in page.blocks if b.block_type == "table")
        aux = 0.1 * sum(1 for b in page.blocks if b.block_type != "text")
        density_var = statistics.pstdev(
            [text_density(b) for b in page.blocks if text_density(b) > 0]
        ) if page.blocks else 0.0
        density_term = min(0.2, density_var)
        score = multi_column + tables + aux + density_term
        score += 0.05 * len(soft) + 0.1 * len(strong)
        return score

    def _signals(self, page: PageLayout) -> Tuple[List[str], List[str]]:
        soft: List[str] = []
        strong: List[str] = []
        col_ids = {
            int(block.attrs.get("col_id", 0))
            for block in page.blocks
            if "col_id" in block.attrs
        }
        if len(col_ids) >= 2:
            soft.append("multi_column")
        footnotes = [b for b in page.blocks if detect_footnote_marker(b.text)]
        if len(footnotes) >= 2:
            soft.append("many_footnotes")
        wrap_candidates = [
            b for b in page.blocks if b.block_type == "text" and b.width < 0.4 * page.width
        ]
        if len(wrap_candidates) >= 2:
            soft.append("narrow_columns")
        for block in page.blocks:
            if block.block_type in {"table", "figure"}:
                strong.append(block.block_type)
            if block.attrs.get("rotated"):
                strong.append("rotated")
            if block.attrs.get("ocr"):
                strong.append("ocr")
        return soft, strong


def compute_column_id(block: Block, page_width: float, bands: Sequence[float]) -> int:
    x0 = block.left / page_width
    x1 = block.right / page_width
    if x1 <= bands[1]:
        return 0
    if x0 >= bands[2]:
        return 2
    return 1


def symmetric_band(block: Block, page_width: float, tolerance: float) -> bool:
    left = block.left / page_width
    right = 1.0 - (block.right / page_width)
    return abs(left - right) <= tolerance


def token_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def slugify(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_").lower()


def iter_blocks_in_reading_order(doc: DocumentLayout) -> Iterator[Tuple[PageLayout, Block]]:
    for page in sorted(doc.pages, key=lambda p: p.page_number):
        for block in sorted(page.blocks, key=lambda b: (b.left, b.top)):
            yield page, block
