"""High-level parsing pipeline coordinating layout stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .anchoring import anchor_captions
from .fsm import FlowState, fsm_stitch_and_buffer
from .grouping import Span, group_spans
from .layout_detector import detect_layout
from .reading_order import order_reading
from .region_fusion import fuse_regions
from .rules_v2 import ClassifierState, rules_v2_classify


@dataclass
class PageInput:
    page_id: str
    number: int
    width: float
    height: float
    spans: List[Span]
    meta: Dict[str, object] = field(default_factory=dict)


def load_pages(source, dpi: int) -> List[PageInput]:
    """Load pages from a synthetic list or raise for real PDFs (out of scope)."""

    if isinstance(source, list):
        return list(source)
    raise RuntimeError("PDF loading not implemented in the reference tests")


def parse_document(source, cfg: Dict[str, object]) -> Dict[str, object]:
    """Parse a document into the normalised DB schema."""

    dpi = int(cfg.get("layout_model", {}).get("dpi", 360)) if isinstance(cfg.get("layout_model"), dict) else 360
    pages = load_pages(source, dpi)
    classifier_state = ClassifierState()
    flow_state = FlowState()
    outputs: List[Dict[str, object]] = []

    for page in pages:
        regions = detect_layout(None, cfg.get("layout_model", {}), page.meta)
        _, blocks = group_spans(page.spans, cfg, page_width=page.width)
        for idx, block in enumerate(blocks):
            block.block_id = f"{page.page_id}_blk_{idx:03d}"
            block.meta.setdefault("page_width", page.width)
            block.meta.setdefault("page_height", page.height)
        fused = fuse_regions(blocks, regions, cfg)
        ordered = order_reading(fused, cfg)
        classified = rules_v2_classify(page, ordered, regions, cfg, classifier_state)
        stitched = fsm_stitch_and_buffer(page.page_id, classified, flow_state, cfg)
        anchored_blocks = anchor_captions(stitched["blocks"], regions, cfg)
        stitched["blocks"] = anchored_blocks
        outputs.append(stitched)

    return {"pages": outputs}
