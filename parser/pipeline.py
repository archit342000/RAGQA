"""High-level parsing pipeline coordinating layout stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:  # pragma: no cover - optional dependency for integration tests
    import fitz  # type: ignore
except Exception:  # pragma: no cover - PyMuPDF might be unavailable in CI
    fitz = None

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


def _iter_pdf_pages(pdf_path: Path, dpi: int) -> Iterable[PageInput]:  # pragma: no cover - heavy I/O
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is required to load PDF files; install the 'pymupdf' package"
        )

    doc = fitz.open(pdf_path)
    for index, page in enumerate(doc):
        text_dict = page.get_text("dict")
        spans: List[Span] = []
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text:
                        continue
                    bbox = tuple(float(v) for v in span.get("bbox", (0, 0, 0, 0)))
                    flags = int(span.get("flags", 0))
                    spans.append(
                        Span(
                            text=text,
                            bbox=bbox,  # coordinates remain in PDF space
                            font_size=float(span.get("size", 0.0)),
                            font_name=span.get("font", ""),
                            bold=bool(flags & 2),
                            italic=bool(flags & 1),
                        )
                    )

        page_id = f"p_{index + 1:04d}"
        meta = {
            "dpi": dpi,
            "rotation": getattr(page, "rotation", 0),
            "number": index,
        }
        yield PageInput(
            page_id=page_id,
            number=index + 1,
            width=float(page.rect.width),
            height=float(page.rect.height),
            spans=spans,
            meta=meta,
        )


def load_pages(source, dpi: int) -> List[PageInput]:
    """Normalise page input for the downstream pipeline."""

    if isinstance(source, list):
        return list(source)

    if isinstance(source, Sequence) and all(isinstance(p, PageInput) for p in source):
        return list(source)

    if isinstance(source, (str, Path)):
        pdf_path = Path(source)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        return list(_iter_pdf_pages(pdf_path, dpi))

    raise TypeError(
        "Unsupported source type for parse_document; expected list of PageInput or path-like object"
    )


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
