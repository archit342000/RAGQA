"""Fuse LayoutParser detections with PyMuPDF blocks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from pipeline.ingest.pdf_parser import DocumentGraph, PDFBlock, PageGraph
from pipeline.layout.router import LayoutRoutingPlan
from pipeline.layout.signals import PageLayoutSignals

logger = logging.getLogger(__name__)

try:  # Optional heavy dependency.
    import layoutparser as lp  # type: ignore
except Exception:  # pragma: no cover - gracefully handle at runtime
    lp = None  # type: ignore

BBox = Tuple[float, float, float, float]


@dataclass(slots=True)
class LPRegion:
    bbox: BBox
    label: str
    score: float
    order: int


@dataclass(slots=True)
class FusedBlock:
    block_id: str
    page_number: int
    text: str
    bbox: BBox
    block_type: str  # "main" or "aux"
    region_label: str
    aux_category: Optional[str]
    anchor: Optional[str]
    column: Optional[int]
    char_start: int
    char_end: int
    avg_font_size: float
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def has_anchor(self) -> bool:
        return bool(self.anchor)


@dataclass(slots=True)
class FusedPage:
    page_number: int
    width: float
    height: float
    main_flow: List[FusedBlock]
    auxiliaries: List[FusedBlock]


@dataclass(slots=True)
class FusedDocument:
    doc_id: str
    pages: List[FusedPage]
    block_index: Dict[str, FusedBlock]


class LayoutParserEngine:
    """Thin wrapper around LayoutParser with caching and graceful fallback."""

    def __init__(self, cache_dir: Optional[Path] = None, max_batch: int = 6) -> None:
        self.cache: Dict[Tuple[str, int, str, int], Tuple[LPRegion, ...]] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_batch = max(1, max_batch)
        self._models: Dict[Tuple[str, int], object] = {}
        self._lp_available = lp is not None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _heuristic_regions(self, page: PageGraph) -> List[LPRegion]:
        regions: List[LPRegion] = []
        order = 0
        for block in page.blocks:
            label = block.class_hint if block.class_hint else ("text" if block.is_text else "figure")
            regions.append(LPRegion(bbox=block.bbox, label=label, score=0.5, order=order))
            order += 1
        return regions

    def _run_layout_parser(self, page: PageGraph, model_name: str, dpi: int) -> List[LPRegion]:
        if not self._lp_available:
            raise RuntimeError("layoutparser is not available")
        key = (model_name, dpi)
        if key not in self._models:
            try:
                if model_name == "docbank":
                    detector = lp.Detectron2LayoutModel("lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config")
                elif model_name == "prima":
                    detector = lp.Detectron2LayoutModel("lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config")
                else:
                    detector = lp.Detectron2LayoutModel("lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config")
            except Exception as exc:  # pragma: no cover - dependency heavy
                raise RuntimeError(f"LayoutParser model load failed: {exc}") from exc
            self._models[key] = detector
        detector = self._models[key]
        try:
            pix = page.raw_dict.get("pixmap")
            if pix is None:
                raise RuntimeError("Page image unavailable for LayoutParser")
            layout = detector.detect(pix)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - runtime guard
            raise RuntimeError(f"LayoutParser inference failed: {exc}") from exc
        regions: List[LPRegion] = []
        for order, region in enumerate(layout):
            bbox = (float(region.block.x_1), float(region.block.y_1), float(region.block.x_2), float(region.block.y_2))
            regions.append(LPRegion(bbox=bbox, label=str(region.type).lower(), score=float(region.score), order=order))
        return regions

    def detect(
        self,
        document: DocumentGraph,
        page_numbers: Sequence[int],
        *,
        model: str,
        dpi: int,
    ) -> Dict[int, List[LPRegion]]:
        outputs: Dict[int, List[LPRegion]] = {}
        for page_number in page_numbers:
            key = (document.doc_id, page_number, model, dpi)
            if key in self.cache:
                outputs[page_number] = [replace(region) for region in self.cache[key]]
                continue
            page = document.get_page(page_number)
            try:
                regions = self._run_layout_parser(page, model, dpi)
            except Exception as exc:
                logger.debug("LayoutParser unavailable for page %s: %s", page_number, exc)
                regions = self._heuristic_regions(page)
            self.cache[key] = tuple(regions)
            outputs[page_number] = [replace(region) for region in regions]
        return outputs


def _iou(a: BBox, b: BBox) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
    union = area_a + area_b - intersection
    return max(0.0, intersection / union)


def _assign_regions(page: PageGraph, regions: Sequence[LPRegion], threshold: float = 0.3) -> Dict[str, LPRegion]:
    mapping: Dict[str, LPRegion] = {}
    for block in page.blocks:
        best_region: Optional[LPRegion] = None
        best_iou = 0.0
        for region in regions:
            score = _iou(block.bbox, region.bbox)
            if score >= threshold and score > best_iou:
                best_iou = score
                best_region = region
        if best_region is not None:
            mapping[block.block_id] = best_region
    return mapping


def _is_off_grid(block: PDFBlock, page: PageGraph, column_assignments: Mapping[str, int]) -> bool:
    column = column_assignments.get(block.block_id)
    if column is None:
        left_margin = block.bbox[0] / max(page.width, 1.0)
        right_margin = (page.width - block.bbox[2]) / max(page.width, 1.0)
        return left_margin < 0.05 or right_margin < 0.05 or block.width / max(page.width, 1.0) < 0.25
    return False


def _determine_aux_category(
    block: PDFBlock,
    region: Optional[LPRegion],
    page: PageGraph,
    signal: PageLayoutSignals,
) -> Tuple[bool, Optional[str]]:
    region_label = (region.label if region else block.class_hint or "text").lower()
    if not block.is_text:
        return True, region_label if region else "figure"
    if region_label in {"table", "figure", "list"}:
        return True, region_label
    if region_label == "title" and _is_off_grid(block, page, signal.extras.column_assignments):
        return True, "title"
    if _is_off_grid(block, page, signal.extras.column_assignments) and block.avg_font_size < signal.extras.dominant_font_size * 0.85:
        return True, "off_grid"
    return False, None


def _attach_anchor(text: str, marker: str) -> Tuple[str, bool]:
    stripped = text.rstrip()
    tail_len = len(text) - len(stripped)
    text = stripped
    for idx in range(len(text) - 1, -1, -1):
        if text[idx] in {".", "?", "!"}:
            return text[: idx + 1] + " " + marker + text[idx + 1 :] + (" " * tail_len), True
    return text + (" " if text else "") + marker + (" " * tail_len), False


def fuse_layout(
    document: DocumentGraph,
    signals: Sequence[PageLayoutSignals],
    routing_plan: LayoutRoutingPlan,
    *,
    engine: Optional[LayoutParserEngine] = None,
) -> FusedDocument:
    if len(document.pages) != len(signals):
        raise ValueError("Signals must align with document pages")

    engine = engine or LayoutParserEngine()
    pages_by_model: Dict[Tuple[str, int], List[int]] = {}
    for decision in routing_plan.decisions:
        if decision.use_layout_parser:
            pages_by_model.setdefault((decision.model, decision.dpi), []).append(decision.page_number)

    lp_outputs: Dict[int, List[LPRegion]] = {}
    for (model, dpi), page_numbers in pages_by_model.items():
        batches = [page_numbers[i : i + engine.max_batch] for i in range(0, len(page_numbers), engine.max_batch)]
        for batch in batches:
            lp_outputs.update(engine.detect(document, batch, model=model, dpi=dpi))

    pages: List[FusedPage] = []
    index: Dict[str, FusedBlock] = {}

    for page, signal in zip(document.pages, signals):
        regions = lp_outputs.get(page.page_number)
        if regions is None:
            regions = engine._heuristic_regions(page)
        region_mapping = _assign_regions(page, regions)
        main_blocks: List[FusedBlock] = []
        aux_blocks: List[FusedBlock] = []

        for block in page.blocks:
            if block.block_type == "text" and not block.text:
                continue
            region = region_mapping.get(block.block_id)
            is_aux, aux_category = _determine_aux_category(block, region, page, signal)
            region_label = (region.label if region else block.class_hint or "text").lower()
            fused = FusedBlock(
                block_id=block.block_id,
                page_number=page.page_number,
                text=block.text,
                bbox=block.bbox,
                block_type="aux" if is_aux else "main",
                region_label=region_label,
                aux_category=aux_category,
                anchor=None,
                column=signal.extras.column_assignments.get(block.block_id),
                char_start=block.char_start,
                char_end=block.char_start + len(block.text),
                avg_font_size=block.avg_font_size,
                metadata={
                    "lp_score": getattr(region, "score", 0.0),
                    "source_block_type": block.block_type,
                },
            )
            if is_aux:
                aux_blocks.append(fused)
            else:
                main_blocks.append(fused)
            index[fused.block_id] = fused

        def order_key(block: FusedBlock) -> Tuple[int, float, float]:
            region = region_mapping.get(block.block_id)
            if region is not None:
                return (region.order, block.bbox[1], block.bbox[0])
            column = block.column if block.column is not None else 99
            return (len(regions) + column * 100, block.bbox[1], block.bbox[0])

        main_blocks.sort(key=order_key)
        aux_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))

        anchor_idx = 1
        for aux in aux_blocks:
            if not main_blocks:
                break
            preceding = None
            for block in main_blocks:
                if block.bbox[1] <= aux.bbox[1]:
                    preceding = block
                else:
                    break
            if preceding is None:
                preceding = main_blocks[0]
            marker = f"[AUX-{page.page_number}-{anchor_idx}]"
            updated_text, _ = _attach_anchor(preceding.text, marker)
            preceding.text = updated_text
            preceding.char_end = preceding.char_start + len(preceding.text)
            anchors = preceding.metadata.setdefault("anchors", [])
            if isinstance(anchors, list):
                anchors.append(marker)
            preceding.metadata["has_anchor_refs"] = True
            aux.anchor = marker
            aux.metadata["anchored_to"] = preceding.block_id
            anchor_idx += 1

        pages.append(
            FusedPage(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                main_flow=main_blocks,
                auxiliaries=aux_blocks,
            )
        )

    return FusedDocument(doc_id=document.doc_id, pages=pages, block_index=index)


__all__ = [
    "FusedDocument",
    "FusedPage",
    "FusedBlock",
    "LayoutParserEngine",
    "fuse_layout",
]
