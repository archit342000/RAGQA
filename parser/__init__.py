"""Public entrypoint for the layout aware document parser."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .classifier import classify_blocks
try:  # pragma: no cover - optional dependency
    import fitz  # type: ignore
except Exception:  # pragma: no cover - keep runtime optional
    fitz = None

from .detector import LayoutDetector
from .exporter import export_docblocks
from .extractor import BaseExtractor, ExtractionPipeline, build_pipeline
from .merger import merge_layouts
from .post_pass_repair import layout_repair
from .pp_structure import pp_structure_layout, tag_blocks_with_regions
from .region_assigner import assign_regions
from .stitcher import stitch_predictions
from .utils import load_config


class DocumentParser:
    """High level orchestrator that produces DocBlocks."""

    def __init__(
        self,
        config: Optional[Dict[str, object]] = None,
        cheap_extractor: Optional[BaseExtractor] = None,
        strong_extractor: Optional[BaseExtractor] = None,
    ) -> None:
        self.config = config or load_config()
        self.pipeline: ExtractionPipeline = build_pipeline(
            config=self.config,
            cheap_extractor=cheap_extractor,
            strong_extractor=strong_extractor,
        )
        self.detector = LayoutDetector(self.config)

    def parse(self, pdf_path: Path) -> List[Dict[str, object]]:
        results = self.pipeline.run(pdf_path)
        cheap_doc = results["cheap"].document
        strong_doc = results["strong"].document if results.get("strong") else None
        merged = merge_layouts(cheap_doc, strong_doc, self.config)
        escalation = self.config.get("escalation", {})
        use_pp = escalation.get("use_pp_structure", True) if isinstance(escalation, dict) else True
        detector_doc = None
        if fitz is not None:
            try:  # pragma: no cover - heavy runtime path
                detector_doc = fitz.open(pdf_path)
            except Exception:
                detector_doc = None
        try:
            for page in merged.pages:
                detections = []
                if detector_doc is not None:
                    try:
                        detections = self.detector.detect_page(detector_doc, page.page_number)
                    except Exception:
                        detections = []
                page.meta["detector_regions"] = detections
                if detections:
                    assign_regions(page, detections, allow_override=True)
                if use_pp:
                    regions = pp_structure_layout(page, pdf_path=pdf_path, cfg=self.config)
                    if detections:
                        missing = [blk for blk in page.blocks if not blk.attrs.get("region_tag")]
                        if missing:
                            tag_blocks_with_regions(missing, regions)
                    else:
                        tag_blocks_with_regions(page.blocks, regions)
        finally:
            if detector_doc is not None:
                detector_doc.close()
        predictions = classify_blocks(merged, config=self.config)
        repaired = layout_repair(predictions, config=self.config)
        stitched = stitch_predictions(repaired, config=self.config)
        return export_docblocks(stitched, config=self.config)


def parse_pdf(
    pdf_path: Path,
    config: Optional[Dict[str, object]] = None,
    cheap_extractor: Optional[BaseExtractor] = None,
    strong_extractor: Optional[BaseExtractor] = None,
) -> List[Dict[str, object]]:
    parser = DocumentParser(
        config=config,
        cheap_extractor=cheap_extractor,
        strong_extractor=strong_extractor,
    )
    return parser.parse(pdf_path)
