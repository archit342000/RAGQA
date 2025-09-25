"""Public entrypoint for the layout aware document parser."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .classifier import classify_blocks
from .exporter import export_docblocks
from .extractor import BaseExtractor, ExtractionPipeline, build_pipeline
from .merger import merge_layouts
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

    def parse(self, pdf_path: Path) -> List[Dict[str, object]]:
        results = self.pipeline.run(pdf_path)
        cheap_doc = results["cheap"].document
        strong_doc = results["strong"].document if results.get("strong") else None
        merged = merge_layouts(cheap_doc, strong_doc, self.config)
        predictions = classify_blocks(merged, config=self.config)
        stitched = stitch_predictions(predictions, config=self.config)
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
