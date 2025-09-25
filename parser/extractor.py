"""Extraction pipeline orchestrating cheap and strong extractors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:  # pragma: no cover - optional runtime dependency
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be missing
    fitz = None

from .utils import (
    Block,
    DocumentLayout,
    LayoutComplexityAnalyzer,
    Line,
    PageLayout,
    Span,
    compute_column_id,
    load_config,
)


@dataclass
class ExtractionResult:
    document: DocumentLayout
    meta: Dict[str, object]


class BaseExtractor:
    """Common interface for PDF extractors."""

    def extract(self, pdf_path: Path, pages: Optional[Sequence[int]] = None) -> ExtractionResult:
        raise NotImplementedError


class CheapPDFExtractor(BaseExtractor):
    """Cheap extractor built on top of PyMuPDF.

    The class intentionally performs only light processing, returning the raw
    layout blocks without attempting heavy canonicalisation.  The heavy lifting
    is performed by the downstream modules which can also reconcile the output
    with a strong extractor when required.
    """

    def __init__(self, config: Optional[Dict[str, object]] = None):
        self.config = config or load_config()
        self.bands = self.config.get("bands", {}).get("margin_x_pct", [0, 0.5, 0.5, 1])

    def extract(self, pdf_path: Path, pages: Optional[Sequence[int]] = None) -> ExtractionResult:
        if fitz is None:  # pragma: no cover - runtime guard for environments without PyMuPDF
            raise RuntimeError("PyMuPDF is required for the cheap extractor")
        doc = fitz.open(pdf_path)  # pragma: no cover - heavy I/O path
        selected = set(pages) if pages is not None else None
        layouts: List[PageLayout] = []
        for idx, page in enumerate(doc):  # pragma: no cover - heavy I/O path
            page_number = idx
            if selected is not None and page_number not in selected:
                continue
            blocks: List[Block] = []
            raw_blocks = page.get_text("dict").get("blocks", [])
            for block in raw_blocks:
                if block.get("type") != 0:
                    block_type = "image" if block.get("type") == 1 else "other"
                    bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
                    blocks.append(Block(lines=[], bbox=bbox, block_type=block_type))
                    continue
                lines: List[Line] = []
                for line_dict in block.get("lines", []):
                    spans = [
                        Span(
                            text=span.get("text", ""),
                            font=span.get("font", ""),
                            size=float(span.get("size", 0.0)),
                            bbox=tuple(span.get("bbox", (0, 0, 0, 0))),
                            flags={
                                "bold": bool(span.get("flags", 0) & 2),
                                "smallcaps": span.get("font", "").isupper(),
                            },
                        )
                        for span in line_dict.get("spans", [])
                    ]
                    if not spans:
                        continue
                    line_bbox = tuple(line_dict.get("bbox", (0, 0, 0, 0)))
                    lines.append(Line(spans=spans, bbox=line_bbox))
                if not lines:
                    continue
                bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
                block_obj = Block(lines=lines, bbox=bbox, block_type="text")
                block_obj.attrs["col_id"] = compute_column_id(block_obj, page.rect.width, self.bands)
                blocks.append(block_obj)
            layouts.append(
                PageLayout(
                    page_number=page_number,
                    width=page.rect.width,
                    height=page.rect.height,
                    blocks=blocks,
                )
            )
        return ExtractionResult(document=DocumentLayout(pages=layouts), meta={"source": "cheap"})


class StrongExtractor(BaseExtractor):
    """Strong extractor facade.

    The class defers to either Docling, Unstructured or OCR, depending on what
    is available in the runtime.  For the unit tests the class is intentionally
    very small so it can be mocked easily.
    """

    def __init__(self, delegate: Optional[BaseExtractor] = None):
        self.delegate = delegate

    def extract(self, pdf_path: Path, pages: Optional[Sequence[int]] = None) -> ExtractionResult:
        if self.delegate is None:
            raise RuntimeError("Strong extractor not configured for this runtime")
        return self.delegate.extract(pdf_path, pages=pages)


class ExtractionPipeline:
    """High level pipeline that manages extractor escalation."""

    def __init__(
        self,
        config: Optional[Dict[str, object]] = None,
        cheap_extractor: Optional[BaseExtractor] = None,
        strong_extractor: Optional[BaseExtractor] = None,
    ) -> None:
        self.config = config or load_config()
        self.cheap = cheap_extractor or CheapPDFExtractor(self.config)
        self.strong = strong_extractor or StrongExtractor()
        self.analyzer = LayoutComplexityAnalyzer(self.config)

    def run(self, pdf_path: Path) -> Dict[str, object]:
        cheap_result = self.cheap.extract(pdf_path)
        analysis = self.analyzer.analyze(cheap_result.document)
        escalate_pages: List[int] = analysis["escalate_pages"]
        strong_result: Optional[ExtractionResult] = None
        if escalate_pages:
            try:
                strong_result = self.strong.extract(pdf_path, pages=escalate_pages)
            except RuntimeError:
                strong_result = None
        return {
            "cheap": cheap_result,
            "strong": strong_result,
            "analysis": analysis,
        }


def build_pipeline(
    config: Optional[Dict[str, object]] = None,
    cheap_extractor: Optional[BaseExtractor] = None,
    strong_extractor: Optional[BaseExtractor] = None,
) -> ExtractionPipeline:
    return ExtractionPipeline(
        config=config,
        cheap_extractor=cheap_extractor,
        strong_extractor=strong_extractor,
    )
