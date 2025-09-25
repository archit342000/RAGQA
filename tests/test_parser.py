from __future__ import annotations

from pathlib import Path

import pytest

from parser import DocumentParser
from parser.classifier import BlockPrediction
from parser.extractor import BaseExtractor, ExtractionResult, ExtractionPipeline
from parser.merger import merge_layouts
from parser.utils import Block, DocumentLayout, Line, PageLayout, Span


class FakeCheapExtractor(BaseExtractor):
    def __init__(self, document: DocumentLayout):
        self.document = document

    def extract(self, pdf_path: Path, pages=None):
        return ExtractionResult(document=self.document, meta={"source": "cheap"})


class FakeStrongExtractor(BaseExtractor):
    def __init__(self, document: DocumentLayout):
        self.document = document
        self.calls = []

    def extract(self, pdf_path: Path, pages=None):
        if pages is None:
            pages = []
        self.calls.append(sorted(pages))
        if pages:
            pages_set = set(pages)
            pages_out = [page for page in self.document.pages if page.page_number in pages_set]
        else:
            pages_out = self.document.pages
        return ExtractionResult(document=DocumentLayout(pages=pages_out), meta={"source": "strong"})


def make_span(text: str, bbox, font_size=11, font="Times", bold=False):
    return Span(text=text, font=font, size=font_size, bbox=bbox, flags={"bold": bold})


def make_block(text: str, bbox, font_size=11, block_type="text", col_id=1, bold=False):
    span = make_span(text, bbox, font_size=font_size, bold=bold)
    line = Line(spans=[span], bbox=bbox)
    block = Block(lines=[line], bbox=bbox, block_type=block_type)
    block.attrs["col_id"] = col_id
    return block


def build_test_document() -> DocumentLayout:
    width, height = 600, 800
    pages = []
    # Page 0
    page0_blocks = [
        make_block(
            "This is the first part of a paragraph that",
            (80, 160, 310, 220),
            col_id=0,
        ),
        make_block(
            "wraps",
            (320, 160, 350, 220),
            col_id=1,
        ),
        make_block(
            "continues after an image on the same page.",
            (360, 160, 580, 220),
            col_id=2,
        ),
        make_block(
            "Illustration block",
            (200, 200, 400, 400),
            block_type="figure",
            col_id=1,
        ),
    ]
    pages.append(PageLayout(page_number=0, width=width, height=height, blocks=page0_blocks))
    # Page 1
    page1_blocks = [
        make_block(
            "INTRODUCTION",
            (60, 60, 540, 110),
            font_size=18,
            bold=True,
            col_id=1,
        ),
        make_block(
            "It then carries on across the page break and",
            (80, 170, 520, 240),
            col_id=1,
        ),
        make_block(
            "Figure 1. Sample caption.",
            (80, 250, 520, 280),
            col_id=1,
        ),
        make_block(
            "1 This is a footnote note.",
            (60, 720, 540, 760),
            col_id=1,
        ),
    ]
    pages.append(PageLayout(page_number=1, width=width, height=height, blocks=page1_blocks))
    # Page 2
    page2_blocks = [
        make_block(
            "REMEMBER",
            (500, 120, 580, 200),
            col_id=2,
        ),
        make_block(
            "finishes two pages later.",
            (80, 170, 520, 240),
            col_id=1,
        ),
        make_block(
            "2 Another footnote.",
            (60, 720, 540, 760),
            col_id=1,
        ),
    ]
    pages.append(PageLayout(page_number=2, width=width, height=height, blocks=page2_blocks))
    # Page 3 with orphaned footnote
    page3_blocks = [
        make_block(
            "3 Footnote on empty page.",
            (60, 720, 540, 760),
            col_id=1,
        ),
    ]
    pages.append(PageLayout(page_number=3, width=width, height=height, blocks=page3_blocks))
    return DocumentLayout(pages=pages)


def test_escalation_bootstrap_and_signals():
    document = build_test_document()
    cheap = FakeCheapExtractor(document)
    strong = FakeStrongExtractor(document)
    pipeline = ExtractionPipeline(config=None, cheap_extractor=cheap, strong_extractor=strong)
    pipeline.run(Path("dummy.pdf"))
    assert strong.calls, "Strong extractor must be invoked"
    assert strong.calls[0] == [0, 1]


def test_document_parser_outputs_normalized_docblocks():
    document = build_test_document()
    cheap = FakeCheapExtractor(document)
    strong = FakeStrongExtractor(document)
    parser = DocumentParser(cheap_extractor=cheap, strong_extractor=strong)
    docblocks = parser.parse(Path("dummy.pdf"))
    # Ensure wraparound block merged into neighbour
    wrap_text_blocks = [block for block in docblocks if "wraps" in block["text"]]
    assert len(wrap_text_blocks) == 1
    assert "This is the first part" in wrap_text_blocks[0]["text"]
    # Headings must survive and be marked
    headings = [block for block in docblocks if block["meta"].get("heading_level")]
    assert headings and headings[0]["kind"] == "main"
    # Stitching across pages with TTL must keep paragraph intact
    stitched = [block for block in docblocks if "finishes two pages later" in block["text"]]
    assert stitched and "It then carries" in stitched[0]["text"]
    assert stitched[0]["attached_across_pages"] is True
    # Aux buffering: caption and footnotes anchor to previous main block
    caption = next(block for block in docblocks if block["aux_type"] == "caption")
    assert caption["anchor_to"], "Caption should anchor to a main block"
    footnote = next(block for block in docblocks if block["text"].startswith("1 This"))
    assert footnote["anchor_to"], "Footnote should anchor"
    orphan_footnote = next(block for block in docblocks if block["text"].startswith("3 Footnote"))
    assert orphan_footnote["anchor_to"], "Orphan footnote must fallback to previous page"
    # Aux leakage guard
    aux_main = [block for block in docblocks if block["kind"] == "main" and "Figure" in block["text"]]
    assert not aux_main, "Caption text must not leak into main flow"
    # Normalised bbox between 0 and 1
    for block in docblocks:
        x0, y0, x1, y1 = block["bbox"]
        assert 0.0 <= x0 <= 1.0
        assert 0.0 <= y0 <= 1.0
        assert 0.0 <= x1 <= 1.0
        assert 0.0 <= y1 <= 1.0
        assert x0 <= x1 and y0 <= y1
