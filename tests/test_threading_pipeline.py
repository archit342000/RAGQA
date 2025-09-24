from __future__ import annotations

from pipeline.audit import thread_audit
from pipeline.ingest import pdf_parser
from pipeline.ingest.pdf_parser import DocumentGraph, PageGraph, PDFBlock, PDFLine, PDFSpan
from pipeline.layout.aux_detection import detect_auxiliary_blocks
from pipeline.layout.lp_fuser import FusedBlock, FusedDocument, FusedPage
from pipeline.layout.signals import PageLayoutSignals, PageSignalExtras
from pipeline.repair.dehyphenate import apply_dehyphenation
from pipeline.threading.threader import Threader


def _make_text_block(block_id: str, text: str, bbox: tuple[float, float, float, float], avg_font: float, zone: str = "body") -> PDFBlock:
    span = PDFSpan(text=text, bbox=bbox, font="Times", font_size=avg_font, flags=0, color=0)
    line = PDFLine(spans=[span], bbox=bbox)
    block = PDFBlock(block_id=block_id, bbox=bbox, block_type="text", lines=[line])
    block.text = text
    block.avg_font_size = avg_font
    block.metadata["zone"] = zone
    block.metadata["char_count"] = len(text)
    block.metadata["line_count"] = 1
    block.metadata["span_count"] = 1
    block.char_start = 0
    block.char_end = len(text)
    return block


def test_header_footer_suppression_recomputes_offsets() -> None:
    header1 = _make_text_block("p1_h", "Unit 1", (0, 0, 400, 40), 12.0, zone="header")
    body1 = _make_text_block("p1_body", "Page one body.", (0, 80, 400, 400), 11.0)
    header2 = _make_text_block("p2_h", "Unit 1", (0, 0, 400, 40), 12.0, zone="header")
    body2 = _make_text_block("p2_body", "Page two body.", (0, 80, 400, 400), 11.0)

    page1 = PageGraph(page_number=1, width=400, height=600, rotation=0, blocks=[header1, body1], raw_dict={}, char_start=0, char_end=body1.char_end)
    page2 = PageGraph(page_number=2, width=400, height=600, rotation=0, blocks=[header2, body2], raw_dict={}, char_start=0, char_end=body2.char_end)

    flagged = pdf_parser._collect_repeating_headers_footers([page1, page2])  # type: ignore[attr-defined]
    suppressed = pdf_parser._suppress_headers_footers([page1, page2], flagged)  # type: ignore[attr-defined]
    total_chars = pdf_parser._recompute_document_offsets([page1, page2])  # type: ignore[attr-defined]

    assert suppressed == 2
    assert header1.text == ""
    assert header2.metadata.get("suppressed") is True
    # After recomputing offsets, body blocks should be contiguous
    assert body1.char_start == 0
    assert body2.char_start > body1.char_end
    assert total_chars == body2.char_end + 1


def test_aux_detection_identifies_caption() -> None:
    caption = _make_text_block("p1_caption", "Figure 1: Example caption", (0, 300, 300, 360), 9.0)
    page = PageGraph(page_number=1, width=400, height=600, rotation=0, blocks=[caption], raw_dict={}, char_start=0, char_end=caption.char_end)
    result = detect_auxiliary_blocks(
        page,
        body_font_size=12.0,
        column_assignments={"p1_caption": 0},
    )
    record = result.blocks.get("p1_caption")
    assert record is not None
    assert record.category == "figure_caption"
    assert record.confidence >= 0.5


def test_dehyphenation_merges_cross_page_word() -> None:
    prev = FusedBlock(
        block_id="prev",
        page_number=1,
        text="Long-",
        bbox=(0.0, 0.0, 100.0, 100.0),
        block_type="main",
        region_label="text",
        aux_category=None,
        anchor=None,
        column=0,
        char_start=0,
        char_end=5,
        avg_font_size=11.0,
        metadata={},
    )
    nxt = FusedBlock(
        block_id="next",
        page_number=2,
        text="word continues",
        bbox=(0.0, 0.0, 100.0, 100.0),
        block_type="main",
        region_label="text",
        aux_category=None,
        anchor=None,
        column=0,
        char_start=0,
        char_end=14,
        avg_font_size=11.0,
        metadata={},
    )
    result = apply_dehyphenation(prev, nxt)
    assert result is not None
    assert prev.text.startswith("Longword")
    assert nxt.text.startswith("continues")


def test_threader_delays_aux_until_sentence_end() -> None:
    heading_block = _make_text_block("p1_b0", "Chapter 1", (0, 0, 400, 40), 18.0)
    para_block = _make_text_block("p1_b1", "This is the lead paragraph introducing the topic.", (0, 80, 400, 200), 12.0)
    aux_block = FusedBlock(
        block_id="p1_aux",
        page_number=1,
        text="See also the diagram on the next page.",
        bbox=(300.0, 200.0, 380.0, 260.0),
        block_type="aux",
        region_label="figure",
        aux_category="figure_caption",
        anchor="[AUX-1-1]",
        column=None,
        char_start=0,
        char_end=0,
        avg_font_size=9.0,
        metadata={"anchored_to": "p1_b1"},
    )

    fused_heading = FusedBlock(
        block_id=heading_block.block_id,
        page_number=1,
        text="Chapter 1",
        bbox=heading_block.bbox,
        block_type="main",
        region_label="title",
        aux_category=None,
        anchor=None,
        column=0,
        char_start=0,
        char_end=9,
        avg_font_size=heading_block.avg_font_size,
        metadata={},
    )
    fused_para = FusedBlock(
        block_id=para_block.block_id,
        page_number=1,
        text="This is the lead [AUX-1-1] paragraph introducing the topic.",
        bbox=para_block.bbox,
        block_type="main",
        region_label="text",
        aux_category=None,
        anchor=None,
        column=0,
        char_start=0,
        char_end=56,
        avg_font_size=para_block.avg_font_size,
        metadata={},
    )

    page_graph = PageGraph(
        page_number=1,
        width=400,
        height=600,
        rotation=0,
        blocks=[heading_block, para_block, _make_text_block("p1_aux", aux_block.text, aux_block.bbox, 9.0)],
        raw_dict={},
        char_start=0,
        char_end=para_block.char_end,
    )
    document = DocumentGraph(
        doc_id="doc",
        file_path="/tmp/doc.pdf",
        pages=[page_graph],
        char_count=para_block.char_end,
        metadata={"page_count": 1},
    )

    fused_page = FusedPage(
        page_number=1,
        width=400,
        height=600,
        main_flow=[fused_heading, fused_para],
        auxiliaries=[aux_block],
    )
    fused_doc = FusedDocument(doc_id="doc", pages=[fused_page], block_index={b.block_id: b for b in [fused_heading, fused_para, aux_block]})

    extras = PageSignalExtras(
        column_count=1,
        column_assignments={"p1_b0": 0, "p1_b1": 0},
        table_overlap_ratio=0.0,
        figure_overlap_ratio=0.0,
        dominant_font_size=12.0,
        footnote_block_ids=[],
        superscript_spans=0,
        total_line_count=32,
        has_normal_density=True,
        char_density=0.002,
        structural_score=0.3,
        intrusion_ratio=0.0,
    )
    signals = [PageLayoutSignals(page_number=1, raw={}, normalized={}, page_score=0.4, extras=extras)]

    threader = Threader()
    threaded_doc, report = threader.thread_document(document, fused_doc, signals)

    para_text = threaded_doc.pages[0].main_flow[1].text
    assert para_text.endswith(aux_block.anchor)
    assert "[AUX-1-1]" not in para_text.split("paragraph")[0]
    assert report.placed_aux >= 1

    findings = thread_audit.run_thread_audit(threaded_doc)
    assert findings == []

