import pytest

from chunking.driver import chunk_documents
from parser.types import ParsedDoc, ParsedPage

from pipeline.layout.lp_fuser import FusedBlock, FusedDocument, FusedPage
from pipeline.layout.router import LayoutRoutingPlan
from pipeline.layout.signals import PageLayoutSignals, PageSignalExtras


@pytest.fixture(autouse=True)
def configure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENIZER_NAME", "hf-internal-testing/llama-tokenizer")
    monkeypatch.setenv("MAX_TOTAL_TOKENS_FOR_CHUNKING", "50000")
    monkeypatch.setenv("PIPELINE_SKIP_EMBEDDER", "true")


_SIGNAL_NAMES = ("CIS", "OGR", "BXS", "DAS", "FVS", "ROJ", "TFI", "MSA", "FNL")


def _make_page(doc_id: str, page_num: int, text: str, file_name: str = "fixture.pdf") -> ParsedPage:
    metadata = {"file_name": file_name, "source": "hybrid", "strategy": "hybrid"}
    return ParsedPage(
        doc_id=doc_id,
        page_num=page_num,
        text=text,
        char_range=(0, len(text)),
        metadata=metadata,
    )


def _make_doc(
    doc_id: str,
    pages: list[ParsedPage],
    fused_pages: list[FusedPage],
    *,
    block_index: dict[str, FusedBlock],
) -> ParsedDoc:
    signals: list[PageLayoutSignals] = []
    for page in fused_pages:
        assignments = {block.block_id: 0 for block in page.main_flow}
        extras = PageSignalExtras(
            column_count=1,
            column_assignments=assignments,
            table_overlap_ratio=0.0,
            figure_overlap_ratio=0.0,
            dominant_font_size=12.0,
            footnote_block_ids=[],
            superscript_spans=0,
            total_line_count=25,
            has_normal_density=True,
            char_density=0.001,
        )
        raw = {name: 0.1 for name in _SIGNAL_NAMES}
        signals.append(PageLayoutSignals(page_number=page.page_number, raw=raw, normalized=raw, page_score=0.2, extras=extras))
    fused_document = FusedDocument(doc_id=doc_id, pages=fused_pages, block_index=block_index)
    plan = LayoutRoutingPlan(doc_id=doc_id, total_pages=len(fused_pages), budget=len(fused_pages), selected_pages=[page.page_number for page in fused_pages], decisions=[], overflow=0)
    return ParsedDoc(
        doc_id=doc_id,
        pages=pages,
        total_chars=sum(len(page.text) for page in pages),
        parser_used="pymupdf-hybrid",
        stats={},
        meta={"file_name": pages[0].metadata.get("file_name", doc_id), "truncated": False},
        fused_document=fused_document,
        layout_signals=signals,
        routing_plan=plan,
    )


def test_hybrid_chunking_produces_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = "doc-sem"
    page = _make_page(doc_id, 1, "Paragraph one. Paragraph two continuing the discussion.")
    main_block = FusedBlock(
        block_id="b1",
        page_number=1,
        text=page.text,
        bbox=(0.0, 0.0, 100.0, 100.0),
        block_type="main",
        region_label="text",
        aux_category=None,
        anchor=None,
        column=0,
        char_start=0,
        char_end=len(page.text),
        avg_font_size=12.0,
        metadata={"section_id": "1", "paragraph_id": "1.001"},
    )
    fused_page = FusedPage(page_number=1, width=612.0, height=792.0, main_flow=[main_block], auxiliaries=[])
    doc = _make_doc(doc_id, [page], [fused_page], block_index={"b1": main_block})

    chunks, stats = chunk_documents([doc], mode="semantic")

    assert chunks, "Expected at least one hybrid chunk"
    assert chunks[0].meta["strategy"] == "hybrid-semantic"
    assert "block_ids" in chunks[0].meta
    assert stats[doc_id]["chunks"] == len(chunks)
    assert stats[doc_id]["lp_pages"] == 1


def test_table_blocks_emit_table_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = "doc-table"
    page = _make_page(doc_id, 1, "Body text before table.")
    main_block = FusedBlock(
        block_id="b1",
        page_number=1,
        text=page.text,
        bbox=(0.0, 0.0, 100.0, 100.0),
        block_type="main",
        region_label="text",
        aux_category=None,
        anchor=None,
        column=0,
        char_start=0,
        char_end=len(page.text),
        avg_font_size=12.0,
        metadata={"section_id": "1", "paragraph_id": "1.001"},
    )
    table_block = FusedBlock(
        block_id="t1",
        page_number=1,
        text="H1 H2\nR1C1 R1C2\nR2C1 R2C2",
        bbox=(0.0, 120.0, 100.0, 200.0),
        block_type="aux",
        region_label="table",
        aux_category="table",
        anchor="[AUX-1-1]",
        column=None,
        char_start=len(page.text),
        char_end=len(page.text) + 20,
        avg_font_size=11.0,
        metadata={"owner_section_id": "1", "section_id": "1"},
    )
    fused_page = FusedPage(page_number=1, width=612.0, height=792.0, main_flow=[main_block], auxiliaries=[table_block])
    doc = _make_doc(doc_id, [page], [fused_page], block_index={"b1": main_block, "t1": table_block})

    chunks, stats = chunk_documents([doc], mode="fixed")

    table_chunks = [chunk for chunk in chunks if chunk.meta.get("table_row_range")]
    assert table_chunks, "Table blocks should yield dedicated table chunks"
    assert stats[doc_id]["tables"] == len(table_chunks)
    assert all(chunk.meta.get("aux_kind") == "table" for chunk in table_chunks)
    assert all(chunk.meta.get("section_id") == "1" for chunk in table_chunks)


def test_auxiliary_blocks_produce_aux_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = "doc-aux"
    page = _make_page(doc_id, 1, "Main paragraph content for attachment.")
    main_block = FusedBlock(
        block_id="b1",
        page_number=1,
        text=page.text,
        bbox=(0.0, 0.0, 100.0, 100.0),
        block_type="main",
        region_label="text",
        aux_category=None,
        anchor=None,
        column=0,
        char_start=0,
        char_end=len(page.text),
        avg_font_size=12.0,
        metadata={"section_id": "1", "paragraph_id": "1.001"},
    )
    aux_block = FusedBlock(
        block_id="a1",
        page_number=1,
        text="Figure caption awaiting attachment",
        bbox=(0.0, 120.0, 100.0, 200.0),
        block_type="aux",
        region_label="figure",
        aux_category="figure",
        anchor="[AUX-1-1]",
        column=None,
        char_start=len(page.text),
        char_end=len(page.text) + 32,
        avg_font_size=11.0,
        metadata={"owner_section_id": "1"},
    )
    fused_page = FusedPage(page_number=1, width=612.0, height=792.0, main_flow=[main_block], auxiliaries=[aux_block])
    doc = _make_doc(doc_id, [page], [fused_page], block_index={"b1": main_block, "a1": aux_block})

    chunks, stats = chunk_documents([doc], mode="semantic")

    aux_chunks = [chunk for chunk in chunks if chunk.meta.get("aux_kind")]
    assert aux_chunks, "Auxiliary blocks should yield dedicated chunks"
    assert all(chunk.meta.get("aux_kind") == "figure" for chunk in aux_chunks)
    assert stats[doc_id]["aux_attached"] == 0
