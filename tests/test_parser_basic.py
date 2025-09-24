"""Regression tests that cover parser heuristics and UI wiring."""

from __future__ import annotations

from importlib import reload
from pathlib import Path

import pytest

from chunking.types import Chunk
from parser.cleaning import remove_headers_footers
from parser.driver import parse_documents, parse_pdf, parse_text
from parser.types import ParsedDoc, ParsedPage, RunReport

from pipeline.ingest.pdf_parser import DocumentGraph, PageGraph, PDFBlock
from pipeline.layout.lp_fuser import FusedBlock, FusedDocument, FusedPage
from pipeline.layout.router import LayoutRoutingPlan, PageRoutingDecision
from pipeline.layout.signals import PageLayoutSignals, PageSignalExtras
from pipeline.repair.repair_pass import RepairStats


@pytest.fixture
def tmp_pdf_path(tmp_path: Path) -> Path:
    """Create a minimal PDF on disk for tests that expect a real file path."""

    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[30 R]/Count 1>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n4 0 obj<</Length 44>>stream\nBT /F1 24 Tf 100 700 Td (stub) Tj ET\nendstream\nendobj\n5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\nxref\n0 6\n0000000000 65535 f \n0000000015 00000 n \n0000000060 00000 n \n0000000111 00000 n \n0000000234 00000 n \n0000000330 00000 n \ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n386\n%%EOF")
    return pdf_path


def test_cleaning_dehyphenation_and_char_ranges(tmp_path: Path) -> None:
    sample_text = "Header\nSection\nLong-\nword continues here\nFooter"
    text_file = tmp_path / "sample.txt"
    text_file.write_text(sample_text, encoding="utf-8")

    doc = parse_text(str(text_file), doc_id="sample")

    assert doc.total_chars == len(doc.pages[0].text)
    assert doc.pages[0].text.count("Longword") == 1
    assert doc.pages[0].char_range == (0, doc.total_chars)


def test_parse_pdf_hybrid_pipeline_metadata(monkeypatch: pytest.MonkeyPatch, tmp_pdf_path: Path) -> None:
    """The hybrid parser should surface routing, repair, and layout metadata."""

    block = PDFBlock(
        block_id="b1",
        bbox=(0.0, 0.0, 100.0, 100.0),
        block_type="text",
        lines=[],
        text="Main paragraph",
        avg_font_size=12.0,
        dominant_fonts={},
        metadata={},
        char_start=0,
        char_end=15,
        class_hint="prose",
    )
    page_graph = PageGraph(
        page_number=1,
        width=612.0,
        height=792.0,
        rotation=0,
        blocks=[block],
        raw_dict={},
        char_start=0,
        char_end=15,
    )
    document_graph = DocumentGraph(
        doc_id="doc",
        file_path=str(tmp_pdf_path),
        pages=[page_graph],
        char_count=15,
        metadata={"page_count": 1, "file_name": tmp_pdf_path.name},
    )

    extras = PageSignalExtras(
        column_count=1,
        column_assignments={"b1": 0},
        table_overlap_ratio=0.0,
        figure_overlap_ratio=0.0,
        dominant_font_size=12.0,
        footnote_block_ids=[],
        superscript_spans=0,
        total_line_count=30,
        has_normal_density=True,
        char_density=0.001,
    )
    raw = {name: 0.1 for name in ("CIS", "OGR", "BXS", "DAS", "FVS", "ROJ", "TFI", "MSA", "FNL")}
    normalized = {**raw, "CIS": 0.65}
    signals = [PageLayoutSignals(page_number=1, raw=raw, normalized=normalized, page_score=0.6, extras=extras)]

    plan = LayoutRoutingPlan(
        doc_id="doc",
        total_pages=1,
        budget=1,
        selected_pages=[1],
        decisions=[
            PageRoutingDecision(
                page_number=1,
                score=0.6,
                triggers=["T1"],
                neighbor=False,
                use_layout_parser=True,
                model="publaynet",
                dpi=180,
            )
        ],
        overflow=0,
    )

    main_block = FusedBlock(
        block_id="b1",
        page_number=1,
        text="Main paragraph",
        bbox=(0.0, 0.0, 100.0, 100.0),
        block_type="main",
        region_label="text",
        aux_category=None,
        anchor=None,
        column=0,
        char_start=0,
        char_end=15,
        avg_font_size=12.0,
        metadata={},
    )
    aux_block = FusedBlock(
        block_id="aux1",
        page_number=1,
        text="Table row",
        bbox=(0.0, 120.0, 100.0, 200.0),
        block_type="aux",
        region_label="table",
        aux_category="table",
        anchor="[AUX-1-1]",
        column=None,
        char_start=15,
        char_end=25,
        avg_font_size=10.0,
        metadata={"has_anchor_refs": True},
    )
    fused_page = FusedPage(
        page_number=1,
        width=612.0,
        height=792.0,
        main_flow=[main_block],
        auxiliaries=[aux_block],
    )
    fused_doc = FusedDocument(doc_id="doc", pages=[fused_page], block_index={"b1": main_block, "aux1": aux_block})
    repair_stats = RepairStats(merged_blocks=1, split_blocks=0, footnotes_linked=1)

    monkeypatch.setattr("parser.driver.parse_pdf_with_pymupdf", lambda *args, **kwargs: document_graph)
    monkeypatch.setattr("parser.driver.compute_layout_signals", lambda _: signals)
    monkeypatch.setattr("parser.driver.plan_layout_routing", lambda *args, **kwargs: plan)
    monkeypatch.setattr("parser.driver.fuse_layout", lambda *args, **kwargs: fused_doc)
    monkeypatch.setattr("parser.driver.run_repair_pass", lambda *args, **kwargs: (fused_doc, repair_stats, {1: 0}))
    monkeypatch.setattr("parser.driver._load_embedder", lambda model_name=None: None)

    doc = parse_pdf(str(tmp_pdf_path), doc_id="doc")

    assert doc.parser_used == "pymupdf-hybrid"
    assert doc.fused_document is fused_doc
    assert doc.layout_signals == signals
    assert doc.routing_plan.selected_pages == [1]
    assert doc.pages[0].metadata["lp_used"] == "true"
    assert doc.pages[0].metadata["layout_score"] == "0.600"
    assert "[TABLE]" in doc.pages[0].text
    assert doc.pages[0].meta["blocks"]["auxiliary_blocks"][0]["category"] == "table"
    assert doc.stats["repair_merged_blocks"] == 1.0
    assert doc.meta["layout_plan"]["selected_pages"] == [1]


def test_hi_res_disabled_blocks_high_accuracy(monkeypatch: pytest.MonkeyPatch, tmp_pdf_path: Path) -> None:
    """Hi-Res requests should record the disabled flag when env forbids it."""

    monkeypatch.setenv("ENABLE_HI_RES", "false")

    block = PDFBlock(
        block_id="b1",
        bbox=(0.0, 0.0, 100.0, 100.0),
        block_type="text",
        lines=[],
        text="Dense text",
        avg_font_size=12.0,
        dominant_fonts={},
        metadata={},
        char_start=0,
        char_end=10,
        class_hint="prose",
    )
    page_graph = PageGraph(
        page_number=1,
        width=612.0,
        height=792.0,
        rotation=0,
        blocks=[block],
        raw_dict={},
        char_start=0,
        char_end=10,
    )
    document_graph = DocumentGraph(
        doc_id="doc",
        file_path=str(tmp_pdf_path),
        pages=[page_graph],
        char_count=10,
        metadata={"page_count": 1, "file_name": tmp_pdf_path.name},
    )

    extras = PageSignalExtras(
        column_count=1,
        column_assignments={"b1": 0},
        table_overlap_ratio=0.0,
        figure_overlap_ratio=0.0,
        dominant_font_size=12.0,
        footnote_block_ids=[],
        superscript_spans=0,
        total_line_count=20,
        has_normal_density=True,
        char_density=0.001,
    )
    raw = {name: 0.0 for name in ("CIS", "OGR", "BXS", "DAS", "FVS", "ROJ", "TFI", "MSA", "FNL")}
    signals = [PageLayoutSignals(page_number=1, raw=raw, normalized=raw, page_score=0.2, extras=extras)]
    plan = LayoutRoutingPlan(
        doc_id="doc",
        total_pages=1,
        budget=1,
        selected_pages=[],
        decisions=[
            PageRoutingDecision(
                page_number=1,
                score=0.2,
                triggers=[],
                neighbor=False,
                use_layout_parser=False,
                model="publaynet",
                dpi=180,
            )
        ],
        overflow=0,
    )
    fused = FusedDocument(
        doc_id="doc",
        pages=[
            FusedPage(
                page_number=1,
                width=612.0,
                height=792.0,
                main_flow=[
                    FusedBlock(
                        block_id="b1",
                        page_number=1,
                        text="Dense text",
                        bbox=(0.0, 0.0, 100.0, 100.0),
                        block_type="main",
                        region_label="text",
                        aux_category=None,
                        anchor=None,
                        column=0,
                        char_start=0,
                        char_end=10,
                        avg_font_size=12.0,
                        metadata={},
                    )
                ],
                auxiliaries=[],
            )
        ],
        block_index={},
    )
    repair_stats = RepairStats()

    monkeypatch.setattr("parser.driver.parse_pdf_with_pymupdf", lambda *args, **kwargs: document_graph)
    monkeypatch.setattr("parser.driver.compute_layout_signals", lambda _: signals)
    monkeypatch.setattr("parser.driver.plan_layout_routing", lambda *args, **kwargs: plan)
    monkeypatch.setattr("parser.driver.fuse_layout", lambda *args, **kwargs: fused)
    monkeypatch.setattr("parser.driver.run_repair_pass", lambda *args, **kwargs: (fused, repair_stats, {}))
    monkeypatch.setattr("parser.driver._load_embedder", lambda model_name=None: None)

    doc = parse_pdf(str(tmp_pdf_path), strategy_env="hi_res", doc_id="doc")

    assert doc.stats["hi_res_blocked"] == 1.0
    assert doc.pages[0].metadata["hi_res_blocked"] == "true"


def test_remove_headers_footers() -> None:
    pages = [
        "Company Header\nActual content page 1\nFooter text",
        "Company Header\nDifferent middle\nFooter text",
        "Company Header\nMore body text\nFooter text",
    ]

    cleaned = remove_headers_footers(pages)

    assert all(not page.startswith("Company Header") for page in cleaned)
    assert all(not page.endswith("Footer text") for page in cleaned)


def _build_stub_doc(doc_id: str, file_name: str, page_count: int) -> ParsedDoc:
    """Construct a lightweight ParsedDoc for monkeypatched driver tests."""

    pages: list[ParsedPage] = []
    offset = 0
    for idx in range(1, page_count + 1):
        text = f"{file_name} page {idx}"
        metadata = {
            "source": "stub",
            "strategy": "stub",
            "file_name": file_name,
        }
        char_range = (offset, offset + len(text))
        pages.append(ParsedPage(doc_id=doc_id, page_num=idx, text=text, char_range=char_range, metadata=metadata))
        offset = char_range[1]

    stats = {
        "total_pages": float(page_count),
        "total_chars": float(offset),
        "fallback_triggered": 0.0,
        "fallback_reason": "none",
        "parse_duration_seconds": 0.0,
        "hi_res_blocked": 0.0,
    }

    return ParsedDoc(
        doc_id=doc_id,
        pages=pages,
        total_chars=offset,
        parser_used="stub",
        stats=stats,
        meta={"file_name": file_name, "truncated": False},
    )


def test_parse_documents_multi_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure aggregate limits skip later documents once the total cap is hit."""

    pdf_one = tmp_path / "doc1.pdf"
    pdf_two = tmp_path / "doc2.pdf"
    txt_doc = tmp_path / "doc3.txt"
    for path in (pdf_one, pdf_two, txt_doc):
        path.write_text("stub")

    monkeypatch.setenv("MAX_PAGES", "5")
    monkeypatch.setenv("MAX_TOTAL_PAGES", "3")
    monkeypatch.setenv("MIN_CHARS_PER_PAGE", "10")

    page_plan = {
        pdf_one.name: 2,
        pdf_two.name: 2,
        txt_doc.name: 1,
    }

    def fake_parse_pdf(
        file_path: str,
        *,
        strategy_env: str | None = None,
        doc_id: str | None = None,
        max_pages_override: int | None = None,
        file_name: str | None = None,
    ) -> ParsedDoc:
        allowed = max_pages_override if max_pages_override is not None else 5
        planned = page_plan[file_name or Path(file_path).name]
        pages = min(allowed, planned)
        return _build_stub_doc(doc_id or "pdf", file_name or "pdf", pages)

    def fake_parse_text(
        file_path: str,
        *,
        doc_id: str | None = None,
        max_pages_override: int | None = None,
        file_name: str | None = None,
    ) -> ParsedDoc:
        allowed = max_pages_override if max_pages_override is not None else 5
        planned = page_plan[file_name or Path(file_path).name]
        pages = min(allowed, planned)
        return _build_stub_doc(doc_id or "text", file_name or "text", pages)

    monkeypatch.setattr("parser.driver.parse_pdf", fake_parse_pdf)
    monkeypatch.setattr("parser.driver.parse_text", fake_parse_text)

    docs, report = parse_documents([str(pdf_one), str(pdf_two), str(txt_doc)], strategy_env="hi_res")

    assert len(docs) == 2  # third document skipped due to total page cap
    assert report.truncated is True
    assert report.total_pages == 3
    assert any("skipped" in msg for msg in report.skipped_docs)


def test_parse_documents_no_skip_when_within_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When under the total cap, every uploaded document should be parsed."""

    pdf_one = tmp_path / "doc1.pdf"
    pdf_two = tmp_path / "doc2.pdf"
    txt_doc = tmp_path / "doc3.txt"
    for path in (pdf_one, pdf_two, txt_doc):
        path.write_text("stub")

    monkeypatch.setenv("MAX_PAGES", "5")
    monkeypatch.setenv("MAX_TOTAL_PAGES", "1200")
    monkeypatch.setenv("MIN_CHARS_PER_PAGE", "10")

    page_plan = {
        pdf_one.name: 2,
        pdf_two.name: 3,
        txt_doc.name: 1,
    }

    def fake_parse_pdf(
        file_path: str,
        *,
        strategy_env: str | None = None,
        doc_id: str | None = None,
        max_pages_override: int | None = None,
        file_name: str | None = None,
    ) -> ParsedDoc:
        planned = page_plan[file_name or Path(file_path).name]
        allowed = max_pages_override if max_pages_override is not None else planned
        pages = min(planned, allowed)
        return _build_stub_doc(doc_id or "pdf", file_name or "pdf", pages)

    def fake_parse_text(
        file_path: str,
        *,
        doc_id: str | None = None,
        max_pages_override: int | None = None,
        file_name: str | None = None,
    ) -> ParsedDoc:
        planned = page_plan[file_name or Path(file_path).name]
        allowed = max_pages_override if max_pages_override is not None else planned
        pages = min(planned, allowed)
        return _build_stub_doc(doc_id or "text", file_name or "text", pages)

    monkeypatch.setattr("parser.driver.parse_pdf", fake_parse_pdf)
    monkeypatch.setattr("parser.driver.parse_text", fake_parse_text)

    docs, report = parse_documents([str(pdf_one), str(pdf_two), str(txt_doc)], strategy_env="fast")

    assert len(docs) == 3
    assert report.truncated is False
    assert report.total_pages == sum(page_plan.values())
    assert not report.message


def test_ui_uses_fixed_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """UI parsing should always rely on the fixed hybrid pipeline without prompts."""

    monkeypatch.setenv("SHOW_DEBUG", "false")

    import app as app_module

    reload(app_module)

    dummy_file = tmp_path / "sample.txt"
    dummy_file.write_text("content")

    captured: Dict[str, object] = {}

    def fake_parse_documents(files, *, strategy_env=None):
        captured["strategy_env"] = strategy_env
        doc = _build_stub_doc("doc", Path(files[0]).name, 1)
        report = RunReport(total_docs=1, total_pages=1, truncated=False, skipped_docs=[], message="")
        return [doc], report

    def fake_chunk_documents(docs, *, mode=None, model_name=None):
        captured["chunk_mode"] = mode
        chunk = Chunk(
            doc_id="doc",
            doc_name="sample.txt",
            page_start=1,
            page_end=1,
            section_title=None,
            text="chunk text",
            token_len=128,
            meta={},
        )
        stats = {
            "doc": {
                "doc_name": "sample.txt",
                "chunks": 1,
                "avg_tokens": 128.0,
                "mode": "semantic",
                "tables": 0,
                "aux_attached": 0,
            }
        }
        return [chunk], stats

    monkeypatch.setattr(app_module, "parse_documents", fake_parse_documents)
    monkeypatch.setattr(app_module, "chunk_documents", fake_chunk_documents)

    state, _, _, chunk_text, status_text, _ = app_module.parse_batch([str(dummy_file)])

    assert captured.get("strategy_env") is None
    assert captured.get("chunk_mode") is None
    assert state["chunks"]
    assert chunk_text == "chunk text"
    assert "chunks" in status_text
