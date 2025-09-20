"""Regression tests that cover parser heuristics and UI wiring."""

from __future__ import annotations

from importlib import reload
from pathlib import Path

import pytest

from parser.cleaning import remove_headers_footers
from parser.driver import parse_documents, parse_pdf, parse_text
from parser.types import ParsedDoc, ParsedPage, RunReport


@pytest.fixture
def tmp_pdf_path(tmp_path: Path) -> Path:
    """Create a minimal PDF on disk for tests that expect a real file path."""
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n4 0 obj<</Length 44>>stream\nBT /F1 24 Tf 100 700 Td (stub) Tj ET\nendstream\nendobj\n5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\nxref\n0 6\n0000000000 65535 f \n0000000015 00000 n \n0000000060 00000 n \n0000000111 00000 n \n0000000234 00000 n \n0000000330 00000 n \ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n386\n%%EOF")
    return pdf_path


def test_cleaning_dehyphenation_and_char_ranges(tmp_path: Path) -> None:
    sample_text = "Header\nSection\nLong-\nword continues here\nFooter"
    text_file = tmp_path / "sample.txt"
    text_file.write_text(sample_text, encoding="utf-8")

    doc = parse_text(str(text_file), doc_id="sample")

    assert doc.total_chars == len(doc.pages[0].text)
    assert doc.pages[0].text.count("Longword") == 1
    assert doc.pages[0].char_range == (0, doc.total_chars)


def test_pdf_parsing_prefers_pypdf(monkeypatch: pytest.MonkeyPatch, tmp_pdf_path: Path) -> None:
    def fake_pypdf(_: str, *, max_pages: int | None = None):
        pages = ["This is a well populated page." * 5, "Another dense page of text." * 5]
        if max_pages is not None:
            pages = pages[:max_pages]
        return pages, {"elapsed": "0.01", "page_count": str(len(pages)), "truncated": "false"}

    def fake_unstructured(*args, **kwargs):
        raise AssertionError("unstructured fallback should not be invoked")

    monkeypatch.setenv("MIN_CHARS_PER_PAGE", "10")
    monkeypatch.setenv("FALLBACK_EMPTY_PAGE_RATIO", "0.3")
    monkeypatch.setattr("parser.driver.extract_pdf_with_pypdf", fake_pypdf)
    monkeypatch.setattr("parser.driver.extract_pdf_with_unstructured", fake_unstructured)

    doc = parse_pdf(str(tmp_pdf_path), doc_id="dense")

    assert doc.parser_used == "pypdf"
    assert doc.stats["empty_or_low_ratio"] < 0.05


def test_pdf_parsing_triggers_fallback(monkeypatch: pytest.MonkeyPatch, tmp_pdf_path: Path) -> None:
    def sparse_pypdf(_: str, *, max_pages: int | None = None):
        pages = ["", "Digits 123\n45 67", " "]
        if max_pages is not None:
            pages = pages[:max_pages]
        return pages, {"elapsed": "0.01", "page_count": str(len(pages)), "truncated": "false"}

    def rich_unstructured(_: str, strategy: str, *, max_pages: int | None = None, enable_hi_res: bool = False):
        pages = ["Full text from fallback page one.", "Fallback page two rich text."]
        if max_pages is not None:
            pages = pages[:max_pages]
        meta = {
            "elapsed": "0.02",
            "requested_strategy": strategy,
            "strategy_used": "fast",
            "warnings": "",
        }
        return pages, meta

    monkeypatch.setenv("MIN_CHARS_PER_PAGE", "200")
    monkeypatch.setenv("FALLBACK_EMPTY_PAGE_RATIO", "0.3")
    monkeypatch.setattr("parser.driver.extract_pdf_with_pypdf", sparse_pypdf)
    monkeypatch.setattr("parser.driver.extract_pdf_with_unstructured", rich_unstructured)

    doc = parse_pdf(str(tmp_pdf_path), doc_id="fallback")

    assert doc.parser_used.startswith("unstructured")
    assert doc.stats["fallback_triggered"] == 1.0
    assert doc.stats["fallback_reason"] == "empty_ratio"


def test_env_min_chars_controls_fallback(monkeypatch: pytest.MonkeyPatch, tmp_pdf_path: Path) -> None:
    def borderline_pypdf(_: str, *, max_pages: int | None = None):
        pages = ["a" * 250, "b" * 250]
        if max_pages is not None:
            pages = pages[:max_pages]
        return pages, {"elapsed": "0.01", "page_count": str(len(pages)), "truncated": "false"}

    def fallback_pages(_: str, strategy: str, *, max_pages: int | None = None, enable_hi_res: bool = False):
        pages = ["Fallback enriched page one."]
        meta = {
            "elapsed": "0.02",
            "requested_strategy": strategy,
            "strategy_used": "fast",
            "warnings": "",
        }
        return pages, meta

    monkeypatch.setenv("MIN_CHARS_PER_PAGE", "300")
    monkeypatch.setenv("FALLBACK_EMPTY_PAGE_RATIO", "0.3")
    monkeypatch.setattr("parser.driver.extract_pdf_with_pypdf", borderline_pypdf)
    monkeypatch.setattr("parser.driver.extract_pdf_with_unstructured", fallback_pages)

    doc = parse_pdf(str(tmp_pdf_path), doc_id="threshold")

    assert doc.parser_used.startswith("unstructured")
    assert doc.stats["fallback_reason"] == "empty_ratio"


def test_hi_res_disabled_blocks_high_accuracy(monkeypatch: pytest.MonkeyPatch, tmp_pdf_path: Path) -> None:
    def dense_pypdf(_: str, *, max_pages: int | None = None):
        pages = ["Dense text" * 20, "More dense text" * 20]
        if max_pages is not None:
            pages = pages[:max_pages]
        return pages, {"elapsed": "0.01", "page_count": str(len(pages)), "truncated": "false"}

    def unexpected_unstructured(*args, **kwargs):
        raise AssertionError("Fallback should not run when pypdf succeeds")

    monkeypatch.setenv("ENABLE_HI_RES", "false")
    monkeypatch.setenv("MIN_CHARS_PER_PAGE", "10")
    monkeypatch.setenv("FALLBACK_EMPTY_PAGE_RATIO", "0.3")
    monkeypatch.setattr("parser.driver.extract_pdf_with_pypdf", dense_pypdf)
    monkeypatch.setattr("parser.driver.extract_pdf_with_unstructured", unexpected_unstructured)

    doc = parse_pdf(str(tmp_pdf_path), strategy_env="hi_res", doc_id="blocked")

    assert doc.parser_used == "pypdf"
    assert doc.stats["hi_res_blocked"] == 1.0
    assert any(page.metadata.get("hi_res_blocked") == "true" for page in doc.pages)


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

    return ParsedDoc(doc_id=doc_id, pages=pages, total_chars=offset, parser_used="stub", stats=stats)


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
    monkeypatch.setenv("FALLBACK_EMPTY_PAGE_RATIO", "0.3")

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
        # Simulate borderline documents whose size depends on the override the
        # driver computed for them.
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
        # Mirror the behaviour of the PDF stub so the driver treats both paths
        # consistently during the tests.
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

    # Generous limits ensure the driver should keep every document.
    monkeypatch.setenv("MAX_PAGES", "5")
    monkeypatch.setenv("MAX_TOTAL_PAGES", "1200")
    monkeypatch.setenv("MIN_CHARS_PER_PAGE", "10")
    monkeypatch.setenv("FALLBACK_EMPTY_PAGE_RATIO", "0.3")

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
        # Each stubbed parse honours the max_pages_override provided by the
        # driver, ensuring we can detect if pages were truncated.
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
        # Text files follow the same plan so that the multi-doc flow is uniform
        # regardless of file type.
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


def test_ui_hides_high_res_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """UI should react gracefully when High-Res is disabled via env flags."""
    monkeypatch.setenv("ENABLE_HI_RES", "false")
    monkeypatch.setenv("SHOW_DEBUG", "false")

    import app as app_module

    reload(app_module)
    assert "High-Res" not in app_module._mode_choices()

    dummy_file = tmp_path / "sample.txt"
    dummy_file.write_text("content")

    def fake_parse_documents(files, *, strategy_env=None):
        doc = _build_stub_doc("doc", Path(files[0]).name, 1)
        report = RunReport(total_docs=1, total_pages=1, truncated=False, skipped_docs=[], message="")
        return [doc], report

    monkeypatch.setattr(app_module, "parse_documents", fake_parse_documents)

    state, _, _, chunk_text, status_text, _ = app_module.parse_batch([str(dummy_file)], "High-Res", "Fixed")

    assert "High-Res disabled" in status_text
    assert state["chunks"]
    assert chunk_text
