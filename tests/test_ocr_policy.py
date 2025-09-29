from __future__ import annotations

from pdf_ingest.config import Config
from pdf_ingest.ocr import classify_pages
from pdf_ingest.pdf_io import PageSignals


def make_signal(
    *,
    index: int,
    glyphs: int,
    density: float,
    unicode_ratio: float,
    has_fonts: bool = True,
    image_coverage: float = 0.0,
    hidden: bool = False,
) -> PageSignals:
    bands = [glyphs]
    return PageSignals(
        index=index,
        glyph_count=glyphs,
        text_density=density,
        unicode_ratio=unicode_ratio,
        has_fonts=has_fonts,
        image_coverage=image_coverage,
        delimiter_ratio=0.0,
        whitespace_ratio=0.0,
        hidden_text_layer=hidden,
        dpi=300.0,
        page_area=1.0,
        band_glyph_counts=bands,
        band_text_density=[density],
    )


def test_promote_none_vs_partial() -> None:
    config = Config()
    signals = [
        make_signal(index=0, glyphs=1000, density=0.5, unicode_ratio=0.99),
        make_signal(index=1, glyphs=100, density=0.01, unicode_ratio=0.5),
    ]
    decisions = classify_pages(signals, config)
    assert decisions[0].mode == "none"
    assert decisions[1].mode == "partial"


def test_hidden_text_prevents_full() -> None:
    config = Config()
    signals = [
        make_signal(index=0, glyphs=50, density=0.01, unicode_ratio=0.95, hidden=True)
    ]
    decisions = classify_pages(signals, config)
    assert decisions[0].mode == "none"


def test_full_persists_with_neighbors() -> None:
    config = Config()
    signals = [
        make_signal(index=0, glyphs=10, density=0.0001, unicode_ratio=0.4, image_coverage=0.6),
        make_signal(index=1, glyphs=12, density=0.0001, unicode_ratio=0.4, image_coverage=0.6),
        make_signal(index=2, glyphs=15, density=0.0001, unicode_ratio=0.4, image_coverage=0.6),
    ]
    decisions = classify_pages(signals, config)
    assert all(dec.mode == "full" for dec in decisions)
