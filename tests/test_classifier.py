from __future__ import annotations

from pdfchunks.config import ParserConfig
from pdfchunks.parsing.baselines import BaselineStats, ColumnBand
from pdfchunks.parsing.block_extractor import Block
from pdfchunks.parsing.classifier import BlockClassifier


def make_block(**kwargs) -> Block:
    defaults = dict(
        block_id="b0",
        text="",
        font_name="Times",
        font_size=10.0,
        line_height=12.0,
        bbox=(70.0, 100.0, 530.0, 150.0),
        page_num=0,
        page_width=600.0,
        page_height=800.0,
        has_border=False,
        has_shading=False,
    )
    defaults.update(kwargs)
    return Block(**defaults)


def baseline() -> BaselineStats:
    density_low = 0.001
    density_high = 0.01
    return BaselineStats(
        body_font_size=10.0,
        body_line_height=12.0,
        density_p20=density_low,
        density_p80=density_high,
        columns=[ColumnBand(index=0, x0=70.0, x1=530.0)],
    )


def test_allow_list_keeps_body_paragraph():
    config = ParserConfig()
    classifier = BlockClassifier(config.classifier)
    block = make_block(text="This is a body paragraph with plenty of words.")
    label = classifier.classify([block], baseline())[0]
    assert label.role == "MAIN"


def test_lexical_cue_forces_aux_caption():
    config = ParserConfig()
    classifier = BlockClassifier(config.classifier)
    block = make_block(block_id="cap", text="Figure 1. A lexical cue caption")
    label = classifier.classify([block], baseline())[0]
    assert label.role == "AUX"
    assert label.subtype == "caption"


def test_heading_exception_allows_top_band_heading():
    config = ParserConfig()
    classifier = BlockClassifier(config.classifier)
    heading = make_block(
        block_id="head",
        text="1 Introduction",
        font_size=14.0,
        line_height=14.0,
        bbox=(55.0, 40.0, 545.0, 80.0),
    )
    label = classifier.classify([heading], baseline())[0]
    assert label.role == "MAIN"
    assert label.is_heading is True

