from __future__ import annotations

from typing import List

from parser.classifier import classify_blocks
from parser.utils import Block, DocumentLayout, Line, PageLayout, Span


def _make_span(text: str, bbox, font_size: float = 11.0, font: str = "Times", bold: bool = False) -> Span:
    return Span(text=text, font=font, size=font_size, bbox=bbox, flags={"bold": bold})


def _make_line(text: str, bbox, font_size: float = 11.0) -> Line:
    spans = [_make_span(text, bbox, font_size=font_size)]
    return Line(spans=spans, bbox=bbox)


def _make_block(
    text: str,
    bbox,
    font_size: float = 11.0,
    block_type: str = "text",
    col_id: int = 1,
    lines: List[str] | None = None,
) -> Block:
    if lines is None:
        lines = [text]
    line_objs: List[Line] = []
    top = bbox[1]
    height = bbox[3] - bbox[1]
    if len(lines) == 1:
        line_objs.append(_make_line(lines[0], bbox, font_size=font_size))
    else:
        line_height = height / max(len(lines), 1)
        for idx, line_text in enumerate(lines):
            line_bbox = (bbox[0], top + idx * line_height, bbox[2], top + (idx + 1) * line_height)
            line_objs.append(_make_line(line_text, line_bbox, font_size=font_size))
    block = Block(lines=line_objs, bbox=bbox, block_type=block_type)
    block.attrs["col_id"] = col_id
    return block


def _build_doc(blocks: List[Block]) -> DocumentLayout:
    page = PageLayout(page_number=0, width=600, height=800, blocks=blocks)
    return DocumentLayout(pages=[page])


def test_fail_safe_promotes_main_classification():
    base_text = "This paragraph contains enough tokens to trigger the fail safe window." * 2
    block_a = _make_block(base_text, (54, 150, 324, 260), font_size=11, col_id=0)
    block_b = _make_block("Supporting paragraph with wide body text." * 3, (20, 320, 580, 460), font_size=11, col_id=1)
    doc = _build_doc([block_a, block_b])
    preds = classify_blocks(doc)
    target = next(p for p in preds if "fail safe" in p.text.lower())
    assert target.kind == "main"
    assert any(reason == "FailSafe" for reason in target.reason)


def test_caption_detection_requires_overlap_and_small_font():
    figure = _make_block("", (200, 200, 420, 420), block_type="figure", col_id=1)
    caption = _make_block("Figure 1. Example diagram.", (210, 430, 410, 470), font_size=8, col_id=1)
    doc = _build_doc([figure, caption])
    preds = classify_blocks(doc)
    cap_pred = next(p for p in preds if "Figure 1" in p.text)
    assert cap_pred.kind == "aux"
    assert cap_pred.aux_type == "caption"
    assert any("Caption" in reason for reason in cap_pred.reason)


def test_sidenote_width_and_font_gate():
    body = _make_block("Body paragraph with regular font size." * 2, (120, 200, 520, 320), font_size=12, col_id=1)
    sidenote = _make_block("Note: reference text.", (10, 210, 110, 320), font_size=8, col_id=0)
    doc = _build_doc([body, sidenote])
    preds = classify_blocks(doc)
    note_pred = next(p for p in preds if "Note:" in p.text)
    assert note_pred.kind == "aux"
    assert note_pred.aux_type == "sidenote"


def test_wrap_protection_biases_main_score():
    figure = _make_block("", (80, 220, 260, 340), block_type="figure", col_id=0)
    lines = ["This paragraph", "is wrapping around the figure"]
    text_block = _make_block(" ".join(lines), (100, 320, 360, 420), font_size=12, col_id=1, lines=lines)
    text_block.lines[0].bbox = (100, 320, 360, 345)
    text_block.lines[1].bbox = (100, 385, 360, 410)
    doc = _build_doc([figure, text_block])
    preds = classify_blocks(doc)
    para_pred = next(p for p in preds if "wrapping" in p.text)
    assert para_pred.kind == "main"
    assert any(reason == "WrapProtect" for reason in para_pred.reason)


def test_implicit_section_control_block_emitted():
    long_text = " ".join(["Intro"] * 50)
    opener = _make_block(long_text, (60, 40, 540, 260), font_size=12, col_id=1)
    opener.attrs["indent"] = 16
    doc = _build_doc([opener])
    preds = classify_blocks(doc)
    assert any(p.kind == "control" and p.subtype == "implicit_section_start" for p in preds)


def test_header_long_text_not_forced_aux():
    header_like = _make_block("This header band paragraph actually belongs to the body text." * 2, (60, 40, 540, 180), font_size=12, col_id=1)
    doc = _build_doc([header_like])
    preds = classify_blocks(doc)
    target = preds[0]
    assert target.kind == "main"


def test_expect_continuation_sets_open_paragraph_id():
    cont_block = _make_block("This line continues across pages" , (80, 300, 520, 360), font_size=12, col_id=1)
    doc = _build_doc([cont_block])
    preds = classify_blocks(doc)
    target = preds[0]
    assert target.expect_continuation is True
    assert target.meta.get("open_paragraph_id")
