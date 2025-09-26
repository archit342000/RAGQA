from parser.classifier import classify_blocks
from parser.utils import Block, DocumentLayout, Line, PageLayout, Span


def _make_caption_block(text: str, bbox) -> Block:
    span = Span(text=text, font="Times", size=9.0, bbox=bbox)
    line = Line(spans=[span], bbox=bbox)
    block = Block(lines=[line], bbox=bbox)
    block.attrs["region_tag"] = "caption"
    block.attrs["region_score"] = 0.92
    return block


def test_detector_caption_forces_aux_classification():
    caption = _make_caption_block("Figure 2. Detector caption", (120, 360, 340, 410))
    page = PageLayout(page_number=0, width=600, height=800, blocks=[caption])
    doc = DocumentLayout(pages=[page])
    predictions = classify_blocks(doc)
    assert len(predictions) == 1
    pred = predictions[0]
    assert pred.kind == "aux"
    assert pred.aux_type == "caption"
    assert any(reason.startswith("Detector:caption") for reason in pred.reason)
