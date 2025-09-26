from parser.classifier import classify_blocks
from parser.utils import Block, DocumentLayout, Line, PageLayout, Span


def _figure_block(bbox) -> Block:
    return Block(lines=[], bbox=bbox, block_type="figure")


def _caption_block(text: str, bbox) -> Block:
    span = Span(text=text, font="Times", size=9.0, bbox=bbox)
    line = Line(spans=[span], bbox=bbox)
    block = Block(lines=[line], bbox=bbox)
    block.attrs["region_tag"] = "caption"
    block.attrs["region_score"] = 0.9
    return block


def test_caption_region_anchors_to_nearest_figure():
    figure = _figure_block((150, 220, 330, 360))
    caption = _caption_block("Figure 3. Anatomy of a leaf", (160, 365, 340, 410))
    page = PageLayout(page_number=0, width=600, height=800, blocks=[figure, caption])
    doc = DocumentLayout(pages=[page])
    predictions = classify_blocks(doc)
    cap_pred = next(pred for pred in predictions if pred.aux_type == "caption")
    assert cap_pred.anchor_hint is not None
    fig_pred = next(pred for pred in predictions if pred.aux_type == "figure")
    assert cap_pred.anchor_hint[0] == fig_pred.page
