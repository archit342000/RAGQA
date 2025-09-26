from parser.region_assigner import assign_regions
from parser.utils import Block, Line, PageLayout, Span


def _make_text_block(text: str, bbox) -> Block:
    span = Span(text=text, font="Times", size=11.0, bbox=bbox)
    line = Line(spans=[span], bbox=bbox)
    block = Block(lines=[line], bbox=bbox)
    return block


def test_assign_regions_prefers_priority_on_tie():
    caption_bbox = (50.0, 100.0, 350.0, 160.0)
    block = _make_text_block("Figure 1. Caption text", caption_bbox)
    page = PageLayout(page_number=0, width=600.0, height=800.0, blocks=[block])
    detections = [
        {"cls": "paragraph", "bbox_pdf": caption_bbox, "score": 0.4},
        {"cls": "caption", "bbox_pdf": caption_bbox, "score": 0.9},
    ]
    assign_regions(page, detections)
    assert block.attrs.get("region_tag") == "caption"


def test_assign_regions_defaults_to_paragraph_when_unlabeled():
    bbox = (40.0, 200.0, 420.0, 320.0)
    block = _make_text_block("Body paragraph continues here." * 2, bbox)
    page = PageLayout(page_number=0, width=600.0, height=800.0, blocks=[block])
    assign_regions(page, [])
    assert block.attrs.get("region_tag") == "paragraph"
