from parser.pp_structure import tag_blocks_with_regions
from parser.utils import Block, Line, Span


def _make_text_block(text: str, bbox):
    span = Span(text=text, font="Times", size=11.0, bbox=bbox, flags={})
    line = Line(spans=[span], bbox=bbox)
    block = Block(lines=[line], bbox=bbox)
    block.attrs["indent"] = 12
    return block


def test_iou_majority_prefers_text_on_tie():
    block = _make_text_block("This is a narrative paragraph with several words.", (0, 0, 100, 40))
    regions = {
        "text": [[0, 0, 100, 40]],
        "figure": [[0, 0, 100, 40]],
        "title": [],
        "table": [],
        "list": [],
    }
    tagged = tag_blocks_with_regions([block], regions)
    assert tagged[0].attrs.get("region_tag") == "text"


def test_default_to_text_when_no_regions():
    block = _make_text_block("Body paragraph continues with sufficient tokens for prose.", (10, 10, 110, 60))
    tag_blocks_with_regions([block], {})
    assert block.attrs.get("region_tag") == "text"
