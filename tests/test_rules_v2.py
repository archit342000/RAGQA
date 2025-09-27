from parser.grouping import Block, Line, Span
from parser.grouping import Block, Line, Span
from parser.rules_v2 import ClassifierState, rules_v2_classify
from parser.utils import DEFAULT_CONFIG


class DummyPage:
    def __init__(self, height: float):
        self.height = height


def make_block(text, bbox, region_tag="text"):
    block = Block(lines=[Line(spans=[Span(text=text, bbox=bbox, font_size=11)], bbox=bbox)], bbox=bbox)
    block.region_tag = region_tag
    block.block_id = f"blk_{hash(text) & 0xffff}"
    return block


def test_rules_force_caption_and_bias_main():
    page = DummyPage(800)
    blocks = [
        make_block("Figure 1. Caption", (10, 400, 200, 430), "text"),
        make_block("Regular paragraph text that is fairly long and descriptive.", (10, 200, 400, 260), "text"),
    ]
    state = ClassifierState()
    result = rules_v2_classify(page, blocks, [], DEFAULT_CONFIG, state)
    types = {blk["text"]: blk for blk in result}
    assert types["Figure 1. Caption"]["type"] == "aux"
    assert types["Regular paragraph text that is fairly long and descriptive."]["type"] == "main"
