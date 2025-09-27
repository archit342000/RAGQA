from parser.grouping import Block, Line, Span
from parser.grouping import Block, Line, Span
from parser.rules_v2 import ClassifierState, rules_v2_classify
from parser.utils import DEFAULT_CONFIG


class DummyPage:
    def __init__(self, height):
        self.height = height


def make_block(text, bbox):
    block = Block(lines=[Line(spans=[Span(text=text, bbox=bbox, font_size=10)], bbox=bbox)], bbox=bbox)
    block.block_id = text
    return block


def test_header_requires_repetition():
    cfg = DEFAULT_CONFIG
    page = DummyPage(800)
    state = ClassifierState()
    blocks = [make_block("Chapter 1", (0, 10, 200, 30))]
    first = rules_v2_classify(page, blocks, [], cfg, state)[0]
    assert first["type"] == "main"
    for _ in range(2):
        result = rules_v2_classify(page, blocks, [], cfg, state)[0]
    assert result["type"] == "aux"
    assert result["subtype"] == "header"
