from parser.grouping import Block, Line, Span
from parser.rules_v2 import ClassifierState, rules_v2_classify
from parser.utils import DEFAULT_CONFIG


class DummyPage:
    def __init__(self, height):
        self.height = height


def test_bias_to_main_window_classifies_main():
    block = Block(
        lines=[Line(spans=[Span(text="Short paragraph continues", bbox=(0, 200, 300, 240), font_size=11)], bbox=(0, 200, 300, 240))],
        bbox=(0, 200, 300, 240),
    )
    block.block_id = "b1"
    block.meta["col_width"] = 300
    page = DummyPage(800)
    result = rules_v2_classify(page, [block], [], DEFAULT_CONFIG, ClassifierState())[0]
    assert result["type"] == "main"
