from parser.grouping import Span, group_spans
from parser.utils import DEFAULT_CONFIG


def test_group_spans_merges_lines_and_blocks():
    spans = [
        Span(text="Hello ", bbox=(10, 10, 60, 20), font_size=11),
        Span(text="world", bbox=(62, 10, 110, 20), font_size=11),
        Span(text="Second line", bbox=(10, 24, 120, 34), font_size=11),
    ]
    lines, blocks = group_spans(spans, DEFAULT_CONFIG, page_width=200)
    assert len(lines) == 2
    assert len(blocks) == 1
    block = blocks[0]
    assert block.meta["col_id"] == 0
    assert block.text.startswith("Hello")
