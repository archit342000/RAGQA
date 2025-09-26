from parser.grouping import Block, Line, Span
from parser.region_fusion import fuse_regions
from parser.utils import DEFAULT_CONFIG


def test_fuse_regions_assigns_priority():
    block = Block(
        lines=[Line(spans=[Span(text="Caption", bbox=(0, 0, 100, 20), font_size=10)], bbox=(0, 0, 100, 20))],
        bbox=(0, 0, 100, 20),
    )
    regions = [
        {"class": "figure", "bbox": (0, 0, 100, 100), "score": 0.9},
        {"class": "caption", "bbox": (0, 0, 100, 20), "score": 0.8},
    ]
    fused = fuse_regions([block], regions, DEFAULT_CONFIG)
    assert fused[0].region_tag == "caption"
