from parser.post_pass_repair import CAPTION_RE, layout_repair
from parser.classifier import BlockPrediction
from parser.utils import Block


def _prediction(
    *,
    page: int,
    index: int,
    kind: str,
    aux_type: str | None,
    text: str,
    bbox,
    block_type: str = "text",
    region_tag: str | None = None,
    confidence: float = 0.6,
) -> BlockPrediction:
    block = Block(lines=[], bbox=bbox, block_type=block_type)
    meta = {
        "region_tag": region_tag,
        "top_pct": 0.1,
        "bottom_pct": 0.2,
    }
    return BlockPrediction(
        page=page,
        index=index,
        kind=kind,
        aux_type=aux_type,
        subtype=None,
        text=text,
        bbox=bbox,
        ms=0.58,
        hs=None,
        reason=[],
        meta=meta,
        flow="aux" if kind == "aux" else "main",
        confidence=confidence,
        source_block=block,
    )


def test_caption_regex_matches_fig_variants():
    assert CAPTION_RE.match("Fig. 2: Star chart")
    assert CAPTION_RE.match("Figure 10 – caption")
    assert CAPTION_RE.match("Table 3 Results")


def test_layout_repair_peels_caption_from_main():
    figure = _prediction(
        page=0,
        index=0,
        kind="aux",
        aux_type="figure",
        text="",
        bbox=(10, 10, 60, 60),
        block_type="figure",
        region_tag="figure",
        confidence=0.9,
    )
    caption_main = _prediction(
        page=0,
        index=1,
        kind="main",
        aux_type=None,
        text="Fig. 1: Solar system diagram with notes.",
        bbox=(12, 62, 120, 90),
        region_tag="figure",
        confidence=0.55,
    )
    repaired = layout_repair([figure, caption_main], config={})
    captions = [block for block in repaired if block.aux_type == "caption"]
    assert captions, "Expected caption to be peeled into aux block"
    assert all(block.kind == "aux" for block in captions)
    assert all(block.anchor_hint == (figure.page, figure.index) for block in captions if block.anchor_hint)
    assert all("PostPass:Peel" in block.reason for block in captions)
