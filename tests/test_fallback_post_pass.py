from parser.classifier import BlockPrediction
from parser.post_pass_repair import layout_repair


def _meta_for_bbox(bbox, page_width=600.0, page_height=800.0):
    x0, y0, x1, y1 = bbox
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    return {
        "page_width": page_width,
        "page_height": page_height,
        "width": width,
        "height": height,
        "left_pct": x0 / page_width,
        "right_pct": x1 / page_width,
        "top_pct": y0 / page_height,
        "bottom_pct": y1 / page_height,
    }


def test_layout_repair_peels_caption_without_detector_regions():
    figure_bbox = (200.0, 220.0, 360.0, 360.0)
    caption_bbox = (210.0, 365.0, 350.0, 410.0)
    figure = BlockPrediction(
        page=0,
        index=0,
        kind="aux",
        aux_type="figure",
        subtype=None,
        text="",
        bbox=figure_bbox,
        ms=0.0,
        hs=None,
        reason=["Detector:figure"],
        meta={"region_tag": "figure", **_meta_for_bbox(figure_bbox)},
        flow="aux",
        confidence=0.9,
        source_block=None,  # type: ignore[arg-type]
    )
    caption_candidate = BlockPrediction(
        page=0,
        index=1,
        kind="main",
        aux_type=None,
        subtype=None,
        text="Figure 5 shows the process in detail.",
        bbox=caption_bbox,
        ms=0.58,
        hs=None,
        reason=[],
        meta=_meta_for_bbox(caption_bbox),
        flow="main",
        confidence=0.55,
        source_block=None,  # type: ignore[arg-type]
    )
    repaired = layout_repair([figure, caption_candidate])
    aux_blocks = [pred for pred in repaired if pred.kind == "aux" and pred.aux_type == "caption"]
    assert aux_blocks, "Caption was not peeled despite fallback repair"
