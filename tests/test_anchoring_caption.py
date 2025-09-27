from parser.anchoring import anchor_captions


def test_anchor_captions_links_to_nearest_figure():
    blocks = [
        {"id": "fig1", "type": "aux", "subtype": "figure", "bbox": [0, 0, 100, 100], "text": "Figure"},
        {"id": "cap1", "type": "aux", "subtype": "caption", "bbox": [0, 110, 100, 140], "text": "Fig. 1"},
    ]
    anchored = anchor_captions(blocks, [], {})
    caption = next(blk for blk in anchored if blk["id"] == "cap1")
    assert caption["links"][0]["target_id"] == "fig1"
