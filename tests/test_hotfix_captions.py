from tests.test_classifier import _build_doc, _make_block

from parser.classifier import classify_blocks


def test_caption_within_ring_forced_aux_and_linked():
    figure = _make_block("", (150, 150, 320, 320), block_type="figure", col_id=1)
    caption = _make_block(
        "Fig. 3 The solar system.",
        (155, 360, 315, 392),
        font_size=9,
        col_id=1,
    )
    doc = _build_doc([figure, caption])
    preds = classify_blocks(doc)
    cap_pred = next(p for p in preds if "Fig. 3" in p.text)
    assert cap_pred.kind == "aux"
    assert cap_pred.aux_type == "caption"
    assert cap_pred.anchor_hint is not None
    assert cap_pred.meta.get("caption_ring_forced") is True
    assert any("CaptionRingHit" in reason or "Caption" in reason for reason in cap_pred.reason)
