from tests.test_classifier import _build_doc, _make_block

from parser.classifier import classify_blocks


def test_activity_block_detected_and_does_not_absorb_main():
    body = _make_block(
        "This paragraph establishes the body style for measurements.",
        (80, 180, 520, 260),
        font_size=12,
        col_id=1,
    )
    body.attrs["indent"] = 18

    callout = _make_block(
        "Activity: Discuss the outcome with your partner.",
        (100, 270, 520, 340),
        font_size=12,
        col_id=1,
    )
    callout.attrs["indent"] = 40

    trailing = _make_block(
        "The main text resumes at the normal margin after the callout.",
        (80, 350, 520, 430),
        font_size=12,
        col_id=1,
    )
    trailing.attrs["indent"] = 18

    doc = _build_doc([body, callout, trailing])
    preds = classify_blocks(doc)
    callout_pred = next(p for p in preds if p.text.startswith("Activity"))
    trailing_pred = next(p for p in preds if "resumes" in p.text)
    assert callout_pred.kind == "aux"
    assert callout_pred.aux_type == "callout"
    assert any("Callout" in reason for reason in callout_pred.reason)
    assert trailing_pred.kind == "main"
