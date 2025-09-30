from pipeline.main_gate import main_gate


def test_main_gate_rejects_caption_and_activity() -> None:
    caption = {"type": "paragraph", "text": "Figure 1. Sample", "aux": {}}
    activity = {"type": "paragraph", "text": "Let's discuss", "aux": {}}
    passed_caption, reasons_caption = main_gate(caption)
    passed_activity, reasons_activity = main_gate(activity)
    assert not passed_caption and "caption_prefix" in reasons_caption
    assert not passed_activity and "lexical_aux" in reasons_activity


def test_main_gate_accepts_narrative_paragraph() -> None:
    paragraph = {"type": "paragraph", "text": "This is a narrative sentence.", "aux": {}}
    passed, reasons = main_gate(paragraph)
    assert passed and not reasons
