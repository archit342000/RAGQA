from parser.classifier import BodyStats, flow_safe_decision, maybe_quarantine


def test_bias_to_main_window_defaults_inside_section():
    block = {
        "ms": 0.58,
        "meta": {"indent": 12.0, "is_continuation": False},
        "aux_type": None,
        "reason": [],
    }
    state = {
        "section_state": "IN_SECTION",
        "continuing_paragraph": False,
        "last_main_meta": {"indent": 12.0},
    }
    stats = BodyStats(font_size=11.0, col_width=320.0, indent=12.0)
    cfg = {"thresholds": {"tau_main": 0.60, "tau_fail_safe_low": 0.52, "tau_bias_high": 0.65}}
    page_ctx = {"margin_left_pct": 0.05, "margin_right_pct": 0.95, "header_band": 0.12, "footer_band": 0.88}
    decision, score, reasons = flow_safe_decision(block, state, stats, cfg, page_ctx)
    assert decision == "main"
    assert score == block["ms"]
    assert "FailSafe" in reasons


def test_maybe_quarantine_marks_low_confidence_aux():
    block = {"kind": "aux", "aux_type": "caption", "confidence": 0.4, "reason": []}
    cfg = {"thresholds": {"quarantined_aux_conf_min": 0.65}}
    maybe_quarantine(block, cfg)
    assert block.get("quarantined") is True
    assert "QuarantinedLowConf" in block["reason"]
