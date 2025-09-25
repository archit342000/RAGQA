from parser.classifier import (
    BodyStats,
    detect_implicit_section_start,
    flow_safe_decision,
    maybe_quarantine,
)


def _base_cfg():
    return {
        "thresholds": {
            "tau_main": 0.6,
            "tau_fail_safe_low": 0.52,
            "quarantined_aux_conf_min": 0.65,
        },
        "implicit_section": {"enable": True, "score_threshold": 0.1, "only_when_no_section": True},
    }


def test_bias_to_main_window_in_active_section():
    cfg = _base_cfg()
    stats = BodyStats(font_size=11.0, col_width=400.0, indent=24.0)
    block = {
        "ms": 0.56,
        "meta": {"indent": 24.0, "is_continuation": True, "col_id": 1},
        "reason": [],
        "kind": "unknown",
    }
    page_ctx = {"margin_left_pct": 0.05, "margin_right_pct": 0.95, "header_band": 0.12, "footer_band": 0.88}
    state = {
        "section_state": "IN_SECTION",
        "continuing_paragraph": True,
        "last_main_meta": {"indent": 24.0, "col_id": 1},
    }
    kind, score, reasons = flow_safe_decision(block, state, stats, cfg, page_ctx)
    assert kind == "main"
    assert 0.55 <= score <= 0.57
    assert "BiasToMainWindow" in reasons


def test_quarantined_aux_flag_sets_reason():
    cfg = _base_cfg()
    block = {"kind": "aux", "confidence": 0.5, "reason": []}
    maybe_quarantine(block, cfg)
    assert block["quarantined"] is True
    assert "QuarantinedLowConf" in block["reason"]


def test_implicit_section_requires_out_of_section_state():
    cfg = _base_cfg()
    stats = BodyStats(font_size=11.0, col_width=400.0, indent=24.0)
    candidate = {
        "page": 1,
        "index": 0,
        "kind": "main",
        "text": "A" * 200,
        "bbox": (0, 0, 100, 100),
        "meta": {"top_pct": 0.1, "char_count": 200, "halo_above": 20.0, "indent": 24.0, "col_id": 0},
        "source": object(),
        "reason": [],
    }
    state = {"section_state": "IN_SECTION", "continuing_paragraph": False}
    assert detect_implicit_section_start([candidate], state, cfg, stats) is None
    state["section_state"] = "OUT_OF_SECTION"
    control = detect_implicit_section_start([candidate], state, cfg, stats)
    assert control is not None
    assert control["kind"] == "control"
