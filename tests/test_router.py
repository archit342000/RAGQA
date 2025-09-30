from pipeline.router.decide import decide_route


def test_decide_route_promote_and_fallback():
    cfg = {
        "router.promote.tablefig_per_page": 0.5,
        "router.promote.ocr_page_pct": 0.10,
        "router.promote.aux_leaks_promote": 2,
        "router.fallback.gpu_denials": 2,
        "router.fallback.underemit_token_ratio": 0.60,
        "router.fallback.flow_fence_hits": 3,
    }

    assert decide_route({"tablefig_per_page": 0.6}, cfg) == "promote"
    assert decide_route({"ocr_page_ratio": 0.2}, cfg) == "promote"
    assert decide_route({"aux_leaks": 3}, cfg) == "promote"

    assert decide_route({"gpu_denials": 3}, cfg) == "fallback"
    assert decide_route({"underemit_ratio_vs_light": 0.5}, cfg) == "fallback"
    assert decide_route({"flow_fence_hits": 3}, cfg) == "fallback"
    assert decide_route({"timeouts": 1}, cfg) == "fallback"

    assert decide_route({}, cfg) == "stay"
