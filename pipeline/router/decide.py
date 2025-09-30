from __future__ import annotations

from typing import Any, Dict


PROMOTE_KEY = "promote"
FALLBACK_KEY = "fallback"
STAY_KEY = "stay"


def decide_route(window_stats: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """Return routing decision for the *next* window.

    The decision follows the Option-C specification.  The keys that are read from
    ``window_stats`` mirror the telemetry fields produced by the router; callers
    may supply a subset and defaults will be used for missing values.
    """

    table_rate = float(window_stats.get("tablefig_per_page", 0.0))
    ocr_ratio = float(window_stats.get("ocr_page_ratio", 0.0))
    aux_leaks = int(window_stats.get("aux_leaks", 0))
    gpu_denials = int(window_stats.get("gpu_denials", 0))
    underemit_ratio = float(window_stats.get("underemit_ratio_vs_light", 1.0))
    flow_fence_hits = int(window_stats.get("flow_fence_hits", 0))
    timeouts = int(window_stats.get("timeouts", 0))

    if (
        table_rate
        >= float(cfg.get("router.promote.tablefig_per_page", 0.5))
        or ocr_ratio >= float(cfg.get("router.promote.ocr_page_pct", 0.10))
        or aux_leaks >= int(cfg.get("router.promote.aux_leaks_promote", 2))
    ):
        return PROMOTE_KEY

    if (
        gpu_denials >= int(cfg.get("router.fallback.gpu_denials", 2))
        or underemit_ratio < float(cfg.get("router.fallback.underemit_token_ratio", 0.60))
        or flow_fence_hits >= int(cfg.get("router.fallback.flow_fence_hits", 3))
        or timeouts > 0
    ):
        return FALLBACK_KEY

    return STAY_KEY
