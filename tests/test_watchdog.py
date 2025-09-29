import time

from pipeline.watchdog import Watchdog


def test_watchdog_triggers_timeout_and_fallback():
    watchdog = Watchdog("slow-stage", timeout_seconds=0.01)

    def slow_fn():
        time.sleep(0.05)
        return "done"

    result, elapsed, timed_out = watchdog.run(slow_fn, on_timeout=lambda: "fallback")

    assert timed_out is True
    assert result == "fallback"
    assert elapsed >= 0
