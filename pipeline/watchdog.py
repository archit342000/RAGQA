from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Callable, Generic, Optional, Tuple, TypeVar


T = TypeVar("T")

logger = logging.getLogger(__name__)


class StageTimeout(RuntimeError):
    """Raised internally when a stage exceeds its watchdog deadline."""


class Watchdog(Generic[T]):
    """Simple thread-pool watchdog that aborts stages after a timeout."""

    def __init__(self, description: str, timeout_seconds: float) -> None:
        self.description = description
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        fn: Callable[[], T],
        *,
        on_timeout: Optional[Callable[[], T]] = None,
    ) -> Tuple[T, float, bool]:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn)
            try:
                result = future.result(timeout=self.timeout_seconds)
                elapsed = (time.perf_counter() - start) * 1000.0
                return result, elapsed, False
            except TimeoutError:
                elapsed = (time.perf_counter() - start) * 1000.0
                logger.warning("Watchdog fired for %s after %.2f ms", self.description, elapsed)
                if on_timeout is not None:
                    return on_timeout(), elapsed, True
                raise StageTimeout(f"Stage {self.description} exceeded {self.timeout_seconds}s")
