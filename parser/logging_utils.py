"""Structured logging helpers for the parser pipeline."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

try:
    import orjson
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without orjson
    orjson = None  # type: ignore


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("parser")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_event(event: str, **payload: Any) -> None:
    logger = configure_logger()
    data: Dict[str, Any] = {"event": event, **payload}
    if orjson:
        message = orjson.dumps(data).decode("utf-8")
    else:
        message = json.dumps(data, ensure_ascii=False)
    logger.info(message)
