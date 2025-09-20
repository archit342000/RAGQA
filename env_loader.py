"""Simple loader for project-wide environment defaults."""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load_dotenv_once(path: str | Path = ".env") -> None:
    """Populate ``os.environ`` with key/value pairs from a ``.env`` file.

    The loader is intentionally tiny to avoid extra dependencies. It supports
    ``KEY=value`` pairs, ignoring blank lines and ``#`` comments. Existing
    environment variables always win so callers (or the test suite) can
    override defaults.
    """

    global _LOADED
    if _LOADED:
        return

    env_path = Path(path)
    if not env_path.exists():
        _LOADED = True
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value

    _LOADED = True
