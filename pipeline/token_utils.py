from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import tiktoken
except Exception:  # pragma: no cover - fallback when tiktoken missing
    tiktoken = None  # type: ignore


if tiktoken is not None:  # pragma: no cover
    try:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODER = None
else:  # pragma: no cover
    _ENCODER = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENCODER is not None:  # pragma: no cover
        return len(_ENCODER.encode(text))
    return max(1, len(text.split()))
