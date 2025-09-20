"""Utilities for constructing bounded text windows from parsed documents."""

from __future__ import annotations

from typing import Dict, List

try:
    import tiktoken
except Exception:  # pragma: no cover - tokenizer optional
    tiktoken = None


def _get_encoder() -> "tiktoken.Encoding | None":
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model("gpt-3.5-turbo")
    except Exception:  # pragma: no cover - fallback path
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover - tokenizer unavailable
            return None


_ENCODER = _get_encoder()


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENCODER is None:
        # Rough heuristic: 4 characters ≈ 1 token
        return max(len(text) // 4, 1)
    try:
        return len(_ENCODER.encode(text))
    except Exception:  # pragma: no cover - defensive fallback
        return max(len(text) // 4, 1)


def make_windows(
    parsed_doc: Dict,
    max_pages: int = 2,
    max_chars: int = 12_000,
    max_tokens: int = 1_800,
) -> List[Dict]:
    """Merge consecutive pages into bounded windows for downstream synthesis."""

    doc_id = parsed_doc.get("doc_id", "doc")
    doc_name = parsed_doc.get("doc_name", doc_id)
    pages = sorted(parsed_doc.get("pages", []), key=lambda p: p.get("page_num", 0))

    windows: List[Dict] = []
    current: List[Dict] = []

    def flush() -> None:
        if not current:
            return
        window_pages = current[:]
        texts = [page.get("text") or "" for page in window_pages]
        window_text = "\n\n".join(texts).strip("\n")
        token_count = _count_tokens(window_text)
        window = {
            "window_id": f"{doc_id}:p{window_pages[0]['page_num']}-p{window_pages[-1]['page_num']}",
            "doc_id": doc_id,
            "doc_name": doc_name,
            "page_start": window_pages[0]["page_num"],
            "page_end": window_pages[-1]["page_num"],
            "window_text": window_text,
            "char_len": len(window_text),
            "token_count": token_count,
        }
        windows.append(window)
        current.clear()

    for page in pages:
        text = page.get("text") or ""
        if not text.strip():
            continue
        candidate_pages = current + [page]
        candidate_text = "\n\n".join(p.get("text") or "" for p in candidate_pages).strip("\n")
        candidate_chars = len(candidate_text)
        candidate_tokens = _count_tokens(candidate_text)
        if (
            len(candidate_pages) > max_pages
            or candidate_chars > max_chars
            or candidate_tokens > max_tokens
        ):
            flush()
            current.append(page)
            continue
        current.append(page)
    flush()
    return windows
