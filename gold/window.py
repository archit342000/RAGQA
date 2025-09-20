"""Windowing utilities for LLM mining."""

from __future__ import annotations

from typing import Dict, List

import tiktoken


def _get_encoder() -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model("gpt-3.5-turbo")
    except Exception:  # pragma: no cover - fallback path
        return tiktoken.get_encoding("cl100k_base")


_ENCODER = _get_encoder()


def _count_tokens(text: str) -> int:
    try:
        return len(_ENCODER.encode(text))
    except Exception:  # pragma: no cover - defensive
        return max(len(text) // 4, 1)


def make_windows(
    parsed_doc: Dict,
    max_pages: int = 2,
    max_chars: int = 12_000,
    max_tokens: int = 1_800,
) -> List[Dict]:
    """Merge consecutive pages into bounded windows for LLM processing."""

    doc_id = parsed_doc.get("doc_id", "doc")
    doc_name = parsed_doc.get("doc_name", doc_id)
    pages = sorted(parsed_doc.get("pages", []), key=lambda p: p.get("page_num", 0))

    windows: List[Dict] = []
    current_pages: List[Dict] = []

    def flush_window() -> None:
        if not current_pages:
            return
        texts = [page["text"] for page in current_pages]
        window_text = "\n\n".join(texts)
        window = {
            "window_id": f"{doc_id}:p{current_pages[0]['page_num']}-p{current_pages[-1]['page_num']}",
            "doc_id": doc_id,
            "doc_name": doc_name,
            "page_start": current_pages[0]["page_num"],
            "page_end": current_pages[-1]["page_num"],
            "window_text": window_text,
        }
        windows.append(window)
        current_pages.clear()

    for page in pages:
        text = page.get("text") or ""
        if not text.strip():
            continue
        candidate_pages = current_pages + [page]
        candidate_text = "\n\n".join(p["text"] for p in candidate_pages)
        if (
            len(candidate_pages) > max_pages
            or len(candidate_text) > max_chars
            or _count_tokens(candidate_text) > max_tokens
        ):
            flush_window()
            current_pages.append(page)
        else:
            current_pages.append(page)
    flush_window()
    return windows
