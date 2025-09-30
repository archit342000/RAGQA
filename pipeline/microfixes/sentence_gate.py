from __future__ import annotations

import re
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..chunker import Chunk

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"\”\’\)]*\s*$")
_ABBR_DENYLIST_RE = re.compile(r"\b(Mr|Mrs|Dr|Prof|Inc|Ltd|Fig|Etc|e\.g|i\.e|No|Vol|pp|vs)\.$", re.IGNORECASE)
_AUX_DELAY_MAX_SEGMENTS = 1


def _is_sentence_closed(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    last_line = stripped.splitlines()[-1]
    if not _SENTENCE_END_RE.search(last_line):
        return False
    return not _ABBR_DENYLIST_RE.search(last_line)


def sentence_closure_gate(chunks: List["Chunk"]) -> Tuple[List["Chunk"], int]:
    """Delay auxiliary chunks until a closing sentence is observed."""

    ordered: List["Chunk"] = []
    pending_aux: List["Chunk"] = []
    delayed_aux = 0
    segments_waited = 0

    def _flush_pending(force: bool = False) -> None:
        nonlocal pending_aux, delayed_aux, segments_waited
        if not pending_aux:
            segments_waited = 0
            return
        if force:
            delayed_aux += len(pending_aux)
        ordered.extend(pending_aux)
        pending_aux = []
        segments_waited = 0

    for chunk in chunks:
        if chunk.is_main_only:
            ordered.append(chunk)
            if not pending_aux:
                continue
            segments_waited += 1
            if _is_sentence_closed(chunk.text) or segments_waited > _AUX_DELAY_MAX_SEGMENTS:
                _flush_pending(force=segments_waited > _AUX_DELAY_MAX_SEGMENTS)
        elif chunk.is_aux_only:
            if not ordered:
                pending_aux.append(chunk)
                continue
            last_main = next((c for c in reversed(ordered) if c.is_main_only), None)
            if last_main and _is_sentence_closed(last_main.text):
                ordered.append(chunk)
            else:
                pending_aux.append(chunk)
        else:
            ordered.append(chunk)

    if pending_aux:
        _flush_pending(force=True)

    return ordered, delayed_aux
