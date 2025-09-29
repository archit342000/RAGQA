from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

try:  # pragma: no cover - optional dependency
    import tiktoken
except Exception:  # pragma: no cover - fall back to whitespace counting
    tiktoken = None  # type: ignore

from .config import PipelineConfig
from .ids import make_chunk_id
from .normalize import Block

logger = logging.getLogger(__name__)

TEXT_TYPES = {"paragraph", "list", "item", "code", "footnote"}
SIDECAR_TYPES = {"table", "figure", "caption"}
DISCOURSE_SPLIT_RE = re.compile(r"\b(Background|Method|Methods|Results|Conclusion|Conclusions)\b", re.IGNORECASE)

if tiktoken is not None:  # pragma: no cover - heavy dependency
    try:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODER = None
else:  # pragma: no cover
    _ENCODER = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENCODER is not None:  # pragma: no cover - depends on tiktoken
        return len(_ENCODER.encode(text))
    return max(1, len(text.split()))


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    page_span: List[int]
    heading_path: List[str]
    text: str
    token_count: int
    sidecars: List[dict]
    evidence_spans: List[dict]
    quality: dict


@dataclass
class _ChunkBuilder:
    doc_id: str
    heading_path: List[str]
    token_cfg: PipelineConfig
    text: str = ""
    evidence_spans: List[dict] = field(default_factory=list)
    sidecars: List[dict] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    ocr_pages: set[int] = field(default_factory=set)
    rescued: bool = False
    notes: set[str] = field(default_factory=set)

    def add_heading_context(self, headings: Iterable[str]) -> None:
        for heading in headings:
            clean = heading.strip()
            if not clean:
                continue
            if self.text:
                self.text += "\n\n"
            self.text += clean

    def add_text(self, block: Block, text: str) -> None:
        snippet = text.strip()
        if not snippet:
            return
        if self.text:
            self.text += "\n\n"
        start = len(self.text)
        self.text += snippet
        end = len(self.text)
        self.evidence_spans.append({
            "para_block_id": block.block_id,
            "start": start,
            "end": end,
        })
        if block.page not in self.pages:
            self.pages.append(block.page)
        if block.source.get("stage") == "ocr":
            self.ocr_pages.add(block.page)
        if block.source.get("stage") == "layout":
            self.rescued = True
        token_len = count_tokens(self.text)
        if token_len > self.token_cfg.chunk.tokens.maximum:
            logger.warning(
                "Chunk exceeded maximum tokens (%s > %s) while building",
                token_len,
                self.token_cfg.chunk.tokens.maximum,
            )
        self.notes.update(filter(None, [block.source.get("notes")]))

    @property
    def token_count(self) -> int:
        return count_tokens(self.text)

    def page_span(self) -> List[int]:
        if not self.pages:
            return [0, 0]
        return [min(self.pages), max(self.pages)]

    def build(self) -> Chunk:
        quality = {
            "ocr_pages": len(self.ocr_pages),
            "rescued": self.rescued,
            "notes": ",".join(sorted(self.notes)) if self.notes else "",
        }
        return Chunk(
            chunk_id=make_chunk_id(),
            doc_id=self.doc_id,
            page_span=self.page_span(),
            heading_path=list(self.heading_path),
            text=self.text,
            token_count=self.token_count,
            sidecars=list(self.sidecars),
            evidence_spans=list(self.evidence_spans),
            quality=quality,
        )

    def attach_sidecar(self, entry: dict) -> None:
        self.sidecars.append(entry)

    def reset_for_next(self) -> None:
        self.text = ""
        self.evidence_spans.clear()
        self.sidecars.clear()
        self.pages.clear()
        self.ocr_pages.clear()
        self.rescued = False
        self.notes.clear()


def _update_heading_path(current: List[str], block: Block) -> List[str]:
    if block.heading_level is None or block.heading_level <= 0:
        if block.text.strip():
            current = list(current)
            if current:
                current[-1] = block.text.strip()
            else:
                current = [block.text.strip()]
        return current
    level_index = block.heading_level - 1
    new_path = current[:level_index]
    new_path.append(block.text.strip())
    return new_path


def _split_for_chunking(text: str, max_tokens: int) -> List[str]:
    tokens = count_tokens(text)
    if tokens <= max_tokens:
        return [text]
    pieces = re.split(r"\n\n+", text)
    if len(pieces) == 1:
        match_positions = [m.start() for m in DISCOURSE_SPLIT_RE.finditer(text)]
        if match_positions:
            splits: List[str] = []
            last = 0
            for pos in match_positions:
                if pos > last:
                    splits.append(text[last:pos])
                    last = pos
            splits.append(text[last:])
            pieces = [s for s in splits if s]
        if len(pieces) <= 1:
            pieces = re.split(r"(?<=[.!?])\s+", text)
    segments: List[str] = []
    for piece in pieces:
        running = piece.strip()
        if not running:
            continue
        if count_tokens(running) <= max_tokens:
            segments.append(running)
            continue
        words = running.split()
        if len(words) > max_tokens:
            step = max_tokens
            for idx in range(0, len(words), step):
                segment = " ".join(words[idx : idx + step])
                if segment:
                    segments.append(segment)
            continue
        cursor = 0
        while cursor < len(running):
            chunk = running[cursor : cursor + max(128, len(running) // 3)]
            segments.append(chunk)
            cursor += len(chunk)
    return segments


def _drain_pending_sidecars(waiting: Dict[int, List[dict]], pages: Iterable[int]) -> List[dict]:
    collected: List[dict] = []
    for page in sorted(set(pages)):
        collected.extend(waiting.pop(page, []))
    return collected


def chunk_blocks(doc_id: str, blocks: Sequence[Block], config: PipelineConfig) -> List[Chunk]:
    tokens_cfg = config
    chunks: List[Chunk] = []
    builder: _ChunkBuilder | None = None
    heading_path: List[str] = []
    pending_heading_texts: List[str] = []
    pending_sidecars: Dict[int, List[dict]] = {}

    def _start_new_chunk() -> _ChunkBuilder:
        nonlocal builder
        builder = _ChunkBuilder(doc_id=doc_id, heading_path=list(heading_path), token_cfg=tokens_cfg)
        if pending_heading_texts:
            builder.add_heading_context(pending_heading_texts)
        return builder

    def _finalise_current_chunk() -> None:
        nonlocal builder
        if builder is None or not builder.text.strip():
            builder = None
            return
        builder.sidecars.extend(_drain_pending_sidecars(pending_sidecars, builder.pages))
        chunk = builder.build()
        if chunk.token_count > config.chunk.tokens.maximum:
            logger.warning("Chunk %s exceeds hard maximum tokens (%s)", chunk.chunk_id, chunk.token_count)
        chunks.append(chunk)
        builder = None

    for block in blocks:
        if block.type == "heading":
            heading_path = _update_heading_path(heading_path, block)
            pending_heading_texts = [block.text.strip()]
            _finalise_current_chunk()
            continue

        if block.type in SIDECAR_TYPES:
            entry = {"type": block.type, "page": block.page, "text": block.text}
            if builder is not None and builder.text.strip():
                builder.attach_sidecar(entry)
            else:
                pending_sidecars.setdefault(block.page, []).append(entry)
            continue

        if block.type not in TEXT_TYPES:
            continue

        segments = _split_for_chunking(block.text, config.chunk.tokens.maximum)
        for segment in segments:
            if builder is None:
                _start_new_chunk()
            elif builder.token_count >= config.chunk.tokens.target:
                projected = count_tokens(builder.text + "\n\n" + segment)
                if projected > config.chunk.tokens.maximum:
                    _finalise_current_chunk()
                    _start_new_chunk()
            builder.add_text(block, segment)
            builder.sidecars.extend(_drain_pending_sidecars(pending_sidecars, [block.page]))
            if builder.token_count >= config.chunk.tokens.target:
                _finalise_current_chunk()
                pending_heading_texts = []

    if builder is not None and builder.text.strip():
        _finalise_current_chunk()

    if pending_sidecars and chunks:
        chunks[-1].sidecars.extend(_drain_pending_sidecars(pending_sidecars, chunks[-1].page_span))

    return chunks
