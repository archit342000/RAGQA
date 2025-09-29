from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .triage import PageTriageResult

try:
    import fitz  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional import
    fitz = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DoclingBlock:
    page_number: int
    block_type: str
    text: str
    bbox: tuple[float, float, float, float] | None = None
    heading_level: Optional[int] = None
    heading_path: List[str] = field(default_factory=list)
    source_stage: str = "docling"
    source_tool: str = "docling"
    source_version: str = "unknown"
    aux: dict = field(default_factory=dict)


def _load_docling() -> Optional[object]:  # pragma: no cover - optional import
    try:
        from docling.document_converter import DocumentConverter  # type: ignore

        return DocumentConverter
    except Exception:
        logger.warning("Docling is not available; falling back to triage text")
        return None


def _normalise_heading_level(level: Optional[int]) -> Optional[int]:
    if level is None:
        return None
    try:
        level_int = int(level)
    except (TypeError, ValueError):
        return None
    return max(1, min(level_int, 6))


def run_docling(triage: Iterable[PageTriageResult]) -> List[DoclingBlock]:
    converter_cls = _load_docling()
    if converter_cls is None:
        return _fallback_blocks(triage)

    # We intentionally defer imports and heavy work until needed so ZeroGPU
    # deployments only pay for Docling when a document is processed.
    try:  # pragma: no cover - external dependency heavy to test
        converter = converter_cls()
    except Exception as exc:  # fallback to triage text when Docling fails
        logger.exception("Docling initialisation failed: %s", exc)
        return _fallback_blocks(triage)

    blocks: List[DoclingBlock] = []
    for page in triage:
        try:
            result = converter.convert_bytes(page.text.encode("utf-8"), "text/plain")
        except Exception as exc:  # fallback on per-page failure
            logger.warning("Docling conversion failed on page %s: %s", page.page_number, exc)
            blocks.extend(_fallback_blocks([page]))
            continue
        try:
            md_blocks = result.document.export_structured()  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("Docling structured export failed on page %s: %s", page.page_number, exc)
            blocks.extend(_fallback_blocks([page]))
            continue
        for seq, md_block in enumerate(md_blocks):
            block_type = str(md_block.get("type", "paragraph"))
            text = str(md_block.get("text", ""))
            bbox = md_block.get("bbox")
            heading_level = _normalise_heading_level(md_block.get("heading_level"))
            heading_path = list(md_block.get("heading_path") or [])
            blocks.append(
                DoclingBlock(
                    page_number=page.page_number,
                    block_type=block_type,
                    text=text,
                    bbox=tuple(bbox) if bbox else None,
                    heading_level=heading_level,
                    heading_path=heading_path,
                    source_stage="docling",
                    source_tool="docling",
                    source_version=getattr(result, "version", "unknown"),
                )
            )
    return blocks


def _fallback_blocks(triage: Iterable[PageTriageResult]) -> List[DoclingBlock]:
    blocks: List[DoclingBlock] = []
    for page in triage:
        if not page.text.strip():
            continue
        blocks.append(
            DoclingBlock(
                page_number=page.page_number,
                block_type="paragraph",
                text=page.text,
                bbox=None,
                heading_level=None,
                heading_path=[],
                source_stage="triage",
                source_tool="pymupdf",
                source_version=getattr(fitz, "__doc__", "unknown") if "fitz" in globals() else "unknown",
            )
        )
    return blocks
