from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Any

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


def run_docling(pdf_bytes: bytes, triage: Iterable[PageTriageResult]) -> List[DoclingBlock]:
    converter_cls = _load_docling()
    if converter_cls is None:
        return fallback_blocks(triage)

    # We intentionally defer imports and heavy work until needed so ZeroGPU
    # deployments only pay for Docling when a document is processed.
    try:  # pragma: no cover - external dependency heavy to test
        converter = converter_cls()
    except Exception as exc:  # fallback to triage text when Docling fails
        logger.exception("Docling initialisation failed: %s", exc)
        return fallback_blocks(triage)

    doc_input = _build_docling_input(pdf_bytes)
    if doc_input is None:
        logger.warning("Docling DocumentInput helpers unavailable; using triage fallback")
        return fallback_blocks(triage)

    try:
        if hasattr(converter, "convert"):
            result = converter.convert(doc_input)
        elif hasattr(converter, "convert_bytes"):
            result = converter.convert_bytes(pdf_bytes, "application/pdf")  # pragma: no cover
        else:
            logger.warning("Docling converter has no supported convert method")
            return fallback_blocks(triage)
    except Exception as exc:
        logger.warning("Docling conversion failed: %s", exc)
        return fallback_blocks(triage)

    document = getattr(result, "document", None)
    if document is None:
        logger.warning("Docling result missing document attribute")
        return fallback_blocks(triage)

    try:
        md_blocks = document.export_structured()  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("Docling structured export failed: %s", exc)
        return fallback_blocks(triage)

    blocks: List[DoclingBlock] = []
    for md_block in md_blocks or []:
        block_type = str(md_block.get("type", "paragraph"))
        text = str(md_block.get("text", ""))
        bbox = md_block.get("bbox")
        heading_level = _normalise_heading_level(md_block.get("heading_level"))
        heading_path = list(md_block.get("heading_path") or [])
        page_number = md_block.get("page_number") or md_block.get("page")
        try:
            page_number_int = int(page_number) if page_number is not None else 1
        except (TypeError, ValueError):
            page_number_int = 1
        blocks.append(
            DoclingBlock(
                page_number=page_number_int,
                block_type=block_type,
                text=text,
                bbox=tuple(bbox) if bbox else None,
                heading_level=heading_level,
                heading_path=heading_path,
                source_stage="docling",
                source_tool="docling",
                source_version=str(getattr(result, "version", "unknown")),
            )
        )

    if not blocks:
        return fallback_blocks(triage)

    return blocks


def _build_docling_input(pdf_bytes: bytes) -> Optional[Any]:  # pragma: no cover - import heavy
    candidates = [
        ("docling.models.document", "DocumentInput"),
        ("docling.models.doc", "DocumentInput"),
        ("docling.document", "DocumentInput"),
        ("docling.document_input", "DocumentInput"),
        ("docling.document_converter", "DocumentInput"),
    ]
    for module_name, attr in candidates:
        try:
            module = __import__(module_name, fromlist=[attr])
            DocumentInput = getattr(module, attr)
        except Exception:
            continue
        for factory in ("from_bytes", "from_pdf_bytes", "from_data"):
            if hasattr(DocumentInput, factory):
                creator = getattr(DocumentInput, factory)
                try:
                    return creator(pdf_bytes, mime_type="application/pdf")
                except TypeError:
                    try:
                        return creator(pdf_bytes, "application/pdf")
                    except TypeError:
                        continue
        try:
            return DocumentInput(pdf_bytes, mime_type="application/pdf")
        except TypeError:
            pass
        except Exception:
            pass
        try:
            return DocumentInput(data=pdf_bytes, mime_type="application/pdf")
        except TypeError:
            pass
        except Exception:
            pass
    return None


def fallback_blocks(triage: Iterable[PageTriageResult]) -> List[DoclingBlock]:
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
