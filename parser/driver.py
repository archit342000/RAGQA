"""Compatibility layer exposing simple parsing helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Sequence, Tuple

from .config import DEFAULT_CONFIG, ParserConfig
from .logging_utils import log_event
from .pdf_parser import PDFParser
from .types import LineSpan, PageParseResult, ParsedDocument, RunReport
from .utils.text import merge_whitespace


def parse_pdf(
    file_path: str,
    *,
    doc_id: str | None = None,
    mode: str = "fast",
    config: ParserConfig | None = None,
    tables_dir: str | Path | None = None,
) -> ParsedDocument:
    cfg = config or DEFAULT_CONFIG
    parser = PDFParser(cfg)
    tables_out = Path(tables_dir) if tables_dir else None
    return parser.parse(file_path, doc_id=doc_id, mode=mode, tables_out=tables_out)


def parse_text(
    file_path: str,
    *,
    doc_id: str | None = None,
    config: ParserConfig | None = None,
) -> ParsedDocument:
    cfg = config or DEFAULT_CONFIG
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    lines = [merge_whitespace(line) for line in text.splitlines()]
    line_objs: List[LineSpan] = []
    cursor = 0
    for idx, line in enumerate(lines):
        if not line:
            continue
        span = LineSpan(
            page_index=0,
            line_index=idx,
            text=line,
            bbox=None,
            char_start=cursor,
            char_end=cursor + len(line),
        )
        cursor += len(line) + 1
        line_objs.append(span)
    page = PageParseResult(
        page_index=0,
        glyph_count=sum(len(line.text) for line in line_objs),
        had_text=bool(line_objs),
        ocr_performed=False,
        lines=line_objs,
    )
    doc_hash = _hash_bytes(text.encode("utf-8"))
    document = ParsedDocument(
        doc_id=doc_id or path.stem,
        file_path=str(path),
        pages=[page],
        config_used=cfg.to_dict(),
        parse_time_s=0.0,
        content_hash=doc_hash,
    )
    return document


def parse_documents(
    files: Sequence[str],
    *,
    mode: str = "fast",
    config: ParserConfig | None = None,
) -> Tuple[List[ParsedDocument], RunReport]:
    cfg = config or DEFAULT_CONFIG
    parsed_docs: List[ParsedDocument] = []
    skipped: List[str] = []
    for file in files:
        path = Path(file)
        try:
            if path.suffix.lower() == ".pdf":
                document = parse_pdf(str(path), doc_id=path.stem, mode=mode, config=cfg)
            else:
                document = parse_text(str(path), doc_id=path.stem, config=cfg)
            parsed_docs.append(document)
        except Exception as exc:  # pragma: no cover - defensive logging
            log_event("parse_error", file=str(path), error=str(exc))
            skipped.append(str(path))
    total_pages = sum(len(doc.pages) for doc in parsed_docs)
    report = RunReport(
        total_docs=len(parsed_docs),
        total_pages=total_pages,
        truncated=False,
        skipped_docs=skipped,
        message="",
    )
    return parsed_docs, report


def _hash_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()
