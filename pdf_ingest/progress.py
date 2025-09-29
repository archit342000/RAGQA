"""Progress tracking helpers supporting resume-safe ingestion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logging import utc_now


def _now_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PageProgress:
    index: int
    status: str = "pending"
    ocr_mode: str = "none"
    chunks_emitted: List[str] = field(default_factory=list)
    tables_emitted: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    dpi_used: int | None = None
    tsv_lines: int = 0
    alerts: List[str] = field(default_factory=list)
    rois_considered: int = 0
    rois_ocrd: int = 0
    scaled_due_to_megapixels: bool = False
    surface_consumed_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "status": self.status,
            "ocr_mode": self.ocr_mode,
            "chunks_emitted": list(self.chunks_emitted),
            "tables_emitted": list(self.tables_emitted),
            "notes": list(self.notes),
            "dpi_used": self.dpi_used,
            "tsv_lines": self.tsv_lines,
            "alerts": list(self.alerts),
            "rois_considered": self.rois_considered,
            "rois_ocrd": self.rois_ocrd,
            "scaled_due_to_megapixels": self.scaled_due_to_megapixels,
            "surface_consumed_ratio": self.surface_consumed_ratio,
        }


@dataclass
class ProgressState:
    doc_id: str
    input_pdf: str
    mode: str
    pages_total: int
    started_at: str = field(default_factory=_now_ts)
    completed_at: Optional[str] = None
    completed: bool = False
    next_chunk_index: int = 0
    last_chunk_id: Optional[str] = None
    pending_paragraphs: List[Dict[str, Any]] = field(default_factory=list)
    counters: Dict[str, Any] = field(default_factory=lambda: {
        "lines_total": 0,
        "tables_emitted": 0,
        "captions_extracted": 0,
        "skipped_tables": 0,
        "noise_dropped": 0,
        "tsv_empty_alerts": 0,
    })
    pages: List[PageProgress] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "input_pdf": self.input_pdf,
            "mode": self.mode,
            "pages_total": self.pages_total,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "completed": self.completed,
            "next_chunk_index": self.next_chunk_index,
            "last_chunk_id": self.last_chunk_id,
            "pending_paragraphs": self.pending_paragraphs,
            "counters": self.counters,
            "pages": [page.to_dict() for page in self.pages],
        }


class ProgressTracker:
    """Persist progress.json after each page to enable resumable runs."""

    def __init__(self, path: Path, *, doc_id: str, input_pdf: Path, mode: str, pages_total: int) -> None:
        self.path = path
        self.state = ProgressState(doc_id=doc_id, input_pdf=str(input_pdf), mode=mode, pages_total=pages_total)
        if path.exists():
            self._load_existing()
        else:
            self._initialise_pages(pages_total)
            self.flush()

    def _load_existing(self) -> None:
        data = json.loads(self.path.read_text())
        if not isinstance(data, dict):
            raise ValueError("progress.json must be a JSON object")
        self.state.doc_id = data.get("doc_id", self.state.doc_id)
        self.state.input_pdf = data.get("input_pdf", self.state.input_pdf)
        self.state.mode = data.get("mode", self.state.mode)
        self.state.pages_total = data.get("pages_total", self.state.pages_total)
        self.state.started_at = data.get("started_at", self.state.started_at)
        self.state.completed = bool(data.get("completed", False))
        self.state.completed_at = data.get("completed_at")
        self.state.next_chunk_index = int(data.get("next_chunk_index", 0))
        self.state.last_chunk_id = data.get("last_chunk_id")
        self.state.pending_paragraphs = list(data.get("pending_paragraphs", []))
        self.state.counters = dict(data.get("counters", self.state.counters))
        pages_payload = data.get("pages", [])
        pages: List[PageProgress] = []
        for idx, payload in enumerate(pages_payload):
            page = PageProgress(index=payload.get("index", idx))
            page.status = payload.get("status", "pending")
            page.ocr_mode = payload.get("ocr_mode", "none")
            page.chunks_emitted = list(payload.get("chunks_emitted", []))
            page.tables_emitted = list(payload.get("tables_emitted", []))
            page.notes = list(payload.get("notes", []))
            page.dpi_used = payload.get("dpi_used")
            page.tsv_lines = int(payload.get("tsv_lines", 0))
            page.alerts = list(payload.get("alerts", []))
            page.rois_considered = int(payload.get("rois_considered", 0))
            page.rois_ocrd = int(payload.get("rois_ocrd", 0))
            page.scaled_due_to_megapixels = bool(payload.get("scaled_due_to_megapixels", False))
            page.surface_consumed_ratio = float(payload.get("surface_consumed_ratio", 0.0))
            pages.append(page)
        if not pages:
            self._initialise_pages(self.state.pages_total)
        else:
            self.state.pages = pages

    def _initialise_pages(self, pages_total: int) -> None:
        self.state.pages = [PageProgress(index=i) for i in range(pages_total)]

    def page_started(self, index: int, *, ocr_mode: str) -> None:
        page = self.state.pages[index]
        page.status = "processing"
        page.ocr_mode = ocr_mode
        page.notes = []
        page.dpi_used = None
        page.tsv_lines = 0
        page.alerts = []
        page.rois_considered = 0
        page.rois_ocrd = 0
        page.scaled_due_to_megapixels = False
        page.surface_consumed_ratio = 0.0
        self.flush()

    def page_completed(
        self,
        index: int,
        *,
        chunks: List[str],
        tables: List[str],
        notes: Optional[List[str]] = None,
        counters: Optional[Dict[str, Any]] = None,
        pending_paragraphs: Optional[List[Dict[str, Any]]] = None,
        last_chunk_id: Optional[str] = None,
        dpi_used: Optional[int] = None,
        tsv_lines: int = 0,
        alerts: Optional[List[str]] = None,
        rois_considered: int = 0,
        rois_ocrd: int = 0,
        scaled_due_to_megapixels: bool = False,
        surface_consumed_ratio: float = 0.0,
    ) -> None:
        page = self.state.pages[index]
        page.status = "done"
        page.chunks_emitted = list(chunks)
        page.tables_emitted = list(tables)
        if notes:
            page.notes = list(notes)
        if alerts:
            page.alerts = list(alerts)
        if dpi_used is not None:
            page.dpi_used = dpi_used
        page.tsv_lines = tsv_lines
        page.rois_considered = rois_considered
        page.rois_ocrd = rois_ocrd
        page.scaled_due_to_megapixels = scaled_due_to_megapixels
        page.surface_consumed_ratio = surface_consumed_ratio
        if counters:
            self.state.counters.update(counters)
        if pending_paragraphs is not None:
            self.state.pending_paragraphs = pending_paragraphs
        if last_chunk_id is not None:
            self.state.last_chunk_id = last_chunk_id
        self.flush()

    def update_next_chunk_index(self, value: int) -> None:
        self.state.next_chunk_index = value
        self.flush()

    def mark_completed(self) -> None:
        self.state.completed = True
        self.state.completed_at = utc_now()
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state.to_dict(), indent=2))

