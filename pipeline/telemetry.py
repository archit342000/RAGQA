from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .ocr import OCRDecision
from .triage import PageTriageSummary


@dataclass(slots=True)
class Telemetry:
    doc_id: str
    file_name: str
    started_at: float = field(default_factory=time.time)
    triage_latency_ms: float = 0.0
    ocr_decisions: List[OCRDecision] = field(default_factory=list)
    layout_pages: List[int] = field(default_factory=list)
    chunk_count: int = 0
    aux_buffered: int = 0
    aux_emitted: int = 0
    aux_discarded: int = 0
    segments: int = 0
    hard_boundaries: int = 0
    soft_boundaries: int = 0
    flags: List[str] = field(default_factory=list)
    extra_metrics: Dict[str, int] = field(default_factory=dict)
    pages: int = 0
    doc_time_ms: float = 0.0
    docling_pages: int = 0
    ocr_pages_cpu: int = 0
    ocr_pages_gpu: int = 0
    emitted_chunks: int = 0
    fallbacks_used: Dict[str, int] = field(
        default_factory=lambda: {"docling_fail": 0, "ocr": 0, "degraded": 0}
    )
    first_error_code: Optional[str] = None
    per_page_rows: List[Dict[str, object]] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "triage_latency_ms": self.triage_latency_ms,
            "ocr_ratio": sum(1 for d in self.ocr_decisions if d.should_ocr) / max(len(self.ocr_decisions), 1),
            "layout_pages": self.layout_pages,
            "chunk_count": self.chunk_count,
            "aux_buffered": self.aux_buffered,
            "aux_emitted": self.aux_emitted,
            "aux_discarded": self.aux_discarded,
            "segments": self.segments,
            "hard_boundaries": self.hard_boundaries,
            "soft_boundaries": self.soft_boundaries,
            "flags": self.flags,
            "extra_metrics": self.extra_metrics,
            "summary": {
                "pages": self.pages,
                "doc_time_ms": int(self.doc_time_ms),
                "docling_pages": self.docling_pages,
                "ocr_pages_cpu": self.ocr_pages_cpu,
                "ocr_pages_gpu": self.ocr_pages_gpu,
                "layout_pages": len(self.layout_pages),
                "emitted_chunks": self.emitted_chunks,
                "fallbacks_used": self.fallbacks_used,
                "first_error_code": self.first_error_code,
            },
            "per_page": self.per_page_rows,
            "stage_timings": self.stage_timings,
            "decisions": [
                {
                    "page": d.page_number,
                    "should_ocr": d.should_ocr,
                    "engine": d.engine,
                    "reason": d.reason,
                }
                for d in self.ocr_decisions
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def inc(self, key: str, amount: int = 1) -> None:
        if hasattr(self, key):
            current = getattr(self, key)
            if isinstance(current, int):
                setattr(self, key, current + amount)
                return
        self.extra_metrics[key] = self.extra_metrics.get(key, 0) + amount

    def flag(self, code: str) -> None:
        self.flags.append(code)
        if self.first_error_code is None:
            self.first_error_code = code

    def record_per_page(
        self,
        *,
        page: int,
        stage_used: str,
        latency_ms: float,
        text_len: int,
        fallback_applied: bool,
        error_codes: List[str],
    ) -> None:
        self.per_page_rows.append(
            {
                "doc_id": self.doc_id,
                "page": page,
                "stage_used": stage_used,
                "latency_ms": int(latency_ms),
                "text_len": text_len,
                "fallback_applied": fallback_applied,
                "error_codes": list(error_codes),
            }
        )


def record_triage(summary: PageTriageSummary, started_at: float) -> Telemetry:
    telemetry = Telemetry(doc_id=summary.doc_id, file_name=summary.file_name)
    telemetry.triage_latency_ms = (time.time() - started_at) * 1000
    telemetry.pages = len(summary.pages)
    return telemetry
