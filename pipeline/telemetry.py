from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List

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

    def to_dict(self) -> Dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "triage_latency_ms": self.triage_latency_ms,
            "ocr_ratio": sum(1 for d in self.ocr_decisions if d.should_ocr) / max(len(self.ocr_decisions), 1),
            "layout_pages": self.layout_pages,
            "chunk_count": self.chunk_count,
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


def record_triage(summary: PageTriageSummary, started_at: float) -> Telemetry:
    telemetry = Telemetry(doc_id=summary.doc_id, file_name=summary.file_name)
    telemetry.triage_latency_ms = (time.time() - started_at) * 1000
    return telemetry
