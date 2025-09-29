from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .chunker import Chunk, chunk_blocks
from .config import PipelineConfig, DEFAULT_CONFIG
from .docling_adapter import run_docling
from .layout_rescue import apply_layout_rescue, plan_layout_rescue
from .normalize import Block, normalise_blocks
from .ocr import apply_ocr, plan_ocr
from .telemetry import Telemetry, record_triage
from .triage import triage_document

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineResult:
    doc_id: str
    doc_name: str
    blocks: List[Block]
    chunks: List[Chunk]
    triage_rows: List[List[str]]
    telemetry: Telemetry

    def blocks_json(self) -> List[dict]:
        return [
            {
                "doc_id": block.doc_id,
                "block_id": block.block_id,
                "page": block.page,
                "order": block.order,
                "type": block.type,
                "text": block.text,
                "bbox": block.bbox,
                "heading_level": block.heading_level,
                "heading_path": block.heading_path,
                "source": block.source,
                "aux": block.aux,
                "role": block.role,
                "aux_subtype": block.aux_subtype,
                "parent_block_id": block.parent_block_id,
                "role_confidence": block.role_confidence,
            }
            for block in self.blocks
        ]

    def chunks_jsonl(self) -> List[dict]:
        return [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "page_span": chunk.page_span,
                "heading_path": chunk.heading_path,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "sidecars": chunk.sidecars,
                "evidence_spans": chunk.evidence_spans,
                "quality": chunk.quality,
                "aux_groups": chunk.aux_groups,
                "notes": chunk.notes,
            }
            for chunk in self.chunks
        ]

    def write_outputs(self, out_dir: str | Path) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        blocks_path = path / f"{self.doc_id}_blocks.json"
        chunks_path = path / f"{self.doc_id}_chunks.jsonl"
        triage_path = path / f"{self.doc_id}_triage.csv"
        telemetry_path = path / f"{self.doc_id}_telemetry.json"

        blocks_path.write_text(json.dumps(self.blocks_json(), indent=2), encoding="utf-8")
        with chunks_path.open("w", encoding="utf-8") as fh:
            for record in self.chunks_jsonl():
                fh.write(json.dumps(record) + "\n")
        with triage_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(self.triage_rows)
        telemetry_path.write_text(self.telemetry.to_json(), encoding="utf-8")


class PipelineService:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG

    def process_pdf(self, pdf_path: str) -> PipelineResult:
        logger.info("Processing document: %s", pdf_path)
        start_time = time.time()
        triage_summary = triage_document(pdf_path, self.config)
        telemetry = record_triage(triage_summary, start_time)
        ocr_decisions = plan_ocr(triage_summary.pages, self.config)
        telemetry.ocr_decisions = ocr_decisions
        triage_pages = apply_ocr(list(triage_summary.pages), ocr_decisions)
        triage_summary.pages = list(triage_pages)
        layout_decisions = plan_layout_rescue(triage_pages)
        telemetry.layout_pages = [d.page_number for d in layout_decisions if d.should_rescue]
        docling_blocks = run_docling(triage_pages)
        rescued_blocks = apply_layout_rescue(docling_blocks, layout_decisions)
        blocks = normalise_blocks(triage_summary.doc_id, rescued_blocks, self.config, telemetry)
        chunks = chunk_blocks(triage_summary.doc_id, blocks, self.config, telemetry)
        telemetry.chunk_count = len(chunks)
        triage_rows = triage_summary.to_csv_rows()
        return PipelineResult(
            doc_id=triage_summary.doc_id,
            doc_name=triage_summary.file_name,
            blocks=blocks,
            chunks=chunks,
            triage_rows=triage_rows,
            telemetry=telemetry,
        )


def parse_to_chunks(pdf_path: str, out_dir: str | Path, config: PipelineConfig | None = None) -> PipelineResult:
    service = PipelineService(config)
    result = service.process_pdf(pdf_path)
    result.write_outputs(out_dir)
    return result
