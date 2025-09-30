from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .chunker import Chunk, chunk_blocks
from .config import PipelineConfig, DEFAULT_CONFIG
from .docling_adapter import run_docling, fallback_blocks
from .layout_rescue import apply_layout_rescue, plan_layout_rescue
from .normalize import Block, normalise_blocks
from .ocr import apply_ocr, plan_ocr
from .telemetry import Telemetry, record_triage
from .triage import triage_document
from .watchdog import Watchdog

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
        doc_start = time.time()
        triage_start = time.time()
        triage_summary = triage_document(pdf_path, self.config)
        telemetry = record_triage(triage_summary, triage_start)
        telemetry.stage_timings["triage_ms"] = telemetry.triage_latency_ms
        telemetry.text_layer_pages = sum(1 for page in triage_summary.pages if page.digital_text)
        ocr_decisions = plan_ocr(triage_summary.pages, self.config)
        telemetry.ocr_decisions = ocr_decisions
        ocr_watchdog = Watchdog("ocr", self.config.timeouts.ocr_seconds)
        triage_pages, ocr_latency, ocr_timed_out = ocr_watchdog.run(
            lambda: apply_ocr(list(triage_summary.pages), ocr_decisions),
            on_timeout=lambda: list(triage_summary.pages),
        )
        telemetry.stage_timings["ocr_ms"] = ocr_latency
        if ocr_timed_out:
            telemetry.fallbacks_used["ocr"] = telemetry.fallbacks_used.get("ocr", 0) + 1
            telemetry.flag("OCR_TIMEOUT")
        triage_summary.pages = list(triage_pages)
        telemetry.ocr_pages_cpu = sum(
            1 for d in ocr_decisions if d.should_ocr and d.engine != "nougat"
        )
        telemetry.ocr_pages_gpu = sum(
            1 for d in ocr_decisions if d.should_ocr and d.engine == "nougat"
        )
        layout_decisions = plan_layout_rescue(triage_pages)
        telemetry.layout_pages = [d.page_number for d in layout_decisions if d.should_rescue]
        docling_watchdog = Watchdog("docling", self.config.timeouts.docling_seconds)
        docling_blocks, docling_latency, docling_timed_out = docling_watchdog.run(
            lambda: run_docling(triage_summary.pdf_bytes, triage_pages),
            on_timeout=lambda: fallback_blocks(triage_pages),
        )
        telemetry.stage_timings["docling_ms"] = docling_latency
        if docling_timed_out:
            telemetry.fallbacks_used["docling_fail"] = telemetry.fallbacks_used.get(
                "docling_fail", 0
            ) + 1
            telemetry.flag("DOCLING_TIMEOUT")
        telemetry.docling_pages = len({block.page_number for block in docling_blocks})
        layout_watchdog = Watchdog("layout", self.config.timeouts.layout_seconds)
        rescued_blocks, layout_latency, layout_timed_out = layout_watchdog.run(
            lambda: apply_layout_rescue(docling_blocks, layout_decisions),
            on_timeout=lambda: docling_blocks,
        )
        telemetry.stage_timings["layout_ms"] = layout_latency
        if layout_timed_out:
            telemetry.flag("LAYOUT_TIMEOUT")
        blocks = normalise_blocks(triage_summary.doc_id, rescued_blocks, self.config, telemetry)
        chunks = chunk_blocks(triage_summary.doc_id, blocks, self.config, telemetry)
        telemetry.chunk_count = len(chunks)
        telemetry.emitted_chunks = len(chunks)
        telemetry.doc_time_ms = (time.time() - doc_start) * 1000.0
        if telemetry.doc_time_ms > self.config.timeouts.doc_cap_seconds * 1000:
            telemetry.flag("DOC_TIMEOUT")

        page_sources: Dict[int, str] = {}
        for block in rescued_blocks:
            existing = page_sources.get(block.page_number)
            if existing == "docling":
                continue
            if block.source_stage == "docling" or existing is None:
                page_sources[block.page_number] = block.source_stage
        docling_page_numbers = {
            page for page, stage in page_sources.items() if stage == "docling"
        }
        ocr_pages = {decision.page_number for decision in ocr_decisions if decision.should_ocr}
        for page in triage_summary.pages:
            if page.page_number in ocr_pages:
                page.stage_used = "ocr"
                page.path_used = "ocr"
            elif page.page_number in docling_page_numbers:
                page.stage_used = "docling"
                page.path_used = "docling"
            else:
                stage = page_sources.get(page.page_number)
                if stage == "extractor":
                    page.stage_used = "extractor"
                    page.path_used = "extractor"
                    page.fallback_applied = True
                else:
                    page.stage_used = "triage"
                    page.path_used = "triage"
                    page.fallback_applied = True
            page.filter_relaxed = telemetry.was_filter_relaxed(page.page_number)
            telemetry.record_per_page(
                page=page.page_number,
                stage_used=page.stage_used,
                latency_ms=page.latency_ms,
                text_len=len(page.text),
                fallback_applied=page.fallback_applied,
                error_codes=page.error_codes or page.errors,
                len_text_fitz=page.len_text_fitz,
                len_text_pdfium=page.len_text_pdfium,
                len_text_pdfminer=page.len_text_pdfminer,
                has_type3=page.has_type3,
                has_cid=page.has_cid,
                has_tounicode=page.has_tounicode,
                path_used=page.path_used,
                filter_relaxed=page.filter_relaxed,
            )
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
