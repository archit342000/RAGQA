"""Minimal FastAPI wrapper for synchronous parsing."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from chunking import Chunker
from parse_and_chunk import _collect_stats, _dump_json, _write_jsonl
from parser import PDFParser, load_config

app = FastAPI(title="PDF Parse and Chunk")


@app.post("/parse")
async def parse_endpoint(
    pdf: UploadFile = File(...),
    config_file: UploadFile | None = File(None),
    mode: str = "fast",
):
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="A PDF file must be provided")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pdf_path = temp_path / pdf.filename
        with pdf_path.open("wb") as handle:
            shutil.copyfileobj(pdf.file, handle)
        config_path = None
        if config_file:
            config_path = temp_path / "config.json"
            with config_path.open("wb") as handle:
                shutil.copyfileobj(config_file.file, handle)
        config = load_config(str(config_path) if config_path else None)
        parser = PDFParser(config)
        tables_dir = temp_path / "tables"
        document = parser.parse(str(pdf_path), mode=mode, tables_out=tables_dir)
        chunker = Chunker(config)
        chunks = chunker.chunk_document(document)

        out_dir = temp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl([chunk.__dict__ for chunk in chunks], out_dir / "chunks.jsonl")
        _dump_json(document.config_used, out_dir / "config_used.json")
        stats = _collect_stats(document, chunks)
        _dump_json(stats, out_dir / "stats.json")

        return {
            "doc_id": document.doc_id,
            "chunks": len(chunks),
            "stats": stats,
            "artifact_dir": str(out_dir),
        }
