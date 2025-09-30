from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from .config import PipelineConfig
from .service import PipelineService

app = FastAPI(title="PDF Blocks Parser")
service = PipelineService(PipelineConfig.from_mapping({}))


@app.post("/parse")
async def parse_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        return JSONResponse({"error": "filename_missing"}, status_code=400)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / file.filename
        data = await file.read()
        path.write_bytes(data)
        result = service.process_pdf(str(path))
    payload = {
        "doc_id": result.doc_id,
        "doc_name": result.doc_name,
        "blocks": result.blocks_json(),
        "chunks": result.chunks_jsonl(),
        "triage": result.triage_rows,
        "telemetry": result.telemetry.to_dict(),
    }
    return JSONResponse(payload)
