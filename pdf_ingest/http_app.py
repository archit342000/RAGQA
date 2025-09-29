"""FastAPI wrapper for the deterministic parser."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

from .cli import parse_and_chunk
from .config import Config

app = FastAPI()


@app.post("/parse")
async def parse_endpoint(
    pdf: UploadFile = File(...),
    outdir: str = Form(...),
    mode: str = Form("fast"),
    config_json: str | None = Form(None),
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pdf = Path(tmp_dir) / pdf.filename
        with tmp_pdf.open("wb") as handle:
            shutil.copyfileobj(pdf.file, handle)
        overrides = json.loads(config_json) if config_json else None
        config = Config.from_sources(overrides=overrides)
        summary = parse_and_chunk(tmp_pdf, Path(outdir), config=config, mode=mode)
        return summary
