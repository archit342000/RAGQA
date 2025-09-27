"""FastAPI wrapper exposing the parsing pipeline over HTTP."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

from .cli import parse_and_chunk
from .config import load_config

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
        config = load_config(None)
        if config_json:
            overrides = json.loads(config_json)
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        summary = parse_and_chunk(tmp_pdf, Path(outdir), config, mode=mode)
        return summary
