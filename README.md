# CPU-First PDF Parsing & Chunking

This repository now ships a lightweight parsing and chunking stack tailored for
Hugging Face Spaces running on ZeroGPU. The pipeline converts uploaded PDFs into
retrieval-ready JSONL chunks, optional CSV tables, and detailed stats while
respecting strict latency budgets.

## Features

- **CPU-first extraction** powered by PyMuPDF with optional page-level OCR via
  Tesseract/pytesseract when glyph counts fall below a configurable threshold.
- **Time-bounded execution** with FAST (≤20s) and THOROUGH (≤40s) modes plus
  graceful early exits that still emit partial outputs and structured logs.
- **Balanced chunking** that grows paragraphs, applies TF–IDF cosine-drop topic
  boundaries, and keeps captions/footnotes as sidecars.
- **Table emission** using a single-pass delimiter confidence score that never
  blocks the pipeline.
- **Evidence-rich artifacts**: `chunks.jsonl`, `/tables/*.csv`, `stats.json`,
  and the resolved `config_used.json` for reproducibility.
- **CLI and optional HTTP wrapper** for easy integration into Spaces or local
  automation.

## Installation

```bash
pip install -r requirements.txt
```

System packages required for OCR:

```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr qpdf ghostscript
```

## CLI Usage

```bash
python parse_and_chunk.py input.pdf --outdir out/my_doc --mode fast
```

Optional flags:

- `--config path/to/config.json` – override defaults.
- `--mode fast|thorough|auto` – adjust time budgets. `auto` uses the thorough
  budget when the document has ≤50 pages.

Outputs follow the mandated layout:

```
/out/<doc_id>/chunks.jsonl
/out/<doc_id>/tables/*.csv
/out/<doc_id>/stats.json
/out/<doc_id>/config_used.json
```

`chunks.jsonl` records comply with `schemas/chunks.schema.json`, including
`page_spans`, `neighbors`, caption/footnote chunks, and provenance hashes.
`stats.json` matches `schemas/stats.schema.json`.

## FastAPI Endpoint

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

POST `/parse` with a PDF (and optional config JSON) to receive a JSON summary
plus the path to generated artifacts.

## Configuration

See `parser/config.py` for the authoritative defaults:

- `glyph_min_for_text_page=200`
- `fast_budget_s=20`
- `thorough_budget_s=40`
- `max_pages=500`
- `table_digit_ratio>=0.4`
- `chunk_token_target=350..600`
- `overlap=0.1..0.15`
- `noise_drop_ratio>0.3`
- `bbox_mode=line`

Override any subset via the CLI `--config` JSON or the HTTP API payload. The
resolved configuration for each run is saved to `config_used.json`.

## Token Estimation

Token counts are estimated using a 4-characters-per-token heuristic, chosen for
speed on CPU-only environments. Swap in a tokenizer if tighter accounting is
required.

## Testing

Run the test suite:

```bash
pytest
```

Unit tests cover glyph routing, caption detection, TF–IDF boundary selection,
and table confidence handling. Integration tests exercise native vs. scanned PDF
fixtures (see `examples/`).

## Examples

Synthetic sample outputs demonstrating the artifact structure live under
`examples/sample_native/` and `examples/sample_scanned/`.
