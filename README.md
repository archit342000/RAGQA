# CPU-First PDF Parser & Chunker

A lightweight, offline-friendly pipeline that converts PDFs into retrieval-ready
artifacts. The system favours CPU execution for Hugging Face Spaces running on
ZeroGPU, opportunistically using OCR when pages lack extractable glyphs.

## Features

- **Time-bounded parsing** with FAST (≤20s) and THOROUGH (≤40s) budgets, plus an
  AUTO mode that promotes short, noisy documents to the thorough budget.
- **On-demand OCR** using OCRmyPDF with conservative flags and early exits when
  the remaining budget would be exceeded.
- **Balanced chunking** that rebuilds paragraphs, applies TF–IDF cosine-drop
  boundaries, enforces 350–600 token targets, and limits overlaps to 10–15% at
  strong boundaries only.
- **Sidecars for captions and footnotes** detected via regex anchors within
  ±3 lines of figure/table markers, preventing leakage into body chunks.
- **Lightweight table detection** driven by delimiter regularity and digit
  ratios; confident tables are exported to CSV while low-confidence candidates
  are skipped with counters logged.
- **Evidence-rich outputs**: chunk page spans, provenance hashes, optional
  table CSV references, and structured stats.

## Installation

System packages for OCR (Ubuntu/Debian):

```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr qpdf ghostscript
```

Python dependencies:

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
```

## CLI Usage

```bash
python -m pdf_ingest.cli parse_and_chunk input.pdf --outdir out/doc --mode fast
```

Optional flags:

- `--mode fast|thorough|auto` – choose the time budget (AUTO promotes docs with
  ≤50 pages to THOROUGH).
- `--config path.json` – JSON overrides for the configuration (see below).

Artifacts are written under `out/doc/`:

```
chunks.jsonl
config_used.json
stats.json
/tables/*.csv
```

Each chunk record follows `schemas/chunks.schema.json`, while `stats.json`
complies with `schemas/stats.schema.json`.

## FastAPI Endpoint

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

POST `/parse` with a PDF (and optional JSON config overrides) to run the same
pipeline and receive a JSON summary.

## Configuration Defaults

Authoritative defaults from `pdf_ingest.config`:

```
glyph_min_for_text_page=200
fast_budget_s=20
thorough_budget_s=40
max_pages=500
table_digit_ratio>=0.4
chunk_token_target=350..600
overlap=0.1..0.15
noise_drop_ratio>0.3
```

Overridden values and runtime mode are persisted to `config_used.json` for
traceability.

## Testing

```bash
pytest
```

Unit tests cover chunk token estimation, overlap behaviour, and table delimiter
scoring. Extend with real PDF fixtures for integration coverage.

## Notes

- Token counts use a 4-characters-per-token heuristic for CPU speed; replace
  with a tokenizer if tighter accounting is required.
- Evidence byte ranges default to `null` when unavailable, but provenance hashes
  are always recorded.
- Table CSVs include `source_page,row_idx,col_idx,cell_bbox,text` columns, and
  `chunk.table_csv` references the CSV path plus `#rows,cols` metadata.
