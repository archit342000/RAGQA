# CPU-First PDF Parser & Chunker

Deterministic, resume-safe ingestion pipeline tuned for Hugging Face Spaces on
ZeroGPU. The parser streams page outputs, honours per-operation caps, and emits
retrieval-ready chunks, confident table CSVs, rich stats, and resumable progress
metadata.

## Features

- **Streaming parse** – pages are processed sequentially, with `chunks.jsonl`,
  `stats.json`, and `progress.json` updated after every page.
- **Per-operation caps** – no global timeout; rasterisations/page, OCR retries,
  and table probes are bounded to keep latency predictable.
- **Selective OCR** – multi-signal heuristics choose `none|partial|full` per
  page with neighbour smoothing. OCR policy, Tesseract flags, ROI caps, and
  concurrency are driven by the runtime config so the parser respects DPI,
  megapixel, and surface-area limits deterministically. When native text is
  missing, eligible ROIs are rasterised and fed through Tesseract TSV with
  DAWGs disabled for determinism.
- **Balanced chunking** – paragraphs rebuilt from lines, TF–IDF cosine drops
  determine topic boundaries, chunks target 350–600 tokens with 10–15% overlap
  only at strong boundaries, and captions/footnotes are routed to sidecars.
- **Table CSV export** – delimiter regularity scoring emits confident tables to
  `/tables/*.csv` with `source_page,row_idx,col_idx,cell_bbox,text` columns;
  low-confidence candidates are logged and skipped.
- **Evidence and provenance** – every chunk carries page spans, evidence hooks,
  and document hashes so downstream retrieval can audit coverage.

## Outputs

For each document the pipeline writes:

```
chunks.jsonl
config_used.json
progress.json
stats.json
/tables/*.csv
```

`stats.json` captures `parse_time_s`, `lines_total`, `tables_emitted`, `captions_extracted`, `chunks_n`, `avg_tokens`,
`noise_ratio`, `skipped_tables_n`, and `tsv_empty_alerts` for troubleshooting OCR coverage.

## Installation

System packages for OCR (Ubuntu/Debian):

```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
```

Python dependencies:

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
```

## CLI Usage

```bash
python -m pdf_ingest.cli input.pdf --outdir out/doc --mode fast
```

Optional flags:

- `--mode fast|thorough` – choose the heuristic profile.
- `--config path.json` – JSON overrides for the configuration (see below).
- `--toml-config path.toml` – Text-only OCR runtime config (TOML or INI).

Artifacts comply with the schemas under `schemas/`. `progress.json` captures the
resume state (per-page status, OCR mode, counters, pending paragraphs).

## FastAPI Endpoint

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

POST `/parse` with a PDF and optional JSON config overrides to run the parser
and receive a stats summary plus artifact directory.

## Configuration Defaults

Authoritative defaults from `pdf_ingest.config`:

```
glyph_min_for_text_page=200
table_digit_ratio>=0.4
table_score_conf>=0.6
chunk_tokens=350..600
overlap=0.1..0.15
ocr_retry=1
rasterizations_per_page<=2
junk_char_ratio>0.3
dpi_full=300
dpi_retry=400
```

Additional OCR controls live in `configs/ocr_runtime.example.toml`:

```
[ocr_policy]
promote_none_glyphs = 800
promote_none_glyphs_with_quality = 400
unicode_quality_threshold = 0.90
full_threshold_s_native = 0.25
probe_selectable_bands = 2

[work_caps]
max_page_megapixels = 10
max_rois_per_page = 3
max_doc_ocr_surface_multiplier = 1.0
dpi = 300
dpi_retry_small_roi = 400
retry_small_roi_height_frac = 0.25

[tesseract]
oem = 1
psm = 6
lang = eng
disable_dawgs = true
whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:%()-"

[concurrency]
ocr_workers = 2

[fail_fast]
roi_min_alnum_chars = 20
roi_max_retries = 1
emit_stub_on_fail = true
```

Config precedence is **CLI overrides > TOML/INI > JSON > built-in defaults**.
The merged configuration snapshot is written to `config_used.json` for auditing.

`progress.json` records ROI counts, megapixel scaling, and OCR surface
consumption so resume runs and monitoring can identify budget hits quickly.

## Testing

```bash
pytest
```

Unit tests cover chunk token estimation, overlap behaviour, OCR/table heuristics,
and streaming safeguards. Extend with PDF fixtures for integration coverage.

## Notes

- Token counts use a 4-characters-per-token heuristic for CPU speed; replace
  with a tokenizer if tighter accounting is required.
- Evidence byte ranges default to `null` when unavailable; provenance hashes are
  always recorded.
- `chunk.table_csv` references CSV paths plus `#rows,cols` metadata for quick
  inspection.
