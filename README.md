# Hybrid Document Parser & HF Spaces Demo

This project ships a hybrid PDF parsing stack tailored for Retrieval-Augmented Generation (RAG) workloads alongside a Gradio UI
suited for Hugging Face Spaces deployments. The pipeline combines PyMuPDF-first extraction, selective LayoutParser zoning, repair
heuristics, and retrieval-aware chunking to preserve narrative flow while keeping auxiliary artefacts such as tables and footnotes
anchored for citation.

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # Optional heavy extras used by the layout pipeline
   pip install "PyMuPDF==1.24.*" "pdfplumber==0.11.*" "camelot-py[cv]==0.11.*" "ocrmypdf==16.*" \
     "layoutparser==0.3.4" "torch==2.3.*" "detectron2==0.6" "sentence-transformers==2.7.*" rapidfuzz==3.* scikit-image
   ```
2. Run the Spaces-oriented demo locally:
   ```bash
   python app.py
   ```
   or via Gradio's CLI:
   ```bash
   gradio app.py
   ```
3. Execute the end-to-end pipeline from the command line:
   ```bash
   python -m pipeline.run --pdf sample_docs/sample.pdf --emit-chunks artifacts/sample_chunks.jsonl
   ```

Upload one or more PDF, TXT, or Markdown files to preview per-page text, inspect routing decisions, and (optionally) view detailed
metrics via the gated debug panel.

## Hugging Face Spaces Notes

- Designed for the CPU runtime; PyMuPDF parsing is fast and LayoutParser routing is gated by heuristics.
- The hi-res OCR path remains gated by an environment flag and falls back gracefully when Detectron2/Tesseract are unavailable.
- Keep uploads small (<800 pages) for responsive demos. The `MAX_PAGES` flag trims larger documents with a visible warning.

## Configuration via Environment Variables

Default configuration values live in the project's `.env` file; edit it to customise behaviour locally or on Spaces. The parser
reads the following variables (defaults in parentheses):

- `MAX_PAGES` (`800`): hard limit on processed pages per document.
- `MAX_TOTAL_PAGES` (`1200`): aggregate ceiling across all uploaded documents in one run.
- `MIN_CHARS_PER_PAGE` (`200`): threshold used to detect sparse pages during heuristic checks.
- `FALLBACK_EMPTY_PAGE_RATIO` (`0.3`): ratio of low-density pages that triggers the legacy `unstructured` fallback (still
  available in the UI).
- `UNSTRUCTURED_STRATEGY` (`fast`): default fallback strategy.
- `ENABLE_HI_RES` (`false`): when set to `true`, enables the "High accuracy" parsing mode in the UI. Otherwise the option is
  disabled and requests are clamped to the fast path.
- `SHOW_DEBUG` (`false`): when `true`, exposes the developer-focused metrics panel in the UI.
- `CHUNK_MODE_DEFAULT` (`semantic`): default chunking strategy when the UI control is untouched.
- `MAX_TOTAL_TOKENS_FOR_CHUNKING` (`300000`): guardrail enforced by the chunker to keep latency bounded.
- `SEMANTIC_MODEL_NAME` (`intfloat/e5-small-v2`): embedding model used by the semantic chunker in the UI.
- `TOKENIZER_NAME` (`hf-internal-testing/llama-tokenizer`): tokenizer used for legacy token accounting.

### Hybrid Pipeline Overview

The updated `pipeline/` package orchestrates a multi-stage PDF workflow designed for robust mixed-layout parsing:

1. **Ingestion (`pipeline.ingest.pdf_parser`)** – PyMuPDF extracts spans, lines, fonts, and bounding boxes per page, emitting a
   structured page graph that preserves block metadata (font statistics, numeric ratios, bullet density, etc.).
2. **Layout Signals (`pipeline.layout.signals`)** – Nine robust signals are computed per page (CIS, OGR, BXS, DAS, FVS, ROJ, TFI,
   MSA, FNL). Values are normalised to [0, 1] via the median/IQR of the first ten pages before a weighted page score is derived.
3. **Routing (`pipeline.layout.router`)** – Pages are queued for LayoutParser when the score ≥0.55 or any of the hard triggers
   fire (dense graphics overlap, table intrusions, column flips, repair loop failures). Neighbour pages with score ≥0.50 are
   included while respecting the ≤30% budget (with a contiguous +5 page overflow allowance).
4. **Layout Fusion (`pipeline.layout.lp_fuser`)** – LayoutParser detections (PubLayNet/PRIMA/DocBank) are fused with PyMuPDF
   blocks using IoU≥0.3. Auxiliary regions (lists, tables, figures, titles off-grid) are stored separately while prose blocks are
   ordered using detected regions and column threading. Anchor markers are injected at the closest preceding sentence to preserve
   references to auxiliary material.
5. **Repair (`pipeline.repair.repair_pass`)** – Adjacent main-flow blocks are stitched when embedding cosine ≥0.80, guarded
   against over-merging across columns with large Δy. Footnotes are linked via superscript markers and retained as separate,
   anchored blocks. A failure counter is returned so the router can escalate LayoutParser coverage if stitching stalls.
6. **Chunking (`pipeline.chunking.chunker`)** – Prose targets ~500 tokens (180–700 bounds, 80-token overlap) with semantic splits
   when sections exceed 600 tokens and sentence-level Δcos >0.15. Procedures respect 250–350 token windows with 40-token overlap.
   Tables are sharded into 6–12 row ranges with duplicated headers and `col:value` flattening, and at most one auxiliary block is
   appended to a chunk when cosine similarity ≥0.55.
7. **Telemetry (`pipeline.telemetry.metrics`)** – Collects LP utilisation ratios, latency per page, score distributions, top-two
   signals per routed page, interleave error rates, repair merge/split percentages, and retrieval deltas (Hit@K/MRR).

### Chunking Metadata

Each emitted chunk records:

- `doc_id`, `page_start`, `page_end`
- Character offsets (`char_start`, `char_end`)
- Source block identifiers and types
- Region type counts and auxiliary attachments
- Table row ranges (for structured chunks)
- Boolean `has_anchor_refs` flag when inline anchors are inserted

These annotations align chunks with the original document for citation, reranking, and gold alignment.

## Telemetry & Monitoring

`TelemetryCollector` aggregates pipeline metrics and provides JSON-friendly summaries plus log-friendly helpers. The CLI driver
logs LP utilisation, latency, and retrieval deltas after each run. Downstream evaluation scripts can ingest the summaries to track
regression suites.

## Testing

The included tests exercise the legacy parser, routing heuristics, and chunking flows:

```bash
pytest -q
```

For integration-style checks, drop sample fixtures under `sample_docs/` as described in `sample_docs/README.md`, then execute the
CLI:

```bash
python -m pipeline.run --pdf sample_docs/sample.pdf --emit-chunks artifacts/sample_chunks.jsonl
```

## Package Overview

- `parser/` – legacy extraction, cleaning, and driver logic used by the UI.
- `pipeline/` – hybrid PyMuPDF/LayoutParser pipeline modules.
- `chunking/` – semantic/fixed chunkers used by the UI plus the new retrieval chunker.
- `app.py` – Gradio Blocks UI for quick inspection.
- `tests/` – pytest suite covering parsing and chunking flows.

## Offline Retrieval Evaluation

1. Ensure you have a gold dataset following `eval/schema.md` and chunk metadata (e.g. `chunks.jsonl`).
2. Install evaluation dependencies: `pip install -r requirements.txt`.
3. Run a baseline evaluation:
   ```bash
   python eval/runner.py --gold path/to/gold.jsonl --chunks path/to/chunks.jsonl --engine all --config eval/config.yaml
   ```
4. Aggregate multiple runs into an HTML report:
   ```bash
   python eval/report.py --runs "runs/**/*.json" --out report/report.html
   ```
5. Execute ablation suites (e.g., varying top-k):
   ```bash
   python eval/ablation.py --suite basic --gold path/to/gold.jsonl --chunks path/to/chunks.jsonl
   ```

Outputs are written to the directory configured in `eval/config.yaml` (default `runs/`). The generated `summary.csv`,
`per_tag.csv`, and `report.html` provide overall comparisons, per-tag slices, and bootstrap-based engine comparisons.
