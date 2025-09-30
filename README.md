# PDF Blocks & Chunking Pipeline

This repository hosts a Hugging Face Spaces-friendly pipeline that turns PDFs into:

- **Blocks JSON** – canonical page-level records following the Step-1 schema.
- **Chunks JSONL** – content-aware retrieval units aligned to headings.
- **Page Triage CSV** – diagnostic metrics for per-page routing.
- **Telemetry JSON** – summarised gate decisions and latencies.

The pipeline powers both a CLI (`parse_to_chunks`) and the Spaces Gradio app (`app.py`). Retrieval stays unchanged.

## Quickstart

1. Install system dependencies (OCR backends are optional locally but required for full fidelity):
   ```bash
   pip install -r requirements.txt
   ```
2. Run unit tests:
   ```bash
   pytest
   ```
3. Launch the Gradio interface:
   ```bash
   python app.py
   ```
4. Use the CLI for offline processing:
   ```bash
   python -m pipeline.cli parse-to-chunks <input.pdf> <out-dir>
   ```

## Pipeline Overview

1. **Page Triage** (`pipeline/triage.py`)
   - Run PyMuPDF, pypdfium2, and pdfminer to vote on text-layer presence and capture extractor lengths.
   - Record font diagnostics (Type3/CID, ToUnicode) and cache extractor text for Docling fallbacks and forced OCR routing.
   - Extract PyMuPDF bounding boxes for coverage metrics and persist the expanded per-page telemetry CSV.
2. **Primary Conversion** (`pipeline/docling_adapter.py`)
   - Docling-first conversion wrapped in a 20s watchdog with deterministic fallback to triage text.
   - Records which pages required the fallback ladder so telemetry can report degraded paths.
3. **Selective OCR Gate** (`pipeline/ocr.py`)
   - Apply the gate formula, choose Nougat vs Tesseract via `math_density`, and respect a 12s watchdog before falling back to triage text.
4. **Layout Rescue** (`pipeline/layout_rescue.py`)
   - Optional layout recovery guarded by an 8s watchdog; failures return the Docling ordering without aborting the document.
5. **Normalisation** (`pipeline/normalize.py`)
   - Materialise canonical Blocks JSON with deterministic IDs, provenance tags, and auxiliary-role metadata.
   - Apply conservative text cleaning and relax header/footer suppression whenever more than 30% of page lines would be dropped.
6. **Reordering & Stitching** (`pipeline/chunker.py` + `pipeline/reorder_stitch/`)
   - Detect per-page columns, score continuity edges, and grow Flow Threads with a constrained path cover.
   - Assemble section narratives from the best threads, partition leftover blocks into an auxiliary pool, and emit **main-first** chunks followed by **aux-only** payloads per section.
   - Invariants (I1–I4) ensure only threaded blocks reach narrative chunks, aux trails its section, and floats never enter Flow Threads while Flow-First limits (`T/S/H/m = 1600/2000/2400/900`) continue to apply.
7. **Telemetry & Output** (`pipeline/service.py`)
   - Emits per-doc summary telemetry, per-page CSV rows, watchdog timings, and bundles artefacts for downstream retrieval.

## Service Surface

- **CLI** – `python -m pipeline.cli parse-to-chunks sample_docs/demo.pdf artifacts/`.
- **Gradio App** – upload a PDF, trigger parsing, then run retrieval over the generated chunks.
- **Schemas** – verbatim copies live under `schemas/` for compliance automation.

## Configuration

Defaults match the provided CONFIG_DEFAULTS and can be overridden via `PipelineConfig.from_mapping`:

```python
from pipeline import PipelineConfig, PipelineService

config = PipelineConfig.from_mapping({
    "ocr": {"gate": {"char_count_threshold": 180}},
    "timeouts": {"doc": {"cap_seconds": 180}},
    "chunk": {"tokens": {"target": 900}},
})
service = PipelineService(config)
result = service.process_pdf("sample.pdf")
```

Key defaults (see `pipeline/config.py` for the full map):

```
timeouts.triage.seconds=5
timeouts.docling.seconds=20
timeouts.ocr.seconds=12
timeouts.layout.seconds=8
timeouts.doc.cap.seconds=240
extractor.vote.char_threshold=150
ocr.psm=[6,4]
ocr.oem=1
ocr.force.dpi.default=200
ocr.force.dpi.math=300
gpu.acquire.max_wait_seconds=20
raster.dpi.default=200
raster.dpi.math=300
raster.max_megapixels=12
chunk.tokens.target=1000
chunk.tokens.min=500
chunk.tokens.max=1400
chunk.degraded.target=1000
chunk.degraded.min=900
chunk.degraded.max=1100
aux.header_footer.repetition_threshold=0.40
aux.header_footer.dropcap.max_fraction=0.30
aux.header_footer.y_band_pct=0.07
aux.segment0.min_chars=150
aux.segment0.font_percentile=0.80
aux.superscript.y_offset_xheight=0.20
aux.soft_boundary.max_deferred_pages=5
aux.callout.column_width_fraction_max=0.60
aux.font_band.small_quantile=0.20
flow.limits.target=1600
flow.limits.soft=2000
flow.limits.hard=2400
flow.limits.min=900
flow.boundary_slack_tokens=200
segments.soft_boundary_pages=5
anchor.lookahead_pages=1
gate5.header_footer.y_band_pct=0.07
gate5.header_footer.repetition_threshold=0.40
gate5.caption_zone.lineheight_multiplier=1.5
gate5.sidebar.min_column_width_fraction=0.60
paragraph_only.min_blocks_across_pages=1
paragraph_only.window_pages=2
diagnostics.enable=true
diagnostics.overlay.max_pages=2
```

## Testing

Two focused suites assert the mandatory heuristics:

- `tests/test_gate.py` – OCR gate math density and routing edge cases.
- `tests/test_chunker.py` – heading-aware packing, token limits, and sidecar attachment.
- `tests/test_watchdog.py` – watchdog fallbacks for long-running stages.
- `tests/test_degraded_chunker.py` – guarantees at least one chunk via the degraded path.

Run them with `pytest`.

## Directories

- `pipeline/` – new Step-1 implementation (triage → chunking).
- `schemas/` – verbatim schema snapshots.
- `retrieval/` – unchanged retrieval engines wired into the Gradio demo.
- `app.py` – Spaces-ready interface now using the new pipeline outputs.

## License

MIT. See `LICENSE` for full terms.
