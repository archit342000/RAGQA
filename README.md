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
   - Extract text + bounding boxes with PyMuPDF under a 5s per-page watchdog.
   - Capture text density stats and persist the mandated per-page CSV (`stage_used`, `fallback_applied`, `error_codes`).
2. **Primary Conversion** (`pipeline/docling_adapter.py`)
   - Docling-first conversion wrapped in a 20s watchdog with deterministic fallback to triage text.
   - Records which pages required the fallback ladder so telemetry can report degraded paths.
3. **Selective OCR Gate** (`pipeline/ocr.py`)
   - Apply the gate formula, choose Nougat vs Tesseract via `math_density`, and respect a 12s watchdog before falling back to triage text.
4. **Layout Rescue** (`pipeline/layout_rescue.py`)
   - Optional layout recovery guarded by an 8s watchdog; failures return the Docling ordering without aborting the document.
5. **Normalisation** (`pipeline/normalize.py`)
   - Materialise canonical Blocks JSON with deterministic IDs, provenance tags, and auxiliary-role metadata.
6. **Content-Aware Chunking** (`pipeline/chunker.py`)
   - Maintains auxiliary deferral, honours heading boundaries, and guarantees at least one chunk via the degraded fallback packer.
   - Tracks evidence spans for every paragraph block and preserves deterministic IDs with token-aware packing.
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
ocr.psm=[6,4]
ocr.oem=1
gpu.acquire.max_wait_seconds=20
raster.dpi.default=200
raster.dpi.math=300
raster.max_megapixels=12
chunk.tokens.target=1000
chunk.tokens.min=500
chunk.tokens.max=1400
aux.header_footer.repetition_threshold=0.50
aux.y_band.pct=0.03
aux.segment0.min_chars=150
aux.segment0.font_percentile=0.80
aux.superscript.y_offset_xheight=0.20
aux.soft_boundary.max_deferred_pages=5
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
