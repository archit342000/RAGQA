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
   - Extract text and bounding boxes with PyMuPDF.
   - Compute `char_count`, `text_coverage`, and capture extraction errors.
   - Persist the per-page CSV schema verbatim.
2. **Primary Conversion** (`pipeline/docling_adapter.py`)
   - Prefer Docling for Markdown + structured blocks.
   - Fall back to triage text when Docling is unavailable and retain provenance.
3. **Selective OCR Gate** (`pipeline/ocr.py`)
   - Apply the mandatory gate formula and select Nougat vs Tesseract based on `math_density`.
   - Stub OCR output when backends are missing so the rest of the pipeline can run in CI.
4. **Layout Rescue** (`pipeline/layout_rescue.py`)
   - Stubbed hooks for LayoutParser/Detectron2 with telemetry-friendly logging.
5. **Normalisation** (`pipeline/normalize.py`)
   - Materialise canonical Blocks JSON with deterministic IDs and provenance tags.
6. **Content-Aware Chunking** (`pipeline/chunker.py`)
   - Respect heading boundaries first, size targets second.
   - Attach table/figure/caption sidecars atomically.
   - Track evidence spans for every paragraph block.
7. **Telemetry & Output** (`pipeline/service.py`)
   - Bundle Blocks/Chunks/Triage/Telemetry artefacts and optionally write them to disk.

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
    "chunk": {"tokens": {"target": 900}},
})
service = PipelineService(config)
result = service.process_pdf("sample.pdf")
```

## Testing

Two focused suites assert the mandatory heuristics:

- `tests/test_gate.py` – OCR gate math density and routing edge cases.
- `tests/test_chunker.py` – heading-aware packing, token limits, and sidecar attachment.

Run them with `pytest`.

## Directories

- `pipeline/` – new Step-1 implementation (triage → chunking).
- `schemas/` – verbatim schema snapshots.
- `retrieval/` – unchanged retrieval engines wired into the Gradio demo.
- `app.py` – Spaces-ready interface now using the new pipeline outputs.

## License

MIT. See `LICENSE` for full terms.
