# Document Parser & HF Spaces Demo

This project provides a lightweight document parsing module tailored for Retrieval-Augmented Generation (RAG) pipelines, plus a Gradio UI suited for Hugging Face Spaces deployments. The parser prioritises fast PDF extraction via `pypdf`, falls back to the `unstructured` library when the layout is challenging, and exposes metrics that help you decide when to escalate processing.

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the demo locally:
   ```bash
   python app.py
   ```
   or rely on Gradio's CLI:
   ```bash
   gradio app.py
   ```

Upload one or more PDF, TXT, or Markdown files to preview per-page text, see fallback decisions, and (optionally) inspect metrics from the gated debug panel.

## Hugging Face Spaces Notes

- Designed for the CPU runtime; no GPU or heavy OCR dependencies are required.
- The hi-res OCR path is gated by an environment flag and currently stubbed to the fast strategy to avoid Detectron2/Tesseract installs.
- Keep uploads small (<800 pages) for responsive demos. The `MAX_PAGES` flag trims larger documents with a visible warning.

## Configuration via Environment Variables

Default configuration values live in the project's `.env` file; edit it to
customise behaviour locally or on Spaces. The parser reads the following
variables (defaults in parentheses):

- `MAX_PAGES` (`800`): hard limit on processed pages per document.
- `MAX_TOTAL_PAGES` (`1200`): aggregate ceiling across all uploaded documents in one run.
- `MIN_CHARS_PER_PAGE` (`200`): threshold used to detect sparse pages during heuristic checks.
- `FALLBACK_EMPTY_PAGE_RATIO` (`0.3`): ratio of low-density pages that triggers the `unstructured` fallback.
- `UNSTRUCTURED_STRATEGY` (`fast`): default fallback strategy. The UI can switch to "High accuracy" when `ENABLE_HI_RES=true`.
- `ENABLE_HI_RES` (`false`): when set to `true`, enables the "High accuracy" parsing mode in the UI. Otherwise the option is disabled and requests are clamped to the fast path.
- `SHOW_DEBUG` (`false`): when `true`, exposes the developer-focused metrics panel in the UI.
- `CHUNK_MODE_DEFAULT` (`semantic`): default chunking strategy when the UI control is untouched.
- `MAX_TOTAL_TOKENS_FOR_CHUNKING` (`300000`): guardrail enforced by the chunker to keep latency bounded.
- `SEMANTIC_MODEL_NAME` (`intfloat/e5-small-v2`): embedding model used by the semantic chunker. Override to match your hardware budget.
- `TOKENIZER_NAME` (`hf-internal-testing/llama-tokenizer`): tokenizer used for token accounting inside the chunker.

### Why hide these knobs?

These thresholds and strategy fields are expert tuning levers that depend on document corpora. Exposing them in the user interface creates confusion without delivering value. They remain fully configurable via environment variables so deployments can tailor behaviour without overwhelming end users.

## Fallback Heuristics

1. Extract text from each page with `pypdf` and compute character counts.
2. If the fraction of pages below the `MIN_CHARS_PER_PAGE` threshold exceeds `FALLBACK_EMPTY_PAGE_RATIO`, switch to `unstructured`.
3. Independently, compute layout metrics (short-line ratio, blank-line ratio, digit-heavy ratio). If the weighted score ≥ 0.6, trigger fallback.
4. The hi-res path is clamped to the fast path unless `ENABLE_HI_RES=true`, and even then it stays stubbed pending OCR dependencies.

## Chunking Strategies

After parsing and cleaning, the app automatically turns page text into
retrieval-sized chunks. Two strategies are available from the "Chunking mode"
dropdown:

- **Semantic (recommended)** – Uses LangChain's `SemanticChunker` with
  `intfloat/e5-small-v2` embeddings to group coherent sentences before packing
  them into ~200–700 token windows. Pages detected as table/list-heavy fall
  back to fixed windowing automatically.
- **Fixed** – Applies deterministic sliding windows (700 tokens with 100-token
  overlap, or 400/40 for table-heavy pages) while keeping page anchors intact.

Chunk metadata records `doc_id`, human-friendly `doc_name`, originating
`page_start`/`page_end`, the first heading encountered, and the token count
used for retrieval. Switch modes via the UI or set `CHUNK_MODE_DEFAULT` in the
`.env` file for unattended deployments.

## Debugging
 
Enable the `SHOW_DEBUG=true` environment flag to surface parser choice,
aggregate metrics, fallback reasons, chunk counts, and timing data. The panel
stays hidden by default to keep the end-user experience focused.

## Testing

The included tests exercise the routing heuristics and cleaning routines:

```bash
pytest
```

For integration-style checks, drop sample fixtures under `sample_docs/` as described in `sample_docs/README.md`.

## Package Overview

- `parser/` – modular extraction, cleaning, metrics, and driver logic.
- `chunking/` – semantic/fixed chunkers plus token packers for retrieval windows.
- `app.py` – Gradio Blocks UI for quick inspection.
- `tests/` – pytest suite covering parsing and chunking flows.

This structure keeps the parser module production-ready while remaining light enough for Hugging Face Spaces cold starts.

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

Outputs are written to the directory configured in `eval/config.yaml` (default `runs/`). The generated `summary.csv`, `per_tag.csv`, and `report.html` provide overall comparisons, per-tag slices, and bootstrap-based engine comparisons.
