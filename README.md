# pdfchunks – High-Fidelity PDF ➜ Chunk Pipeline

This repository provides a ground-up rewrite of the PDF parsing and chunking
pipeline tailored for Retrieval-Augmented Generation (RAG) workloads. The new
architecture extracts layout-aware blocks with PyMuPDF, enforces a strict
allow-list for MAIN prose, isolates AUX payloads until their owning section is
sealed, and emits retrieval chunks with deterministic ordering guarantees.

## Pipeline Overview

1. **Block Extraction** – `BlockExtractor` converts PyMuPDF span output into
   rich block objects (`text`, `font`, `bbox`, `page`, `line_height`). Repeated
   headers/footers (appearing on ≥5 pages) are suppressed automatically.
2. **Baseline Estimation** – `BaselineEstimator` infers column bands, body font
   medians, and text density percentiles to anchor later classification steps.
3. **Classification** – `BlockClassifier` applies a strict allow-list:
   column overlap ≥90 %, width ≥80 % of the band, typography within ±10 / 12 %,
   density within P20–P80, and no lexical cues such as `Figure`, `Activity`,
   `Keywords`, etc. Heading overrides promote legitimate headers to MAIN even
   when they sit inside exclusion bands.
4. **Ownership** – Non-MAIN blocks become AUX with subtypes (captions,
   activities, callouts, footnotes). Each payload inherits an
   `owner_section_seq` based on the nearest heading or explicit metadata.
5. **Threading** – `Threader` keeps MAIN paragraphs continuous with
   cross-page dehyphenation and sentence segmentation. Section transactions
   buffer AUX content in per-section queues, flushing only after the section is
   sealed. Every emitted AUX sentence is wrapped in `<aux>...</aux>`.
6. **Serialization** – A single `Serializer` enforces monotonic
   `(doc_id, section_seq, para_seq, sent_seq, emit_phase)` order keys, blocking
   any 1→0 regressions.
7. **Chunking** – `Chunker` packs MAIN-only retrieval windows (~500 tokens,
   overlap 80/40) and, after seals, emits AUX-only chunks grouped by
   section/subtype. AUX payloads never mingle with MAIN text.
8. **Audits & Telemetry** – `run_audits` asserts AUX isolation, monotonic order
   keys, and `<aux>` wrapping for both units and chunks while logging key
   telemetry. `metrics.compute_metrics` exposes lightweight counters for higher
   level dashboards.

All behaviour is configurable through `configs/parser.yaml`. The default values
mirror the requirements described in the project brief and can be overridden at
runtime or via the CLI.

## Getting Started

```bash
pip install -e .
python -m pdfchunks.cli path/to/document.pdf --json
```

The CLI drives the full pipeline: extraction → classification → threading →
chunking, runs guardrail audits, and prints JSON metrics (when `--json` is
provided). Logging is enabled via `--log-level`.

## Key Modules

- `src/pdfchunks/parsing/block_extractor.py` – PyMuPDF block extraction.
- `src/pdfchunks/parsing/baselines.py` – Column and density baselines.
- `src/pdfchunks/parsing/classifier.py` – MAIN allow-list & AUX tagging.
- `src/pdfchunks/parsing/ownership.py` – Section sequencing & AUX ownership.
- `src/pdfchunks/threading/transactions.py` – Section transactions & AUX queues.
- `src/pdfchunks/threading/threader.py` – MAIN threading with AUX isolation.
- `src/pdfchunks/serialize/serializer.py` – Monotonic order enforcement.
- `src/pdfchunks/chunking/chunker.py` – MAIN-only packing & AUX grouping.
- `src/pdfchunks/audit/guards.py` – Guardrails (`<aux>` audits, ordering).
- `src/pdfchunks/telemetry/metrics.py` – Lightweight metric aggregation.

## Testing

The pytest suite covers:

- classifier allow-list rules and heading overrides,
- transaction guardrails (no AUX before lead paragraphs),
- serializer monotonic ordering enforcement,
- chunker MAIN-only packing and AUX grouping,
- `<aux>` wrapping audits for every AUX unit and chunk.

Run the tests with:

```bash
pytest
```

## Configuration

`configs/parser.yaml` exposes thresholds for block extraction, classifier
constraints, threading behaviour, chunk sizes, and audit logging. Adjust these
values to tune the pipeline for different document corpora or AUX policies.

