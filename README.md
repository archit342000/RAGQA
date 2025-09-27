# Document Parser (Layout-Aware)

This repository contains a CPU-only reference implementation of a layout-aware
PDF parser that produces `DocBlock` JSON suitable for retrieval augmented
generation (RAG) pipelines.  The parser performs a two-tier extraction using a
cheap PyMuPDF pass with optional escalation to a strong extractor, classifies
layout blocks into narrative versus auxiliary material, stitches paragraphs
across page breaks and exports normalised blocks with provenance metadata.

## Features

* **Two-tier extraction:** run a fast PyMuPDF-based pass and escalate to a
  configurable strong extractor when layout complexity signals are detected.
* **Layout complexity scoring:** bootstrap pages are always escalated and a
  mix of soft/strong signals (multi-column, tables, OCR markers, etc.) drive
  additional escalation.
* **False wrap-around handling:** narrow symmetric bands are re-attached to the
  nearest column to prevent spurious auxiliary blocks.
* **Mainness/heading classification:** heuristic scoring uses fonts, whitespace
  halos, lexical cues and geometry bands to separate main narrative from
  auxiliary content, including captions, footnotes and callouts.
* **Paragraph stitching:** a paragraph state buffer (TTL = 2 pages) bridges
  page breaks and auxiliary interruptions so long-form paragraphs stay intact.
* **Flow-safe classification:** ambiguous blocks inside an active section bias
  to `main` when paragraph geometry continues, preventing figure-adjacent body
  text from being demoted.
* **PP-Structure region tagging:** optionally run PaddleOCR PP-Structure in
  layout-only mode to obtain `{title,text,figure,table,list}` masks.  PyMuPDF
  text remains authoritative while regions gate the auxiliary pre-pass.
* **Auxiliary buffering and anchoring:** aux blocks are buffered and flushed
  only at section boundaries, anchored to the nearest narrative block with a
  footnote fallback to the previous page.
* **Page zoning & running header guards:** advisory header/footer/body bands
  combine with cross-page repetition tracking to demote running heads/feet
  while allowing genuine paragraph continuations inside the top/bottom bands.
* **Caption ring anchoring:** a 32–120 px caption ring around detected figures
  forces nearby small-font or cue-matched captions into `aux(caption)` and
  relinks them to the owning figure, eliminating caption leakage into main.
* **Callout anti-absorption:** activity/callout blocks require dual cues
  (lexical + inset/leading) and snap shut when lines realign to the body margin,
  preventing footer mislabels from swallowing surrounding paragraphs.
* **Continuation guard & aux parking:** an open-paragraph state machine parks
  page-top aux blocks until the continuation arrives, then flushes them only
  after the paragraph closes or a new section header is confirmed.
* **Conflict resolver with `aux_shadow`:** ordered heuristics (captions → running
  headers/footers → callouts → continuation guard) resolve disagreements while
  tagging borderline footer-as-main cases with `aux_shadow=true` for downstream
  auditing.
* **Quarantined auxiliary mode:** low-confidence aux blocks are tagged as
  `quarantined` so they can be surfaced to consumers without perturbing the
  stitching or section state machine.
* **Layout repair post-pass:** caption peel/bound trimming, header/footer
  demotion and activity grouping run after classification to eliminate caption
  leakage and over-eager header detection.
* **DocBlock export:** emits normalised bounding boxes and structured metadata
  (including `region_tag`, `quarantined`, `aux_shadow` and `anchor_to`) compliant with the
  provided schema.

## Setup

```bash
pip install -r requirements.txt  # baseline runtime
# optional extras for full pipeline
pip install paddleocr opencv-python-headless scikit-learn
sudo apt-get install tesseract-ocr  # for OCR escalation
```

The unit tests rely purely on the Python implementation and do not require any
native dependencies.

## Usage

Instantiate the high level parser and call `parse` with a PDF path:

```python
from pathlib import Path
from parser import DocumentParser

parser = DocumentParser()
docblocks = parser.parse(Path("input.pdf"))
```

The parser loads configuration from `parser/config.yaml`.  Override thresholds
or escalation behaviour by editing the file or supplying a custom dictionary to
`DocumentParser`.

### Configuration highlights

The default configuration exposes the following key groups:

* `thresholds`: includes `tau_main`, `tau_main_page_confident`,
  `tau_fail_safe_low`, `tau_bias_high`, heading thresholds, caption overlap
  ratio, wrap heuristics and the `quarantined_aux_conf_min` guard.
* `bands`: normalised margin/header/footer bands used by the classifier
  pre-pass.
* `caption` / `sidenote`: fine-tune extended caption merging, image proximity
  checks and margin-based sidenote detection.
* `implicit_section`: controls the implicit section start detector (score
  threshold, top-of-page window, whitespace halo multiplier, drop-cap toggle
  and indent delta) and whether it can fire while a section is active.
* `stitching`: parameters for the paragraph state buffer (split confidence,
  top-of-page continuation lookahead and TTL values).
* `buffers`: limits for auxiliary anchors, including per-anchor quotas and
  top-window allowances for page-turn captions/sidenotes.
* `activity`: regex lexicon used to deterministically tag activity/callout
  prompts in conjunction with margin bands.
* `post_pass`: ordered list of layout repair stages (`peel_captions`,
  `shrink_caption_bounds`, `demote_headers`, etc.).
* `source_label`: strategy for anchoring source/courtesy lines (free floating,
  attach to nearest excerpt, or attach to the closest figure).

### Export format

Auxiliary text is wrapped in `<aux>...</aux>` and section boundaries are surfaced
as `kind="control"` blocks with `subtype="implicit_section_start"`.  Exported
blocks expose `region_tag`, `quarantined`, `anchor_to` and `attached_across_pages`
fields alongside normalised bounding boxes for downstream consumers.

## Testing

```bash
pytest tests/
```

The synthetic tests exercise escalation routing, wrap-around remediation,
heading detection, paragraph stitching and auxiliary anchoring.
