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
* **Auxiliary buffering and anchoring:** aux blocks are buffered and flushed
  only at section boundaries, anchored to the nearest narrative block with a
  footnote fallback to the previous page.
* **DocBlock export:** emits normalised bounding boxes and structured metadata
  compliant with the provided schema.

## Setup

```bash
pip install -r requirements.txt  # optional, only if you intend to use real extractors
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
  `tau_fail_safe_low`, heading thresholds, caption overlap ratio and
  wrap/sidenote heuristics.
* `bands`: normalised margin/header/footer bands used by the classifier
  pre-pass.
* `implicit_section`: controls the implicit section start detector (score
  threshold, top-of-page window, whitespace halo multiplier, drop-cap toggle
  and indent delta).
* `buffering`: limits for auxiliary anchors and next-page windows for captions,
  sidenotes and callouts.
* `stitching`: parameters for the paragraph state buffer (split confidence and
  TTL windows).

### Export format

Auxiliary text is wrapped in `<aux>...</aux>` and section boundaries are surfaced
as `kind="control"` blocks with `subtype="implicit_section_start"` for
downstream consumers.

## Testing

```bash
pytest tests/
```

The synthetic tests exercise escalation routing, wrap-around remediation,
heading detection, paragraph stitching and auxiliary anchoring.
