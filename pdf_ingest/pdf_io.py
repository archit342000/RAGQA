"""PDF loading and page signal extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import fitz  # type: ignore


@dataclass
class Line:
    page_index: int
    line_index: int
    text: str
    bbox: Tuple[float, float, float, float] | None
    x_center: float | None
    source: str = "native"
    confidence: float | None = None


@dataclass
class PageSignals:
    index: int
    glyph_count: int
    text_density: float
    unicode_ratio: float
    has_fonts: bool
    image_coverage: float
    delimiter_ratio: float
    whitespace_ratio: float
    hidden_text_layer: bool
    dpi: float
    page_area: float
    band_glyph_counts: List[int]
    band_text_density: List[float]


@dataclass
class PagePayload:
    index: int
    lines: List[Line]
    glyph_count: int


def open_document(path: Path) -> fitz.Document:
    return fitz.open(path)


def iterate_pages(doc: fitz.Document, *, max_pages: int | None = None) -> Iterator[fitz.Page]:
    total = len(doc)
    limit = total if max_pages is None else min(total, max_pages)
    for page_index in range(limit):
        yield doc.load_page(page_index)


def collect_page_lines(page: fitz.Page) -> Tuple[List[Line], int]:
    textpage = page.get_textpage()
    raw = textpage.extractDICT()
    lines: List[Line] = []
    glyphs = 0
    line_index = 0
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            for span in spans:
                content = span.get("text", "")
                if not content.strip():
                    continue
                bbox = tuple(span.get("bbox", (0.0, 0.0, 0.0, 0.0)))
                x_center = (bbox[0] + bbox[2]) / 2 if bbox else None
                glyphs += len(content)
                lines.append(
                    Line(
                        page_index=page.number,
                        line_index=line_index,
                        text=content.strip(),
                        bbox=bbox if bbox != (0.0, 0.0, 0.0, 0.0) else None,
                        x_center=x_center,
                        source="native",
                        confidence=1.0,
                    )
                )
                line_index += 1
    return lines, glyphs


def compute_page_signals(
    page: fitz.Page,
    lines: Sequence[Line],
    glyph_count: int,
    *,
    probe_bands: int = 2,
) -> PageSignals:
    width, height = page.rect.width, page.rect.height
    area = max(width * height, 1.0)
    text_density = glyph_count / area
    ascii_chars = sum(ch.isascii() for line in lines for ch in line.text)
    total_chars = sum(len(line.text) for line in lines) or 1
    unicode_ratio = ascii_chars / total_chars
    fonts = page.get_fonts(full=True)
    has_fonts = bool(fonts)
    images = page.get_images(full=True)
    image_coverage = 0.0
    for image in images:
        bbox = page.get_image_bbox(image)
        image_area = bbox.width * bbox.height
        image_coverage += image_area / area
    image_coverage = min(image_coverage, 1.0)
    delimiter_lines = 0
    whitespace_samples = []
    hidden_text_layer = False
    for line in lines:
        if any(delim in line.text for delim in ",;|\t"):
            delimiter_lines += 1
        whitespace_samples.extend([c for c in line.text if c.isspace()])
        if "\uFFFD" in line.text:
            hidden_text_layer = True
    delimiter_ratio = delimiter_lines / max(len(lines), 1)
    whitespace_ratio = len(whitespace_samples) / max(total_chars, 1)
    dpi = (page.rect.width / (page.mediabox.width or 1)) * 72

    bands = max(1, probe_bands)
    band_glyph_counts = [0 for _ in range(bands)]
    if lines:
        for line in lines:
            if not line.bbox:
                continue
            y_mid = (line.bbox[1] + line.bbox[3]) / 2.0
            relative = 0.0 if height == 0 else max(0.0, min(0.9999, (y_mid - page.rect.y0) / height))
            band_index = min(bands - 1, int(relative * bands))
            band_glyph_counts[band_index] += len(line.text)
    band_height = height / bands if bands else height
    band_area = width * band_height if band_height else width * height
    band_text_density = [
        (band_glyph_counts[i] / max(band_area, 1.0)) if band_area else 0.0 for i in range(bands)
    ]
    return PageSignals(
        index=page.number,
        glyph_count=glyph_count,
        text_density=text_density,
        unicode_ratio=unicode_ratio,
        has_fonts=has_fonts,
        image_coverage=image_coverage,
        delimiter_ratio=delimiter_ratio,
        whitespace_ratio=whitespace_ratio,
        hidden_text_layer=hidden_text_layer,
        dpi=dpi,
        page_area=area,
        band_glyph_counts=band_glyph_counts,
        band_text_density=band_text_density,
    )


def load_document_payload(
    path: Path,
    *,
    max_pages: int | None = None,
    probe_bands: int = 2,
) -> Tuple[List[PagePayload], List[PageSignals]]:
    with open_document(path) as doc:
        pages: List[PagePayload] = []
        signals: List[PageSignals] = []
        for page in iterate_pages(doc, max_pages=max_pages):
            lines, glyphs = collect_page_lines(page)
            pages.append(PagePayload(index=page.number, lines=lines, glyph_count=glyphs))
            signals.append(compute_page_signals(page, lines, glyphs, probe_bands=probe_bands))
        return pages, signals

