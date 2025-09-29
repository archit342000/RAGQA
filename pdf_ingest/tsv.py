"""Helpers for converting Tesseract TSV/hOCR output into :class:`Line` records."""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path
from typing import List, Sequence, Tuple

from difflib import SequenceMatcher

from .pdf_io import Line


def _apply_clip_offset(
    bbox: Tuple[float, float, float, float],
    clip_bbox: Tuple[float, float, float, float] | None,
) -> Tuple[float, float, float, float]:
    if clip_bbox is None:
        return bbox
    x0, y0, x1, y1 = bbox
    cx0, cy0, _, _ = clip_bbox
    return (x0 + cx0, y0 + cy0, x1 + cx0, y1 + cy0)


def _scale_bbox(
    bbox: Tuple[float, float, float, float],
    *,
    pdf_width: float,
    pdf_height: float,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    if image_width == 0 or image_height == 0:
        return bbox
    sx = pdf_width / float(image_width)
    sy = pdf_height / float(image_height)
    x0, y0, x1, y1 = bbox
    return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)


def tsv_to_lines(
    page_idx: int,
    tsv_path: Path,
    *,
    pdf_width: float,
    pdf_height: float,
    image_width: int,
    image_height: int,
    dpi: int,
    clip_bbox: Tuple[float, float, float, float] | None = None,
) -> List[Line]:
    lines: List[Line] = []
    if not tsv_path.exists():
        return lines

    key = None
    buffer: List[Tuple[str, Tuple[float, float, float, float], float]] = []
    _ = dpi  # reserved for future use (tsv confidence scaling)

    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            level = int(row.get("level", "0"))
            if level < 5:
                continue
            block = row.get("block_num", "0")
            par = row.get("par_num", "0")
            line = row.get("line_num", "0")
            new_key = (block, par, line)
            left = float(row.get("left", "0") or 0)
            top = float(row.get("top", "0") or 0)
            width = float(row.get("width", "0") or 0)
            height = float(row.get("height", "0") or 0)
            conf = float(row.get("conf", "0") or 0)
            if key is None:
                key = new_key
            if new_key != key:
                lines.extend(
                    _flush_tsv_buffer(
                        page_idx,
                        key,
                        buffer,
                        pdf_width,
                        pdf_height,
                        image_width,
                        image_height,
                        clip_bbox,
                    )
                )
                buffer = []
                key = new_key
            bbox = (left, top, left + width, top + height)
            buffer.append((text, bbox, conf))

    if buffer and key is not None:
        lines.extend(
            _flush_tsv_buffer(
                page_idx,
                key,
                buffer,
                pdf_width,
                pdf_height,
                image_width,
                image_height,
                clip_bbox,
            )
        )

    for idx, line in enumerate(lines):
        line.line_index = idx
        if line.bbox:
            x0, _, x1, _ = line.bbox
            line.x_center = (x0 + x1) / 2
    return lines


def _flush_tsv_buffer(
    page_idx: int,
    key: Tuple[str, str, str],
    buffer: Sequence[Tuple[str, Tuple[float, float, float, float], float]],
    pdf_width: float,
    pdf_height: float,
    image_width: int,
    image_height: int,
    clip_bbox: Tuple[float, float, float, float] | None,
) -> List[Line]:
    if not buffer:
        return []
    texts = [item[0] for item in buffer]
    joined = " ".join(texts).strip()
    if not joined:
        return []
    x0 = min(item[1][0] for item in buffer)
    y0 = min(item[1][1] for item in buffer)
    x1 = max(item[1][2] for item in buffer)
    y1 = max(item[1][3] for item in buffer)
    bbox = _scale_bbox(
        (x0, y0, x1, y1),
        pdf_width=pdf_width,
        pdf_height=pdf_height,
        image_width=image_width,
        image_height=image_height,
    )
    bbox = _apply_clip_offset(bbox, clip_bbox)
    avg_conf = sum(max(item[2], 0.0) for item in buffer) / len(buffer)
    return [
        Line(
            page_index=page_idx,
            line_index=0,
            text=joined,
            bbox=bbox,
            x_center=(bbox[0] + bbox[2]) / 2,
            source="ocr-tsv",
            confidence=avg_conf / 100.0,
        )
    ]


HOCR_NS = {"html": "http://www.w3.org/1999/xhtml"}


def hocr_to_lines(
    page_idx: int,
    hocr_path: Path,
    *,
    pdf_width: float,
    pdf_height: float,
    image_width: int,
    image_height: int,
    clip_bbox: Tuple[float, float, float, float] | None = None,
) -> List[Line]:
    if not hocr_path.exists():
        return []
    try:
        tree = ET.parse(hocr_path)
    except ET.ParseError:
        return []
    root = tree.getroot()
    lines: List[Line] = []
    for elem in root.findall(".//html:span", HOCR_NS):
        cls = elem.attrib.get("class", "")
        if "ocr_line" not in cls:
            continue
        title = elem.attrib.get("title", "")
        bbox_match = re.search(r"bbox (\d+) (\d+) (\d+) (\d+)", title)
        conf_match = re.search(r"x_wconf (\d+)", title)
        if not bbox_match:
            continue
        x0, y0, x1, y1 = [float(val) for val in bbox_match.groups()]
        bbox = _scale_bbox((x0, y0, x1, y1), pdf_width=pdf_width, pdf_height=pdf_height, image_width=image_width, image_height=image_height)
        bbox = _apply_clip_offset(bbox, clip_bbox)
        text = " ".join(unescape(part.strip()) for part in elem.itertext()).strip()
        if not text:
            continue
        confidence = float(conf_match.group(1)) / 100.0 if conf_match else None
        lines.append(
            Line(
                page_index=page_idx,
                line_index=len(lines),
                text=text,
                bbox=bbox,
                x_center=(bbox[0] + bbox[2]) / 2,
                source="ocr-hocr",
                confidence=confidence,
            )
        )
    return lines


def merge_lines(native_lines: Sequence[Line], ocr_lines: Sequence[Line]) -> List[Line]:
    if not ocr_lines:
        return list(native_lines)
    merged: List[Line] = list(native_lines)
    for o_line in ocr_lines:
        duplicate = False
        for n_line in native_lines:
            if _line_similarity(n_line, o_line) >= 0.8:
                duplicate = True
                break
        if not duplicate:
            merged.append(o_line)
    merged.sort(key=lambda ln: (
        ln.page_index,
        ln.bbox[1] if ln.bbox else float(ln.line_index),
        ln.bbox[0] if ln.bbox else 0.0,
    ))
    for idx, line in enumerate(merged):
        line.line_index = idx
        if line.bbox:
            line.x_center = (line.bbox[0] + line.bbox[2]) / 2
    return merged


def _line_similarity(a: Line, b: Line) -> float:
    text_ratio = SequenceMatcher(None, a.text, b.text).ratio() if a.text and b.text else 0.0
    if text_ratio >= 0.8:
        return text_ratio
    if a.bbox and b.bbox:
        iou = _bbox_iou(a.bbox, b.bbox)
        if iou >= 0.5:
            return max(text_ratio, iou)
    return text_ratio


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
    area_b = max((bx1 - bx0) * (by1 - by0), 1e-6)
    return inter_area / (area_a + area_b - inter_area)
