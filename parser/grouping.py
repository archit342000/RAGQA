"""Span→line→block grouping utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

BBox = Tuple[float, float, float, float]


@dataclass
class Span:
    """Light-weight representation of a text span."""

    text: str
    bbox: BBox
    font_size: float
    font_name: str = "Times"
    bold: bool = False
    italic: bool = False


@dataclass
class Line:
    spans: List[Span]
    bbox: BBox
    meta: Dict[str, float] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(span.text for span in self.spans).strip()


@dataclass
class Block:
    lines: List[Line]
    bbox: BBox
    meta: Dict[str, float | int | bool] = field(default_factory=dict)
    region_tag: str | None = None
    ro_index: int = -1
    block_id: str = ""

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text)


def _merge_bbox(bboxes: Iterable[BBox]) -> BBox:
    boxes = list(bboxes)
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


def _line_gap(line: Line, next_line: Line) -> float:
    return max(0.0, next_line.bbox[1] - line.bbox[3])


def group_spans(
    spans: Iterable[Span],
    cfg: Dict[str, object],
    page_width: float | None = None,
) -> Tuple[List[Line], List[Block]]:
    """Group spans into lines and blocks with simple geometric heuristics."""

    ordered: List[Span] = sorted(spans, key=lambda s: (s.bbox[1], s.bbox[0]))
    if not ordered:
        return [], []

    slope_tol = float(cfg.get("grouping", {}).get("baseline_slope_tol_deg", 2.0)) if isinstance(cfg.get("grouping"), dict) else 2.0
    _ = slope_tol  # currently unused but kept for API parity
    y_gap_max_lh = float(cfg.get("grouping", {}).get("line_y_gap_max_lh", 1.4)) if isinstance(cfg.get("grouping"), dict) else 1.4
    margin_tol = float(cfg.get("grouping", {}).get("left_margin_merge_tol_px", 6)) if isinstance(cfg.get("grouping"), dict) else 6.0

    lines: List[Line] = []
    current: List[Span] = []
    for span in ordered:
        if not current:
            current.append(span)
            continue
        prev = current[-1]
        prev_height = prev.bbox[3] - prev.bbox[1]
        tol = max(6.0, prev_height * 0.6)
        if abs(span.bbox[1] - prev.bbox[1]) <= tol:
            current.append(span)
        else:
            line_bbox = _merge_bbox(s.bbox for s in current)
            line_meta = {
                "font_size": sum(s.font_size for s in current) / max(1, len(current)),
                "left_margin": line_bbox[0],
            }
            lines.append(Line(spans=list(current), bbox=line_bbox, meta=line_meta))
            current = [span]
    if current:
        line_bbox = _merge_bbox(s.bbox for s in current)
        line_meta = {
            "font_size": sum(s.font_size for s in current) / max(1, len(current)),
            "left_margin": line_bbox[0],
        }
        lines.append(Line(spans=list(current), bbox=line_bbox, meta=line_meta))

    for idx, line in enumerate(lines[:-1]):
        gap = _line_gap(line, lines[idx + 1])
        line.meta["leading"] = gap
    if lines:
        lines[-1].meta.setdefault("leading", 0.0)

    blocks: List[Block] = []
    block_lines: List[Line] = []
    for line in lines:
        if not block_lines:
            block_lines.append(line)
            continue
        prev_line = block_lines[-1]
        gap = _line_gap(prev_line, line)
        prev_height = prev_line.bbox[3] - prev_line.bbox[1]
        gap_limit = (prev_height if prev_height > 0 else 12.0) * y_gap_max_lh
        margin_delta = abs(line.meta.get("left_margin", 0.0) - prev_line.meta.get("left_margin", 0.0))
        if gap > gap_limit or margin_delta > margin_tol:
            block_bbox = _merge_bbox(ln.bbox for ln in block_lines)
            block_meta = _block_meta(block_lines, page_width)
            blocks.append(Block(lines=list(block_lines), bbox=block_bbox, meta=block_meta))
            block_lines = [line]
        else:
            block_lines.append(line)
    if block_lines:
        block_bbox = _merge_bbox(ln.bbox for ln in block_lines)
        block_meta = _block_meta(block_lines, page_width)
        blocks.append(Block(lines=list(block_lines), bbox=block_bbox, meta=block_meta))

    return lines, blocks


def _block_meta(lines: Sequence[Line], page_width: float | None) -> Dict[str, float | int]:
    left = min(line.meta.get("left_margin", line.bbox[0]) for line in lines)
    widths = [line.bbox[2] - line.bbox[0] for line in lines]
    font_sizes = [line.meta.get("font_size", 0.0) for line in lines]
    avg_font = sum(font_sizes) / max(1, len(font_sizes))
    block_bbox = _merge_bbox(line.bbox for line in lines)
    block_width = block_bbox[2] - block_bbox[0]
    col_id = 0
    if page_width and page_width > 0:
        norm_left = left / page_width
        if norm_left >= 0.66:
            col_id = 2
        elif norm_left >= 0.33:
            col_id = 1
    return {
        "font_size": avg_font,
        "left_margin": left,
        "col_width": block_width,
        "col_id": col_id,
        "line_count": len(lines),
        "avg_width": sum(widths) / max(1, len(widths)),
    }
