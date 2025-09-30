from __future__ import annotations

from typing import Dict, Iterable, List

from html import escape

from .normalize import Block


def make_diagnostic_row(block: Block, decision: str, reason: str) -> Dict[str, object]:
    bbox = block.bbox or {}
    width = float(bbox.get("x1", 0.0)) - float(bbox.get("x0", 0.0))
    return {
        "block_id": block.block_id,
        "page": block.page,
        "y": bbox.get("y0", 0.0),
        "x": bbox.get("x0", 0.0),
        "width": width,
        "type": block.type,
        "text_prefix_64": (block.text or "")[:64],
        "decision": decision,
        "reason": reason,
    }


def build_overlay_html(blocks: Iterable[Block], max_pages: int = 2) -> str:
    pages: List[int] = sorted({block.page for block in blocks})[:max_pages]
    html: List[str] = ["<html><body>"]
    for page in pages:
        html.append(f"<h3>Page {page}</h3>")
        html.append("<div style='position:relative;border:1px solid #ccc;'>")
        for block in blocks:
            if block.page != page:
                continue
            bbox = block.bbox or {}
            color = block.aux.get("diagnostic_color", "gray")
            width = float(bbox.get("x1", 0.0)) - float(bbox.get("x0", 0.0))
            height = float(bbox.get("y1", 0.0)) - float(bbox.get("y0", 0.0))
            title = escape((block.text or "")[:120])
            html.append(
                "<div style="
                "position:absolute;border:2px solid {color};left:{left}px;top:{top}px;"
                "width:{width}px;height:{height}px".format(
                    color=color,
                    left=bbox.get("x0", 0.0),
                    top=bbox.get("y0", 0.0),
                    width=width,
                    height=height,
                )
                + f""" title='{title}'></div>"""
            )
        html.append("</div>")
    html.append("</body></html>")
    return "\n".join(html)
