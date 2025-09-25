"""Block classification and heading detection."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import statistics

from .utils import (
    Block,
    DocumentLayout,
    DocBlock,
    PageLayout,
    bbox_union,
    detect_footnote_marker,
    ends_with_terminal_punct,
    font_stats,
    is_all_caps,
    line_spacing,
    load_config,
    normalize_bbox,
    slugify,
    symmetric_band,
    text_density,
    token_count,
    whitespace_halo,
)


@dataclass
class BlockPrediction:
    page: int
    index: int
    kind: str
    aux_type: Optional[str]
    text: str
    bbox: Tuple[float, float, float, float]
    ms: float
    hs: Optional[float]
    reason: List[str]
    meta: Dict[str, object]
    flow: str
    confidence: float
    source_block: Block
    expect_continuation: bool = False

    def copy(self) -> "BlockPrediction":
        return replace(self, reason=list(self.reason), meta=dict(self.meta))


class BlockClassifier:
    def __init__(self, config: Optional[Dict[str, object]] = None) -> None:
        self.config = config or load_config()
        thresholds = self.config.get("thresholds", {})
        self.tau_main = float(thresholds.get("tau_main", 0.6))
        self.tau_main_confident = float(thresholds.get("tau_main_page_confident", 0.55))
        self.tau_heading_h12 = float(thresholds.get("tau_heading_h12", 0.75))
        self.tau_heading_h3 = float(thresholds.get("tau_heading_h3", 0.60))
        bands = self.config.get("bands", {})
        self.header_band = float(bands.get("header_y_pct", 0.12))
        self.footer_band = float(bands.get("footer_y_pct", 0.88))
        margin = bands.get("margin_x_pct", [0.0, 0.12, 0.88, 1.0])
        self.margin_left = float(margin[1])
        self.margin_right = float(margin[2])
        wrapping = self.config.get("wrapping", {})
        self.wrap_width_pct = float(wrapping.get("wrap_text_width_max_pct_of_col", 0.6))
        self.symmetric_tol = float(wrapping.get("symmetric_band_tolerance_pct", 0.10))

    def classify(self, doc: DocumentLayout) -> List[BlockPrediction]:
        predictions: List[BlockPrediction] = []
        for page in doc.pages:
            predictions.extend(self._classify_page(page))
        return predictions

    def _classify_page(self, page: PageLayout) -> List[BlockPrediction]:
        text_blocks = [b for b in page.blocks if b.block_type == "text"]
        page_font_sizes: List[float] = []
        for block in text_blocks:
            page_font_sizes.extend(block.font_sizes)
        page_font_mean = statistics.mean(page_font_sizes) if page_font_sizes else 0.0
        page_font_max = max(page_font_sizes) if page_font_sizes else 0.0
        results: List[BlockPrediction] = []
        for idx, block in enumerate(page.blocks):
            block_meta = self._block_meta(block, page)
            ms, reason = self._mainness(block, page, page_font_mean, page_font_max)
            hs, heading_reasons = self._heading_score(block, page, page_font_mean, page_font_max)
            reason.extend(heading_reasons)
            heading_level: Optional[int] = None
            if hs is not None:
                block_meta["heading_score"] = hs
                if hs >= self.tau_heading_h12:
                    heading_level = 1
                elif hs >= self.tau_heading_h3:
                    heading_level = 2
                if heading_level is not None:
                    block_meta["heading_level"] = heading_level
            kind = "main"
            flow = "main"
            aux_type: Optional[str] = None
            tau = self.tau_main if page_font_mean else self.tau_main_confident
            if ms < tau or block.block_type != "text":
                kind = "aux"
                flow = "aux"
                aux_type = self._aux_type(block, page, block_meta)
                reason.append("BelowTauMain")
            elif heading_level is None and self._force_aux(block, page, block_meta):
                kind = "aux"
                flow = "aux"
                aux_type = self._aux_type(block, page, block_meta)
                reason.append("ForcedAux")
                ms = min(ms, 0.45)
            if heading_level is not None and kind == "aux":
                kind = "main"
                flow = "main"
                aux_type = None
                reason.append("HeadingOverride")
                ms = max(ms, self.tau_main_confident)
            expect_continuation = self._expect_continuation(block, block_meta)
            prediction = BlockPrediction(
                page=page.page_number,
                index=idx,
                kind=kind,
                aux_type=aux_type,
                text=block.text,
                bbox=block.bbox,
                ms=ms,
                hs=hs,
                reason=reason,
                meta=block_meta,
                flow=flow,
                confidence=self._confidence(kind, ms, hs),
                source_block=block,
                expect_continuation=expect_continuation,
            )
            results.append(prediction)
        return results

    def _block_meta(self, block: Block, page: PageLayout) -> Dict[str, object]:
        mean_size, std_size = font_stats(block)
        meta = {
            "font_size": mean_size,
            "font_size_std": std_size,
            "font_name": block.fonts[0] if block.fonts else None,
            "line_spacing": line_spacing(block),
            "text_density": text_density(block),
            "col_id": block.attrs.get("col_id"),
            "page_width": page.width,
            "page_height": page.height,
            "indent": block.lines[0].bbox[0] - block.left if block.lines else 0.0,
            "block_type": block.block_type,
            "token_count": token_count(block.text),
        }
        return meta

    def _mainness(
        self,
        block: Block,
        page: PageLayout,
        page_font_mean: float,
        page_font_max: float,
    ) -> Tuple[float, List[str]]:
        score = 0.45 if block.block_type == "text" else 0.1
        reasons: List[str] = []
        width_pct = block.width / page.width if page.width else 0.0
        height_pct = block.height / page.height if page.height else 0.0
        density = text_density(block)
        if block.block_type == "text":
            score += min(0.2, density * 0.02)
            if len(block.lines) >= 2:
                score += 0.1
                reasons.append("MultiLine")
            if token_count(block.text) >= 8:
                score += 0.1
                reasons.append("LongText")
            if width_pct >= 0.45:
                score += 0.15
                reasons.append("Wide")
            if block.left / page.width > self.margin_left and block.right / page.width < self.margin_right:
                score += 0.1
                reasons.append("InsideMargins")
            if block.attrs.get("col_id") in {0, 2} and width_pct >= 0.3:
                score += 0.12
                reasons.append("ColumnAligned")
        col_id = block.attrs.get("col_id")
        if width_pct < self.wrap_width_pct:
            if col_id in {0, 2}:
                score -= 0.05
                reasons.append("NarrowColumn")
            else:
                score -= 0.25
                reasons.append("NarrowBand")
        if width_pct <= self.wrap_width_pct and symmetric_band(
            block, page.width, self.symmetric_tol
        ):
            score -= 0.2
            reasons.append("SymmetricBand")
        if detect_footnote_marker(block.text):
            score -= 0.2
            reasons.append("FootnoteMarker")
        if block.top / page.height <= self.header_band or block.bottom / page.height >= self.footer_band:
            score -= 0.25
            reasons.append("HeaderFooterBand")
        if block.block_type in {"table", "figure", "image"}:
            score -= 0.4
            reasons.append("NonText")
        score = max(0.0, min(1.0, score))
        return score, reasons

    def _force_aux(self, block: Block, page: PageLayout, meta: Dict[str, object]) -> bool:
        width_pct = block.width / page.width if page.width else 0.0
        if meta["token_count"] <= 3 and is_all_caps(block.text):
            return True
        if width_pct < self.wrap_width_pct and symmetric_band(block, page.width, self.symmetric_tol):
            return True
        if block.block_type in {"figure", "table"}:
            return True
        if detect_footnote_marker(block.text) and block.top / page.height > self.footer_band - 0.05:
            return True
        lowered = block.text.strip().lower()
        if lowered.startswith(("figure", "fig.", "table")):
            return True
        return False

    def _heading_score(
        self,
        block: Block,
        page: PageLayout,
        page_font_mean: float,
        page_font_max: float,
    ) -> Tuple[Optional[float], List[str]]:
        if block.block_type != "text" or not block.text:
            return None, []
        score = 0.0
        reasons: List[str] = []
        mean_size, std_size = font_stats(block)
        if mean_size and page_font_mean and mean_size >= page_font_mean:
            ratio = (mean_size - page_font_mean) / (page_font_mean or 1)
            score += min(0.6, ratio * 0.8)
            reasons.append("FontJump")
        if mean_size and mean_size >= page_font_max:
            score += 0.2
            reasons.append("MaxFont")
        if block.lines and len(block.lines) == 1 and token_count(block.text) <= 12:
            score += 0.1
            reasons.append("ShortLine")
        if is_all_caps(block.text):
            score += 0.1
            reasons.append("AllCaps")
        if block.lines and any(span.bold for span in block.lines[0].spans):
            score += 0.05
            reasons.append("Bold")
        above, below = whitespace_halo(page, block)
        if above >= 12 and below >= 12:
            score += 0.15
            reasons.append("WhitespaceHalo")
        if ends_with_terminal_punct(block.text):
            score -= 0.2
            reasons.append("TrailingPunctuation")
        if detect_footnote_marker(block.text):
            score -= 0.25
        if score <= 0:
            return None, reasons
        return min(1.0, max(0.0, score)), reasons

    def _aux_type(self, block: Block, page: PageLayout, meta: Dict[str, object]) -> Optional[str]:
        if block.block_type in {"figure", "image"}:
            return "figure"
        if block.block_type == "table":
            return "table"
        top_pct = block.top / page.height if page.height else 0.0
        bottom_pct = block.bottom / page.height if page.height else 0.0
        center_x = (block.left + block.right) / 2 / page.width if page.width else 0.5
        text = block.text.strip()
        if detect_footnote_marker(text):
            return "footnote"
        if top_pct <= self.header_band:
            if text.isdigit() or len(text) <= 4:
                return "page_number"
            return "header"
        if bottom_pct >= self.footer_band:
            if detect_footnote_marker(text):
                return "footnote"
            if text.isdigit():
                return "page_number"
            return "footer"
        if text.lower().startswith(("figure", "fig.", "table")):
            return "caption"
        if len(text) <= 40 and is_all_caps(text):
            return "callout"
        width_pct = block.width / page.width if page.width else 0.0
        if width_pct < self.wrap_width_pct and symmetric_band(block, page.width, self.symmetric_tol):
            return "sidenote"
        return "other"

    def _expect_continuation(self, block: Block, meta: Dict[str, object]) -> bool:
        if not block.text:
            return False
        if block.text.endswith("-"):
            return True
        if not ends_with_terminal_punct(block.text) and meta.get("token_count", 0) >= 3:
            return True
        return False

    def _confidence(self, kind: str, ms: float, hs: Optional[float]) -> float:
        if kind == "main":
            return max(ms, hs or 0.5)
        return max(0.6, 1.0 - ms / 2)


def classify_blocks(doc: DocumentLayout, config: Optional[Dict[str, object]] = None) -> List[BlockPrediction]:
    classifier = BlockClassifier(config=config)
    return classifier.classify(doc)
