"""Post-thread audits to guarantee auxiliary anchors land on legal boundaries."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from pipeline.layout.lp_fuser import FusedBlock, FusedDocument

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AuditFinding:
    page_number: int
    block_id: str
    issue: str


def _find_anchor_index(block: FusedBlock, anchor: str) -> int:
    if not block.text or not anchor:
        return -1
    return block.text.find(anchor)


def _ensure_sentence_boundary(block: FusedBlock, anchor: str) -> bool:
    index = _find_anchor_index(block, anchor)
    if index == -1:
        return False
    prefix = block.text[:index].rstrip()
    if not prefix:
        return False
    last_char = prefix[-1]
    if last_char in {".", "?", "!", "\n", "\r", "\u201d", "\u2019"}:
        return False
    block.text = block.text.replace(anchor, "", 1)
    block.text = prefix + ("." if last_char not in {".", "?", "!"} else "") + " " + anchor + block.text[index:].lstrip()
    block.metadata.setdefault("anchors", [])
    if anchor not in block.metadata["anchors"]:
        block.metadata["anchors"].append(anchor)
    return True


def _move_anchor(anchor: str, source: FusedBlock, target: FusedBlock) -> None:
    if anchor in source.text:
        source.text = source.text.replace(anchor, "").strip()
    anchors = source.metadata.get("anchors")
    if isinstance(anchors, list) and anchor in anchors:
        anchors.remove(anchor)
    text = target.text.rstrip()
    if text and text[-1] not in {".", "?", "!", "\u201d"}:
        text = text + "."
    target.text = (text + " " + anchor).strip()
    target_anchors = target.metadata.get("anchors")
    if not isinstance(target_anchors, list):
        target_anchors = []
    if anchor not in target_anchors:
        target_anchors.append(anchor)
    target.metadata["anchors"] = target_anchors
    target.metadata["has_anchor_refs"] = True


def run_thread_audit(document: FusedDocument) -> List[AuditFinding]:
    """Validate anchor placement and reposition offenders when possible."""

    findings: List[AuditFinding] = []
    section_last_block: Dict[str, FusedBlock] = {}
    last_main_block: Optional[FusedBlock] = None
    for page in document.pages:
        for block in page.main_flow:
            if not block.metadata.get("is_heading"):
                section_id = str(block.metadata.get("section_id") or "0")
                section_last_block[section_id] = block
                last_main_block = block
    for page in document.pages:
        for block in page.main_flow:
            anchors = block.metadata.get("anchors") if isinstance(block.metadata.get("anchors"), list) else []
            for anchor in list(anchors):
                section_id = str(block.metadata.get("section_id") or "0")
                target = section_last_block.get(section_id) or last_main_block
                moved = False
                if block.metadata.get("section_lead") and target and target is not block:
                    _move_anchor(anchor, block, target)
                    findings.append(
                        AuditFinding(
                            page_number=page.page_number,
                            block_id=block.block_id,
                            issue="anchor-moved-from-lead-paragraph",
                        )
                    )
                    moved = True
                elif target and target is not block:
                    _move_anchor(anchor, block, target)
                    findings.append(
                        AuditFinding(
                            page_number=page.page_number,
                            block_id=block.block_id,
                            issue="anchor-moved-to-section-end",
                        )
                    )
                    moved = True
                if moved:
                    continue
                if _ensure_sentence_boundary(block, anchor):
                    findings.append(
                        AuditFinding(
                            page_number=page.page_number,
                            block_id=block.block_id,
                            issue="anchor-moved-to-boundary",
                        )
                    )
    if findings:
        logger.debug("Thread audit repositioned %d anchors", len(findings))
    return findings


__all__ = ["AuditFinding", "run_thread_audit"]
