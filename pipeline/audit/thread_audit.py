"""Post-thread audits to guarantee auxiliary anchors land on legal boundaries."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

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


def run_thread_audit(document: FusedDocument) -> List[AuditFinding]:
    """Validate anchor placement and reposition offenders when possible."""

    findings: List[AuditFinding] = []
    for page in document.pages:
        for block in page.main_flow:
            anchors = block.metadata.get("anchors") if isinstance(block.metadata.get("anchors"), list) else []
            for anchor in list(anchors):
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
