"""Public entrypoint exposing the simplified parsing pipeline."""
from __future__ import annotations

from typing import Dict, Optional

from .exporter import export_doc
from .pipeline import PageInput, parse_document
from .utils import DEFAULT_CONFIG
import copy


class DocumentParser:
    """High-level facade used by the tests."""

    def __init__(self, config: Optional[Dict[str, object]] = None) -> None:
        self.config = config or copy.deepcopy(DEFAULT_CONFIG)

    def parse(self, pages) -> Dict[str, object]:
        doc = parse_document(pages, self.config)
        return export_doc(doc)


__all__ = ["DocumentParser", "PageInput"]
