"""PDF extraction via unstructured."""

# When pypdf struggles with complex layouts, we hand the document to
# ``unstructured``. The dependency is heavier, so the driver only invokes it
# when necessary.

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from unstructured.partition.pdf import partition_pdf
except ImportError as exc:  # pragma: no cover - dependency managed via requirements
    partition_pdf = None  # type: ignore[assignment]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

logger = logging.getLogger(__name__)
_VALID_STRATEGIES = {"fast", "auto", "hi_res"}


def extract_pdf_with_unstructured(
    file_path: str,
    strategy: str,
    *,
    max_pages: int | None = None,
    enable_hi_res: bool = False,
) -> Tuple[List[str], Dict[str, str]]:
    """Extract PDFs using the unstructured partitioner.

    Parameters mirror :func:`extract_pdf_with_pypdf`, but we expose the
    ``strategy`` knob so the driver can respect the user-configured mode while
    still clamping to CPU-safe defaults.

    Returns
    -------
    tuple[list[str], dict[str, str]]
        ``unstructured`` emits one or more layout-aware ``Element`` objects per
        page; we reassemble them into plain strings and return a metadata dict
        that captures timing and any warnings emitted by the library.
    """

    if partition_pdf is None:  # pragma: no cover - ensures clearer error at runtime
        raise RuntimeError(
            "unstructured.partition.pdf is unavailable. Install unstructured to enable fallback."
        ) from IMPORT_ERROR

    pdf_path = Path(file_path)
    requested = (strategy or "fast").lower()
    if requested not in _VALID_STRATEGIES:
        logger.warning("Unknown unstructured strategy '%s'; defaulting to fast", requested)
        requested = "fast"

    warnings: list[str] = []
    strategy_to_use = requested
    if requested == "hi_res":
        warnings.append("hi_res strategy stubbed to fast; heavy OCR dependencies are disabled")
        strategy_to_use = "fast"
    elif requested == "auto" and not enable_hi_res:
        # auto may choose hi_res internally; remind user it is constrained
        warnings.append("auto strategy limited to CPU fast path; hi_res flag disabled")

    start = time.perf_counter()
    kwargs: Dict[str, object] = {
        "filename": str(pdf_path),
        "strategy": strategy_to_use,
        "extract_images_in_pdf": False,
        "include_page_breaks": False,
        "infer_table_structure": True,
    }
    if max_pages is not None:
        kwargs["maxpages"] = max_pages

    elements = partition_pdf(**kwargs)

    pages_map: dict[int, list[str]] = defaultdict(list)
    max_page_seen = 0
    for element in elements:
        page_number = getattr(getattr(element, "metadata", None), "page_number", None) or 1
        text = getattr(element, "text", "") or ""
        pages_map[page_number].append(text)
        max_page_seen = max(max_page_seen, page_number)

    if max_page_seen == 0:
        logger.warning("unstructured returned no elements for %s", pdf_path.name)
        return [], {
            "elapsed": f"{time.perf_counter() - start:.4f}",
            "requested_strategy": requested,
            "strategy_used": strategy_to_use,
            "warnings": "|".join(warnings),
        }

    ordered_pages: List[str] = []
    for page_number in range(1, max_page_seen + 1):
        chunk = "\n".join(pages_map.get(page_number, []))
        ordered_pages.append(chunk)

    elapsed = time.perf_counter() - start
    meta: Dict[str, str] = {
        "elapsed": f"{elapsed:.4f}",
        "requested_strategy": requested,
        "strategy_used": strategy_to_use,
        "warnings": "|".join(warnings),
    }
    logger.debug(
        "unstructured extraction for %s produced %d pages in %.3fs (strategy=%s)",
        pdf_path.name,
        len(ordered_pages),
        elapsed,
        strategy_to_use,
    )
    return ordered_pages, meta
