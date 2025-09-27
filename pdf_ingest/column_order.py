"""Multi-column repair utilities."""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.cluster import KMeans

from .pdf_io import Line


def _line_sort_key(line: Line) -> tuple[float, float]:
    y = line.bbox[1] if line.bbox else float(line.line_index)
    x = line.x_center if line.x_center is not None else 0.0
    return (y, x)


def reorder_page_lines(lines: List[Line]) -> List[Line]:
    with_coords = [ln for ln in lines if ln.x_center is not None and ln.bbox is not None]
    if len(with_coords) < 8:
        return sorted(lines, key=_line_sort_key)

    X = np.array([[ln.x_center] for ln in with_coords], dtype=float)
    try:
        models = [KMeans(n_clusters=k, n_init=5, random_state=17).fit(X) for k in (1, 2)]
    except Exception:
        return sorted(lines, key=_line_sort_key)

    best_order: List[Line] | None = None
    for model in models:
        labels = model.labels_
        centers = model.cluster_centers_.ravel()
        if len(centers) == 1:
            ordered = sorted(with_coords, key=_line_sort_key)
        else:
            left_cluster = int(np.argmin(centers))
            right_cluster = 1 - left_cluster
            left_lines = sorted([ln for ln, label in zip(with_coords, labels) if label == left_cluster], key=_line_sort_key)
            right_lines = sorted([ln for ln, label in zip(with_coords, labels) if label == right_cluster], key=_line_sort_key)
            ordered = left_lines + right_lines
        if best_order is None or len(ordered) > len(best_order):
            best_order = ordered

    ordered = list(best_order or sorted(with_coords, key=_line_sort_key))
    without_coords = [ln for ln in lines if ln not in with_coords]
    ordered.extend(sorted(without_coords, key=lambda ln: ln.line_index))
    return ordered

