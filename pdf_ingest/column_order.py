"""Utilities for repairing multi-column reading order."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sklearn.cluster import KMeans

from .pdf_io import Line


def reorder_page_lines(lines: List[Line]) -> List[Line]:
    """Attempt to repair two-column layouts using k-means on x-centroids."""

    with_coords = [ln for ln in lines if ln.x_center is not None and ln.y_top is not None]
    if len(with_coords) < 6:
        return sorted(lines, key=_line_sort_key)

    X = np.array([[ln.x_center] for ln in with_coords], dtype=float)
    try:
        model = KMeans(n_clusters=2, n_init=5, random_state=13)
        labels = model.fit_predict(X)
    except Exception:
        return sorted(lines, key=_line_sort_key)

    centers = model.cluster_centers_.ravel()
    left_cluster = int(np.argmin(centers))
    right_cluster = 1 - left_cluster

    grouped = {left_cluster: [], right_cluster: []}
    for ln, label in zip(with_coords, labels):
        grouped.setdefault(label, []).append(ln)

    ordered: List[Line] = []
    for cluster in (left_cluster, right_cluster):
        ordered.extend(sorted(grouped.get(cluster, []), key=_line_sort_key))

    # Append lines that lacked coordinate data in their original order
    without_coords = [ln for ln in lines if ln not in with_coords]
    ordered.extend(sorted(without_coords, key=lambda ln: ln.line_index))

    return ordered


def _line_sort_key(line: Line) -> tuple[float, float]:
    y = line.y_top if line.y_top is not None else float(line.line_index)
    x = line.x_center if line.x_center is not None else 0.0
    return (y, x)
