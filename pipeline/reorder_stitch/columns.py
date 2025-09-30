from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans

from ..normalize import Block


def detect_columns(
    blocks: Sequence[Block], page_width: float
) -> Tuple[List[Tuple[float, float]], Dict[str, int]]:
    """Detect column boundaries and assign a column index per block."""

    usable = [block for block in blocks if block.bbox and block.bbox["x1"] > block.bbox["x0"]]
    if not usable:
        return [(0.0, page_width)], {block.block_id: 0 for block in blocks}

    centers = np.array(
        [
            [((b.bbox["x0"] + b.bbox["x1"]) / 2.0)]
            for b in usable
        ]
    )

    max_k = min(3, len(usable))
    best_model: KMeans | None = None
    best_inertia = float("inf")
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=0)
        model.fit(centers)
        if model.inertia_ < best_inertia:
            best_inertia = model.inertia_
            best_model = model
    assert best_model is not None  # for type checkers

    labels = list(best_model.labels_)
    label_to_blocks: Dict[int, List[Block]] = {}
    for block, label in zip(usable, labels):
        label_to_blocks.setdefault(int(label), []).append(block)

    label_to_center = {
        label: float(np.mean([((b.bbox["x0"] + b.bbox["x1"]) / 2.0) for b in group]))
        for label, group in label_to_blocks.items()
    }
    ordered_labels = [label for label, _ in sorted(label_to_center.items(), key=lambda item: item[1])]

    accepted: List[int] = []
    columns: List[Tuple[float, float]] = []
    min_width = max(page_width * 0.25, 1.0)

    for label in ordered_labels:
        xs = [
            (block.bbox["x0"], block.bbox["x1"])
            for block in label_to_blocks.get(label, [])
        ]
        if not xs:
            continue
        x0 = min(pair[0] for pair in xs)
        x1 = max(pair[1] for pair in xs)
        width = x1 - x0
        if width >= min_width or len(ordered_labels) == 1:
            accepted.append(label)
            columns.append((x0, x1))

    if not columns:
        columns = [(0.0, page_width)]
        accepted = ordered_labels[:1]

    assignments: Dict[str, int] = {}
    for block, label in zip(usable, labels):
        if label in accepted:
            idx = accepted.index(label)
        else:
            idx = _closest_column_index(block, columns)
        assignments[block.block_id] = idx

    for block in blocks:
        assignments.setdefault(block.block_id, _closest_column_index(block, columns))

    return columns, assignments


def _closest_column_index(block: Block, columns: Sequence[Tuple[float, float]]) -> int:
    if not block.bbox:
        return 0
    xmid = (block.bbox["x0"] + block.bbox["x1"]) / 2.0
    centres = [((x0 + x1) / 2.0) for x0, x1 in columns]
    return min(range(len(columns)), key=lambda idx: abs(xmid - centres[idx]))
