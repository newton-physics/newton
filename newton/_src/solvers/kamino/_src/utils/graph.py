# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host-side graph utilities shared by the Kamino solvers."""

from __future__ import annotations

import numpy as np

###
# Module interface
###

__all__ = ["connected_component_labels"]


###
# Functions
###


def connected_component_labels(count: int, bid_a: np.ndarray, bid_b: np.ndarray) -> np.ndarray:
    """Label each vertex with the lowest vertex index of its connected component.

    Uses min-label propagation with pointer jumping, vectorized over all edges.
    Labels only decrease and stay inside their component, so the fixed point is
    the component's lowest vertex index. Replicated worlds converge in a few
    passes, unlike a per-edge union-find loop.

    Args:
        count: Number of vertices.
        bid_a: First endpoint of each edge.
        bid_b: Second endpoint of each edge.

    Returns:
        The ``int32`` component label of each vertex.
    """
    labels = np.arange(count, dtype=np.int32)
    if bid_a.size == 0:
        return labels
    while True:
        endpoint_labels = np.minimum(labels[bid_a], labels[bid_b])
        updated = labels.copy()
        np.minimum.at(updated, bid_a, endpoint_labels)
        np.minimum.at(updated, bid_b, endpoint_labels)
        updated = updated[updated]
        if np.array_equal(updated, labels):
            return labels
        labels = updated
