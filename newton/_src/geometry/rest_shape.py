# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rest geometry calculations shared by model construction and USD import.

Callers own validation, topology mapping, and fallback policies. Calculations
preserve the input precision and do not access builder or USD state.
"""

from __future__ import annotations

import numpy as np


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors along the last axis, leaving zero vectors unchanged."""
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return vectors / lengths


def _triangle_rest_geometry(qp: np.ndarray, rp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return triangle rest bases and areas from two edges, preserving input precision."""
    normal = _normalize_vectors(np.cross(qp, rp))
    e1 = _normalize_vectors(qp)
    e2 = _normalize_vectors(np.cross(normal, e1))
    frame = np.stack((e1, e2), axis=-1)
    edges = np.stack((qp, rp), axis=-1)
    basis = np.swapaxes(frame, -2, -1) @ edges
    return basis, np.linalg.det(basis) / 2.0


def _tet_rest_geometry(qp: np.ndarray, rp: np.ndarray, sp: np.ndarray) -> tuple[np.ndarray, np.ndarray | np.floating]:
    """Return tetrahedron rest matrices and signed volumes without changing edge precision."""
    # Avoid batch-stacking overhead in add_tetrahedron's per-element path.
    matrices = np.array((qp, rp, sp)).T if qp.ndim == 1 else np.stack((qp, rp, sp), axis=-1)
    return matrices, np.linalg.det(matrices) / 6.0


def _dihedral_angles(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    x4: np.ndarray,
    x3_other: np.ndarray,
    x4_other: np.ndarray,
) -> np.ndarray:
    """Return signed dihedrals for triangles (x1, x3, x4) and (x2, x4_other, x3_other).

    The second triangle can use separate copies of the shared edge for disjoint rest
    topology. Welded geometry passes x3 and x4 for both copies.
    """
    n1 = _normalize_vectors(np.cross(x3 - x1, x4 - x1))
    n2 = _normalize_vectors(np.cross(x4_other - x2, x3_other - x2))
    edge = _normalize_vectors(x4 - x3)
    cos_theta = np.clip((n1 * n2).sum(axis=-1), -1.0, 1.0)
    sin_theta = (np.cross(n1, n2) * edge).sum(axis=-1)
    return np.arctan2(sin_theta, cos_theta)
