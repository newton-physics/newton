# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Rest geometry and per-frame strain evidence for the cached jacket trials."""

from __future__ import annotations

import numpy as np

__all__ = ["ClothStrain", "mesh_statistics"]


def mesh_statistics(vertices, triangles):
    """Measure edge lengths [m], triangle areas [m²], and mesh conditioning."""
    v, t = np.asarray(vertices, dtype=np.float64), np.asarray(triangles)
    e = np.sort(np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]])), axis=1)
    e = np.unique(e, axis=0)
    lengths = np.linalg.norm(v[e[:, 1]] - v[e[:, 0]], axis=1)
    a, b = v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]
    area = np.linalg.norm(np.cross(a, b), axis=1) * 0.5
    qs = [0, 0.01, 0.5, 0.95, 0.99, 1]
    return {
        "vertex_count": len(v),
        "triangle_count": len(t),
        "quantile_probabilities": qs,
        "edge_length_m_quantiles": np.quantile(lengths, qs).tolist(),
        "triangle_area_m2_quantiles": np.quantile(area, qs).tolist(),
        "surface_area_m2": float(area.sum()),
        "dimensions_m": np.ptp(v, axis=0).tolist(),
    }


class ClothStrain:
    """Measure strain against ModelBuilder's actual imported triangle rest basis."""

    def __init__(self, model):
        self.triangles = model.tri_indices.numpy()
        self.rest_inverse = model.tri_poses.numpy().astype(np.float64)
        self.mass = model.particle_mass.numpy().astype(np.float64)

    def sample(self, positions):
        """Return principal stretch and area ratios; one means no deformation."""
        q = np.asarray(positions, dtype=np.float64)
        triangles = q[self.triangles]
        ds = np.stack((triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=2)
        deformation = ds @ self.rest_inverse
        singular = np.linalg.svd(deformation, compute_uv=False)
        area_ratio = singular[:, 0] * singular[:, 1]
        return {
            "center_of_mass_m": np.average(q, axis=0, weights=self.mass).tolist(),
            "min_z_m": float(q[:, 2].min()),
            "max_z_m": float(q[:, 2].max()),
            "principal_stretch_p50": float(np.quantile(singular[:, 0], 0.5)),
            "principal_stretch_p95": float(np.quantile(singular[:, 0], 0.95)),
            "principal_stretch_p99": float(np.quantile(singular[:, 0], 0.99)),
            "principal_stretch_max": float(singular[:, 0].max()),
            "area_ratio_min": float(area_ratio.min()),
            "area_ratio_p99": float(np.quantile(area_ratio, 0.99)),
            "area_ratio_max": float(area_ratio.max()),
            "triangles_over_10_percent_stretch": int(np.sum(singular[:, 0] > 1.1)),
        }
