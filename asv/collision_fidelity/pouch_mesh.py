# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pouch tessellation that resolves fingertips without over-refining thin sides."""

import hashlib

import numpy as np

from .assets import AssetMesh, cloth_mesh

__all__ = ["build_pouch_mesh"]


def build_pouch_mesh():
    """Retessellate the original 150 by 200 by 18 mm open pouch in metres.

    Front and back cells are 6.25 by 8.33 mm. The side and bottom strips
    retain 4.5 mm spacing across the 18 mm depth. Geometry and area are
    unchanged; this prevents a 2 mm contact shell from nearly filling those
    narrow cells after uniform refinement.
    """
    original = cloth_mesh("pouch", resolution="medium")
    xs = np.linspace(-0.075, 0.075, 25)
    ys = np.linspace(-0.009, 0.009, 5)
    zs = np.linspace(0.0, 0.2, 25)
    vertices, faces, index = [], [], {}

    def panel(positions, *, flip=False):
        rows = []
        for row in positions:
            ids = []
            for point in row:
                key = tuple(np.round(point, 10))
                if key not in index:
                    index[key] = len(vertices)
                    vertices.append(point)
                ids.append(index[key])
            rows.append(ids)
        for j in range(len(rows) - 1):
            for i in range(len(rows[j]) - 1):
                a, b, c, d = rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i]
                pair = [(a, b, c), (a, c, d)]
                faces.extend([tuple(reversed(t)) for t in pair] if flip else pair)

    for sign in (-1, 1):
        panel([[(x, sign * 0.009, z) for x in xs] for z in zs], flip=sign == 1)
        panel([[(sign * 0.075, y, z) for y in ys] for z in zs], flip=sign == -1)
    panel([[(x, y, 0.0) for x in xs] for y in ys], flip=True)
    q = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int32)
    metadata = dict(original.metadata)
    metadata.update(
        name="pouch_grip_refined",
        resolution="front and back refined; depth strips retain coarse spacing",
        body_grid=[24, 4, 24],
        construction="Original welded pouch panels; unchanged dimensions, area, open rim, and material",
        source_mesh="Original procedural pouch in assets.py",
        geometry_change="Retessellation only: 24 cells in X/Z, 4 across narrow Y depth",
    )
    landmarks = {
        "rim": np.flatnonzero(np.isclose(q[:, 2], 0.2)).tolist(),
        "bottom": np.flatnonzero(np.isclose(q[:, 2], 0.0)).tolist(),
    }
    metadata.update(
        vertex_count=len(q),
        triangle_count=len(triangles),
        geometry_sha256=hashlib.sha256(q.astype("<f4").tobytes() + triangles.astype("<i4").tobytes()).hexdigest(),
        dimensions_m=np.ptp(q, axis=0).tolist(),
        bounds_m=[q.min(axis=0).tolist(), q.max(axis=0).tolist()],
        landmarks=landmarks,
        synthetic_material={
            "density_kg_m2": 0.22,
            "particle_radius_m": 0.001,
            "tri_ke": 3000.0,
            "tri_ka": 3000.0,
            "tri_kd": 0.025,
            "edge_ke": 0.025,
            "edge_kd": 0.0005,
        },
    )
    return AssetMesh("pouch_grip_refined", q, triangles, landmarks=landmarks, metadata=metadata)
