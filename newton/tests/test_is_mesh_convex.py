# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`newton._src.geometry.utils.is_mesh_convex`.

Covers:
- convex closed meshes (box, tetrahedron) with both windings
- non-convex closed meshes (U-channel built from three boxes)
- open/planar meshes (no volume, not a violation)
- degenerate and trivial inputs
- the ``max_face_vertex_pairs`` cost guard
"""

import unittest

import numpy as np

from newton._src.geometry.utils import is_mesh_convex


def box_mesh(cx=0.0, cy=0.0, cz=0.0, hx=1.0, hy=1.0, hz=1.0):
    v = np.array(
        [[sx * hx + cx, sy * hy + cy, sz * hz + cz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=np.float32,
    )
    f = [
        (0, 1, 3),
        (0, 3, 2),
        (4, 6, 7),
        (4, 7, 5),
        (0, 4, 5),
        (0, 5, 1),
        (2, 3, 7),
        (2, 7, 6),
        (0, 2, 6),
        (0, 6, 4),
        (1, 5, 7),
        (1, 7, 3),
    ]
    return v, np.array(f, dtype=np.int32)


def u_channel_mesh():
    # Floor plus two walls as one closed triangle mesh; the cavity between the
    # walls makes the surface non-convex.
    parts = [
        box_mesh(0, 0, 0.05, 0.6, 0.6, 0.05),
        box_mesh(0, 0.40, 0.30, 0.6, 0.2, 0.20),
        box_mesh(0, -0.40, 0.30, 0.6, 0.2, 0.20),
    ]
    verts, faces, off = [], [], 0
    for v, f in parts:
        verts.append(v)
        faces.append(f + off)
        off += len(v)
    return np.concatenate(verts), np.concatenate(faces)


class TestIsMeshConvex(unittest.TestCase):
    def test_box_is_convex(self):
        """Return True for a closed, convex box mesh."""
        verts, faces = box_mesh()
        self.assertTrue(is_mesh_convex(verts, faces))

    def test_box_with_inverted_winding_is_convex(self):
        """Return True for a convex box regardless of triangle winding."""
        verts, faces = box_mesh()
        self.assertTrue(is_mesh_convex(verts, faces[:, ::-1]))

    def test_tetrahedron_is_convex(self):
        """Return True for a minimal closed tetrahedron."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        faces = np.array([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3], dtype=np.int32)
        self.assertTrue(is_mesh_convex(verts, faces))

    def test_u_channel_is_not_convex(self):
        """Return False for a closed mesh with a cavity between its walls."""
        verts, faces = u_channel_mesh()
        self.assertFalse(is_mesh_convex(verts, faces))

    def test_scaled_u_channel_is_not_convex(self):
        """Return False for a non-convex mesh regardless of uniform scale."""
        # Convexity is scale-invariant; the solver relies on this for its cache.
        verts, faces = u_channel_mesh()
        self.assertFalse(is_mesh_convex(verts * 7.5, faces))

    def test_downscaled_u_channel_is_not_convex(self):
        """Return False for a non-convex mesh below the epsilon floor."""
        # Raw plane distances scale with the square of mesh size while eps
        # scales with extent; an unnormalized comparison would swallow a tiny
        # non-convex mesh entirely.
        verts, faces = u_channel_mesh()
        self.assertFalse(is_mesh_convex(verts * 1e-3, faces))

    def test_enlarged_box_is_convex(self):
        """Return True for a convex mesh at large scales."""
        verts, faces = box_mesh()
        self.assertTrue(is_mesh_convex(verts * 1e5, faces))

    def test_flat_open_mesh_is_convex(self):
        """Return True for an open planar grid with no separating face plane."""
        xs, ys = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
        verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(16)], axis=1).astype(np.float32)
        faces = []
        for y in range(3):
            for x in range(3):
                i = y * 4 + x
                faces += [i, i + 1, i + 4, i + 1, i + 5, i + 4]
        self.assertTrue(is_mesh_convex(verts, np.array(faces, dtype=np.int32)))

    def test_none_indices_assumed_convex(self):
        """Return True when triangle indices are missing."""
        verts, _ = u_channel_mesh()
        self.assertTrue(is_mesh_convex(verts, None))

    def test_degenerate_inputs_assumed_convex(self):
        """Return True instead of raising for empty or degenerate inputs."""
        verts, faces = u_channel_mesh()
        self.assertTrue(is_mesh_convex(verts, np.array([], dtype=np.int32).reshape(0, 3)))
        self.assertTrue(is_mesh_convex(np.zeros((3, 3), dtype=np.float32), faces[:3]))

    def test_cost_guard_returns_true(self):
        """Return True instead of failing when the pair bound is exceeded."""
        verts, faces = u_channel_mesh()
        # A bound below faces x vertices skips the exact test instead of failing it.
        self.assertTrue(is_mesh_convex(verts, faces, max_face_vertex_pairs=10))

    def test_missing_faces_within_guard(self):
        """Skip zero-area triangles instead of treating them as planes."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 1, 1], [1, 2, 2], [0, 0, 0]], dtype=np.int32)
        self.assertTrue(is_mesh_convex(verts, faces))


if __name__ == "__main__":
    unittest.main()
