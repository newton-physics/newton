# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

import numpy as np

import newton


ASSET_PATH = Path(__file__).resolve().parents[1] / "examples" / "assets" / "bunny_tet.npz"


class TestBunnyTetAsset(unittest.TestCase):
    def test_preserves_source_topology_and_orientation(self):
        """Preserve the source bunny topology and positive tetrahedron orientation."""
        mesh = newton.TetMesh.create_from_file(str(ASSET_PATH))
        tetrahedra = mesh.tet_indices.reshape(-1, 4)
        edges = np.stack(
            (
                mesh.vertices[tetrahedra[:, 1]] - mesh.vertices[tetrahedra[:, 0]],
                mesh.vertices[tetrahedra[:, 2]] - mesh.vertices[tetrahedra[:, 0]],
                mesh.vertices[tetrahedra[:, 3]] - mesh.vertices[tetrahedra[:, 0]],
            ),
            axis=2,
        )

        self.assertEqual(mesh.vertex_count, 1869)
        self.assertEqual(mesh.tet_count, 7356)
        self.assertEqual(len(mesh.surface_tri_indices) // 3, 2152)
        self.assertGreater(float(np.linalg.det(edges).min()), 0.0)


if __name__ == "__main__":
    unittest.main()
