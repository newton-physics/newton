# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerNull

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


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestLimxArapBunnyTableExample(unittest.TestCase):
    def test_uses_approved_contact_and_solver_configuration(self):
        """Use the approved undamped one-step LIMX table-contact configuration."""
        module = importlib.import_module("newton.examples.softbody.example_softbody_limx_arap_bunny_table")
        example = module.Example(ViewerNull(num_frames=1), None)

        self.assertEqual(example.frame_dt, 0.01)
        self.assertEqual(example.model.particle_count, 1869)
        self.assertEqual(example.model.tet_count, 7356)
        self.assertEqual(example.model.body_count, 0)
        self.assertGreater(example.initial_minimum_height, 0.14)
        self.assertLess(example.initial_minimum_height, 0.16)
        self.assertEqual(example.solver.nonlinear_iterations, 1)
        self.assertEqual(example.solver.linear_iterations, 128)
        self.assertEqual(example.solver.velocity_damping, 1.0)
        self.assertIs(example.solver.dynamic_operator, example.table_contact)
        self.assertEqual(example.table_contact.thickness, 0.003)
        self.assertEqual(example.table_contact.stiffness, 2.0e4)
        self.assertEqual(example.table_contact.normal_damping, 0.0)
        self.assertEqual(example.table_contact.friction, 0.05)


if __name__ == "__main__":
    unittest.main()
