# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton.examples.multiphysics.example_softbody_limx_arap_bunny_cloth import Example
from newton.viewer import ViewerNull


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestSoftbodyLimxArapBunnyCloth(unittest.TestCase):
    def test_configures_unified_bunny_cloth_model(self):
        """Configure one LIMX model with ARAP bunny and four-corner cloth."""
        with wp.ScopedDevice("cuda:0"):
            example = Example(ViewerNull(num_frames=1), None)
            for _ in range(100):
                example.step()
            positions = example.state_0.particle_q.numpy()
            velocities = example.state_0.particle_qd.numpy()

        self.assertEqual(example.cloth_cells, 40)
        self.assertEqual(example.cloth_particle_count, 41 * 41)
        self.assertEqual(len(example.cloth_anchor_indices), 4)
        self.assertEqual(example.bunny_particle_stop, example.cloth_particle_start)
        self.assertGreater(len(example.bunny_tetrahedra), 0)
        self.assertEqual(len(example.cloth_triangles), 2 * 40 * 40)
        self.assertEqual(example.solver.nonlinear_iterations, 1)
        self.assertEqual(example.solver.linear_iterations, 50)
        self.assertEqual(example.solver.velocity_damping, 1.0)
        self.assertFalse(example.self_collision.use_outward_normals)
        self.assertEqual(example.self_collision.geometry_radius_scale, 0.25)
        self.assertTrue(example.self_collision.geometry_radius_topology_local_only)
        self.assertEqual(example.self_collision.max_contacts, 262144)
        tetrahedra = example.bunny_tetrahedra
        edges = np.stack(
            (
                positions[tetrahedra[:, 1]] - positions[tetrahedra[:, 0]],
                positions[tetrahedra[:, 2]] - positions[tetrahedra[:, 0]],
                positions[tetrahedra[:, 3]] - positions[tetrahedra[:, 0]],
            ),
            axis=2,
        )
        self.assertGreater(float(np.linalg.det(edges).min()), 0.0)
        self.assertTrue(np.isfinite(positions).all())
        self.assertTrue(np.isfinite(velocities).all())


if __name__ == "__main__":
    unittest.main()
