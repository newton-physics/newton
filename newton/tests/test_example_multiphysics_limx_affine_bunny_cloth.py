# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest

import numpy as np
import warp as wp

from newton.solvers import ConstraintAnchor, ConstraintDihedralBending, ConstraintTriangleElastic
from newton.viewer import ViewerNull


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestMultiphysicsLimxAffineBunnyClothExample(unittest.TestCase):
    def test_uses_approved_affine_cloth_configuration(self):
        """Configure the approved 100-by-100 four-corner mixed-contact scene."""
        module = importlib.import_module(
            "newton.examples.multiphysics.example_multiphysics_limx_affine_bunny_cloth"
        )
        device = wp.get_cuda_devices()[0]

        with wp.ScopedDevice(device):
            example = module.Example(ViewerNull(num_frames=1), None)

        self.assertEqual(example.cloth_cells, 100)
        self.assertEqual(example.model.particle_count, 101 * 101)
        self.assertEqual(example.model.tri_count, 2 * 100 * 100)
        self.assertEqual(len(example.cloth_anchor_indices), 4)
        self.assertEqual(example.frame_dt, 0.01)
        self.assertEqual(example.solver.nonlinear_iterations, 1)
        self.assertEqual(example.solver.linear_iterations, 50)
        self.assertEqual(example.solver.velocity_damping, 1.0)
        self.assertEqual(example.contact.thickness, 0.003)
        self.assertEqual(example.contact.stiffness, 2.0e4)
        self.assertEqual(example.contact.normal_damping, 0.0)
        self.assertEqual(example.contact.friction, 0.01)
        self.assertEqual(example.contact.friction_epsilon, 1.0e-4)
        self.assertEqual(example.contact.max_contacts, 262144)
        self.assertAlmostEqual(float(example.solver.q.numpy()[0, 2]), 0.55, places=6)
        self.assertIs(example.solver.dynamic_operator, example.contact)
        self.assertEqual(sum(isinstance(item, ConstraintAnchor) for item in example.solver.constraints), 1)
        self.assertEqual(sum(isinstance(item, ConstraintTriangleElastic) for item in example.solver.constraints), 1)
        self.assertEqual(sum(isinstance(item, ConstraintDihedralBending) for item in example.solver.constraints), 1)
        self.assertIsNotNone(example.graph)
        self.assertEqual(module.Example.create_parser().parse_args([]).num_frames, 300)

    def test_supports_bunny_over_300_frames(self):
        """Support the affine bunny without anchor drift, overflow, or deep penetration."""
        module = importlib.import_module(
            "newton.examples.multiphysics.example_multiphysics_limx_affine_bunny_cloth"
        )
        device = wp.get_cuda_devices()[0]

        with wp.ScopedDevice(device):
            example = module.Example(ViewerNull(num_frames=300), None)
            for _ in range(300):
                example.step()
                example.test_post_step()
            example.test_final()

        positions = example.state_0.particle_q.numpy()
        anchor_error = np.linalg.norm(
            positions[example.cloth_anchor_indices] - example.cloth_anchor_targets,
            axis=1,
        )
        self.assertLess(float(anchor_error.max()), 1.0e-4)
        self.assertTrue(example.contact_observed)
        self.assertEqual(example.maximum_contact_overflow, 0)
        self.assertLess(example.maximum_contact_depth, 0.012)
        self.assertGreater(example.initial_cloth_center_height - example.minimum_cloth_center_height, 0.02)
        self.assertGreater(
            float(example.solver.q.numpy()[0, 2] - positions[example.cloth_center_index, 2]),
            0.03,
        )
        self.assertGreater(example.minimum_affine_determinant, 0.0)


if __name__ == "__main__":
    unittest.main()
