# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest

import numpy as np
import warp as wp

from newton.viewer import ViewerNull


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestBasicLimxAffineBunnyGroundExample(unittest.TestCase):
    def test_uses_approved_affine_contact_configuration(self):
        """Use the approved rigid affine bunny and frictional ground settings."""
        module = importlib.import_module("newton.examples.basic.example_basic_limx_affine_bunny_ground")
        device = wp.get_cuda_devices()[0]

        with wp.ScopedDevice(device):
            example = module.Example(ViewerNull(num_frames=1), None)

        self.assertEqual(example.frame_dt, 0.01)
        self.assertEqual(example.solver.nonlinear_iterations, 1)
        self.assertEqual(example.solver.linear_iterations, 50)
        self.assertEqual(example.solver.velocity_damping, 1.0)
        self.assertEqual(example.body_model.surface_vertex_count, 1078)
        self.assertEqual(example.body_model.surface_triangle_count, 2152)
        self.assertAlmostEqual(example.body_mass, 2.81, delta=0.03)
        self.assertAlmostEqual(float(example.solver.q.numpy()[0, 2]), 0.65, places=6)
        self.assertEqual(example.ground_contact.thickness, 0.003)
        self.assertEqual(example.ground_contact.stiffness, 2.0e4)
        self.assertEqual(example.ground_contact.normal_damping, 0.5)
        self.assertEqual(example.ground_contact.friction, 0.5)
        self.assertEqual(example.ground_contact.friction_epsilon, 1.0e-4)
        self.assertIs(example.solver.dynamic_operator, example.ground_contact)
        self.assertIsNotNone(example.graph)
        self.assertEqual(module.Example.create_parser().parse_args([]).num_frames, 300)

    def test_drops_contacts_and_settles_over_300_frames(self):
        """Drop onto the ground without losing rigidity or sliding indefinitely."""
        module = importlib.import_module("newton.examples.basic.example_basic_limx_affine_bunny_ground")
        device = wp.get_cuda_devices()[0]

        with wp.ScopedDevice(device):
            example = module.Example(ViewerNull(num_frames=300), None)
            for _ in range(300):
                example.step()
                example.test_post_step()
            example.test_final()

        self.assertTrue(example.contact_activated)
        self.assertLessEqual(example.minimum_height, example.ground_contact.thickness)
        self.assertGreaterEqual(example.minimum_height, -0.006)
        self.assertGreater(example.initial_center_height - example.center_heights[-1], 0.20)
        self.assertGreater(example.minimum_determinant, 0.0)
        self.assertLess(example.maximum_singular_value_error, 0.02)
        self.assertLess(float(np.mean(example.tangential_speeds[-30:])), 0.05)


if __name__ == "__main__":
    unittest.main()
