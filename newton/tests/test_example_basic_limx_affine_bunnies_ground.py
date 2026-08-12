# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest

import numpy as np
import warp as wp

from newton.viewer import ViewerNull


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestBasicLimxAffineBunniesGroundExample(unittest.TestCase):
    def test_uses_approved_multi_body_contact_configuration(self):
        """Use the approved affine pile and frictional contact settings."""
        module = importlib.import_module("newton.examples.basic.example_basic_limx_affine_bunnies_ground")
        device = wp.get_cuda_devices()[0]

        with wp.ScopedDevice(device):
            example = module.Example(ViewerNull(num_frames=1), None)

        self.assertEqual(example.body_model.body_count, 8)
        self.assertEqual(example.frame_dt, 0.01)
        self.assertEqual(example.solver.nonlinear_iterations, 1)
        self.assertEqual(example.solver.linear_iterations, 50)
        self.assertEqual(example.solver.velocity_damping, 1.0)
        self.assertEqual(example.body_contact.thickness, 0.003)
        self.assertEqual(example.body_contact.stiffness, 2.0e4)
        self.assertEqual(example.body_contact.normal_damping, 0.5)
        self.assertEqual(example.body_contact.friction, 0.5)
        self.assertEqual(example.ground_contact.friction, 0.5)
        self.assertEqual(module.Example.create_parser().parse_args([]).num_frames, 300)
        self.assertIsNotNone(example.graph)

    def test_stacks_eight_bunnies_over_300_frames(self):
        """Stack eight affine bunnies without inversion, overflow, or deep contact."""
        module = importlib.import_module("newton.examples.basic.example_basic_limx_affine_bunnies_ground")
        device = wp.get_cuda_devices()[0]

        with wp.ScopedDevice(device):
            example = module.Example(ViewerNull(num_frames=300), None)
            for _ in range(300):
                example.step()
                example.test_post_step()
            example.test_final()

        final_centers = example.center_heights[-1]
        self.assertTrue(np.all(example.initial_center_heights - final_centers > 0.03))
        self.assertGreater(example.minimum_determinant, 0.0)
        self.assertLess(example.maximum_singular_value_error, 0.02)
        self.assertGreaterEqual(example.minimum_height, -0.006)
        self.assertEqual(example.maximum_vf_overflow, 0)
        self.assertEqual(example.maximum_ee_overflow, 0)
        self.assertLess(example.maximum_contact_depth, 0.012)
        self.assertTrue(example.cross_body_contact_observed)
        self.assertGreater(float(np.mean(example.support_margins[-30:])), 0.10)


if __name__ == "__main__":
    unittest.main()
