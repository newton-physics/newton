# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest

import warp as wp

from newton.viewer import ViewerNull


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestLimxArapBunniesBoxExample(unittest.TestCase):
    def test_uses_fixed_thickness_multi_bunny_configuration(self):
        """Use eight bunnies with fixed 3 mm VF/EE and one undamped Newton step."""
        module = importlib.import_module("newton.examples.softbody.example_softbody_limx_arap_bunnies_box")
        example = module.Example(ViewerNull(num_frames=1), None)

        self.assertEqual(example.bunny_count, 8)
        self.assertEqual(example.particles_per_bunny, 1869)
        self.assertEqual(example.model.particle_count, 14952)
        self.assertEqual(example.model.tet_count, 58848)
        self.assertEqual(example.model.tri_count, 17216)
        self.assertEqual(example.model.body_count, 0)
        self.assertEqual(example.self_collision.thickness, 0.003)
        self.assertIsNone(example.self_collision.geometry_radius_scale)
        self.assertIsNone(example.self_collision.stiffness)
        self.assertEqual(example.self_collision.stiffness_factors, (0.5, 0.3, 1.5))
        self.assertEqual(example.self_collision.friction, 0.05)
        self.assertEqual(example.self_collision.max_contacts, 262144)
        self.assertEqual(len(example.box_contacts), 5)
        self.assertTrue(all(contact.thickness == 0.003 for contact in example.box_contacts))
        self.assertTrue(all(contact.normal_damping == 0.0 for contact in example.box_contacts))
        self.assertEqual(example.solver.nonlinear_iterations, 1)
        self.assertEqual(example.solver.linear_iterations, 50)
        self.assertEqual(example.solver.velocity_damping, 1.0)


if __name__ == "__main__":
    unittest.main()
