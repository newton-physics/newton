# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest

import warp as wp

from newton.viewer import ViewerNull


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestLimxArapBunniesBoxExample(unittest.TestCase):
    def test_uses_automatic_topology_local_collision_thickness(self):
        """Estimate nonlocal VF/EE thickness from the two-ring clearance."""
        module = importlib.import_module("newton.examples.softbody.example_softbody_limx_arap_bunnies_box")
        example = module.Example(ViewerNull(num_frames=1), None)

        self.assertEqual(example.bunny_count, 8)
        self.assertEqual(example.particles_per_bunny, 1869)
        self.assertEqual(example.model.particle_count, 14952)
        self.assertEqual(example.model.tet_count, 58848)
        self.assertEqual(example.model.tri_count, 17216)
        self.assertEqual(len(example.surface_vertex_indices), 8624)
        self.assertEqual(example.model.body_count, 0)
        self.assertEqual(example.model.shape_count, 1)
        self.assertTrue(example.self_collision.thickness_was_estimated)
        self.assertAlmostEqual(example.self_collision.thickness, 0.0029901, places=7)
        self.assertLessEqual(example.self_collision.thickness, 0.005)
        self.assertEqual(set(example.arap_constraint.host_stiffnesses), {3.0e5})
        self.assertEqual(example.self_collision.geometry_radius_scale, 0.25)
        self.assertTrue(example.self_collision.geometry_radius_topology_local_only)
        self.assertIsNone(example.self_collision.stiffness)
        self.assertEqual(example.self_collision.stiffness_factors, (0.5, 0.3, 1.5))
        self.assertEqual(example.self_collision.friction, 0.0)
        self.assertEqual(example.self_collision.max_contacts, 262144)
        self.assertFalse(example.self_collision.enable_edge_face)
        self.assertTrue(example.self_collision.use_outward_normals)
        self.assertEqual(len(example.box_contacts), 5)
        self.assertEqual(example.self_collision.surface_vertex_count, 8624)
        self.assertTrue(all(contact.contact_particle_count == 8624 for contact in example.box_contacts))
        self.assertTrue(
            all(
                (contact.particle_indices.numpy() == example.surface_vertex_indices).all()
                for contact in example.box_contacts
            )
        )
        self.assertTrue(all(contact.thickness == 0.003 for contact in example.box_contacts))
        self.assertTrue(all(contact.normal_damping == 0.0 for contact in example.box_contacts))
        self.assertTrue(all(contact.friction == 0.05 for contact in example.box_contacts))
        self.assertEqual(example.solver.nonlinear_iterations, 1)
        self.assertEqual(example.solver.linear_iterations, 50)
        self.assertEqual(example.solver.velocity_damping, 1.0)

    def test_remains_stable_for_300_frames(self):
        """Keep oriented bunny contact stable through the full example rollout."""
        module = importlib.import_module("newton.examples.softbody.example_softbody_limx_arap_bunnies_box")
        example = module.Example(ViewerNull(num_frames=300), None)

        for _ in range(300):
            example.step()
            example.test_post_step()
        example.test_final()

        self.assertGreater(example.minimum_determinant, 0.0)
        self.assertEqual(int(example.saw_cross_bunny_contact.numpy()[0]), 1)


if __name__ == "__main__":
    unittest.main()
