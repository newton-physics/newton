# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify physical invariants needed by persistent FeatherPGS contacts."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.friction_patches import contact_tangent_basis
from newton.solvers import SolverFeatherPGS
from newton.tests.test_contact_reduction_body_pairs import _cylinder_foot


class TestFeatherPGSPersistentContacts(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "requires CUDA graph capture")
    def test_combined_history_reset_replays_with_changed_world_mask(self):
        """Capture reduction, matching and solver resets with a live device mask."""
        template = newton.ModelBuilder()
        _cylinder_foot(template, wp.vec3(0.0, 0.0, 0.015))
        builder = newton.ModelBuilder()
        builder.replicate(template, 2, spacing=(1.0, 0.0, 0.0))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_cuda_device())
        pipeline = newton.CollisionPipeline(
            model,
            contact_matching="latest",
            reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(
                body_pairs=True,
                body_pair_hysteresis=0.001,
            ),
        )
        solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_warmstart=True)
        state, output = model.state(), model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
        mask = wp.array([False, False, False], dtype=wp.bool, device=model.device)
        solver_mask = mask[:2]
        # Compile both reset paths before capture.
        solver.reset(state, solver_mask)
        pipeline.reset_contact_matching(mask)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.reset(state, solver_mask)
            pipeline.reset_contact_matching(mask)
            pipeline.collide(state, contacts)
        for selected in (0, 1):
            values = np.zeros(3, dtype=bool)
            values[selected] = True
            mask.assign(values)
            solver._ws_prev_mf_impulses.fill_(1.0)
            wp.capture_launch(capture.graph)
            previous = solver._ws_prev_mf_impulses.numpy()
            self.assertTrue(np.all(previous[selected] == 0.0))
            self.assertTrue(np.all(previous[1 - selected] == 1.0))
            count = int(contacts.rigid_contact_count.numpy()[0])
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = contacts.rigid_contact_shape1.numpy()[:count]
            worlds = np.maximum(model.shape_world.numpy()[shape0], model.shape_world.numpy()[shape1])
            matches = contacts.rigid_contact_match_index.numpy()[:count]
            self.assertTrue(np.any(worlds == selected) and np.any(worlds == 1 - selected))
            self.assertTrue(np.all(matches[worlds == selected] < 0))
            self.assertTrue(np.all(matches[worlds == 1 - selected] >= 0))

    def _seed_rotating_contact(self, friction, *, oblique=False, rotated=False, device="cpu"):
        """Prepare a cached friction impulse and move its normal across a basis seam."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.X, gravity=wp.vec3(0.0))
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        model.shape_material_mu.fill_(1.0)
        solver = SolverFeatherPGS(model, pgs_warmstart=True, pgs_iterations=0)
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        pipeline.collide(state_in, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        contacts.rigid_contact_count.fill_(1)
        old_normal = wp.normalize(wp.vec3(-1.0, 0.001, 0.0))
        new_normal = wp.normalize(wp.vec3(-1.0, -0.001, 0.0))
        if oblique:
            old_normal = wp.normalize(wp.vec3(0.2, 0.3, 0.93))
            new_normal = wp.normalize(wp.vec3(0.3, 0.2, 0.93))
        if rotated:
            old_normal = wp.vec3(0.0, 0.0, 1.0)
            new_normal = wp.vec3(0.5 / np.sqrt(2.0), 0.5 / np.sqrt(2.0), np.sqrt(3.0) / 2.0)
        normals = contacts.rigid_contact_normal.numpy()
        normals[0] = old_normal
        contacts.rigid_contact_normal.assign(normals)
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
        slot = int(solver.contact_slot.numpy()[0])
        self.assertGreaterEqual(slot, 0)
        jacobian = solver.mf_J_a if int(solver.mf_body_a.numpy()[0, slot]) >= 0 else solver.mf_J_b
        old_direction = jacobian.numpy()[0, slot + 1, :3].copy()
        previous = solver._ws_prev_mf_impulses.numpy()
        previous[0, slot : slot + 3] = (1.0, 0.5, 0.0)
        solver._ws_prev_mf_impulses.assign(previous)
        # Seed both owners of the previous solve's tangent impulse. Persistent
        # anchors take precedence over the contact-matched cache when carried.
        patches = solver._friction_patches
        tangent, _ = contact_tangent_basis(-old_normal)
        tangent *= 0.5
        if patches.current.flipped.numpy()[0]:
            tangent = -tangent
        anchor_body = int(patches.current.body_a.numpy()[0])
        if anchor_body >= 0:
            pose = state_in.body_q.numpy()[anchor_body]
            transform = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
            tangent = wp.transform_vector(wp.transform_inverse(transform), tangent)
        cached = patches.previous.tangent_impulse.numpy()
        cached[0] = tangent
        patches.previous.tangent_impulse.assign(cached)
        normals[0] = new_normal
        contacts.rigid_contact_normal.assign(normals)
        contacts.rigid_contact_match_index.fill_(-1)
        indices = contacts.rigid_contact_match_index.numpy()
        indices[0] = 0
        contacts.rigid_contact_match_index.assign(indices)
        model.shape_material_mu.fill_(friction)
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
        slot = int(solver.contact_slot.numpy()[0])
        impulses = solver.mf_impulses.numpy()[0, slot : slot + 3]
        tangent_rows = jacobian.numpy()[0, slot + 1 : slot + 3, :3]
        world_tangent = impulses[1:] @ tangent_rows
        # Independent world-space projection; solver rows use B-to-A normals.
        n = np.asarray(new_normal)
        projected_direction = old_direction - n * np.dot(n, old_direction)
        return impulses, world_tangent, projected_direction

    def test_cached_friction_preserves_world_direction(self):
        """Transport a cached tangent impulse through a discontinuous tangent basis."""
        _, world_tangent, old_direction = self._seed_rotating_contact(1.0)
        np.testing.assert_allclose(world_tangent, 0.5 * old_direction, atol=1.0e-6)

    def test_cached_friction_uses_solver_normal_convention(self):
        """Project friction in the frame used by the actual contact Jacobians."""
        devices = ["cpu", wp.get_cuda_device()] if wp.is_cuda_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=str(device)):
                _, world_tangent, projected_direction = self._seed_rotating_contact(1.0, oblique=True, device=device)
                np.testing.assert_allclose(world_tangent, 0.5 * projected_direction, atol=1.0e-6)

    def test_cached_friction_obeys_changed_material(self):
        """Clamp carried friction to the current material before applying velocity."""
        impulses, world_tangent, old_direction = self._seed_rotating_contact(0.1)
        self.assertLessEqual(float(np.linalg.norm(impulses[1:])), 0.1 * impulses[0] + 1.0e-6)
        np.testing.assert_allclose(world_tangent, 0.1 * old_direction, atol=1.0e-6)

    def test_rotating_contact_clamps_the_projected_friction_cone(self):
        """Project through a 30-degree normal change and saturate the current cone."""
        devices = ["cpu", wp.get_cuda_device()] if wp.is_cuda_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=str(device)):
                impulses, tangent, projected = self._seed_rotating_contact(0.1, rotated=True, device=device)
                expected = 0.1 * projected / np.linalg.norm(projected)
                np.testing.assert_allclose(tangent, expected, atol=1.0e-6)
                self.assertAlmostEqual(float(np.linalg.norm(impulses[1:])), 0.1 * impulses[0], delta=1.0e-6)

    def test_matching_retains_world_distance_policy(self):
        """Moving beyond the existing world-distance threshold must cold-start."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        for z in (0.1, 0.29):
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        state = model.state()
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0)
        poses = state.body_q.numpy()
        poses[:, 0] += 1.0
        state.body_q.assign(poses)
        pipeline.collide(state, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), count)
        self.assertTrue(np.all(contacts.rigid_contact_match_index.numpy()[:count] < 0))

    def test_reduced_contacts_preserve_warmstart_identity(self):
        """Carry only retained contact identities through reduction and solver steps."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        _cylinder_foot(builder, wp.vec3(0.0, 0.0, 0.015))
        model = builder.finalize(device="cpu")
        solver = SolverFeatherPGS(model, pgs_warmstart=True, pgs_iterations=8)
        pipeline = newton.CollisionPipeline(
            model,
            contact_matching="latest",
            reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(body_pairs=True),
        )
        contacts = pipeline.contacts()
        state_in, state_out = model.state(), model.state()
        matched = 0
        for _ in range(40):
            pipeline.collide(state_in, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            indices = contacts.rigid_contact_match_index.numpy()[:count]
            valid = indices[indices >= 0]
            self.assertEqual(len(valid), len(np.unique(valid)), "a cached impulse was assigned more than once")
            matched += len(valid)
            self.assertEqual(int(pipeline._contact_matcher.prev_contact_count.numpy()[0]), count)
            state_in.clear_forces()
            solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
            state_in, state_out = state_out, state_in
        stats = pipeline.body_pair_reduction_stats()
        self.assertGreater(stats["sum_contacts_in"], stats["sum_contacts_kept"])
        self.assertGreater(matched, 20)
        self.assertTrue(np.isfinite(state_in.body_q.numpy()).all())
        self.assertGreater(float(state_in.body_q.numpy()[0, 2]), 0.012)

    def test_matching_reset_also_invalidates_reduction_history(self):
        """Partial resets preserve other worlds; material changes start cold."""
        template = newton.ModelBuilder()
        _cylinder_foot(template, wp.vec3(0.0, 0.0, 0.015))
        builder = newton.ModelBuilder()
        builder.replicate(template, 2, spacing=(1.0, 0.0, 0.0))
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(
            model,
            contact_matching="latest",
            reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(
                body_pairs=True,
                body_pair_hysteresis=0.001,
            ),
        )
        state, contacts = model.state(), pipeline.contacts()
        pipeline.collide(state, contacts)
        pipeline.collide(state, contacts)
        generations = pipeline._body_pair_reducer.history_generation.numpy().copy()
        pipeline.reset_contact_matching(wp.array([True, False, False], dtype=wp.bool, device=model.device))
        after = pipeline._body_pair_reducer.history_generation.numpy()
        self.assertGreater(after[0], generations[0])
        self.assertEqual(after[1], generations[1])
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        worlds = np.maximum(model.shape_world.numpy()[shape0], model.shape_world.numpy()[shape1])
        matches = contacts.rigid_contact_match_index.numpy()[:count]
        self.assertTrue(np.any(worlds == 0) and np.any(worlds == 1))
        self.assertTrue(np.all(matches[worlds == 0] < 0))
        self.assertTrue(np.all(matches[worlds == 1] >= 0))
        model.shape_material_mu.fill_(0.1)
        pipeline.refresh_body_pair_reduction_groups()
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertTrue(np.all(contacts.rigid_contact_match_index.numpy()[:count] < 0))


if __name__ == "__main__":
    unittest.main()
