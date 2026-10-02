# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify physical invariants needed by persistent FeatherPGS contacts."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.friction import contact_tangent_basis
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_test_devices

_DEVICE = "cuda:0"


def _cylinder_foot(builder, pos, mass=1.0, num_cyl=7, radius=0.02, half_height=0.015):
    """Add one free body whose collision is ``num_cyl`` small cylinders in a row.

    All cylinders reach the ground at nearly the same height, so a plane contact
    produces a multiple of ``num_cyl`` contacts for a single region.
    """
    body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), mass=mass)
    for i in range(num_cyl):
        x = (i - (num_cyl - 1) / 2.0) * (2.2 * radius)
        builder.add_shape_cylinder(
            body,
            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            radius=radius,
            half_height=half_height,
        )
    return body


@unittest.skipUnless(wp.is_cuda_available(), "SolverFeatherPGS requires CUDA")
class TestFeatherPGSPersistentContacts(unittest.TestCase):
    def test_combined_history_reset_replays_with_changed_world_mask(self):
        """Capture contact-matching and solver resets with a live device mask."""
        template = newton.ModelBuilder()
        _cylinder_foot(template, wp.vec3(0.0, 0.0, 0.015))
        builder = newton.ModelBuilder()
        builder.replicate(template, 2, spacing=(1.0, 0.0, 0.0))
        builder.add_ground_plane()
        model = builder.finalize(device=_DEVICE)
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        solver = SolverFeatherPGS(model, pgs_warmstart=True)
        state, output = model.state(), model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
        # The solver and the pipeline share the (world_count + 1,) reset-mask layout.
        mask = wp.array([False, False, False], dtype=wp.bool, device=model.device)
        # Compile both reset paths before capture.
        solver.reset(state, mask)
        pipeline.reset_contact_matching(mask)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.reset(state, mask)
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

    def _seed_rotating_contact(self, friction, *, oblique=False, rotated=False, device=_DEVICE):
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
        _, world_tangent, projected_direction = self._seed_rotating_contact(1.0, oblique=True)
        np.testing.assert_allclose(world_tangent, 0.5 * projected_direction, atol=1.0e-6)

    def test_cached_friction_obeys_changed_material(self):
        """Clamp carried friction to the current material before applying velocity."""
        impulses, world_tangent, old_direction = self._seed_rotating_contact(0.1)
        self.assertLessEqual(float(np.linalg.norm(impulses[1:])), 0.1 * impulses[0] + 1.0e-6)
        np.testing.assert_allclose(world_tangent, 0.1 * old_direction, atol=1.0e-6)

    def test_rotating_contact_clamps_the_projected_friction_cone(self):
        """Project through a 30-degree normal change and saturate the current cone."""
        impulses, tangent, projected = self._seed_rotating_contact(0.1, rotated=True)
        expected = 0.1 * projected / np.linalg.norm(projected)
        np.testing.assert_allclose(tangent, expected, atol=1.0e-6)
        self.assertAlmostEqual(float(np.linalg.norm(impulses[1:])), 0.1 * impulses[0], delta=1.0e-6)


def test_matching_retains_world_distance_policy(test, device):
    """Moving beyond the existing world-distance threshold must cold-start."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for z in (0.1, 0.29):
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    state = model.state()
    pipeline = newton.CollisionPipeline(model, contact_matching="latest")
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    poses = state.body_q.numpy()
    poses[:, 0] += 1.0
    state.body_q.assign(poses)
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), count)
    test.assertTrue(np.all(contacts.rigid_contact_match_index.numpy()[:count] < 0))


add_function_test(
    TestFeatherPGSPersistentContacts,
    "test_matching_retains_world_distance_policy",
    test_matching_retains_world_distance_policy,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
