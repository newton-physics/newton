# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.viewer.picking import Picking
from newton._src.viewer.viewer_null import ViewerNull
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices


def _make_single_sphere_model(device=None, *, is_kinematic: bool = False, body_com=None):
    """Model with one body and one sphere at origin (radius 0.5)."""
    builder = newton.ModelBuilder()
    builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        com=body_com,
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=body_com is not None,
        is_kinematic=is_kinematic,
    )
    builder.add_shape_sphere(body=0, radius=0.5)
    return builder.finalize(device=device)


def _make_kinematic_front_dynamic_back_model(device=None):
    """Model with a kinematic sphere in front and dynamic sphere behind it."""
    builder = newton.ModelBuilder()
    builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        is_kinematic=True,
    )
    builder.add_shape_sphere(body=0, radius=0.5)

    builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    builder.add_shape_sphere(body=1, radius=0.5)
    return builder.finalize(device=device)


def _make_model_no_shapes(device=None):
    """Model with one body and no shapes (shape_count == 0)."""
    builder = newton.ModelBuilder()
    builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    return builder.finalize(device=device)


class TestPickingSetup(unittest.TestCase):
    """Tests for the Picking setup (construction, release, pick, update, apply_force)."""

    def test_init_state(self):
        """Picking initializes with no body picked and default torque behavior."""
        model = _make_single_sphere_model(device="cpu")
        picking = Picking(model, pick_stiffness=100.0, pick_damping=10.0)

        self.assertFalse(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], -1)
        self.assertEqual(picking.pick_stiffness, 100.0)
        self.assertEqual(picking.pick_damping, 10.0)
        self.assertEqual(picking.pick_torque_scale, 1.0)
        self.assertIsNotNone(picking.pick_state)
        self.assertEqual(picking.pick_state.shape[0], 1)

    def test_release_clears_state(self):
        """release() clears pick_body and sets picking_active to False."""
        model = _make_single_sphere_model(device="cpu")
        state = model.state()
        picking = Picking(model)

        # Ray from above origin going down hits the sphere
        ray_start = wp.vec3(0.0, 0.0, 2.0)
        ray_dir = wp.vec3(0.0, 0.0, -1.0)
        picking.pick(state, ray_start, ray_dir)
        self.assertTrue(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], 0)

        picking.release()
        self.assertFalse(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], -1)

    def test_pick_miss_remains_inactive(self):
        """pick() with a ray that misses all geometry leaves picking inactive."""
        model = _make_single_sphere_model(device="cpu")
        state = model.state()
        picking = Picking(model)

        # Ray far from the sphere
        ray_start = wp.vec3(10.0, 10.0, 0.0)
        ray_dir = wp.vec3(0.0, 0.0, 1.0)
        picking.pick(state, ray_start, ray_dir)

        self.assertFalse(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], -1)

    def test_pick_hit_activates_picking(self):
        """pick() with a ray that hits the sphere activates picking and sets pick_body."""
        model = _make_single_sphere_model(device="cpu")
        state = model.state()
        picking = Picking(model)

        # Ray from -Z toward origin hits the sphere (center at origin, radius 0.5)
        ray_start = wp.vec3(0.0, 0.0, -2.0)
        ray_dir = wp.vec3(0.0, 0.0, 1.0)
        picking.pick(state, ray_start, ray_dir)

        self.assertTrue(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], 0)
        self.assertGreater(picking.pick_dist, 0.0)
        self.assertLess(picking.pick_dist, 1.0e10)

    def test_pick_kinematic_body_remains_inactive(self):
        """pick() ignores kinematic bodies so no body is selected."""
        model = _make_single_sphere_model(device="cpu", is_kinematic=True)
        state = model.state()
        picking = Picking(model)

        ray_start = wp.vec3(0.0, 0.0, -2.0)
        ray_dir = wp.vec3(0.0, 0.0, 1.0)
        picking.pick(state, ray_start, ray_dir)

        self.assertFalse(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], -1)

    def test_pick_kinematic_occludes_dynamic(self):
        """pick() does not pick dynamic bodies occluded by kinematic bodies."""
        model = _make_kinematic_front_dynamic_back_model(device="cpu")
        state = model.state()
        picking = Picking(model)

        ray_start = wp.vec3(0.0, 0.0, -3.0)
        ray_dir = wp.vec3(0.0, 0.0, 1.0)
        picking.pick(state, ray_start, ray_dir)

        self.assertFalse(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], -1)

    def test_pick_empty_model_no_crash(self):
        """pick() with a model that has no shapes returns without error."""
        model = _make_model_no_shapes(device="cpu")
        state = model.state()
        picking = Picking(model)

        ray_start = wp.vec3(0.0, 0.0, -2.0)
        ray_dir = wp.vec3(0.0, 0.0, 1.0)
        picking.pick(state, ray_start, ray_dir)

        self.assertFalse(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], -1)

    def test_update_when_not_picking_no_op(self):
        """update() when not picking does not change state."""
        model = _make_single_sphere_model(device="cpu")
        picking = Picking(model)

        self.assertFalse(picking.is_picking())
        picking.update(wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0))
        self.assertFalse(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], -1)

    def test_apply_picking_force_when_not_picking(self):
        """_apply_picking_force() when not picking runs kernel without modifying body_f."""
        model = _make_single_sphere_model(device="cpu")
        state = model.state()
        picking = Picking(model)

        state.body_f.zero_()
        picking._apply_picking_force(state)

        # No body picked -> no force applied
        forces = state.body_f.numpy()
        assert_np_equal(forces, np.zeros_like(forces), tol=1e-9)

    def test_apply_picking_force_when_picking(self):
        """_apply_picking_force() when picking runs; force is non-zero after update() moves target."""
        model = _make_single_sphere_model(device="cpu")
        state = model.state()
        picking = Picking(model)

        # Activate picking with a hit (target at hit point on sphere)
        ray_start = wp.vec3(0.0, 0.0, -2.0)
        ray_dir = wp.vec3(0.0, 0.0, 1.0)
        picking.pick(state, ray_start, ray_dir)
        self.assertTrue(picking.is_picking())

        # Move target by updating with a ray offset from center so target != attachment point
        picking.update(wp.vec3(0.5, 0.0, -2.0), wp.vec3(0.0, 0.0, 1.0))
        state.body_f.zero_()
        picking._apply_picking_force(state)

        forces = state.body_f.numpy()
        self.assertEqual(forces.shape[0], model.body_count)
        self.assertFalse(np.allclose(forces[0], np.zeros(6), atol=1e-9))

    def test_viewer_picking_torque_scale(self):
        """Viewer-level torque control is safe without a picker and forwards when available."""
        viewer = ViewerNull(num_frames=0)
        viewer.set_picking_torque_scale(0.5)
        for scale in (-0.1, 1.1):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "must be in"):
                viewer.set_picking_torque_scale(scale)

        model = _make_single_sphere_model(device="cpu")
        viewer.picking = Picking(model)

        viewer.set_picking_torque_scale(0.25)
        self.assertEqual(viewer.picking.pick_torque_scale, 0.25)

    def test_world_offsets_optional(self):
        """Picking can be constructed with optional world_offsets."""
        model = _make_single_sphere_model(device="cpu")
        picking = Picking(model, world_offsets=None)
        self.assertIsNone(picking.world_offsets)

        offsets = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=model.device)
        picking_with_offsets = Picking(model, world_offsets=offsets)
        self.assertIsNotNone(picking_with_offsets.world_offsets)
        self.assertEqual(picking_with_offsets.world_offsets.shape[0], 1)


def test_picking_setup_device(test: TestPickingSetup, device):
    """Picking setup works on the given device (CPU or CUDA)."""
    model = _make_single_sphere_model(device=device)
    state = model.state()
    picking = Picking(model)

    test.assertFalse(picking.is_picking())
    test.assertEqual(picking.pick_body.numpy()[0], -1)

    # Hit the sphere
    ray_start = wp.vec3(0.0, 0.0, -2.0)
    ray_dir = wp.vec3(0.0, 0.0, 1.0)
    picking.pick(state, ray_start, ray_dir)

    test.assertTrue(picking.is_picking())
    test.assertEqual(picking.pick_body.numpy()[0], 0)

    # update and apply_force should not crash
    picking.update(ray_start, ray_dir)
    picking._apply_picking_force(state)

    picking.release()
    test.assertFalse(picking.is_picking())
    test.assertEqual(picking.pick_body.numpy()[0], -1)


def test_picking_torque_scale(test: TestPickingSetup, device):
    """Picking scales torque while preserving consistent point-force behavior."""
    model = _make_single_sphere_model(device=device, body_com=wp.vec3(0.0))
    state = model.state()
    picking = Picking(model, pick_stiffness=4.0, pick_damping=2.0, pick_max_acceleration=1.0e6)

    def apply_scale(scale: float) -> np.ndarray:
        picking.set_torque_scale(scale)
        state.body_f.zero_()
        picking._apply_picking_force(state)
        return state.body_f.numpy()[0].copy()

    ray_start = wp.vec3(0.0, 0.0, -2.0)
    ray_dir = wp.vec3(0.0, 0.0, 1.0)
    picking.pick(state, ray_start, ray_dir)
    test.assertTrue(picking.is_picking())

    picking.update(wp.vec3(0.5, 0.0, -2.0), ray_dir)
    normal_force = apply_scale(1.0)
    test.assertFalse(np.allclose(normal_force[:3], np.zeros(3), atol=1e-9))
    test.assertFalse(np.allclose(normal_force[3:], np.zeros(3), atol=1e-9))

    half_torque_force = apply_scale(0.5)
    assert_np_equal(half_torque_force[:3], normal_force[:3], tol=1e-6)
    assert_np_equal(half_torque_force[3:], 0.5 * normal_force[3:], tol=1e-6)

    force_only = apply_scale(0.0)
    assert_np_equal(force_only[:3], normal_force[:3], tol=1e-6)
    assert_np_equal(force_only[3:], np.zeros(3), tol=1e-9)

    # Exercise the previous point-force formula and an intermediate virtual
    # attachment with nonzero rotation, linear/angular velocity, and damping.
    state.body_q.assign(
        wp.array(
            [
                wp.transform(
                    wp.vec3(0.1, 0.2, 0.3),
                    wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * wp.pi),
                )
            ],
            dtype=wp.transform,
            device=device,
        )
    )
    state.body_qd.assign(
        wp.array(
            [wp.spatial_vector(0.2, -0.1, 0.3, 1.0, 0.0, 0.0)],
            dtype=wp.spatial_vector,
            device=device,
        )
    )
    pick_state = picking.pick_state.numpy()
    pick_state[0]["picking_target_world"] = (0.4, 0.6, 0.2)
    picking.pick_state.assign(pick_state)

    force_multiplier = 10.0 + model.body_mass.numpy()[0]

    scale_one_force = force_multiplier * np.array([0.8, -0.2, -2.0])
    scale_one_expected = np.concatenate((scale_one_force, np.cross([0.0, 0.5, 0.0], scale_one_force)))
    assert_np_equal(apply_scale(1.0), scale_one_expected, tol=1e-5)

    half_scale_force = force_multiplier * np.array([0.8, 0.8, -0.5])
    half_scale_expected = np.concatenate((half_scale_force, np.cross([0.0, 0.25, 0.0], half_scale_force)))
    assert_np_equal(apply_scale(0.5), half_scale_expected, tol=1e-5)


def test_zero_torque_scale_tracks_com(test: TestPickingSetup, device):
    """Zero torque scale tracks the COM independently of body rotation."""
    model = _make_single_sphere_model(device=device, body_com=wp.vec3(0.0, 0.1, 0.0))
    state = model.state()
    picking = Picking(model, pick_stiffness=100.0, pick_damping=5.0, pick_torque_scale=0.0)

    ray_start = wp.vec3(0.0, 0.0, -2.0)
    ray_dir = wp.vec3(0.0, 0.0, 1.0)
    picking.pick(state, ray_start, ray_dir)
    test.assertTrue(picking.is_picking())

    rotated_q = wp.array(
        [
            wp.transform(
                wp.vec3(0.0, 0.1, -0.1),
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * wp.pi),
            )
        ],
        dtype=wp.transform,
        device=device,
    )
    angular_qd = wp.array(
        [wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)],
        dtype=wp.spatial_vector,
        device=device,
    )
    state.body_q.assign(rotated_q)
    state.body_qd.assign(angular_qd)

    state.body_f.zero_()
    picking._apply_picking_force(state)
    assert_np_equal(state.body_f.numpy()[0], np.zeros(6), tol=1e-5)

    picking.update(wp.vec3(0.5, 0.0, -2.0), ray_dir)
    state.body_f.zero_()
    picking._apply_picking_force(state)
    translated_force = state.body_f.numpy()[0]
    test.assertFalse(np.allclose(translated_force[:3], np.zeros(3), atol=1e-9))
    assert_np_equal(translated_force[3:], np.zeros(3), tol=1e-9)


# Device-parameterized tests
add_function_test(
    TestPickingSetup,
    "test_picking_setup_device",
    test_picking_setup_device,
    devices=get_test_devices(),
)
add_function_test(
    TestPickingSetup,
    "test_picking_torque_scale",
    test_picking_torque_scale,
    devices=get_test_devices(),
)
add_function_test(
    TestPickingSetup,
    "test_zero_torque_scale_tracks_com",
    test_zero_torque_scale_tracks_com,
    devices=get_test_devices(),
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
