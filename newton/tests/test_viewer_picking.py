# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.viewer.picking import Picking
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices


def _make_single_sphere_model(
    device=None, *, is_kinematic: bool = False, body_com=None, body_inertia=None, body_rotation=None
):
    """Create a model containing one body and a sphere at the origin.

    Args:
        device: Device on which to finalize the model.
        is_kinematic: Whether to make the body kinematic.
        body_com: Optional body-frame center of mass.
        body_inertia: Optional body-frame inertia tensor.
        body_rotation: Optional initial body orientation.
    """
    lock_inertia = body_com is not None or body_inertia is not None
    if body_inertia is None:
        body_inertia = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if body_rotation is None:
        body_rotation = wp.quat_identity()
    builder = newton.ModelBuilder()
    builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), body_rotation),
        com=body_com,
        mass=1.0,
        inertia=body_inertia,
        lock_inertia=lock_inertia,
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

    def test_pick_max_acceleration_validation(self):
        """Picking rejects negative and non-finite acceleration limits."""
        model = _make_single_sphere_model(device="cpu")
        for value in (-0.1, np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                Picking(model, pick_max_acceleration=value)

        Picking(model, pick_max_acceleration=0.0)

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


def _apply_picking_target(picking: Picking, state: newton.State, target: tuple[float, float, float]) -> np.ndarray:
    pick_state = picking.pick_state.numpy()
    pick_state[0]["picking_target_world"] = target
    picking.pick_state.assign(pick_state)
    state.body_f.zero_()
    picking._apply_picking_force(state)
    return state.body_f.numpy()[0].copy()


def test_picking_torque_limit(test: TestPickingSetup, device):
    """Picking limits inertia-weighted angular response without changing force."""
    inertia = wp.mat33(1.0e-4, 0.0, 0.0, 0.0, 2.0e-4, 0.0, 0.0, 0.0, 3.0e-4)
    model = _make_single_sphere_model(device=device, body_com=wp.vec3(0.0), body_inertia=inertia)
    state = model.state()
    picking = Picking(model, pick_stiffness=100.0, pick_damping=0.0, pick_max_acceleration=5.0)

    picking.pick(state, wp.vec3(0.0, 0.0, -2.0), wp.vec3(0.0, 0.0, 1.0))
    test.assertTrue(picking.is_picking())
    wrench = _apply_picking_target(picking, state, (0.5, 0.0, -0.5))

    mass = model.body_mass.numpy()[0]
    inv_inertia = model.body_inv_inertia.numpy()[0]
    max_acceleration = 5.0 * 9.81
    rotational_acceleration_sq = wrench[3:] @ inv_inertia @ wrench[3:] / mass

    assert_np_equal(wrench[:3], np.array([max_acceleration * mass, 0.0, 0.0]), tol=1.0e-5)
    test.assertLessEqual(rotational_acceleration_sq, max_acceleration**2 * (1.0 + 2.0e-5))

    raw_torque = np.cross(np.array([0.0, 0.0, -0.5]), wrench[:3])
    test.assertGreater(np.dot(raw_torque, wrench[3:]), 0.0)
    assert_np_equal(np.cross(raw_torque, wrench[3:]), np.zeros(3), tol=1.0e-6)
    test.assertLess(np.linalg.norm(wrench[3:]), np.linalg.norm(raw_torque))


def test_picking_torque_limit_is_noop_below_limit(test: TestPickingSetup, device):
    """Picking preserves the original point-force wrench below the angular limit."""
    model = _make_single_sphere_model(device=device, body_com=wp.vec3(0.0))
    state = model.state()
    picking = Picking(model, pick_stiffness=100.0, pick_damping=0.0, pick_max_acceleration=5.0)

    picking.pick(state, wp.vec3(0.0, 0.0, -2.0), wp.vec3(0.0, 0.0, 1.0))
    wrench = _apply_picking_target(picking, state, (0.01, 0.0, -0.5))
    expected_force = np.array([(10.0 + model.body_mass.numpy()[0]) * 100.0 * 0.01, 0.0, 0.0])
    expected_torque = np.cross(np.array([0.0, 0.0, -0.5]), expected_force)
    assert_np_equal(wrench, np.concatenate((expected_force, expected_torque)), tol=1.0e-5)


def test_picking_torque_limit_rotates_with_inertia(test: TestPickingSetup, device):
    """Picking evaluates anisotropic inertia in the body's local frame."""
    inertia = wp.mat33(1.0e-4, 0.0, 0.0, 0.0, 2.0e-4, 0.0, 0.0, 0.0, 3.0e-4)

    def apply_with_rotation(rotation: wp.quat) -> np.ndarray:
        model = _make_single_sphere_model(
            device=device,
            body_com=wp.vec3(0.0),
            body_inertia=inertia,
            body_rotation=rotation,
        )
        state = model.state()
        picking = Picking(model, pick_stiffness=100.0, pick_damping=0.0, pick_max_acceleration=5.0)
        picking.pick(state, wp.vec3(0.0, 0.0, -2.0), wp.vec3(0.0, 0.0, 1.0))
        return _apply_picking_target(picking, state, (0.5, 0.0, -0.5))

    identity_wrench = apply_with_rotation(wp.quat_identity())
    rotated_wrench = apply_with_rotation(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * wp.pi))

    assert_np_equal(rotated_wrench[:3], identity_wrench[:3], tol=1.0e-5)
    test.assertAlmostEqual(
        np.linalg.norm(rotated_wrench[3:]) / np.linalg.norm(identity_wrench[3:]),
        np.sqrt(3.0 / 2.0),
        delta=2.0e-5,
    )


def test_picking_torque_limit_cable(test: TestPickingSetup, device):
    """Default picking keeps a low-inertia cable's angular speed bounded."""
    num_links = 12
    segment_length = 0.05
    builder = newton.ModelBuilder(gravity=0.0)
    points = [wp.vec3(-0.5 * num_links * segment_length + i * segment_length, 0.0, 0.3) for i in range(num_links + 1)]
    quaternions = newton.utils.create_parallel_transport_cable_quaternions(points, twist_total=0.0)
    bodies, _ = builder.add_rod(
        positions=points,
        quaternions=quaternions,
        radius=0.012,
        stretch_stiffness=5.0e5,
        bend_stiffness=20.0,
        bend_damping=20.0,
        body_frame_origin="com",
    )
    builder.color()
    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    solver = newton.solvers.SolverVBD(model, iterations=5)
    picking = Picking(model, pick_stiffness=100.0, pick_damping=0.0, pick_max_acceleration=5.0)

    picking.pick(state_in, wp.vec3(0.025, 0.0, 1.0), wp.vec3(0.0, 0.0, -1.0))
    picked_body = int(picking.pick_body.numpy()[0])
    test.assertEqual(picked_body, bodies[num_links // 2])
    pick_state = picking.pick_state.numpy()
    anchor = np.array(pick_state[0]["picking_target_world"], dtype=np.float32)

    dt = 1.0 / 600.0
    peak_angular_speed = 0.0
    for step in range(300):
        target = anchor + np.array([0.0, 0.5 * min(1.0, step / 150.0), 0.0], dtype=np.float32)
        pick_state[0]["picking_target_world"] = target
        picking.pick_state.assign(pick_state)
        state_in.clear_forces()
        picking._apply_picking_force(state_in)
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in

        body_q = state_in.body_q.numpy()
        body_qd = state_in.body_qd.numpy()
        test.assertTrue(np.all(np.isfinite(body_q)))
        test.assertTrue(np.all(np.isfinite(body_qd)))
        peak_angular_speed = max(peak_angular_speed, float(np.max(np.linalg.norm(body_qd[:, 3:], axis=1))))

    test.assertLess(peak_angular_speed, 3.0)


# Device-parameterized tests
add_function_test(
    TestPickingSetup,
    "test_picking_setup_device",
    test_picking_setup_device,
    devices=get_test_devices(),
)
add_function_test(
    TestPickingSetup,
    "test_picking_torque_limit",
    test_picking_torque_limit,
    devices=get_test_devices(),
)
add_function_test(
    TestPickingSetup,
    "test_picking_torque_limit_is_noop_below_limit",
    test_picking_torque_limit_is_noop_below_limit,
    devices=get_test_devices(),
)
add_function_test(
    TestPickingSetup,
    "test_picking_torque_limit_rotates_with_inertia",
    test_picking_torque_limit_rotates_with_inertia,
    devices=get_test_devices(),
)
add_function_test(
    TestPickingSetup,
    "test_picking_torque_limit_cable",
    test_picking_torque_limit_cable,
    devices=get_test_devices(),
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
