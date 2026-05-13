# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Tests for body velocity stepping with non-zero center of mass offsets.

This module tests that when applying angular velocity to a body with a non-zero
center of mass (CoM) offset, the body rotates about its CoM, not about the body
frame origin. This is verified by checking that the CoM position stays stationary
when only angular velocity is applied.

For generalized coordinate solvers (MuJoCo, Featherstone), velocity is set via joint_qd.
For maximal coordinate solvers (XPBD, SemiImplicit), velocity is set via body_qd.

Note on tolerances:
- MuJoCo/Featherstone use body origin velocity internally, which introduces small
  numerical integration errors when converting back to CoM velocity (~1e-3 after 10 steps).
- Maximal coordinate solvers (XPBD, SemiImplicit) directly integrate CoM velocity,
  so they have much tighter numerical precision (~1e-6).
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.viewer.kernels import compute_com_positions
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestBodyVelocity(unittest.TestCase):
    pass


def _add_free_distance_joint(builder, joint_type, parent, child, parent_xform, child_xform):
    if joint_type == newton.JointType.FREE:
        return builder.add_joint_free(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
    if joint_type == newton.JointType.DISTANCE:
        return builder.add_joint_distance(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            min_distance=-1.0,
            max_distance=-1.0,
        )
    raise AssertionError(f"Unsupported joint type: {joint_type}")


def _joint_type_name(joint_type):
    if joint_type == newton.JointType.FREE:
        return "free"
    if joint_type == newton.JointType.DISTANCE:
        return "distance"
    raise AssertionError(f"Unsupported joint type: {joint_type}")


def _build_rotated_anchor_descendant_model(device, joint_type, parent_kinematic):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)
    base = builder.add_link(is_kinematic=parent_kinematic, mass=1.0)
    child = builder.add_link(mass=1.0)
    builder.add_shape_sphere(base, radius=0.1)
    builder.add_shape_sphere(child, radius=0.1)
    builder.body_com[child] = wp.vec3(0.25, 0.11, -0.17)

    root_parent_rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, -0.2, 1.0)), 0.55)
    if parent_kinematic:
        j0 = builder.add_joint_fixed(
            parent=-1,
            child=base,
            parent_xform=wp.transform(wp.vec3(0.2, -0.1, 0.3), root_parent_rot),
            child_xform=wp.transform_identity(),
        )
    else:
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=base,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.2, -0.1, 0.3), root_parent_rot),
            child_xform=wp.transform_identity(),
        )

    parent_xform = wp.transform(
        wp.vec3(0.7, -0.2, 0.4),
        wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.2, 1.0, -0.3)), 0.7),
    )
    child_xform = wp.transform(
        wp.vec3(0.15, -0.05, 0.2),
        wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, -0.2, 0.4)), -0.9),
    )
    j1 = _add_free_distance_joint(
        builder=builder,
        joint_type=joint_type,
        parent=base,
        child=child,
        parent_xform=parent_xform,
        child_xform=child_xform,
    )
    builder.add_articulation([j0, j1])
    return builder.finalize(device=device), base, child, j0, j1


def compute_com_world_position(body_q, body_com, body_world, world_offsets=None, body_index: int = 0) -> np.ndarray:
    """Compute the center of mass position in world frame."""
    com_world = wp.zeros(body_q.shape[0], dtype=wp.vec3, device=body_q.device)
    wp.launch(
        kernel=compute_com_positions,
        dim=body_q.shape[0],
        inputs=[body_q, body_com, body_world, world_offsets, None],
        outputs=[com_world],
        device=body_q.device,
    )
    return com_world.numpy()[body_index]


def test_angular_velocity_com_stationary(
    test: TestBodyVelocity,
    device,
    solver_fn,
    uses_generalized_coords: bool,
    com_offset: tuple[float, float, float],
    angular_velocity: tuple[float, float, float],
    tolerance: float,
):
    """Test that angular velocity causes rotation about CoM, not body origin.

    When a body has a non-zero CoM offset and we apply angular velocity with zero
    linear velocity (at the CoM), the CoM should stay stationary while the body
    rotates around it.

    Args:
        test: Test case instance
        device: Compute device
        solver_fn: Function that creates a solver given a model
        uses_generalized_coords: If True, set velocity via joint_qd; else via body_qd
        com_offset: Center of mass offset in body frame (x, y, z)
        angular_velocity: Angular velocity in world frame (wx, wy, wz)
        tolerance: Maximum allowed CoM drift
    """
    builder = newton.ModelBuilder(gravity=0.0)

    # Create a body with the specified CoM offset
    initial_pos = wp.vec3(1.0, 2.0, 3.0)
    b = builder.add_body(xform=wp.transform(initial_pos, wp.quat_identity()))
    builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
    builder.body_com[b] = wp.vec3(*com_offset)

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()

    # Compute initial FK
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Set angular velocity (linear velocity = 0 at CoM)
    # joint_qd for FREE joint: [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
    # body_qd: [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
    velocity = np.array([0.0, 0.0, 0.0, *angular_velocity], dtype=np.float32)

    if uses_generalized_coords:
        # MuJoCo, Featherstone: set joint_qd
        state_0.joint_qd.assign(velocity)
        # Also need to update body_qd via FK for the solver
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    else:
        # XPBD, SemiImplicit: set body_qd directly
        state_0.body_qd.assign(velocity.reshape(1, 6))

    # Get initial CoM position in world frame
    body_q_initial = state_0.body_q.numpy()[0].copy()
    com_initial = compute_com_world_position(state_0.body_q, model.body_com, model.body_world)

    # Step simulation
    sim_dt = 0.01
    num_steps = 10

    for _ in range(num_steps):
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0

    # Get final CoM position
    body_q_final = state_0.body_q.numpy()[0]
    com_final = compute_com_world_position(state_0.body_q, model.body_com, model.body_world)

    # CoM should stay stationary (within numerical tolerance)
    com_drift = np.linalg.norm(com_final - com_initial)
    test.assertLess(
        com_drift,
        tolerance,
        f"CoM drifted by {com_drift:.6f} (expected < {tolerance}). Initial CoM: {com_initial}, Final CoM: {com_final}",
    )

    # Verify that the body actually rotated (quaternion changed)
    quat_initial = body_q_initial[3:7]
    quat_final = body_q_final[3:7]
    quat_diff = np.abs(np.dot(quat_initial, quat_final))
    test.assertLess(
        quat_diff,
        0.9999,
        "Body should have rotated but quaternion barely changed",
    )


def test_linear_velocity_com_moves(
    test: TestBodyVelocity,
    device,
    solver_fn,
    uses_generalized_coords: bool,
    com_offset: tuple[float, float, float],
    linear_velocity: tuple[float, float, float],
    tolerance: float,
):
    """Test that linear velocity causes CoM to move as expected.

    When a body has a non-zero CoM offset and we apply linear velocity at the CoM
    with zero angular velocity, the CoM should translate at the specified velocity.

    Args:
        test: Test case instance
        device: Compute device
        solver_fn: Function that creates a solver given a model
        uses_generalized_coords: If True, set velocity via joint_qd; else via body_qd
        com_offset: Center of mass offset in body frame (x, y, z)
        linear_velocity: Linear velocity in world frame (vx, vy, vz)
        tolerance: Maximum allowed displacement error
    """
    builder = newton.ModelBuilder(gravity=0.0)

    initial_pos = wp.vec3(0.0, 0.0, 1.0)
    b = builder.add_body(xform=wp.transform(initial_pos, wp.quat_identity()))
    builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
    builder.body_com[b] = wp.vec3(*com_offset)

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Set linear velocity (angular velocity = 0)
    velocity = np.array([*linear_velocity, 0.0, 0.0, 0.0], dtype=np.float32)

    if uses_generalized_coords:
        state_0.joint_qd.assign(velocity)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    else:
        state_0.body_qd.assign(velocity.reshape(1, 6))

    # Get initial CoM position
    com_initial = compute_com_world_position(state_0.body_q, model.body_com, model.body_world)

    # Step simulation
    sim_dt = 0.01
    num_steps = 10
    total_time = sim_dt * num_steps

    for _ in range(num_steps):
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0

    # Get final CoM position
    com_final = compute_com_world_position(state_0.body_q, model.body_com, model.body_world)

    # Expected displacement = velocity * time
    expected_displacement = np.array(linear_velocity) * total_time
    actual_displacement = com_final - com_initial

    # Check that displacement matches expected
    displacement_error = np.linalg.norm(actual_displacement - expected_displacement)
    test.assertLess(
        displacement_error,
        tolerance,
        f"CoM displacement error: {displacement_error:.6f} (expected < {tolerance}). "
        f"Expected: {expected_displacement}, Actual: {actual_displacement}",
    )


def test_combined_velocity(
    test: TestBodyVelocity,
    device,
    solver_fn,
    uses_generalized_coords: bool,
    com_offset: tuple[float, float, float],
    tolerance: float,
):
    """Test combined linear and angular velocity with non-zero CoM offset.

    When both linear and angular velocities are applied, the CoM should translate
    at the linear velocity rate while the body rotates.

    Args:
        test: Test case instance
        device: Compute device
        solver_fn: Function that creates a solver given a model
        uses_generalized_coords: If True, set velocity via joint_qd; else via body_qd
        com_offset: Center of mass offset in body frame (x, y, z)
        tolerance: Maximum allowed displacement error
    """
    builder = newton.ModelBuilder(gravity=0.0)

    initial_pos = wp.vec3(0.0, 0.0, 1.0)
    b = builder.add_body(xform=wp.transform(initial_pos, wp.quat_identity()))
    builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
    builder.body_com[b] = wp.vec3(*com_offset)

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Set both linear and angular velocity
    linear_velocity = (0.1, 0.0, 0.0)
    angular_velocity = (0.0, 0.0, 1.0)
    velocity = np.array([*linear_velocity, *angular_velocity], dtype=np.float32)

    if uses_generalized_coords:
        state_0.joint_qd.assign(velocity)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    else:
        state_0.body_qd.assign(velocity.reshape(1, 6))

    # Get initial CoM position
    body_q_initial = state_0.body_q.numpy()[0].copy()
    com_initial = compute_com_world_position(state_0.body_q, model.body_com, model.body_world)

    # Step simulation
    sim_dt = 0.01
    num_steps = 10
    total_time = sim_dt * num_steps

    for _ in range(num_steps):
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0

    # Get final CoM position
    body_q_final = state_0.body_q.numpy()[0]
    com_final = compute_com_world_position(state_0.body_q, model.body_com, model.body_world)

    # Expected displacement = linear_velocity * time (rotation shouldn't affect CoM position)
    expected_displacement = np.array(linear_velocity) * total_time
    actual_displacement = com_final - com_initial

    # The CoM should have moved only due to linear velocity, not angular
    displacement_error = np.linalg.norm(actual_displacement - expected_displacement)
    test.assertLess(
        displacement_error,
        tolerance,
        f"CoM displacement error: {displacement_error:.6f} (expected < {tolerance}). "
        f"Expected: {expected_displacement}, Actual: {actual_displacement}",
    )

    # Verify body rotated
    quat_initial = body_q_initial[3:7]
    quat_final = body_q_final[3:7]
    quat_diff = np.abs(np.dot(quat_initial, quat_final))
    test.assertLess(quat_diff, 0.9999, "Body should have rotated")


def test_root_free_joint_under_rotated_parent_xform_uses_parent_frame_qd(
    test: TestBodyVelocity,
    device,
    solver_fn,
):
    """Root FREE joint with a rotated ``parent_xform`` must report ``joint_qd``
    in the parent joint frame and ``body_qd`` in world frame at the COM
    (regression for #2704 — the MuJoCo bridge previously wrote both in world
    frame).
    """
    builder = newton.ModelBuilder(gravity=-10.0, up_axis=newton.Axis.Z)
    parent_xform = wp.transform(wp.vec3(0.5, 0.6, 0.7), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1))
    joint = builder.add_joint_free(parent=-1, child=body, parent_xform=parent_xform)
    builder.add_articulation([joint])

    model = builder.finalize(device=device)
    solver = solver_fn(model)
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    dt = 1e-2
    solver.step(state_0, state_1, model.control(), model.contacts(), dt)

    # World gravity along -Z rotated into the parent frame R(x, 90°) is along -Y.
    np.testing.assert_allclose(state_1.joint_qd.numpy()[0:3], (0.0, -10.0 * dt, 0.0), atol=1e-5)
    np.testing.assert_allclose(state_1.joint_qd.numpy()[3:6], (0.0, 0.0, 0.0), atol=1e-5)
    # body_qd stays world-frame at the COM regardless of joint anchor rotations.
    np.testing.assert_allclose(state_1.body_qd.numpy()[body, 0:3], (0.0, 0.0, -10.0 * dt), atol=1e-5)
    np.testing.assert_allclose(state_1.body_qd.numpy()[body, 3:6], (0.0, 0.0, 0.0), atol=1e-5)


def test_featherstone_free_descendant_joint_qd_round_trip_under_rotated_parent(
    test: TestBodyVelocity,
    device,
):
    """Featherstone should preserve descendant FREE joint_qd in parent-frame coordinates."""
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)
    parent = builder.add_link(mass=1.0)
    child = builder.add_link(mass=1.0)
    builder.body_com[child] = wp.vec3(0.2, 0.0, 0.0)
    builder.add_shape_sphere(parent, radius=0.1)
    builder.add_shape_sphere(child, radius=0.1)

    j0 = builder.add_joint_revolute(parent=-1, child=parent, axis=newton.Axis.Z)
    j1 = builder.add_joint_free(
        parent=parent,
        child=child,
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j0, j1])

    model = builder.finalize(device=device)
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    state_0 = model.state()
    state_1 = model.state()

    q = state_0.joint_q.numpy()
    qd = state_0.joint_qd.numpy()
    q[:] = 0.0
    q[0] = np.pi / 2.0
    q[4:8] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    qd[:] = 0.0
    qd[1:7] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    state_0.joint_q.assign(q)
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    solver.step(state_0, state_1, model.control(), None, 0.01)

    np.testing.assert_allclose(state_1.joint_qd.numpy()[1:7], qd[1:7], atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(
        state_1.body_qd.numpy()[child],
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        atol=1.0e-5,
        rtol=1.0e-6,
    )


def test_featherstone_free_distance_descendant_angular_velocity_keeps_com_stationary_with_rotated_anchors(
    test: TestBodyVelocity,
    device,
    joint_type,
):
    """A rotated-anchor FREE/DISTANCE child with pure angular velocity should rotate about its COM."""
    model, _base, child, _j0, j1 = _build_rotated_anchor_descendant_model(
        device=device,
        joint_type=joint_type,
        parent_kinematic=True,
    )
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    state_0 = model.state()
    state_1 = model.state()

    q = model.joint_q.numpy().copy()
    qd = model.joint_qd.numpy().copy()
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()

    q[q_start[j1] : q_start[j1] + 3] = np.array([0.4, -0.25, 0.3], dtype=np.float32)
    q_child_rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.5, -0.2)), 0.35)
    q[q_start[j1] + 3 : q_start[j1] + 7] = np.array(
        [q_child_rot[0], q_child_rot[1], q_child_rot[2], q_child_rot[3]],
        dtype=np.float32,
    )
    qd[qd_start[j1] : qd_start[j1] + 6] = np.array([0.0, 0.0, 0.0, 0.3, -0.4, 0.5], dtype=np.float32)

    state_0.joint_q.assign(q)
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    com_initial = compute_com_world_position(state_0.body_q, model.body_com, model.body_world, body_index=child)

    for _ in range(10):
        solver.step(state_0, state_1, None, None, 0.01)
        state_0, state_1 = state_1, state_0

    com_final = compute_com_world_position(state_0.body_q, model.body_com, model.body_world, body_index=child)
    com_drift = np.linalg.norm(com_final - com_initial)
    test.assertLess(
        com_drift,
        2.0e-4,
        f"{_joint_type_name(joint_type)} child COM drifted under pure angular velocity: {com_drift}",
    )

    body_qd = state_0.body_qd.numpy()[child]
    np.testing.assert_allclose(body_qd[:3], np.zeros(3, dtype=np.float32), atol=2.0e-4, rtol=1.0e-6)
    np.testing.assert_allclose(
        state_0.joint_qd.numpy()[qd_start[j1] : qd_start[j1] + 6],
        qd[qd_start[j1] : qd_start[j1] + 6],
        atol=2.0e-4,
        rtol=1.0e-6,
    )


def test_featherstone_free_distance_descendant_stays_inertial_under_parent_torque(
    test: TestBodyVelocity,
    device,
    joint_type,
):
    """A FREE/DISTANCE descendant should stay inertial in world space while its parent accelerates."""
    model, base, child, j0, j1 = _build_rotated_anchor_descendant_model(
        device=device,
        joint_type=joint_type,
        parent_kinematic=False,
    )
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    q = model.joint_q.numpy().copy()
    qd = model.joint_qd.numpy().copy()
    joint_f = control.joint_f.numpy().copy()
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()

    q[q_start[j1] : q_start[j1] + 3] = np.array([0.4, -0.25, 0.3], dtype=np.float32)
    q_child_rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.5, -0.2)), 0.35)
    q[q_start[j1] + 3 : q_start[j1] + 7] = np.array(
        [q_child_rot[0], q_child_rot[1], q_child_rot[2], q_child_rot[3]],
        dtype=np.float32,
    )
    qd[:] = 0.0
    joint_f[:] = 0.0
    joint_f[qd_start[j0]] = 7.5

    state_0.joint_q.assign(q)
    state_0.joint_qd.assign(qd)
    control.joint_f.assign(joint_f)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    child_q_initial = state_0.body_q.numpy()[child].copy()

    solver.step(state_0, state_1, control, None, 0.01)

    base_qd = state_1.body_qd.numpy()[base]
    test.assertGreater(np.linalg.norm(base_qd[3:]), 1.0e-2, "Parent torque did not drive the base as intended")

    child_qd = state_1.body_qd.numpy()[child]
    np.testing.assert_allclose(child_qd, np.zeros(6, dtype=np.float32), atol=3.0e-4, rtol=1.0e-6)

    child_q_final = state_1.body_q.numpy()[child]
    np.testing.assert_allclose(child_q_final[:3], child_q_initial[:3], atol=1.0e-5, rtol=1.0e-6)

    quat_dot = abs(np.dot(child_q_initial[3:7], child_q_final[3:7]))
    test.assertGreater(quat_dot, 1.0 - 1.0e-5, f"{_joint_type_name(joint_type)} child orientation drifted in world")

    child_joint_qd = state_1.joint_qd.numpy()[qd_start[j1] : qd_start[j1] + 6]
    test.assertGreater(
        np.linalg.norm(child_joint_qd),
        1.0e-1,
        "Descendant joint state did not pick up the compensating relative motion",
    )


def test_featherstone_free_distance_descendant_matches_ping_pong_when_stepping_in_place(
    test: TestBodyVelocity,
    device,
    joint_type,
):
    """In-place stepping should match ping-pong stepping for descendant FREE/DISTANCE motion."""
    model, _base, child, j0, j1 = _build_rotated_anchor_descendant_model(
        device=device,
        joint_type=joint_type,
        parent_kinematic=False,
    )
    solver_ping_pong = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    solver_in_place = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    control_ping_pong = model.control()
    control_in_place = model.control()

    def _initialize_state(state, control):
        q = model.joint_q.numpy().copy()
        qd = model.joint_qd.numpy().copy()
        joint_f = control.joint_f.numpy().copy()
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()

        q[q_start[j1] : q_start[j1] + 3] = np.array([0.4, -0.25, 0.3], dtype=np.float32)
        q_child_rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.5, -0.2)), 0.35)
        q[q_start[j1] + 3 : q_start[j1] + 7] = np.array(
            [q_child_rot[0], q_child_rot[1], q_child_rot[2], q_child_rot[3]],
            dtype=np.float32,
        )
        qd[:] = 0.0
        joint_f[:] = 0.0
        joint_f[qd_start[j0]] = 7.5

        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        control.joint_f.assign(joint_f)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    state_pp_0 = model.state()
    state_pp_1 = model.state()
    state_in_place = model.state()
    _initialize_state(state_pp_0, control_ping_pong)
    _initialize_state(state_in_place, control_in_place)

    steps = 20
    for _ in range(steps):
        solver_ping_pong.step(state_pp_0, state_pp_1, control_ping_pong, None, 0.01)
        state_pp_0, state_pp_1 = state_pp_1, state_pp_0
        solver_in_place.step(state_in_place, state_in_place, control_in_place, None, 0.01)

    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    np.testing.assert_allclose(
        state_in_place.body_q.numpy()[child],
        state_pp_0.body_q.numpy()[child],
        atol=5.0e-5,
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        state_in_place.joint_q.numpy()[q_start[j1] : q_start[j1] + 7],
        state_pp_0.joint_q.numpy()[q_start[j1] : q_start[j1] + 7],
        atol=5.0e-5,
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        state_in_place.joint_qd.numpy()[qd_start[j1] : qd_start[j1] + 6],
        state_pp_0.joint_qd.numpy()[qd_start[j1] : qd_start[j1] + 6],
        atol=5.0e-4,
        rtol=1.0e-6,
    )


def test_featherstone_free_distance_descendant_correction_path_refreshes_stale_body_pose(
    test: TestBodyVelocity,
    device,
    joint_type,
):
    """The descendant FREE/DISTANCE correction path should ignore stale body poses."""
    model, _base, child, j0, j1 = _build_rotated_anchor_descendant_model(
        device=device,
        joint_type=joint_type,
        parent_kinematic=False,
    )
    solver_fresh = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    solver_stale = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    control_fresh = model.control()
    control_stale = model.control()

    def _initialize(state, control, refresh_fk):
        q = model.joint_q.numpy().copy()
        qd = model.joint_qd.numpy().copy()
        joint_f = control.joint_f.numpy().copy()
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()

        q[q_start[j1] : q_start[j1] + 3] = np.array([0.4, -0.25, 0.3], dtype=np.float32)
        q_child_rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.5, -0.2)), 0.35)
        q[q_start[j1] + 3 : q_start[j1] + 7] = np.array(
            [q_child_rot[0], q_child_rot[1], q_child_rot[2], q_child_rot[3]],
            dtype=np.float32,
        )
        qd[:] = 0.0
        joint_f[:] = 0.0
        joint_f[qd_start[j0]] = 7.5

        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        control.joint_f.assign(joint_f)
        if refresh_fk:
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        else:
            stale_body_q = np.full_like(state.body_q.numpy(), 123.0, dtype=np.float32)
            stale_body_qd = np.full_like(state.body_qd.numpy(), -321.0, dtype=np.float32)
            state.body_q.assign(stale_body_q)
            state.body_qd.assign(stale_body_qd)

    state_fresh_0 = model.state()
    state_fresh_1 = model.state()
    state_stale_0 = model.state()
    state_stale_1 = model.state()
    _initialize(state_fresh_0, control_fresh, refresh_fk=True)
    _initialize(state_stale_0, control_stale, refresh_fk=False)

    solver_fresh.step(state_fresh_0, state_fresh_1, control_fresh, None, 0.01)
    solver_stale.step(state_stale_0, state_stale_1, control_stale, None, 0.01)

    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    np.testing.assert_allclose(
        state_stale_1.body_q.numpy()[child],
        state_fresh_1.body_q.numpy()[child],
        atol=5.0e-5,
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        state_stale_1.body_qd.numpy()[child],
        state_fresh_1.body_qd.numpy()[child],
        atol=5.0e-4,
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        state_stale_1.joint_q.numpy()[q_start[j1] : q_start[j1] + 7],
        state_fresh_1.joint_q.numpy()[q_start[j1] : q_start[j1] + 7],
        atol=5.0e-5,
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        state_stale_1.joint_qd.numpy()[qd_start[j1] : qd_start[j1] + 6],
        state_fresh_1.joint_qd.numpy()[qd_start[j1] : qd_start[j1] + 6],
        atol=5.0e-4,
        rtol=1.0e-6,
    )


devices = get_test_devices()

solvers = {
    "featherstone": (
        lambda model: newton.solvers.SolverFeatherstone(model, angular_damping=0.0),
        True,
        1e-3,  # Internal free-joint speeds differ, but the public boundary is COM-based.
    ),
    "mujoco_cpu": (
        lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True),
        True,
        1e-3,  # Higher tolerance due to body origin velocity integration
    ),
    "mujoco_warp": (
        lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True),
        True,
        1e-3,  # Higher tolerance due to body origin velocity integration
    ),
    "xpbd": (
        lambda model: newton.solvers.SolverXPBD(model, angular_damping=0.0),
        False,
        1e-4,  # Tighter tolerance - directly integrates CoM velocity
    ),
    "semi_implicit": (
        lambda model: newton.solvers.SolverSemiImplicit(model, angular_damping=0.0),
        False,
        1e-4,  # Tighter tolerance - directly integrates CoM velocity
    ),
}

# Test configurations: different CoM offsets and velocity directions
com_offsets = [
    (0.5, 0.0, 0.0),  # X offset
    (0.0, 0.3, 0.0),  # Y offset
    (0.0, 0.0, 0.4),  # Z offset
    (0.2, 0.3, 0.1),  # Combined offset
]

angular_velocities = [
    (0.0, 0.0, 1.0),  # Z rotation
    (0.0, 1.0, 0.0),  # Y rotation
    (1.0, 0.0, 0.0),  # X rotation
]

linear_velocities = [
    (0.7, 0.0, 0.0),  # X translation
    (0.0, 0.7, 0.0),  # Y translation
    (0.0, 0.0, 0.7),  # Z translation
]

for device in devices:
    for solver_name, (solver_fn, uses_gen_coords, tolerance) in solvers.items():
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue

        # Test angular velocity with various CoM offsets
        for i, com_offset in enumerate(com_offsets):
            for j, angular_vel in enumerate(angular_velocities):
                add_function_test(
                    TestBodyVelocity,
                    f"test_angular_com_stationary_{solver_name}_com{i}_ang{j}",
                    test_angular_velocity_com_stationary,
                    devices=[device],
                    solver_fn=solver_fn,
                    uses_generalized_coords=uses_gen_coords,
                    com_offset=com_offset,
                    angular_velocity=angular_vel,
                    tolerance=tolerance,
                )

        # Test linear velocity with various CoM offsets
        for i, com_offset in enumerate(com_offsets):
            for j, linear_vel in enumerate(linear_velocities):
                add_function_test(
                    TestBodyVelocity,
                    f"test_linear_com_moves_{solver_name}_com{i}_lin{j}",
                    test_linear_velocity_com_moves,
                    devices=[device],
                    solver_fn=solver_fn,
                    uses_generalized_coords=uses_gen_coords,
                    com_offset=com_offset,
                    linear_velocity=linear_vel,
                    tolerance=tolerance,
                )

        # Test combined velocity with various CoM offsets
        for i, com_offset in enumerate(com_offsets):
            add_function_test(
                TestBodyVelocity,
                f"test_combined_velocity_{solver_name}_com{i}",
                test_combined_velocity,
                devices=[device],
                solver_fn=solver_fn,
                uses_generalized_coords=uses_gen_coords,
                com_offset=com_offset,
                tolerance=tolerance,
            )

    add_function_test(
        TestBodyVelocity,
        "test_featherstone_free_descendant_joint_qd_round_trip_under_rotated_parent",
        test_featherstone_free_descendant_joint_qd_round_trip_under_rotated_parent,
        devices=[device],
    )
    for solver_name in ("featherstone", "mujoco_cpu", "mujoco_warp"):
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue
        add_function_test(
            TestBodyVelocity,
            f"test_root_free_joint_under_rotated_parent_xform_uses_parent_frame_qd_{solver_name}",
            test_root_free_joint_under_rotated_parent_xform_uses_parent_frame_qd,
            devices=[device],
            solver_fn=solvers[solver_name][0],
        )
    for joint_type in (newton.JointType.FREE, newton.JointType.DISTANCE):
        joint_name = _joint_type_name(joint_type)
        add_function_test(
            TestBodyVelocity,
            f"test_featherstone_{joint_name}_descendant_angular_velocity_keeps_com_stationary_with_rotated_anchors",
            test_featherstone_free_distance_descendant_angular_velocity_keeps_com_stationary_with_rotated_anchors,
            devices=[device],
            joint_type=joint_type,
        )
        add_function_test(
            TestBodyVelocity,
            f"test_featherstone_{joint_name}_descendant_stays_inertial_under_parent_torque",
            test_featherstone_free_distance_descendant_stays_inertial_under_parent_torque,
            devices=[device],
            joint_type=joint_type,
        )
        add_function_test(
            TestBodyVelocity,
            f"test_featherstone_{joint_name}_descendant_matches_ping_pong_when_stepping_in_place",
            test_featherstone_free_distance_descendant_matches_ping_pong_when_stepping_in_place,
            devices=[device],
            joint_type=joint_type,
        )
        add_function_test(
            TestBodyVelocity,
            f"test_featherstone_{joint_name}_descendant_correction_path_refreshes_stale_body_pose",
            test_featherstone_free_distance_descendant_correction_path_refreshes_stale_body_pose,
            devices=[device],
            joint_type=joint_type,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
