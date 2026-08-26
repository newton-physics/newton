# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the operational-space controller family.

Math kernels are tested standalone first, independent of any Controller class,
following the pattern in ``test_jacobian_mass_matrix.py``. Controller-level
tests are added once the surrounding ``Controller`` classes exist.

Kernel launches are written out directly in each test rather than behind a
shared helper whenever the launch itself has real per-test configuration
(index arrays, gains, ...) worth seeing. A launch is only factored out when
it's a single kernel with no derived arguments (e.g. ``_pose_error``), so the
helper hides nothing beyond boilerplate.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.controllers.impl.operational_space._common import (
    _apply_mass_matrix_inv_on_right_kernel,
    _apply_spatial_matrix_kernel,
    _closed_loop_wrench_command_kernel,
    _invert_spd_block_kernel,
    _jacobian_times_jacobian_transpose_kernel,
    _jacobian_transpose_force_kernel,
    _null_space_projector_kernel,
    _operational_space_mass_matrix_inverse_kernel,
    _pose_error_kernel,
    _rotate_selection_matrix_kernel,
    _shift_jacobian_to_tool_kernel,
    _task_matrix_times_jacobian_kernel,
    _task_space_pd_kernel,
    _tool_pose_and_twist_kernel,
)
from newton._src.controllers.impl.operational_space.model_free import ControllerOperationalSpaceModelFree
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()


def _build_two_link_arm_with_tool_site(device):
    """Two-revolute-joint planar arm with a tool site offset from the tip body's COM.

    Returns:
        Tuple of (model, state, tool_body, coordinate_change_body_from_tool).
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)

    b1 = builder.add_link(mass=1.3)
    b2 = builder.add_link(mass=0.9)
    builder.add_shape_box(b1, hx=0.2, hy=0.1, hz=0.1)
    builder.add_shape_box(b2, hx=0.15, hy=0.1, hz=0.08)
    builder.body_com[b1] = wp.vec3(0.5, 0.0, 0.0)
    builder.body_com[b2] = wp.vec3(0.4, 0.0, 0.0)

    j1 = builder.add_joint_revolute(
        parent=-1,
        child=b1,
        axis=newton.Axis.Z,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    j2 = builder.add_joint_revolute(
        parent=b1,
        child=b2,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j1, j2], label="arm")

    coordinate_change_body_from_tool = wp.transform(wp.vec3(0.3, 0.05, -0.1), wp.quat_identity())
    builder.add_site(b2, xform=coordinate_change_body_from_tool, label="tool_site")

    model = builder.finalize(device=device)
    state = model.state()
    return model, state, b2, coordinate_change_body_from_tool


def _build_six_dof_arm_with_tool_site(device):
    """Six-revolute-joint spatial arm (alternating Z/Y axes) with a tool site at the tip.

    Six independent, non-parallel joint axes give a generically full-rank 6x6
    Jacobian, which the operational-space mass matrix Lambda needs to be
    invertible — a planar 2-DOF arm's Jacobian can't span all 6 task dims.

    Returns:
        Tuple of (model, state, tool_body, coordinate_change_body_from_tool).
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)

    axes = [newton.Axis.Z, newton.Axis.Y, newton.Axis.Z, newton.Axis.Y, newton.Axis.Z, newton.Axis.Y]
    joints = []
    parent_body = -1
    for link_idx, axis in enumerate(axes):
        body = builder.add_link(mass=1.0 + 0.1 * link_idx)
        builder.add_shape_box(body, hx=0.1, hy=0.08, hz=0.06)
        builder.body_com[body] = wp.vec3(0.15, 0.02, -0.01)
        parent_xform = wp.transform_identity() if parent_body == -1 else wp.transform(wp.vec3(0.3, 0.0, 0.0))
        joints.append(
            builder.add_joint_revolute(
                parent=parent_body,
                child=body,
                axis=axis,
                parent_xform=parent_xform,
                child_xform=wp.transform_identity(),
            )
        )
        parent_body = body
    tool_body = parent_body
    builder.add_articulation(joints, label="arm")

    coordinate_change_body_from_tool = wp.transform(wp.vec3(0.2, 0.0, 0.05), wp.quat_identity())
    builder.add_site(tool_body, xform=coordinate_change_body_from_tool, label="tool_site")

    model = builder.finalize(device=device)
    state = model.state()
    return model, state, tool_body, coordinate_change_body_from_tool


def _build_seven_dof_arm_with_tool_site(device):
    """Seven-revolute-joint spatial arm (alternating Z/Y axes) with a tool site at the tip.

    One more DOF than the 6D task, i.e. a redundant manipulator — needed for
    the null-space projector to have a nontrivial (nonzero) null space to
    project onto. The 6-DOF arm above can't be used for this: with exactly
    6 DOF, J is square and (generically) invertible, so both pseudo-inverse
    variants degenerate to the exact inverse and the projector is always
    zero, which wouldn't distinguish them.

    Returns:
        Tuple of (model, state, tool_body, coordinate_change_body_from_tool).
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)

    axes = [
        newton.Axis.Z,
        newton.Axis.Y,
        newton.Axis.Z,
        newton.Axis.Y,
        newton.Axis.Z,
        newton.Axis.Y,
        newton.Axis.Z,
    ]
    joints = []
    parent_body = -1
    for link_idx, axis in enumerate(axes):
        body = builder.add_link(mass=1.0 + 0.1 * link_idx)
        builder.add_shape_box(body, hx=0.1, hy=0.08, hz=0.06)
        builder.body_com[body] = wp.vec3(0.15, 0.02, -0.01)
        parent_xform = wp.transform_identity() if parent_body == -1 else wp.transform(wp.vec3(0.3, 0.0, 0.0))
        joints.append(
            builder.add_joint_revolute(
                parent=parent_body,
                child=body,
                axis=axis,
                parent_xform=parent_xform,
                child_xform=wp.transform_identity(),
            )
        )
        parent_body = body
    tool_body = parent_body
    builder.add_articulation(joints, label="arm")

    coordinate_change_body_from_tool = wp.transform(wp.vec3(0.2, 0.0, 0.05), wp.quat_identity())
    builder.add_site(tool_body, xform=coordinate_change_body_from_tool, label="tool_site")

    model = builder.finalize(device=device)
    state = model.state()
    return model, state, tool_body, coordinate_change_body_from_tool


def test_invert_spd_block_matches_numpy_inverse(test, device):
    """The Cholesky-based inverse kernel matches numpy's inverse, for two differently-sized SPD blocks.

    Heterogeneous block sizes exercise the same per-robot padding this
    kernel will see in practice, where each robot's controlled-DOF count
    (or the fixed 6-dim task, for Lambda) differs.
    """
    rng = np.random.default_rng(seed=0)
    block_sizes = [3, 2]
    max_dim = 4

    # Build two random SPD matrices of different sizes, embedded in a shared padded buffer.
    spd_matrix_np = np.zeros((2, max_dim, max_dim), dtype=np.float32)
    expected_inv_np = np.zeros((2, max_dim, max_dim), dtype=np.float32)
    for block_idx, n in enumerate(block_sizes):
        random_matrix = rng.standard_normal((n, n)).astype(np.float32)
        spd_matrix_np[block_idx, :n, :n] = random_matrix @ random_matrix.T + n * np.eye(n, dtype=np.float32)
        expected_inv_np[block_idx, :n, :n] = np.linalg.inv(spd_matrix_np[block_idx, :n, :n])

    # Preallocate scratch and outputs, then launch the kernel under test.
    spd_matrix = wp.array(spd_matrix_np, dtype=float, device=device)
    block_dim = wp.array(block_sizes, dtype=wp.int32, device=device)
    cholesky_factor = wp.zeros((2, max_dim, max_dim), dtype=float, device=device)
    spd_matrix_inv = wp.zeros((2, max_dim, max_dim), dtype=float, device=device)
    wp.launch(
        _invert_spd_block_kernel,
        dim=2,
        inputs=[spd_matrix, block_dim, cholesky_factor],
        outputs=[spd_matrix_inv],
        device=device,
    )

    # Compare: only the top-left n x n submatrix of each block is meaningful.
    for block_idx, n in enumerate(block_sizes):
        np.testing.assert_allclose(
            spd_matrix_inv.numpy()[block_idx, :n, :n], expected_inv_np[block_idx, :n, :n], atol=1e-4
        )


def test_operational_space_mass_matrix_matches_numpy(test, device):
    """Lambda = (J M^-1 J^T)^-1, computed via the kernels, matches a direct numpy computation.

    Also checks that Lambda comes out symmetric and positive-definite, like
    any valid mass matrix should.
    """
    model, state, tool_body, coordinate_change_body_from_tool = _build_six_dof_arm_with_tool_site(device)
    device = model.device

    # Move to a non-identity configuration and compute ground-truth dynamics
    # quantities: the joint-space mass matrix and the COM-referenced Jacobian.
    state.joint_q.assign([0.3, -0.4, 0.6, 0.2, -0.5, 0.35])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    mass_matrix = newton.eval_mass_matrix(model, state)
    jacobian_com_world = newton.eval_jacobian(model, state)

    # Shift the Jacobian to the tool point.
    max_dofs = model.max_dofs_per_articulation
    jacobian_tool_world = wp.zeros((1, 6, max_dofs), dtype=float, device=device)
    wp.launch(
        _shift_jacobian_to_tool_kernel,
        dim=(1, max_dofs),
        inputs=[
            jacobian_com_world,
            state.body_q,
            model.body_com,
            wp.array([tool_body], dtype=wp.int32, device=device),
            wp.array([coordinate_change_body_from_tool], dtype=wp.transform, device=device),
            wp.array([0], dtype=wp.int32, device=device),  # robot_articulation: one robot, articulation 0
            wp.array([5], dtype=wp.int32, device=device),  # robot_link_idx: tool_body is link 5 (the 6th joint's child)
        ],
        outputs=[jacobian_tool_world],
        device=device,
    )

    # Invert the joint-space mass matrix.
    dof_count = wp.array([max_dofs], dtype=wp.int32, device=device)
    mass_matrix_cholesky = wp.zeros((1, max_dofs, max_dofs), dtype=float, device=device)
    mass_matrix_inv = wp.zeros((1, max_dofs, max_dofs), dtype=float, device=device)
    wp.launch(
        _invert_spd_block_kernel,
        dim=1,
        inputs=[mass_matrix, dof_count, mass_matrix_cholesky],
        outputs=[mass_matrix_inv],
        device=device,
    )

    # Lambda^-1 = J M^-1 J^T.
    operational_space_mass_matrix_inv = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _operational_space_mass_matrix_inverse_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian_tool_world, mass_matrix_inv, dof_count],
        outputs=[operational_space_mass_matrix_inv],
        device=device,
    )

    # Lambda: invert that 6x6 result to get the operational-space mass matrix itself.
    task_dim = wp.array([6], dtype=wp.int32, device=device)
    operational_space_mass_matrix_cholesky = wp.zeros((1, 6, 6), dtype=float, device=device)
    operational_space_mass_matrix = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _invert_spd_block_kernel,
        dim=1,
        inputs=[operational_space_mass_matrix_inv, task_dim, operational_space_mass_matrix_cholesky],
        outputs=[operational_space_mass_matrix],
        device=device,
    )
    lambda_result = operational_space_mass_matrix.numpy()[0]

    # Expected: the same formula, computed directly with numpy.
    mass_matrix_np = mass_matrix.numpy()[0]
    jacobian_tool_np = jacobian_tool_world.numpy()[0]
    expected_lambda_inv = jacobian_tool_np @ np.linalg.inv(mass_matrix_np) @ jacobian_tool_np.T
    expected_lambda = np.linalg.inv(expected_lambda_inv)

    np.testing.assert_allclose(lambda_result, expected_lambda, atol=1e-3)
    np.testing.assert_allclose(lambda_result, lambda_result.T, atol=1e-5)  # symmetric
    test.assertGreater(np.linalg.eigvalsh(lambda_result).min(), 0.0)  # positive-definite


def test_tool_pose_matches_body_and_site_composition(test, device):
    """The tool pose is exactly body_q[tool_body] * coordinate_change_body_from_tool."""
    model, state, tool_body, coordinate_change_body_from_tool = _build_two_link_arm_with_tool_site(device)
    device = model.device

    # Move the arm to a non-identity configuration and run ground-truth FK.
    state.joint_q.assign([0.4, -0.9])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Preallocate outputs and launch the kernel under test.
    tool_pose_world = wp.zeros(1, dtype=wp.transform, device=device)
    tool_twist_world = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(
        _tool_pose_and_twist_kernel,
        dim=1,
        inputs=[
            state.body_q,
            state.body_qd,
            model.body_com,
            wp.array([tool_body], dtype=wp.int32, device=device),
            wp.array([coordinate_change_body_from_tool], dtype=wp.transform, device=device),
        ],
        outputs=[tool_pose_world, tool_twist_world],
        device=device,
    )

    # Expected: compose the body's FK pose with the tool site's fixed offset by hand.
    coordinate_change_world_from_body = wp.transform(*state.body_q.numpy()[tool_body])
    expected = coordinate_change_world_from_body * coordinate_change_body_from_tool

    np.testing.assert_allclose(tool_pose_world.numpy()[0], np.array(expected), atol=1e-6)


def test_jacobian_tool_shift_matches_twist(test, device):
    """jacobian_tool_world @ joint_qd must reproduce the independently computed tool twist.

    This is the core internal-consistency check: the twist the Jacobian-shift
    kernel predicts from joint velocities must agree with the twist the
    pose/twist kernel computes directly from state.body_qd, away from the
    identity configuration where a sign or axis-order bug could hide.
    """
    model, state, tool_body, coordinate_change_body_from_tool = _build_two_link_arm_with_tool_site(device)
    device = model.device

    # Move the arm to a non-identity configuration with nonzero joint velocity,
    # then run ground-truth FK and the Jacobian these kernels shift to the tool.
    joint_q = np.array([0.6, 1.1])
    joint_qd = np.array([-0.4, 0.85])
    state.joint_q.assign(joint_q)
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    jacobian_com_world = newton.eval_jacobian(model, state)

    # Index arrays shared by both kernel launches below: one robot, whose tool
    # is the child body of the articulation's 2nd (i.e. last) joint.
    tool_body_arr = wp.array([tool_body], dtype=wp.int32, device=device)
    coordinate_change_body_from_tool_arr = wp.array(
        [coordinate_change_body_from_tool], dtype=wp.transform, device=device
    )

    # Ground truth: the tool twist computed directly from state.body_qd.
    tool_pose_world = wp.zeros(1, dtype=wp.transform, device=device)
    tool_twist_world = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(
        _tool_pose_and_twist_kernel,
        dim=1,
        inputs=[state.body_q, state.body_qd, model.body_com, tool_body_arr, coordinate_change_body_from_tool_arr],
        outputs=[tool_pose_world, tool_twist_world],
        device=device,
    )

    # Under test: the Jacobian shifted to the tool point.
    max_dofs = model.max_dofs_per_articulation
    jacobian_tool_world = wp.zeros((1, 6, max_dofs), dtype=float, device=device)
    wp.launch(
        _shift_jacobian_to_tool_kernel,
        dim=(1, max_dofs),
        inputs=[
            jacobian_com_world,
            state.body_q,
            model.body_com,
            tool_body_arr,
            coordinate_change_body_from_tool_arr,
            wp.array([0], dtype=wp.int32, device=device),  # robot_articulation: one robot, articulation 0
            wp.array([1], dtype=wp.int32, device=device),  # robot_link_idx: tool_body is link 1 (the 2nd joint's child)
        ],
        outputs=[jacobian_tool_world],
        device=device,
    )

    # Compare: jacobian_tool_world @ joint_qd should reproduce the ground-truth twist.
    predicted_twist = jacobian_tool_world.numpy()[0] @ joint_qd
    np.testing.assert_allclose(predicted_twist, tool_twist_world.numpy()[0], atol=1e-6)


def test_jacobian_tool_shift_matches_finite_difference(test, device):
    """The tool point's world position, finite-differenced over time, matches jacobian_tool_world's linear rows."""
    model, state, tool_body, coordinate_change_body_from_tool = _build_two_link_arm_with_tool_site(device)
    device = model.device

    q0 = np.array([0.35, -0.5])
    qd = np.array([0.5, -0.9])

    def tool_position_world(joint_q):
        state.joint_q.assign(joint_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        coordinate_change_world_from_body = wp.transform(*state.body_q.numpy()[tool_body])
        return np.array(
            wp.transform_get_translation(coordinate_change_world_from_body * coordinate_change_body_from_tool)
        )

    # Ground truth: finite-difference the tool's world position across a small
    # step along qd, independent of the Jacobian machinery entirely.
    # float32 body_q makes a too-small dt amplify rounding noise once divided
    # back out below, so dt is chosen well above float32 ULP at this magnitude.
    dt = 1e-3
    finite_diff_velocity = (tool_position_world(q0 + dt * qd) - tool_position_world(q0 - dt * qd)) / (2 * dt)

    # Run ground-truth FK and the Jacobian at the midpoint configuration, then
    # preallocate the output and launch the kernel under test.
    state.joint_q.assign(q0)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    jacobian_com_world = newton.eval_jacobian(model, state)

    max_dofs = model.max_dofs_per_articulation
    jacobian_tool_world = wp.zeros((1, 6, max_dofs), dtype=float, device=device)
    wp.launch(
        _shift_jacobian_to_tool_kernel,
        dim=(1, max_dofs),
        inputs=[
            jacobian_com_world,
            state.body_q,
            model.body_com,
            wp.array([tool_body], dtype=wp.int32, device=device),
            wp.array([coordinate_change_body_from_tool], dtype=wp.transform, device=device),
            wp.array([0], dtype=wp.int32, device=device),  # robot_articulation: one robot, articulation 0
            wp.array([1], dtype=wp.int32, device=device),  # robot_link_idx: tool_body is link 1 (the 2nd joint's child)
        ],
        outputs=[jacobian_tool_world],
        device=device,
    )

    # Compare: the Jacobian's predicted linear velocity should match the finite difference.
    predicted_velocity = (jacobian_tool_world.numpy()[0] @ qd)[:3]
    np.testing.assert_allclose(predicted_velocity, finite_diff_velocity, atol=1e-3)


def test_tool_twist_angular_part_matches_body(test, device):
    """The tool frame's angular velocity is the body's angular velocity, unshifted."""
    model, state, tool_body, coordinate_change_body_from_tool = _build_two_link_arm_with_tool_site(device)
    device = model.device

    # Move the arm to a non-identity configuration with nonzero joint velocity
    # and run ground-truth FK, keeping the body's own twist for comparison.
    state.joint_q.assign([0.2, 0.7])
    state.joint_qd.assign([0.4, -0.3])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    body_twist_com_world = state.body_qd.numpy()[tool_body]

    # Preallocate outputs and launch the kernel under test.
    tool_pose_world = wp.zeros(1, dtype=wp.transform, device=device)
    tool_twist_world = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(
        _tool_pose_and_twist_kernel,
        dim=1,
        inputs=[
            state.body_q,
            state.body_qd,
            model.body_com,
            wp.array([tool_body], dtype=wp.int32, device=device),
            wp.array([coordinate_change_body_from_tool], dtype=wp.transform, device=device),
        ],
        outputs=[tool_pose_world, tool_twist_world],
        device=device,
    )

    # Compare: the tool twist's angular part (last 3 components) should be unchanged.
    np.testing.assert_allclose(tool_twist_world.numpy()[0][3:], body_twist_com_world[3:], atol=1e-6)


def _pose_error(current_pos, current_quat, desired_pos, desired_quat, device):
    """Launch _pose_error_kernel for a single robot and return the 6D error as numpy."""
    current = wp.array([wp.transform(wp.vec3(*current_pos), current_quat)], dtype=wp.transform, device=device)
    desired = wp.array([wp.transform(wp.vec3(*desired_pos), desired_quat)], dtype=wp.transform, device=device)
    pose_error_world = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(_pose_error_kernel, dim=1, inputs=[current, desired], outputs=[pose_error_world], device=device)
    return pose_error_world.numpy()[0]


def test_pose_error_is_zero_when_poses_match(test, device):
    """Identical current and desired poses give exactly zero error, including at the near-identity singularity."""
    quat = wp.quat_from_axis_angle(wp.vec3(0.3, 0.6, -0.2), 1.1)
    error = _pose_error((1.0, -2.0, 0.5), quat, (1.0, -2.0, 0.5), quat, device)
    np.testing.assert_allclose(error, np.zeros(6), atol=1e-7)


def test_pose_error_position_is_desired_minus_current(test, device):
    """The position half of the error is a plain desired-minus-current difference, independent of orientation."""
    quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4)
    error = _pose_error((1.0, 2.0, 3.0), quat, (1.5, 2.0, 2.0), quat, device)
    np.testing.assert_allclose(error[:3], [0.5, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(error[3:], [0.0, 0.0, 0.0], atol=1e-6)


def test_pose_error_orientation_matches_known_rotations(test, device):
    """The orientation error is the axis-angle rotation that carries current onto desired.

    Each case gives (current axis-angle, desired axis-angle, expected error
    axis-angle), hand-computed rather than derived from the kernel itself:

    - 90-degree case: identity to a 90-degree turn about Z gives exactly that turn.
    - Small-angle case: exercises the near-identity Taylor-expansion branch,
      rather than the general atan2 branch.
    - Reversed case: swapping current and desired negates the error, checking
      the sign convention isn't accidentally symmetric.
    - Large-angle case: 170 degrees stays well short of the axis-undefined
      180-degree singularity, but exercises the general branch away from zero.
    """
    identity = wp.quat_identity()
    ninety_about_z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2)
    ten_deg_about_x = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.deg2rad(10.0)))
    one_seventy_about_y = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.deg2rad(170.0)))

    cases = [
        (identity, ninety_about_z, [0.0, 0.0, np.pi / 2]),
        (identity, ten_deg_about_x, [np.deg2rad(10.0), 0.0, 0.0]),
        (ninety_about_z, identity, [0.0, 0.0, -np.pi / 2]),
        (identity, one_seventy_about_y, [0.0, np.deg2rad(170.0), 0.0]),
    ]
    for current_quat, desired_quat, expected_orientation_error in cases:
        error = _pose_error((0.0, 0.0, 0.0), current_quat, (0.0, 0.0, 0.0), desired_quat, device)
        np.testing.assert_allclose(error[:3], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(error[3:], expected_orientation_error, atol=1e-5)


def test_task_space_pd_matches_formula(test, device):
    """The PD kernel computes Kp .* pose_error + Kd .* (desired_twist - current_twist), axis by axis."""
    pose_error_world = wp.array(
        [wp.spatial_vector(0.1, -0.2, 0.3, 0.01, -0.02, 0.03)], dtype=wp.spatial_vector, device=device
    )
    tool_twist_world = wp.array(
        [wp.spatial_vector(0.5, 0.0, -0.5, 0.1, 0.0, -0.1)], dtype=wp.spatial_vector, device=device
    )
    desired_twist_world = wp.array(
        [wp.spatial_vector(0.0, 0.5, 0.0, 0.0, 0.1, 0.0)], dtype=wp.spatial_vector, device=device
    )
    stiffness = wp.array([wp.spatial_vector(10.0, 20.0, 30.0, 1.0, 2.0, 3.0)], dtype=wp.spatial_vector, device=device)
    damping = wp.array([wp.spatial_vector(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)], dtype=wp.spatial_vector, device=device)

    desired_task_acceleration_world = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(
        _task_space_pd_kernel,
        dim=1,
        inputs=[pose_error_world, tool_twist_world, desired_twist_world, stiffness, damping],
        outputs=[desired_task_acceleration_world],
        device=device,
    )

    kp = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])
    kd = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    pose_error_np = np.array([0.1, -0.2, 0.3, 0.01, -0.02, 0.03])
    twist_error_np = np.array([0.0, 0.5, 0.0, 0.0, 0.1, 0.0]) - np.array([0.5, 0.0, -0.5, 0.1, 0.0, -0.1])
    expected = kp * pose_error_np + kd * twist_error_np

    np.testing.assert_allclose(desired_task_acceleration_world.numpy()[0], expected, atol=1e-6)


def test_apply_spatial_matrix_matches_matvec(test, device):
    """The shared 6x6-matrix-times-spatial-vector kernel computes matrix @ vector.

    Used for inertial decoupling (Lambda @ acceleration) and for selection
    masking (selection_matrix @ force) alike, since it's the same operation.
    """
    rng = np.random.default_rng(seed=1)
    matrix_np = rng.standard_normal((6, 6)).astype(np.float32)
    vector_np = rng.standard_normal(6).astype(np.float32)

    matrix = wp.array(matrix_np.reshape(1, 6, 6), dtype=float, device=device)
    vector = wp.array([wp.spatial_vector(*vector_np.tolist())], dtype=wp.spatial_vector, device=device)
    result = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(
        _apply_spatial_matrix_kernel,
        dim=1,
        inputs=[matrix, vector],
        outputs=[result],
        device=device,
    )

    expected = matrix_np @ vector_np
    np.testing.assert_allclose(result.numpy()[0], expected, atol=1e-4)


def test_jacobian_transpose_force_matches_matvec(test, device):
    """The force-mapping kernel computes jacobian_tool_world^T @ force, straight into the compact per-DOF layout.

    Two robots with different controlled-DOF counts (3 and 5, Jacobian
    padded to max_dofs=5) check that a robot's padding columns are never
    read — only ``robot_of_dof``/``slot_of_dof``-addressed compact DOFs are.
    """
    rng = np.random.default_rng(seed=2)
    dof_counts = [3, 5]
    max_dofs = 5
    total_controlled_dofs = sum(dof_counts)

    jacobian_np = np.zeros((2, 6, max_dofs), dtype=np.float32)
    force_np = rng.standard_normal((2, 6)).astype(np.float32)
    for robot_idx, n in enumerate(dof_counts):
        jacobian_np[robot_idx, :, :n] = rng.standard_normal((6, n)).astype(np.float32)

    jacobian_tool_world = wp.array(jacobian_np, dtype=float, device=device)
    task_space_force_world = wp.array(
        [wp.spatial_vector(*force_np[0].tolist()), wp.spatial_vector(*force_np[1].tolist())],
        dtype=wp.spatial_vector,
        device=device,
    )
    # Compact-DOF lookup tables: robot 0's 3 DOFs first, then robot 1's 5.
    robot_of_dof = wp.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=wp.int32, device=device)
    slot_of_dof = wp.array([0, 1, 2, 0, 1, 2, 3, 4], dtype=wp.int32, device=device)

    joint_torque = wp.zeros(total_controlled_dofs, dtype=float, device=device)
    wp.launch(
        _jacobian_transpose_force_kernel,
        dim=total_controlled_dofs,
        inputs=[jacobian_tool_world, task_space_force_world, robot_of_dof, slot_of_dof],
        outputs=[joint_torque],
        device=device,
    )

    expected = np.concatenate(
        [jacobian_np[robot_idx, :, :n].T @ force_np[robot_idx] for robot_idx, n in enumerate(dof_counts)]
    )
    np.testing.assert_allclose(joint_torque.numpy(), expected, atol=1e-4)


def test_jacobian_times_jacobian_transpose_matches_numpy(test, device):
    """J @ J^T, the purely kinematic factor the Moore-Penrose pseudo-inverse transpose needs, matches numpy."""
    model, state, tool_body, coordinate_change_body_from_tool = _build_seven_dof_arm_with_tool_site(device)
    device = model.device

    state.joint_q.assign([0.3, -0.4, 0.6, 0.2, -0.5, 0.35, 0.15])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    jacobian_com_world = newton.eval_jacobian(model, state)

    max_dofs = model.max_dofs_per_articulation
    jacobian_tool_world = wp.zeros((1, 6, max_dofs), dtype=float, device=device)
    wp.launch(
        _shift_jacobian_to_tool_kernel,
        dim=(1, max_dofs),
        inputs=[
            jacobian_com_world,
            state.body_q,
            model.body_com,
            wp.array([tool_body], dtype=wp.int32, device=device),
            wp.array([coordinate_change_body_from_tool], dtype=wp.transform, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([6], dtype=wp.int32, device=device),  # tool_body is link 6 (the 7th joint's child)
        ],
        outputs=[jacobian_tool_world],
        device=device,
    )

    dof_count = wp.array([max_dofs], dtype=wp.int32, device=device)
    jacobian_times_jacobian_transpose = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _jacobian_times_jacobian_transpose_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian_tool_world, dof_count],
        outputs=[jacobian_times_jacobian_transpose],
        device=device,
    )

    jacobian_np = jacobian_tool_world.numpy()[0]
    expected = jacobian_np @ jacobian_np.T
    np.testing.assert_allclose(jacobian_times_jacobian_transpose.numpy()[0], expected, atol=1e-3)


def test_null_space_projector_zeroes_task_response_only_when_dynamically_consistent(test, device):
    """The null-space projector's defining property: null-space torques must not move the tool.

    A joint torque entirely in the null space, tau_null = N @ M @ a for any
    joint acceleration a, must produce zero task-space acceleration when N is
    built from the dynamically-consistent pseudo-inverse transpose. Algebraically
    this reduces to one identity: J @ M^-1 @ N @ M == 0 (a 6 x n zero matrix,
    true for every a simultaneously, not just one example).

    This does *not* hold for the Moore-Penrose variant (which ignores the
    robot's inertia) unless M happens to be proportional to identity, so it's
    checked here too as a contrast — confirming this test would actually
    catch the two variants being mixed up, not just that the formulas run.
    """
    model, state, tool_body, coordinate_change_body_from_tool = _build_seven_dof_arm_with_tool_site(device)
    device = model.device

    # Ground-truth dynamics quantities at a non-identity configuration.
    state.joint_q.assign([0.3, -0.4, 0.6, 0.2, -0.5, 0.35, 0.15])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    mass_matrix = newton.eval_mass_matrix(model, state)
    jacobian_com_world = newton.eval_jacobian(model, state)

    tool_body_arr = wp.array([tool_body], dtype=wp.int32, device=device)
    coordinate_change_body_from_tool_arr = wp.array(
        [coordinate_change_body_from_tool], dtype=wp.transform, device=device
    )
    max_dofs = model.max_dofs_per_articulation
    dof_count = wp.array([max_dofs], dtype=wp.int32, device=device)
    task_dim = wp.array([6], dtype=wp.int32, device=device)

    jacobian_tool_world = wp.zeros((1, 6, max_dofs), dtype=float, device=device)
    wp.launch(
        _shift_jacobian_to_tool_kernel,
        dim=(1, max_dofs),
        inputs=[
            jacobian_com_world,
            state.body_q,
            model.body_com,
            tool_body_arr,
            coordinate_change_body_from_tool_arr,
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([6], dtype=wp.int32, device=device),  # tool_body is link 6 (the 7th joint's child)
        ],
        outputs=[jacobian_tool_world],
        device=device,
    )

    mass_matrix_cholesky = wp.zeros((1, max_dofs, max_dofs), dtype=float, device=device)
    mass_matrix_inv = wp.zeros((1, max_dofs, max_dofs), dtype=float, device=device)
    wp.launch(
        _invert_spd_block_kernel,
        dim=1,
        inputs=[mass_matrix, dof_count, mass_matrix_cholesky],
        outputs=[mass_matrix_inv],
        device=device,
    )

    # Lambda = (J M^-1 J^T)^-1, for the dynamically-consistent variant.
    operational_space_mass_matrix_inv = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _operational_space_mass_matrix_inverse_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian_tool_world, mass_matrix_inv, dof_count],
        outputs=[operational_space_mass_matrix_inv],
        device=device,
    )
    operational_space_mass_matrix_cholesky = wp.zeros((1, 6, 6), dtype=float, device=device)
    operational_space_mass_matrix = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _invert_spd_block_kernel,
        dim=1,
        inputs=[operational_space_mass_matrix_inv, task_dim, operational_space_mass_matrix_cholesky],
        outputs=[operational_space_mass_matrix],
        device=device,
    )

    # (J @ J^T)^-1, for the Moore-Penrose variant.
    jacobian_times_jacobian_transpose = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _jacobian_times_jacobian_transpose_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian_tool_world, dof_count],
        outputs=[jacobian_times_jacobian_transpose],
        device=device,
    )
    jacobian_times_jacobian_transpose_cholesky = wp.zeros((1, 6, 6), dtype=float, device=device)
    jacobian_times_jacobian_transpose_inv = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _invert_spd_block_kernel,
        dim=1,
        inputs=[jacobian_times_jacobian_transpose, task_dim, jacobian_times_jacobian_transpose_cholesky],
        outputs=[jacobian_times_jacobian_transpose_inv],
        device=device,
    )

    def build_projector(task_matrix, apply_mass_matrix_inv):
        task_matrix_times_jacobian = wp.zeros((1, 6, max_dofs), dtype=float, device=device)
        wp.launch(
            _task_matrix_times_jacobian_kernel,
            dim=(1, 6, max_dofs),
            inputs=[task_matrix, jacobian_tool_world, dof_count],
            outputs=[task_matrix_times_jacobian],
            device=device,
        )
        if apply_mass_matrix_inv:
            jacobian_pinv_transpose = wp.zeros((1, 6, max_dofs), dtype=float, device=device)
            wp.launch(
                _apply_mass_matrix_inv_on_right_kernel,
                dim=(1, 6, max_dofs),
                inputs=[task_matrix_times_jacobian, mass_matrix_inv, dof_count],
                outputs=[jacobian_pinv_transpose],
                device=device,
            )
        else:
            jacobian_pinv_transpose = task_matrix_times_jacobian

        null_space_projector = wp.zeros((1, max_dofs, max_dofs), dtype=float, device=device)
        wp.launch(
            _null_space_projector_kernel,
            dim=(1, max_dofs, max_dofs),
            inputs=[jacobian_tool_world, jacobian_pinv_transpose, dof_count],
            outputs=[null_space_projector],
            device=device,
        )
        return null_space_projector.numpy()[0][:7, :7]

    dynamically_consistent_projector = build_projector(operational_space_mass_matrix, apply_mass_matrix_inv=True)
    moore_penrose_projector = build_projector(jacobian_times_jacobian_transpose_inv, apply_mass_matrix_inv=False)

    jacobian_np = jacobian_tool_world.numpy()[0][:, :7]
    mass_matrix_inv_np = mass_matrix_inv.numpy()[0][:7, :7]
    mass_matrix_np = mass_matrix.numpy()[0][:7, :7]

    # Both are valid projectors (idempotent), regardless of which pseudo-inverse built them.
    np.testing.assert_allclose(
        dynamically_consistent_projector @ dynamically_consistent_projector,
        dynamically_consistent_projector,
        atol=1e-3,
    )
    np.testing.assert_allclose(moore_penrose_projector @ moore_penrose_projector, moore_penrose_projector, atol=1e-3)

    # Only the dynamically-consistent projector zeroes the task-space response to a null-space torque.
    dynamically_consistent_response = (
        jacobian_np @ mass_matrix_inv_np @ dynamically_consistent_projector @ mass_matrix_np
    )
    moore_penrose_response = jacobian_np @ mass_matrix_inv_np @ moore_penrose_projector @ mass_matrix_np

    np.testing.assert_allclose(dynamically_consistent_response, np.zeros((6, 7)), atol=1e-3)
    test.assertGreater(np.abs(moore_penrose_response).max(), 0.1)


def test_rotate_selection_matrix_matches_numpy(test, device):
    """The rotated selection matrix is block-diagonal, each block R @ diag(axes) @ R^T, cross-blocks zero."""
    quat = wp.quat_from_axis_angle(wp.vec3(0.3, -0.6, 0.2), 1.1)
    tool_pose_world = wp.array([wp.transform(wp.vec3(0.0, 0.0, 0.0), quat)], dtype=wp.transform, device=device)
    # Select only the local x linear axis and the local y,z angular axes.
    linear_axes_np = np.array([1.0, 0.0, 0.0])
    angular_axes_np = np.array([0.0, 1.0, 1.0])
    selection_axes_tool = wp.array(
        [wp.spatial_vector(*linear_axes_np.tolist(), *angular_axes_np.tolist())],
        dtype=wp.spatial_vector,
        device=device,
    )

    selection_matrix_world = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _rotate_selection_matrix_kernel,
        dim=1,
        inputs=[tool_pose_world, selection_axes_tool],
        outputs=[selection_matrix_world],
        device=device,
    )

    rotation_np = np.array(wp.quat_to_matrix(quat)).reshape(3, 3)
    expected_linear_block = rotation_np @ np.diag(linear_axes_np) @ rotation_np.T
    expected_angular_block = rotation_np @ np.diag(angular_axes_np) @ rotation_np.T

    result = selection_matrix_world.numpy()[0]
    np.testing.assert_allclose(result[0:3, 0:3], expected_linear_block, atol=1e-5)
    np.testing.assert_allclose(result[3:6, 3:6], expected_angular_block, atol=1e-5)
    np.testing.assert_allclose(result[0:3, 3:6], np.zeros((3, 3)))
    np.testing.assert_allclose(result[3:6, 0:3], np.zeros((3, 3)))


def test_closed_loop_wrench_command_matches_formula(test, device):
    """The full wrench (force and moment) gets closed-loop feedback, desired + Kp .* (desired - measured)."""
    desired_wrench_world = wp.array(
        [wp.spatial_vector(10.0, -5.0, 2.0, 1.0, -0.5, 0.25)], dtype=wp.spatial_vector, device=device
    )
    measured_wrench_world = wp.array(
        [wp.spatial_vector(8.0, -6.0, 2.5, 0.8, -0.6, 0.1)], dtype=wp.spatial_vector, device=device
    )
    stiffness = wp.array([wp.spatial_vector(2.0, 3.0, 1.0, 0.5, 0.5, 0.5)], dtype=wp.spatial_vector, device=device)

    wrench_command_world = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(
        _closed_loop_wrench_command_kernel,
        dim=1,
        inputs=[desired_wrench_world, measured_wrench_world, stiffness],
        outputs=[wrench_command_world],
        device=device,
    )

    desired_np = np.array([10.0, -5.0, 2.0, 1.0, -0.5, 0.25])
    measured_np = np.array([8.0, -6.0, 2.5, 0.8, -0.6, 0.1])
    kp_np = np.array([2.0, 3.0, 1.0, 0.5, 0.5, 0.5])
    expected = desired_np + kp_np * (desired_np - measured_np)

    np.testing.assert_allclose(wrench_command_world.numpy()[0], expected, atol=1e-5)


class TestOperationalSpaceKernels(unittest.TestCase):
    pass


add_function_test(
    TestOperationalSpaceKernels,
    "test_invert_spd_block_matches_numpy_inverse",
    test_invert_spd_block_matches_numpy_inverse,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_operational_space_mass_matrix_matches_numpy",
    test_operational_space_mass_matrix_matches_numpy,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_tool_pose_matches_body_and_site_composition",
    test_tool_pose_matches_body_and_site_composition,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_jacobian_tool_shift_matches_twist",
    test_jacobian_tool_shift_matches_twist,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_jacobian_tool_shift_matches_finite_difference",
    test_jacobian_tool_shift_matches_finite_difference,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_tool_twist_angular_part_matches_body",
    test_tool_twist_angular_part_matches_body,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_pose_error_is_zero_when_poses_match",
    test_pose_error_is_zero_when_poses_match,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_pose_error_position_is_desired_minus_current",
    test_pose_error_position_is_desired_minus_current,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_pose_error_orientation_matches_known_rotations",
    test_pose_error_orientation_matches_known_rotations,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_task_space_pd_matches_formula",
    test_task_space_pd_matches_formula,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_apply_spatial_matrix_matches_matvec",
    test_apply_spatial_matrix_matches_matvec,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_jacobian_transpose_force_matches_matvec",
    test_jacobian_transpose_force_matches_matvec,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_jacobian_times_jacobian_transpose_matches_numpy",
    test_jacobian_times_jacobian_transpose_matches_numpy,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_null_space_projector_zeroes_task_response_only_when_dynamically_consistent",
    test_null_space_projector_zeroes_task_response_only_when_dynamically_consistent,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_rotate_selection_matrix_matches_numpy",
    test_rotate_selection_matrix_matches_numpy,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_closed_loop_wrench_command_matches_formula",
    test_closed_loop_wrench_command_matches_formula,
    devices=devices,
)


# ---------------------------------------------------------------------------
# ControllerOperationalSpaceModelFree
# ---------------------------------------------------------------------------


def _dofs_arr(dofs_list, device):
    """Return a wp.array[int32] from a list of per-robot DOF counts."""
    return wp.array(np.array(dofs_list, dtype=np.int32), device=device)


def _poses(poses, device):
    """Return a wp.array[wp.transform] from a list of wp.transform."""
    return wp.array(poses, dtype=wp.transform, device=device)


def _twists(twists, device):
    """Return a wp.array[wp.spatial_vector] from a list of 6-tuples."""
    return wp.array([wp.spatial_vector(*t) for t in twists], dtype=wp.spatial_vector, device=device)


def _make_model_free(*, dofs_list, kp, kd, device, use_inertia=True):
    """Construct a ControllerOperationalSpaceModelFree with scalar-broadcast gains."""
    return ControllerOperationalSpaceModelFree(
        controlled_dofs_per_robot=_dofs_arr(dofs_list, device),
        motion_stiffness=kp,
        motion_damping=kd,
        use_inertia_decoupling=use_inertia,
        device=device,
    )


def _run_model_free(
    ctrl, *, current_poses, current_twists, desired_poses, desired_twists, jacobian, device, mass_matrix=None
):
    """Run one step on a ControllerOperationalSpaceModelFree and return the compact torque array."""
    ins = ctrl.input()
    ins.tool_pose_world = _poses(current_poses, device)
    ins.tool_twist_world = _twists(current_twists, device)
    ins.desired_tool_pose_world = _poses(desired_poses, device)
    ins.desired_twist_world = _twists(desired_twists, device)
    ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
    if mass_matrix is not None:
        ins.mass_matrix = wp.array(mass_matrix, dtype=wp.float32, device=device)
    outs = ctrl.output()
    ctrl.step(inputs=ins, outputs=outs, dt=0.01)
    return outs.joint_f.numpy()


class TestControllerOperationalSpaceModelFree(unittest.TestCase):
    def test_zero_error_gives_zero_torque(self):
        """Identical current and desired poses/twists produce zero torque."""
        device = wp.get_device()
        ctrl = _make_model_free(dofs_list=[7], kp=100.0, kd=10.0, device=device, use_inertia=False)
        identity_pose = wp.transform(wp.vec3(0.1, 0.2, 0.3), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5))
        rng = np.random.default_rng(0)
        jacobian = rng.standard_normal((1, 6, 7)).astype(np.float32)
        tau = _run_model_free(
            ctrl,
            current_poses=[identity_pose],
            current_twists=[(0, 0, 0, 0, 0, 0)],
            desired_poses=[identity_pose],
            desired_twists=[(0, 0, 0, 0, 0, 0)],
            jacobian=jacobian,
            device=device,
        )
        np.testing.assert_allclose(tau, np.zeros(7), atol=1e-5)

    def test_position_error_matches_formula_without_inertia_decoupling(self):
        """tau = J^T @ (Kp * pose_error), when inertial decoupling is off."""
        device = wp.get_device()
        kp = 100.0
        ctrl = _make_model_free(dofs_list=[7], kp=kp, kd=10.0, device=device, use_inertia=False)
        current_pose = wp.transform_identity()
        desired_pose = wp.transform(wp.vec3(0.1, -0.05, 0.02), wp.quat_identity())
        rng = np.random.default_rng(1)
        jacobian = rng.standard_normal((1, 6, 7)).astype(np.float32)

        tau = _run_model_free(
            ctrl,
            current_poses=[current_pose],
            current_twists=[(0, 0, 0, 0, 0, 0)],
            desired_poses=[desired_pose],
            desired_twists=[(0, 0, 0, 0, 0, 0)],
            jacobian=jacobian,
            device=device,
        )

        pose_error = np.array([0.1, -0.05, 0.02, 0.0, 0.0, 0.0])
        expected = jacobian[0].T @ (kp * pose_error)
        np.testing.assert_allclose(tau, expected, atol=1e-3)

    def test_inertia_decoupling_matches_formula(self):
        """tau = J^T @ Lambda @ (Kp * pose_error), the full chain, matches a from-scratch numpy computation."""
        device = wp.get_device()
        kp = 50.0
        ctrl = _make_model_free(dofs_list=[7], kp=kp, kd=10.0, device=device, use_inertia=True)
        current_pose = wp.transform_identity()
        desired_pose = wp.transform(wp.vec3(0.1, -0.05, 0.02), wp.quat_identity())
        rng = np.random.default_rng(2)
        jacobian = rng.standard_normal((1, 6, 7)).astype(np.float32)
        random_matrix = rng.standard_normal((7, 7)).astype(np.float32)
        mass_matrix = (random_matrix @ random_matrix.T + 7 * np.eye(7, dtype=np.float32)).reshape(1, 7, 7)

        tau = _run_model_free(
            ctrl,
            current_poses=[current_pose],
            current_twists=[(0, 0, 0, 0, 0, 0)],
            desired_poses=[desired_pose],
            desired_twists=[(0, 0, 0, 0, 0, 0)],
            jacobian=jacobian,
            mass_matrix=mass_matrix,
            device=device,
        )

        pose_error = np.array([0.1, -0.05, 0.02, 0.0, 0.0, 0.0])
        mass_matrix_inv = np.linalg.inv(mass_matrix[0])
        lambda_inv = jacobian[0] @ mass_matrix_inv @ jacobian[0].T
        operational_space_mass_matrix = np.linalg.inv(lambda_inv)
        expected = jacobian[0].T @ (operational_space_mass_matrix @ (kp * pose_error))
        np.testing.assert_allclose(tau, expected, atol=1e-2)

    def test_rejects_under_six_dof_with_inertia_decoupling(self):
        """Fewer than 6 controlled DOFs with inertial decoupling raises at construction, not silently at runtime."""
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerOperationalSpaceModelFree(
                controlled_dofs_per_robot=_dofs_arr([3], device),
                motion_stiffness=1.0,
                motion_damping=1.0,
                use_inertia_decoupling=True,
                device=device,
            )

    def test_heterogeneous_fleet_matches_per_robot_formulas(self):
        """Two robots with different controlled-DOF counts (6 and 8) are computed independently and correctly."""
        device = wp.get_device()
        kp = 80.0
        ctrl = _make_model_free(dofs_list=[6, 8], kp=kp, kd=10.0, device=device, use_inertia=False)

        current_poses = [wp.transform_identity(), wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity())]
        desired_poses = [
            wp.transform(wp.vec3(0.05, 0.0, 0.0), wp.quat_identity()),
            wp.transform(wp.vec3(1.0, 0.1, -0.05), wp.quat_identity()),
        ]
        rng = np.random.default_rng(3)
        jacobian = np.zeros((2, 6, 8), dtype=np.float32)
        jacobian[0, :, :6] = rng.standard_normal((6, 6)).astype(np.float32)
        jacobian[1, :, :8] = rng.standard_normal((6, 8)).astype(np.float32)

        tau = _run_model_free(
            ctrl,
            current_poses=current_poses,
            current_twists=[(0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)],
            desired_poses=desired_poses,
            desired_twists=[(0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)],
            jacobian=jacobian,
            device=device,
        )

        pose_error_0 = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
        pose_error_1 = np.array([0.0, 0.1, -0.05, 0.0, 0.0, 0.0])
        expected_0 = jacobian[0, :, :6].T @ (kp * pose_error_0)
        expected_1 = jacobian[1, :, :8].T @ (kp * pose_error_1)

        np.testing.assert_allclose(tau[:6], expected_0, atol=1e-3)
        np.testing.assert_allclose(tau[6:], expected_1, atol=1e-3)

    def test_live_gains_read_from_inputs_each_step(self):
        """Passing motion_stiffness=None at construction reads inputs.motion_stiffness each step."""
        device = wp.get_device()
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=_dofs_arr([7], device),
            motion_stiffness=None,
            motion_damping=10.0,
            use_inertia_decoupling=False,
            device=device,
        )
        current_pose = wp.transform_identity()
        desired_pose = wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity())
        rng = np.random.default_rng(4)
        jacobian = rng.standard_normal((1, 6, 7)).astype(np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = _poses([current_pose], device)
        ins.tool_twist_world = _twists([(0, 0, 0, 0, 0, 0)], device)
        ins.desired_tool_pose_world = _poses([desired_pose], device)
        ins.desired_twist_world = _twists([(0, 0, 0, 0, 0, 0)], device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        ins.motion_stiffness = wp.array(
            [wp.spatial_vector(30.0, 30.0, 30.0, 5.0, 5.0, 5.0)], dtype=wp.spatial_vector, device=device
        )
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        pose_error = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        kp = np.array([30.0, 30.0, 30.0, 5.0, 5.0, 5.0])
        expected = jacobian[0].T @ (kp * pose_error)
        np.testing.assert_allclose(outs.joint_f.numpy(), expected, atol=1e-3)

    def test_output_scatters_to_indexed_view(self):
        """outputs.joint_f may be bound to an indexed view of a larger simulation-sized array."""
        device = wp.get_device()
        ctrl = _make_model_free(dofs_list=[7], kp=100.0, kd=10.0, device=device, use_inertia=False)
        current_pose = wp.transform_identity()
        desired_pose = wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity())
        rng = np.random.default_rng(5)
        jacobian = rng.standard_normal((1, 6, 7)).astype(np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = _poses([current_pose], device)
        ins.tool_twist_world = _twists([(0, 0, 0, 0, 0, 0)], device)
        ins.desired_tool_pose_world = _poses([desired_pose], device)
        ins.desired_twist_world = _twists([(0, 0, 0, 0, 0, 0)], device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)

        # A larger simulation-sized joint-force array; only indices [2:9) belong to this robot.
        sim_joint_f = wp.zeros(12, dtype=wp.float32, device=device)
        selection = wp.array(np.arange(2, 9, dtype=np.int32), device=device)
        outs = ctrl.output()
        outs.joint_f = sim_joint_f[selection]
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        pose_error = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        expected = jacobian[0].T @ (100.0 * pose_error)
        np.testing.assert_allclose(sim_joint_f.numpy()[2:9], expected, atol=1e-3)
        np.testing.assert_allclose(sim_joint_f.numpy()[:2], 0.0)
        np.testing.assert_allclose(sim_joint_f.numpy()[9:], 0.0)

    def test_transform_and_spatial_vector_inputs_accept_indexed_views(self):
        """Every input port, not just outputs.joint_f, may be bound to an indexed view of a larger array.

        Binds tool_pose_world, tool_twist_world,
        desired_tool_pose_world, desired_twist_world, and
        motion_stiffness/motion_damping (live) to views selecting robot 1 out
        of a larger 3-robot simulation-sized array, and checks the result
        matches a plain-array run with the same values.
        """
        device = wp.get_device()
        kp_vec = (30.0, 30.0, 30.0, 5.0, 5.0, 5.0)
        kd_vec = (2.0, 2.0, 2.0, 0.5, 0.5, 0.5)
        current_pose = wp.transform_identity()
        desired_pose = wp.transform(wp.vec3(0.1, -0.05, 0.02), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.2))
        rng = np.random.default_rng(6)
        jacobian = rng.standard_normal((1, 6, 7)).astype(np.float32)

        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=_dofs_arr([7], device),
            motion_stiffness=None,
            motion_damping=None,
            use_inertia_decoupling=False,
            device=device,
        )

        # A larger, 3-robot simulation-sized set of per-robot arrays; only
        # index 1 belongs to this controller's one robot.
        selection = wp.array(np.array([1], dtype=np.int32), device=device)
        sim_pose = wp.array(
            [wp.transform_identity(), current_pose, wp.transform_identity()], dtype=wp.transform, device=device
        )
        sim_desired_pose = wp.array(
            [wp.transform_identity(), desired_pose, wp.transform_identity()], dtype=wp.transform, device=device
        )
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sim_twist = wp.array([zero_twist, zero_twist, zero_twist], dtype=wp.spatial_vector, device=device)
        sim_desired_twist = wp.array([zero_twist, zero_twist, zero_twist], dtype=wp.spatial_vector, device=device)
        sim_stiffness = wp.array([wp.spatial_vector(*kp_vec)] * 3, dtype=wp.spatial_vector, device=device)
        sim_damping = wp.array([wp.spatial_vector(*kd_vec)] * 3, dtype=wp.spatial_vector, device=device)

        ins = ctrl.input()
        ins.tool_pose_world = sim_pose[selection]
        ins.tool_twist_world = sim_twist[selection]
        ins.desired_tool_pose_world = sim_desired_pose[selection]
        ins.desired_twist_world = sim_desired_twist[selection]
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        ins.motion_stiffness = sim_stiffness[selection]
        ins.motion_damping = sim_damping[selection]
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        pose_error = np.array([0.1, -0.05, 0.02, 0.0, 0.0, 0.2])
        kp = np.array(kp_vec)
        expected = jacobian[0].T @ (kp * pose_error)
        np.testing.assert_allclose(outs.joint_f.numpy(), expected, atol=1e-2)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
