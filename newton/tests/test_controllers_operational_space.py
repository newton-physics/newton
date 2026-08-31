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
    _wrench_feedback_only_kernel,
    _wrench_feedforward_and_feedback_kernel,
)
from newton._src.controllers.impl.operational_space.model_based import ControllerOperationalSpace
from newton._src.controllers.impl.operational_space.model_free import ControllerOperationalSpaceModelFree
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()

# Operational frame coincides with world frame, for every test not
# specifically exercising operational_frame_pose_world itself.
_IDENTITY_TRANSFORM = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())


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
            wp.array(
                [np.arange(max_dofs, dtype=np.int32)], dtype=wp.int32, device=device
            ),  # articulation_dof_idx_of_padded_dof_idx: every DOF controlled, in order
            wp.array([max_dofs], dtype=wp.int32, device=device),  # controlled_dofs_per_robot
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
            wp.array(
                [np.arange(max_dofs, dtype=np.int32)], dtype=wp.int32, device=device
            ),  # articulation_dof_idx_of_padded_dof_idx: every DOF controlled, in order
            wp.array([max_dofs], dtype=wp.int32, device=device),  # controlled_dofs_per_robot
        ],
        outputs=[jacobian_tool_world],
        device=device,
    )

    # Compare: jacobian_tool_world @ joint_qd should reproduce the ground-truth twist.
    predicted_twist = jacobian_tool_world.numpy()[0] @ joint_qd
    np.testing.assert_allclose(predicted_twist, tool_twist_world.numpy()[0], atol=1e-6)


def test_jacobian_tool_shift_articulation_dof_idx_of_padded_dof_idx_remaps_non_prefix_subset(test, device):
    """articulation_dof_idx_of_padded_dof_idx correctly remaps a non-prefix, non-contiguous controlled-DOF subset.

    Controls joints [0, 1, 3, 5, 6] of the 7-joint arm -- skipping joints 2
    and 4 -- so the controlled DOFs are not the first N columns of the
    articulation's own Jacobian. Each compact output column must equal the
    full-Jacobian column articulation_dof_idx_of_padded_dof_idx maps it to, not
    the column at its own padded index.
    """
    model, state, tool_body, coordinate_change_body_from_tool = _build_seven_dof_arm_with_tool_site(device)
    device = model.device

    state.joint_q.assign([0.3, -0.4, 0.6, 0.2, -0.5, 0.35, 0.15])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    jacobian_com_world = newton.eval_jacobian(model, state)

    tool_body_arr = wp.array([tool_body], dtype=wp.int32, device=device)
    coordinate_change_body_from_tool_arr = wp.array(
        [coordinate_change_body_from_tool], dtype=wp.transform, device=device
    )
    robot_articulation_arr = wp.array([0], dtype=wp.int32, device=device)
    robot_link_idx_arr = wp.array([6], dtype=wp.int32, device=device)  # tool_body is link 6 (the 7th joint's child)

    # Ground truth: the full, every-DOF-controlled shift, already verified
    # correct by test_jacobian_tool_shift_matches_twist/finite_difference.
    full_dof_count = model.max_dofs_per_articulation
    jacobian_tool_world_full = wp.zeros((1, 6, full_dof_count), dtype=float, device=device)
    wp.launch(
        _shift_jacobian_to_tool_kernel,
        dim=(1, full_dof_count),
        inputs=[
            jacobian_com_world,
            state.body_q,
            model.body_com,
            tool_body_arr,
            coordinate_change_body_from_tool_arr,
            robot_articulation_arr,
            robot_link_idx_arr,
            wp.array([np.arange(full_dof_count, dtype=np.int32)], dtype=wp.int32, device=device),
            wp.array([full_dof_count], dtype=wp.int32, device=device),
        ],
        outputs=[jacobian_tool_world_full],
        device=device,
    )

    # Under test: only joints [0, 1, 3, 5, 6] controlled, 5 compact columns.
    controlled_local_dofs = [0, 1, 3, 5, 6]
    controlled_dof_count = len(controlled_local_dofs)
    articulation_dof_idx_of_padded_dof_idx = wp.array([controlled_local_dofs], dtype=wp.int32, device=device)
    controlled_dofs_per_robot = wp.array([controlled_dof_count], dtype=wp.int32, device=device)
    jacobian_tool_world_compact = wp.zeros((1, 6, controlled_dof_count), dtype=float, device=device)
    wp.launch(
        _shift_jacobian_to_tool_kernel,
        dim=(1, controlled_dof_count),
        inputs=[
            jacobian_com_world,
            state.body_q,
            model.body_com,
            tool_body_arr,
            coordinate_change_body_from_tool_arr,
            robot_articulation_arr,
            robot_link_idx_arr,
            articulation_dof_idx_of_padded_dof_idx,
            controlled_dofs_per_robot,
        ],
        outputs=[jacobian_tool_world_compact],
        device=device,
    )

    expected = jacobian_tool_world_full.numpy()[0][:, controlled_local_dofs]
    np.testing.assert_allclose(jacobian_tool_world_compact.numpy()[0], expected, atol=1e-6)


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
            wp.array(
                [np.arange(max_dofs, dtype=np.int32)], dtype=wp.int32, device=device
            ),  # articulation_dof_idx_of_padded_dof_idx: every DOF controlled, in order
            wp.array([max_dofs], dtype=wp.int32, device=device),  # controlled_dofs_per_robot
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
    """The PD kernel computes Kp .* pose_error + Kd .* (desired_twist - current_twist), axis by axis.

    At an identity tool orientation the tool-local frame coincides with
    world, so rotating errors into the tool frame, applying the gain, and
    rotating the result back (the kernel's actual implementation) reduces
    to exactly this per-axis formula.
    """
    identity_pose = wp.array(
        [wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())], dtype=wp.transform, device=device
    )
    pose_error_world = wp.array(
        [wp.spatial_vector(0.1, -0.2, 0.3, 0.01, -0.02, 0.03)], dtype=wp.spatial_vector, device=device
    )
    tool_twist_world = wp.array(
        [wp.spatial_vector(0.5, 0.0, -0.5, 0.1, 0.0, -0.1)], dtype=wp.spatial_vector, device=device
    )
    desired_twist_world = wp.array(
        [wp.spatial_vector(0.0, 0.5, 0.0, 0.0, 0.1, 0.0)], dtype=wp.spatial_vector, device=device
    )
    kp = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])
    kd = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    stiffness = wp.array([wp.spatial_vector(*kp.tolist())], dtype=wp.spatial_vector, device=device)
    damping = wp.array([wp.spatial_vector(*kd.tolist())], dtype=wp.spatial_vector, device=device)

    desired_task_acceleration_world = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(
        _task_space_pd_kernel,
        dim=1,
        inputs=[identity_pose, pose_error_world, tool_twist_world, desired_twist_world, stiffness, damping],
        outputs=[desired_task_acceleration_world],
        device=device,
    )

    pose_error_np = np.array([0.1, -0.2, 0.3, 0.01, -0.02, 0.03])
    twist_error_np = np.array([0.0, 0.5, 0.0, 0.0, 0.1, 0.0]) - np.array([0.5, 0.0, -0.5, 0.1, 0.0, -0.1])
    expected = kp * pose_error_np + kd * twist_error_np

    np.testing.assert_allclose(desired_task_acceleration_world.numpy()[0], expected, atol=1e-5)


def test_task_space_pd_applies_gain_in_operational_frame_not_world(test, device):
    """The kernel's own responsibility is rotating its result from the operational frame back to world.

    Upstream (_tool_state_to_operational_kernel, _pose_error_kernel) already
    puts the pose/twist error in the operational frame before this kernel
    runs, so this test feeds a pose error already expressed there and checks
    only the remaining rotate-out: with the operational frame rotated 45
    degrees about world Z and an anisotropic operational-frame-local Kp
    (stiff along local X only), a purely-local-X error produces a world
    result with equal X and Y components -- not purely along world X.
    """
    quat_45_about_z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 4.0)
    operational_frame_pose_world = wp.array(
        [wp.transform(wp.vec3(0.0, 0.0, 0.0), quat_45_about_z)], dtype=wp.transform, device=device
    )
    stiffness = wp.array([wp.spatial_vector(100.0, 0.0, 0.0, 0.0, 0.0, 0.0)], dtype=wp.spatial_vector, device=device)
    damping = wp.array([wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)], dtype=wp.spatial_vector, device=device)

    zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # A pose error purely along operational-frame-local X.
    pose_error_operational = wp.array(
        [wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)], dtype=wp.spatial_vector, device=device
    )
    tool_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
    desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)

    desired_task_acceleration_world = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(
        _task_space_pd_kernel,
        dim=1,
        inputs=[
            operational_frame_pose_world,
            pose_error_operational,
            tool_twist_operational,
            desired_twist_operational,
            stiffness,
            damping,
        ],
        outputs=[desired_task_acceleration_world],
        device=device,
    )

    # accel_operational = kp .* pose_error_operational = (100, 0, 0). Rotated
    # to world by 45 degrees about Z: 100 * (cos45, sin45, 0) = (70.7, 70.7, 0).
    expected_world = 100.0 * np.array([np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0])
    np.testing.assert_allclose(
        desired_task_acceleration_world.numpy()[0], [*expected_world, 0.0, 0.0, 0.0], atol=1e-4
    )


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
            wp.array(
                [np.arange(max_dofs, dtype=np.int32)], dtype=wp.int32, device=device
            ),  # articulation_dof_idx_of_padded_dof_idx: every DOF controlled, in order
            wp.array([max_dofs], dtype=wp.int32, device=device),  # controlled_dofs_per_robot
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
    np.testing.assert_allclose(jacobian_times_jacobian_transpose.numpy()[0], expected, atol=1e-5)


def test_null_space_projector_zeroes_task_response_only_when_dynamically_consistent(test, device):
    """The null-space projector's defining property: null-space torques must not move the tool.

    A joint torque entirely in the null space, tau_null = N @ M @ a for any
    joint acceleration a, must produce zero task-space acceleration when N is
    built from the dynamically-consistent pseudo-inverse transpose. Algebraically
    this reduces to one identity: J @ M^-1 @ N == 0 (a 6 x n zero matrix) --
    M is always invertible, so this is equivalent to (and tighter than)
    checking J @ M^-1 @ N @ M == 0 for every a simultaneously.

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
            wp.array(
                [np.arange(max_dofs, dtype=np.int32)], dtype=wp.int32, device=device
            ),  # articulation_dof_idx_of_padded_dof_idx: every DOF controlled, in order
            wp.array([max_dofs], dtype=wp.int32, device=device),  # controlled_dofs_per_robot
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

    # Both are valid projectors (idempotent), regardless of which pseudo-inverse built them.
    np.testing.assert_allclose(
        dynamically_consistent_projector @ dynamically_consistent_projector,
        dynamically_consistent_projector,
        atol=1e-4,
    )
    np.testing.assert_allclose(moore_penrose_projector @ moore_penrose_projector, moore_penrose_projector, atol=1e-4)

    # Only the dynamically-consistent projector zeroes J @ M^-1 @ N -- the
    # identity that guarantees tau_null = N @ M @ a never disturbs the task,
    # for every a simultaneously (M is invertible, so multiplying through by
    # it, as tau_null literally does, changes nothing about which projector
    # zeroes this out).
    dynamically_consistent_response = jacobian_np @ mass_matrix_inv_np @ dynamically_consistent_projector
    moore_penrose_response = jacobian_np @ mass_matrix_inv_np @ moore_penrose_projector

    np.testing.assert_allclose(dynamically_consistent_response, np.zeros((6, 7)), atol=1e-4)
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


def test_wrench_feedforward_and_feedback_matches_formula(test, device):
    """The full wrench (force and moment) gets feedforward + feedback, desired + Kp .* (desired - measured).

    An identity tool orientation makes the tool-local Kp coincide with
    world axes, reducing the kernel's rotate/apply/rotate-back to exactly
    this per-axis formula.
    """
    identity_pose = wp.array(
        [wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())], dtype=wp.transform, device=device
    )
    desired_wrench_world = wp.array(
        [wp.spatial_vector(10.0, -5.0, 2.0, 1.0, -0.5, 0.25)], dtype=wp.spatial_vector, device=device
    )
    measured_wrench_world = wp.array(
        [wp.spatial_vector(8.0, -6.0, 2.5, 0.8, -0.6, 0.1)], dtype=wp.spatial_vector, device=device
    )
    kp_np = np.array([2.0, 3.0, 1.0, 0.5, 0.5, 0.5])
    stiffness = wp.array([wp.spatial_vector(*kp_np.tolist())], dtype=wp.spatial_vector, device=device)

    wrench_command_world = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(
        _wrench_feedforward_and_feedback_kernel,
        dim=1,
        inputs=[identity_pose, desired_wrench_world, measured_wrench_world, stiffness],
        outputs=[wrench_command_world],
        device=device,
    )

    desired_np = np.array([10.0, -5.0, 2.0, 1.0, -0.5, 0.25])
    measured_np = np.array([8.0, -6.0, 2.5, 0.8, -0.6, 0.1])
    expected = desired_np + kp_np * (desired_np - measured_np)

    np.testing.assert_allclose(wrench_command_world.numpy()[0], expected, atol=1e-5)


def test_wrench_feedback_only_matches_formula(test, device):
    """Feedback alone, with no feedforward term, is Kp .* (desired - measured).

    Same identity-orientation note as
    :func:`test_wrench_feedforward_and_feedback_matches_formula` above.
    """
    identity_pose = wp.array(
        [wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())], dtype=wp.transform, device=device
    )
    desired_wrench_world = wp.array(
        [wp.spatial_vector(10.0, -5.0, 2.0, 1.0, -0.5, 0.25)], dtype=wp.spatial_vector, device=device
    )
    measured_wrench_world = wp.array(
        [wp.spatial_vector(8.0, -6.0, 2.5, 0.8, -0.6, 0.1)], dtype=wp.spatial_vector, device=device
    )
    kp_np = np.array([2.0, 3.0, 1.0, 0.5, 0.5, 0.5])
    stiffness = wp.array([wp.spatial_vector(*kp_np.tolist())], dtype=wp.spatial_vector, device=device)

    wrench_command_world = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(
        _wrench_feedback_only_kernel,
        dim=1,
        inputs=[identity_pose, desired_wrench_world, measured_wrench_world, stiffness],
        outputs=[wrench_command_world],
        device=device,
    )

    desired_np = np.array([10.0, -5.0, 2.0, 1.0, -0.5, 0.25])
    measured_np = np.array([8.0, -6.0, 2.5, 0.8, -0.6, 0.1])
    expected = kp_np * (desired_np - measured_np)

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
    "test_jacobian_tool_shift_articulation_dof_idx_of_padded_dof_idx_remaps_non_prefix_subset",
    test_jacobian_tool_shift_articulation_dof_idx_of_padded_dof_idx_remaps_non_prefix_subset,
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
    "test_task_space_pd_applies_gain_in_operational_frame_not_world",
    test_task_space_pd_applies_gain_in_operational_frame_not_world,
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
    "test_wrench_feedforward_and_feedback_matches_formula",
    test_wrench_feedforward_and_feedback_matches_formula,
    devices=devices,
)
add_function_test(
    TestOperationalSpaceKernels,
    "test_wrench_feedback_only_matches_formula",
    test_wrench_feedback_only_matches_formula,
    devices=devices,
)


# ---------------------------------------------------------------------------
# ControllerOperationalSpaceModelFree
# ---------------------------------------------------------------------------


class TestControllerOperationalSpaceModelFree(unittest.TestCase):
    def test_zero_error_gives_zero_torque(self):
        """Identical current and desired poses/twists produce zero torque."""
        device = wp.get_device()
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
            motion_stiffness=100.0,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            device=device,
        )
        identity_pose = wp.transform(wp.vec3(0.1, 0.2, 0.3), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5))
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rng = np.random.default_rng(0)
        jacobian = rng.standard_normal((1, 6, 7)).astype(np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array([identity_pose], dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array([identity_pose], dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        np.testing.assert_allclose(outs.joint_f.numpy(), np.zeros(7), atol=1e-5)

    def test_position_error_matches_formula_without_inertia_decoupling(self):
        """tau = J^T @ (Kp * pose_error), when inertial decoupling is off."""
        device = wp.get_device()
        kp = 100.0
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
            motion_stiffness=kp,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            device=device,
        )
        current_pose = wp.transform_identity()
        desired_pose = wp.transform(wp.vec3(0.1, -0.05, 0.02), wp.quat_identity())
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rng = np.random.default_rng(1)
        jacobian = rng.standard_normal((1, 6, 7)).astype(np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array([current_pose], dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array([desired_pose], dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        pose_error = np.array([0.1, -0.05, 0.02, 0.0, 0.0, 0.0])
        expected = jacobian[0].T @ (kp * pose_error)
        np.testing.assert_allclose(outs.joint_f.numpy(), expected, atol=1e-3)

    def test_inertia_decoupling_matches_formula(self):
        """tau = J^T @ Lambda @ (Kp * pose_error), the full chain, matches a from-scratch numpy computation."""
        device = wp.get_device()
        kp = 50.0
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
            motion_stiffness=kp,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=True,
            device=device,
        )
        current_pose = wp.transform_identity()
        desired_pose = wp.transform(wp.vec3(0.1, -0.05, 0.02), wp.quat_identity())
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # A fixed, well-conditioned Jacobian and a fixed, genuinely coupled SPD mass matrix
        # (A @ A^T + 5*I for a fixed, non-diagonal A) -- rather than a random Gram matrix, whose
        # poor conditioning would amplify the float32 rounding difference between this kernel's
        # Cholesky-based inverse and numpy's LU-based one enough to need a much looser tolerance
        # below to avoid test flakiness.
        jacobian = np.array(
            [
                [2, 0, 0, 1, 0, 1, 0],
                [0, 3, 0, 0, 1, 0, 1],
                [0, 0, 1, 2, 1, 0, 0],
                [1, 1, 0, 0, 0, 3, 1],
                [0, 1, 2, 1, 0, 0, 1],
                [1, 0, 1, 0, 2, 1, 0],
            ],
            dtype=np.float32,
        ).reshape(1, 6, 7)
        mass_matrix_seed = np.array(
            [
                [1, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 1, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 2, 0, 1, 0],
                [0, 0, 1, 0, 1, 0, 1],
                [1, 1, 0, 1, 0, 2, 0],
                [0, 0, 0, 0, 1, 0, 1],
            ],
            dtype=np.float32,
        )
        mass_matrix = (mass_matrix_seed @ mass_matrix_seed.T + 5.0 * np.eye(7, dtype=np.float32)).reshape(1, 7, 7)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array([current_pose], dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array([desired_pose], dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        ins.mass_matrix = wp.array(mass_matrix, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        pose_error = np.array([0.1, -0.05, 0.02, 0.0, 0.0, 0.0])
        mass_matrix_inv = np.linalg.inv(mass_matrix[0])
        lambda_inv = jacobian[0] @ mass_matrix_inv @ jacobian[0].T
        operational_space_mass_matrix = np.linalg.inv(lambda_inv)
        expected = jacobian[0].T @ (operational_space_mass_matrix @ (kp * pose_error))
        np.testing.assert_allclose(outs.joint_f.numpy(), expected, rtol=1e-4, atol=1e-4)

    def test_rejects_under_six_dof_with_inertia_decoupling(self):
        """Fewer than 6 controlled DOFs with inertial decoupling raises at construction, not silently at runtime."""
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerOperationalSpaceModelFree(
                controlled_dofs_per_robot=wp.array(np.array([3], dtype=np.int32), device=device),
                motion_stiffness=1.0,
                motion_damping=1.0,
                operational_frame_pose_world=_IDENTITY_TRANSFORM,
                use_inertia_decoupling=True,
                device=device,
            )

    def test_heterogeneous_fleet_matches_per_robot_formulas(self):
        """Two robots with different controlled-DOF counts (6 and 8) are computed independently and correctly."""
        device = wp.get_device()
        kp = 80.0
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([6, 8], dtype=np.int32), device=device),
            motion_stiffness=kp,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            device=device,
        )

        current_poses = [wp.transform_identity(), wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity())]
        desired_poses = [
            wp.transform(wp.vec3(0.05, 0.0, 0.0), wp.quat_identity()),
            wp.transform(wp.vec3(1.0, 0.1, -0.05), wp.quat_identity()),
        ]
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rng = np.random.default_rng(3)
        jacobian = np.zeros((2, 6, 8), dtype=np.float32)
        jacobian[0, :, :6] = rng.standard_normal((6, 6)).astype(np.float32)
        jacobian[1, :, :8] = rng.standard_normal((6, 8)).astype(np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array(current_poses, dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist, zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array(desired_poses, dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist, zero_twist], dtype=wp.spatial_vector, device=device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        pose_error_0 = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
        pose_error_1 = np.array([0.0, 0.1, -0.05, 0.0, 0.0, 0.0])
        expected_0 = jacobian[0, :, :6].T @ (kp * pose_error_0)
        expected_1 = jacobian[1, :, :8].T @ (kp * pose_error_1)

        tau = outs.joint_f.numpy()
        np.testing.assert_allclose(tau[:6], expected_0, atol=1e-3)
        np.testing.assert_allclose(tau[6:], expected_1, atol=1e-3)

    def test_live_gains_read_from_inputs_each_step(self):
        """Passing motion_stiffness=None at construction reads inputs.motion_stiffness each step."""
        device = wp.get_device()
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
            motion_stiffness=None,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            device=device,
        )
        current_pose = wp.transform_identity()
        desired_pose = wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity())
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rng = np.random.default_rng(4)
        jacobian = rng.standard_normal((1, 6, 7)).astype(np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array([current_pose], dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array([desired_pose], dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
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
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
            motion_stiffness=100.0,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            device=device,
        )
        current_pose = wp.transform_identity()
        desired_pose = wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity())
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rng = np.random.default_rng(5)
        jacobian = rng.standard_normal((1, 6, 7)).astype(np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array([current_pose], dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array([desired_pose], dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
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
        desired_tool_pose_operational, desired_twist_operational, and
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
            controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
            motion_stiffness=None,
            motion_damping=None,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
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
        ins.desired_tool_pose_operational = sim_desired_pose[selection]
        ins.desired_twist_operational = sim_desired_twist[selection]
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        ins.motion_stiffness = sim_stiffness[selection]
        ins.motion_damping = sim_damping[selection]
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        pose_error = np.array([0.1, -0.05, 0.02, 0.0, 0.0, 0.2])
        kp = np.array(kp_vec)
        expected = jacobian[0].T @ (kp * pose_error)
        np.testing.assert_allclose(outs.joint_f.numpy(), expected, atol=1e-2)

    def test_use_wrench_feedforward_requires_selection_axes(self):
        """use_wrench_feedforward=True without wrench_selection_axes_tool raises at construction."""
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerOperationalSpaceModelFree(
                controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
                motion_stiffness=1.0,
                motion_damping=1.0,
                operational_frame_pose_world=_IDENTITY_TRANSFORM,
                use_inertia_decoupling=False,
                use_wrench_feedforward=True,
                device=device,
            )

    def test_wrench_params_rejected_without_wrench_enabled(self):
        """wrench_selection_axes_tool set without use_wrench_feedforward/use_wrench_feedback raises at construction."""
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerOperationalSpaceModelFree(
                controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
                motion_stiffness=1.0,
                motion_damping=1.0,
                operational_frame_pose_world=_IDENTITY_TRANSFORM,
                use_inertia_decoupling=False,
                wrench_selection_axes_tool=wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                device=device,
            )

    def test_wrench_stiffness_rejects_a_value_that_is_not_a_gain_shape(self):
        """A wrench_stiffness that is neither a float, wp.spatial_vector, nor wp.array raises at construction."""
        device = wp.get_device()
        with self.assertRaises(TypeError):
            ControllerOperationalSpaceModelFree(
                controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
                motion_stiffness=1.0,
                motion_damping=1.0,
                operational_frame_pose_world=_IDENTITY_TRANSFORM,
                use_inertia_decoupling=False,
                use_wrench_feedback=True,
                wrench_selection_axes_tool=wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                wrench_stiffness=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                device=device,
            )

    def test_wrench_stiffness_accepts_a_bare_spatial_vector(self):
        """A wrench_stiffness passed as a wp.spatial_vector is broadcast to every robot, same as motion_stiffness."""
        device = wp.get_device()
        kp = 60.0
        wrench_kp_vec = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([6], dtype=np.int32), device=device),
            motion_stiffness=kp,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_wrench_feedback=True,
            motion_selection_axes_tool=wp.spatial_vector(1.0, 1.0, 0.0, 1.0, 1.0, 1.0),
            wrench_selection_axes_tool=wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            wrench_stiffness=wp.spatial_vector(*wrench_kp_vec),
            device=device,
        )
        current_pose = wp.transform_identity()
        desired_pose = wp.transform_identity()
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        desired_wrench = wp.spatial_vector(0.0, 0.0, 20.0, 0.0, 0.0, 0.0)
        measured_wrench = wp.spatial_vector(0.0, 0.0, 15.0, 0.0, 0.0, 0.0)
        rng = np.random.default_rng(10)
        jacobian = rng.standard_normal((1, 6, 6)).astype(np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array([current_pose], dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array([desired_pose], dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        ins.desired_wrench_world = wp.array([desired_wrench], dtype=wp.spatial_vector, device=device)
        ins.measured_wrench_world = wp.array([measured_wrench], dtype=wp.spatial_vector, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        # Zero pose error, so the motion term contributes nothing. The z-axis wrench gain is
        # wrench_kp_vec[2] = 3.0, not a uniform scalar, proving the per-axis spatial_vector was used.
        wrench_command_z = wrench_kp_vec[2] * (20.0 - 15.0)
        masked_wrench_force = np.array([0.0, 0.0, wrench_command_z, 0.0, 0.0, 0.0])
        expected = jacobian[0].T @ masked_wrench_force
        np.testing.assert_allclose(outs.joint_f.numpy(), expected, atol=1e-2)

    def test_wrench_feedforward_only_and_motion_selection_matches_formula(self):
        """Hybrid motion/wrench control: tau = J^T @ (S_motion @ F_motion) + J^T @ (S_wrench @ desired_wrench).

        Uses a peg-in-hole-style split (translation z and rotation open to
        force control, everything else motion-controlled) at a non-identity
        tool orientation, so the world-frame selection matrices actually mix
        axes rather than reducing to a fixed 0/1 mask.
        """
        device = wp.get_device()
        kp = 60.0
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([6], dtype=np.int32), device=device),
            motion_stiffness=kp,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_wrench_feedforward=True,
            motion_selection_axes_tool=wp.spatial_vector(1.0, 1.0, 0.0, 1.0, 1.0, 1.0),
            wrench_selection_axes_tool=wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            device=device,
        )
        # quat_from_axis_angle does not normalize its axis; an un-normalized
        # one (unlike test_rotate_selection_matrix_matches_numpy's use of the
        # same values, which is self-consistently checked against
        # wp.quat_to_matrix either way) would make "rotating" an isotropic
        # gain fail to be a true rotation, so isotropic Kp would no longer
        # come back out unchanged as this test's expected motion_force
        # assumes -- normalize explicitly.
        axis = np.array([0.3, -0.6, 0.2])
        axis = axis / np.linalg.norm(axis)
        quat = wp.quat_from_axis_angle(wp.vec3(*axis.tolist()), 1.1)
        current_pose = wp.transform(wp.vec3(0.2, 0.1, -0.1), quat)
        desired_pose = wp.transform(wp.vec3(0.25, 0.05, -0.08), quat)
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        desired_wrench = wp.spatial_vector(3.0, -2.0, 20.0, 0.4, -0.3, 0.6)
        rng = np.random.default_rng(7)
        jacobian = rng.standard_normal((1, 6, 6)).astype(np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array([current_pose], dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array([desired_pose], dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        ins.desired_wrench_world = wp.array([desired_wrench], dtype=wp.spatial_vector, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        pose_error = np.array([0.05, -0.05, 0.02, 0.0, 0.0, 0.0])
        motion_force = kp * pose_error
        desired_wrench_np = np.array([3.0, -2.0, 20.0, 0.4, -0.3, 0.6])

        rotation_np = np.array(wp.quat_to_matrix(quat)).reshape(3, 3)
        motion_linear_block = rotation_np @ np.diag([1.0, 1.0, 0.0]) @ rotation_np.T
        motion_angular_block = rotation_np @ np.diag([1.0, 1.0, 1.0]) @ rotation_np.T
        wrench_linear_block = rotation_np @ np.diag([0.0, 0.0, 1.0]) @ rotation_np.T
        wrench_angular_block = rotation_np @ np.diag([0.0, 0.0, 0.0]) @ rotation_np.T

        masked_motion_force = np.concatenate(
            [motion_linear_block @ motion_force[:3], motion_angular_block @ motion_force[3:]]
        )
        masked_wrench_force = np.concatenate(
            [wrench_linear_block @ desired_wrench_np[:3], wrench_angular_block @ desired_wrench_np[3:]]
        )
        expected = jacobian[0].T @ masked_motion_force + jacobian[0].T @ masked_wrench_force
        np.testing.assert_allclose(outs.joint_f.numpy(), expected, atol=1e-2)

    def test_wrench_feedforward_and_feedback_control_matches_formula(self):
        """With both enabled, wrench control adds Kp .* (desired - measured) to the desired wrench before masking."""
        device = wp.get_device()
        kp = 60.0
        wrench_kp = 2.0
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([6], dtype=np.int32), device=device),
            motion_stiffness=kp,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_wrench_feedforward=True,
            use_wrench_feedback=True,
            motion_selection_axes_tool=wp.spatial_vector(1.0, 1.0, 0.0, 1.0, 1.0, 1.0),
            wrench_selection_axes_tool=wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            wrench_stiffness=wrench_kp,
            device=device,
        )
        current_pose = wp.transform_identity()
        desired_pose = wp.transform_identity()
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        desired_wrench = wp.spatial_vector(0.0, 0.0, 20.0, 0.0, 0.0, 0.0)
        measured_wrench = wp.spatial_vector(0.0, 0.0, 15.0, 0.0, 0.0, 0.0)
        rng = np.random.default_rng(8)
        jacobian = rng.standard_normal((1, 6, 6)).astype(np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array([current_pose], dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array([desired_pose], dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        ins.desired_wrench_world = wp.array([desired_wrench], dtype=wp.spatial_vector, device=device)
        ins.measured_wrench_world = wp.array([measured_wrench], dtype=wp.spatial_vector, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        # Zero pose error, so the motion term contributes nothing.
        wrench_command_z = 20.0 + wrench_kp * (20.0 - 15.0)
        masked_wrench_force = np.array([0.0, 0.0, wrench_command_z, 0.0, 0.0, 0.0])
        expected = jacobian[0].T @ masked_wrench_force
        np.testing.assert_allclose(outs.joint_f.numpy(), expected, atol=1e-2)

    def test_wrench_feedback_only_control_matches_formula(self):
        """With only use_wrench_feedback, the command is Kp .* (desired - measured), with no feedforward term."""
        device = wp.get_device()
        kp = 60.0
        wrench_kp = 2.0
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([6], dtype=np.int32), device=device),
            motion_stiffness=kp,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_wrench_feedback=True,
            motion_selection_axes_tool=wp.spatial_vector(1.0, 1.0, 0.0, 1.0, 1.0, 1.0),
            wrench_selection_axes_tool=wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            wrench_stiffness=wrench_kp,
            device=device,
        )
        current_pose = wp.transform_identity()
        desired_pose = wp.transform_identity()
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        desired_wrench = wp.spatial_vector(0.0, 0.0, 20.0, 0.0, 0.0, 0.0)
        measured_wrench = wp.spatial_vector(0.0, 0.0, 15.0, 0.0, 0.0, 0.0)
        rng = np.random.default_rng(9)
        jacobian = rng.standard_normal((1, 6, 6)).astype(np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array([current_pose], dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array([desired_pose], dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        ins.desired_wrench_world = wp.array([desired_wrench], dtype=wp.spatial_vector, device=device)
        ins.measured_wrench_world = wp.array([measured_wrench], dtype=wp.spatial_vector, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        # Zero pose error, so the motion term contributes nothing. No "+ desired" feedforward term this time.
        wrench_command_z = wrench_kp * (20.0 - 15.0)
        masked_wrench_force = np.array([0.0, 0.0, wrench_command_z, 0.0, 0.0, 0.0])
        expected = jacobian[0].T @ masked_wrench_force
        np.testing.assert_allclose(outs.joint_f.numpy(), expected, atol=1e-2)

    def test_gravity_compensation_adds_directly_to_torque(self):
        """inputs.gravity_force is added directly to the summed joint torque, with zero pose error isolating it."""
        device = wp.get_device()
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
            motion_stiffness=100.0,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_gravity_compensation=True,
            device=device,
        )
        identity_pose = wp.transform_identity()
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rng = np.random.default_rng(11)
        jacobian = rng.standard_normal((1, 6, 7)).astype(np.float32)
        gravity_force = rng.standard_normal(7).astype(np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array([identity_pose], dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array([identity_pose], dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        ins.gravity_force = wp.array(gravity_force, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        np.testing.assert_allclose(outs.joint_f.numpy(), gravity_force, atol=1e-5)

    def test_gravity_force_rejected_without_use_gravity_compensation(self):
        """inputs.gravity_force set without use_gravity_compensation=True raises at step."""
        device = wp.get_device()
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
            motion_stiffness=100.0,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_gravity_compensation=False,
            device=device,
        )
        identity_pose = wp.transform_identity()
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rng = np.random.default_rng(12)
        jacobian = rng.standard_normal((1, 6, 7)).astype(np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array([identity_pose], dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array([identity_pose], dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        ins.gravity_force = wp.zeros(7, dtype=wp.float32, device=device)
        outs = ctrl.output()
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=outs, dt=0.01)

    def test_null_space_control_requires_redundant_manipulator(self):
        """use_null_space_control=True with 6 or fewer controlled DOFs raises at construction."""
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerOperationalSpaceModelFree(
                controlled_dofs_per_robot=wp.array(np.array([6], dtype=np.int32), device=device),
                motion_stiffness=1.0,
                motion_damping=1.0,
                operational_frame_pose_world=_IDENTITY_TRANSFORM,
                use_inertia_decoupling=True,
                use_null_space_control=True,
                null_space_stiffness=1.0,
                null_space_damping=1.0,
                device=device,
            )

    def test_null_space_control_dynamically_consistent_matches_formula(self):
        """With use_inertia_decoupling=True, tau_null = N @ (M @ (Kp*(q_des_null-q) + Kd*(qd_des_null-qd))).

        N is built from the dynamically-consistent pseudo-inverse transpose,
        Lambda @ J @ M^-1. Zero pose error isolates the null-space term from
        the motion term.
        """
        device = wp.get_device()
        null_kp = 20.0
        null_kd = 4.0
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
            motion_stiffness=100.0,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=True,
            use_null_space_control=True,
            null_space_stiffness=null_kp,
            null_space_damping=null_kd,
            device=device,
        )
        identity_pose = wp.transform_identity()
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # A fixed, well-conditioned Jacobian (rank 6, singular values ~1.1-4.6) and a fixed,
        # genuinely coupled SPD mass matrix (A @ A^T + 5*I for a fixed, non-diagonal A) -- rather
        # than a random Gram matrix, whose poor conditioning would amplify the float32 rounding
        # difference between this kernel's Cholesky-based inverse and numpy's LU-based one enough
        # to need a much looser tolerance below to avoid test flakiness.
        jacobian = np.array(
            [
                [2, 0, 0, 1, 0, 1, 0],
                [0, 3, 0, 0, 1, 0, 1],
                [0, 0, 1, 2, 1, 0, 0],
                [1, 1, 0, 0, 0, 3, 1],
                [0, 1, 2, 1, 0, 0, 1],
                [1, 0, 1, 0, 2, 1, 0],
            ],
            dtype=np.float32,
        ).reshape(1, 6, 7)
        mass_matrix_seed = np.array(
            [
                [1, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 1, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 2, 0, 1, 0],
                [0, 0, 1, 0, 1, 0, 1],
                [1, 1, 0, 1, 0, 2, 0],
                [0, 0, 0, 0, 1, 0, 1],
            ],
            dtype=np.float32,
        )
        mass_matrix = (mass_matrix_seed @ mass_matrix_seed.T + 5.0 * np.eye(7, dtype=np.float32)).reshape(1, 7, 7)
        joint_q = np.array([0.1, -0.2, 0.3, -0.1, 0.05, -0.15, 0.2], dtype=np.float32)
        joint_qd = np.array([0.05, 0.02, -0.03, 0.01, -0.02, 0.04, -0.01], dtype=np.float32)
        joint_q_des_null = np.zeros(7, dtype=np.float32)
        joint_qd_des_null = np.zeros(7, dtype=np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array([identity_pose], dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array([identity_pose], dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        ins.mass_matrix = wp.array(mass_matrix, dtype=wp.float32, device=device)
        ins.joint_q = wp.array(joint_q, dtype=wp.float32, device=device)
        ins.joint_qd = wp.array(joint_qd, dtype=wp.float32, device=device)
        ins.joint_q_des_null = wp.array(joint_q_des_null, dtype=wp.float32, device=device)
        ins.joint_qd_des_null = wp.array(joint_qd_des_null, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        mass_matrix_inv = np.linalg.inv(mass_matrix[0])
        lambda_inv = jacobian[0] @ mass_matrix_inv @ jacobian[0].T
        operational_space_mass_matrix = np.linalg.inv(lambda_inv)
        jacobian_pinv_transpose = operational_space_mass_matrix @ jacobian[0] @ mass_matrix_inv
        null_space_projector = np.eye(7) - jacobian[0].T @ jacobian_pinv_transpose

        posture_acc = null_kp * (joint_q_des_null - joint_q) + null_kd * (joint_qd_des_null - joint_qd)
        expected = null_space_projector @ (mass_matrix[0] @ posture_acc)
        np.testing.assert_allclose(outs.joint_f.numpy(), expected, rtol=1e-4, atol=1e-4)

    def test_null_space_control_moore_penrose_matches_formula(self):
        """With use_inertia_decoupling=False, N is built from the kinematics-only Moore-Penrose pseudo-inverse.

        tau_null = N @ (Kp*(q_des_null-q) + Kd*(qd_des_null-qd)) directly,
        with no mass-matrix premultiply, and no mass_matrix input needed.
        """
        device = wp.get_device()
        null_kp = 15.0
        null_kd = 3.0
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
            motion_stiffness=100.0,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_null_space_control=True,
            null_space_stiffness=null_kp,
            null_space_damping=null_kd,
            device=device,
        )
        identity_pose = wp.transform_identity()
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # Same fixed, well-conditioned Jacobian as the dynamically-consistent test above.
        jacobian = np.array(
            [
                [2, 0, 0, 1, 0, 1, 0],
                [0, 3, 0, 0, 1, 0, 1],
                [0, 0, 1, 2, 1, 0, 0],
                [1, 1, 0, 0, 0, 3, 1],
                [0, 1, 2, 1, 0, 0, 1],
                [1, 0, 1, 0, 2, 1, 0],
            ],
            dtype=np.float32,
        ).reshape(1, 6, 7)
        joint_q = np.array([0.1, -0.2, 0.3, -0.1, 0.05, -0.15, 0.2], dtype=np.float32)
        joint_qd = np.array([0.05, 0.02, -0.03, 0.01, -0.02, 0.04, -0.01], dtype=np.float32)
        joint_q_des_null = np.zeros(7, dtype=np.float32)
        joint_qd_des_null = np.zeros(7, dtype=np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array([identity_pose], dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array([identity_pose], dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        ins.joint_q = wp.array(joint_q, dtype=wp.float32, device=device)
        ins.joint_qd = wp.array(joint_qd, dtype=wp.float32, device=device)
        ins.joint_q_des_null = wp.array(joint_q_des_null, dtype=wp.float32, device=device)
        ins.joint_qd_des_null = wp.array(joint_qd_des_null, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        jjt = jacobian[0] @ jacobian[0].T
        jacobian_pinv_transpose = np.linalg.inv(jjt) @ jacobian[0]
        null_space_projector = np.eye(7) - jacobian[0].T @ jacobian_pinv_transpose

        posture_acc = null_kp * (joint_q_des_null - joint_q) + null_kd * (joint_qd_des_null - joint_qd)
        expected = null_space_projector @ posture_acc
        np.testing.assert_allclose(outs.joint_f.numpy(), expected, rtol=1e-4, atol=1e-4)

    def test_partial_inertia_decoupling_requires_full_inertia_decoupling(self):
        """use_partial_inertia_decoupling=True with use_inertia_decoupling=False raises at construction."""
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerOperationalSpaceModelFree(
                controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
                motion_stiffness=1.0,
                motion_damping=1.0,
                operational_frame_pose_world=_IDENTITY_TRANSFORM,
                use_inertia_decoupling=False,
                use_partial_inertia_decoupling=True,
                device=device,
            )

    def test_partial_inertia_decoupling_matches_block_diagonal_formula(self):
        """With use_partial_inertia_decoupling=True, Lambda is two independent 3x3 inversions, block-diagonal.

        tau = J^T @ Lambda_partial @ (Kp * pose_error), where Lambda_partial
        is built from separately inverting the translational and rotational
        3x3 blocks of J M^-1 J^T, ignoring their coupling -- unlike the full
        Lambda, which inverts the whole 6x6 at once.
        """
        device = wp.get_device()
        kp = 50.0
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
            motion_stiffness=kp,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=True,
            use_partial_inertia_decoupling=True,
            device=device,
        )
        current_pose = wp.transform_identity()
        desired_pose = wp.transform(wp.vec3(0.1, -0.05, 0.02), wp.quat_identity())
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # Same fixed, well-conditioned Jacobian and genuinely coupled SPD mass matrix as the
        # null-space tests above.
        jacobian = np.array(
            [
                [2, 0, 0, 1, 0, 1, 0],
                [0, 3, 0, 0, 1, 0, 1],
                [0, 0, 1, 2, 1, 0, 0],
                [1, 1, 0, 0, 0, 3, 1],
                [0, 1, 2, 1, 0, 0, 1],
                [1, 0, 1, 0, 2, 1, 0],
            ],
            dtype=np.float32,
        ).reshape(1, 6, 7)
        mass_matrix_seed = np.array(
            [
                [1, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 1, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 2, 0, 1, 0],
                [0, 0, 1, 0, 1, 0, 1],
                [1, 1, 0, 1, 0, 2, 0],
                [0, 0, 0, 0, 1, 0, 1],
            ],
            dtype=np.float32,
        )
        mass_matrix = (mass_matrix_seed @ mass_matrix_seed.T + 5.0 * np.eye(7, dtype=np.float32)).reshape(1, 7, 7)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array([current_pose], dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array([desired_pose], dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        ins.mass_matrix = wp.array(mass_matrix, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        pose_error = np.array([0.1, -0.05, 0.02, 0.0, 0.0, 0.0])
        jacobian_np = jacobian[0]
        mass_matrix_inv = np.linalg.inv(mass_matrix[0])
        lambda_linear = np.linalg.inv(jacobian_np[0:3] @ mass_matrix_inv @ jacobian_np[0:3].T)
        lambda_angular = np.linalg.inv(jacobian_np[3:6] @ mass_matrix_inv @ jacobian_np[3:6].T)
        lambda_partial = np.zeros((6, 6))
        lambda_partial[0:3, 0:3] = lambda_linear
        lambda_partial[3:6, 3:6] = lambda_angular

        expected = jacobian_np.T @ (lambda_partial @ (kp * pose_error))
        np.testing.assert_allclose(outs.joint_f.numpy(), expected, rtol=1e-4, atol=1e-4)

    def test_null_space_control_falls_back_to_moore_penrose_with_partial_inertia_decoupling(self):
        """With partial inertia decoupling, the null-space projector uses Moore-Penrose, not dynamically-consistent.

        A block-diagonal (partially-decoupled) Lambda does not have the
        property the dynamically-consistent pseudo-inverse formula needs, so
        the projector falls back to the kinematics-only Moore-Penrose
        variant even though use_inertia_decoupling=True and a mass matrix is
        available -- the posture term is still premultiplied by the mass
        matrix, since that stays valid regardless.
        """
        device = wp.get_device()
        null_kp = 20.0
        null_kd = 4.0
        ctrl = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.array(np.array([7], dtype=np.int32), device=device),
            motion_stiffness=100.0,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=True,
            use_partial_inertia_decoupling=True,
            use_null_space_control=True,
            null_space_stiffness=null_kp,
            null_space_damping=null_kd,
            device=device,
        )
        identity_pose = wp.transform_identity()
        zero_twist = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        jacobian = np.array(
            [
                [2, 0, 0, 1, 0, 1, 0],
                [0, 3, 0, 0, 1, 0, 1],
                [0, 0, 1, 2, 1, 0, 0],
                [1, 1, 0, 0, 0, 3, 1],
                [0, 1, 2, 1, 0, 0, 1],
                [1, 0, 1, 0, 2, 1, 0],
            ],
            dtype=np.float32,
        ).reshape(1, 6, 7)
        mass_matrix_seed = np.array(
            [
                [1, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 1, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 2, 0, 1, 0],
                [0, 0, 1, 0, 1, 0, 1],
                [1, 1, 0, 1, 0, 2, 0],
                [0, 0, 0, 0, 1, 0, 1],
            ],
            dtype=np.float32,
        )
        mass_matrix = (mass_matrix_seed @ mass_matrix_seed.T + 5.0 * np.eye(7, dtype=np.float32)).reshape(1, 7, 7)
        joint_q = np.array([0.1, -0.2, 0.3, -0.1, 0.05, -0.15, 0.2], dtype=np.float32)
        joint_qd = np.array([0.05, 0.02, -0.03, 0.01, -0.02, 0.04, -0.01], dtype=np.float32)
        joint_q_des_null = np.zeros(7, dtype=np.float32)
        joint_qd_des_null = np.zeros(7, dtype=np.float32)

        ins = ctrl.input()
        ins.tool_pose_world = wp.array([identity_pose], dtype=wp.transform, device=device)
        ins.tool_twist_world = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.desired_tool_pose_operational = wp.array([identity_pose], dtype=wp.transform, device=device)
        ins.desired_twist_operational = wp.array([zero_twist], dtype=wp.spatial_vector, device=device)
        ins.jacobian_tool_world = wp.array(jacobian, dtype=wp.float32, device=device)
        ins.mass_matrix = wp.array(mass_matrix, dtype=wp.float32, device=device)
        ins.joint_q = wp.array(joint_q, dtype=wp.float32, device=device)
        ins.joint_qd = wp.array(joint_qd, dtype=wp.float32, device=device)
        ins.joint_q_des_null = wp.array(joint_q_des_null, dtype=wp.float32, device=device)
        ins.joint_qd_des_null = wp.array(joint_qd_des_null, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        jacobian_np = jacobian[0]
        jjt = jacobian_np @ jacobian_np.T
        jacobian_pinv_transpose = np.linalg.inv(jjt) @ jacobian_np
        null_space_projector = np.eye(7) - jacobian_np.T @ jacobian_pinv_transpose

        posture_acc = null_kp * (joint_q_des_null - joint_q) + null_kd * (joint_qd_des_null - joint_qd)
        expected = null_space_projector @ (mass_matrix[0] @ posture_acc)
        np.testing.assert_allclose(outs.joint_f.numpy(), expected, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# ControllerOperationalSpace (model-based): construction/selection only.
# step() is not implemented yet, so these tests only exercise __init__.
# ---------------------------------------------------------------------------


def _build_heterogeneous_fleet_with_tool_sites(device):
    """Two robots: a 1-DOF robot and a 3-DOF robot, each with a tool site on its last link.

    Returns:
        Tuple of (model, robot_0_tip_body, robot_1_tip_body).
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)

    robot_0_body = builder.add_link(mass=1.0)
    builder.add_shape_box(robot_0_body, hx=0.1, hy=0.1, hz=0.1)
    robot_0_joint = builder.add_joint_revolute(
        parent=-1,
        child=robot_0_body,
        axis=newton.Axis.Z,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([robot_0_joint], label="robot_0")
    builder.add_site(robot_0_body, xform=wp.transform_identity(), label="tool_site")

    robot_1_body_1 = builder.add_link(mass=1.0)
    robot_1_body_2 = builder.add_link(mass=1.0)
    robot_1_body_3 = builder.add_link(mass=1.0)
    builder.add_shape_box(robot_1_body_1, hx=0.1, hy=0.1, hz=0.1)
    builder.add_shape_box(robot_1_body_2, hx=0.1, hy=0.1, hz=0.1)
    builder.add_shape_box(robot_1_body_3, hx=0.1, hy=0.1, hz=0.1)
    robot_1_joint_1 = builder.add_joint_revolute(
        parent=-1,
        child=robot_1_body_1,
        axis=newton.Axis.Z,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    robot_1_joint_2 = builder.add_joint_revolute(
        parent=robot_1_body_1,
        child=robot_1_body_2,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    robot_1_joint_3 = builder.add_joint_revolute(
        parent=robot_1_body_2,
        child=robot_1_body_3,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([robot_1_joint_1, robot_1_joint_2, robot_1_joint_3], label="robot_1")
    builder.add_site(robot_1_body_3, xform=wp.transform_identity(), label="tool_site")

    model = builder.finalize(device=device)
    return model, robot_0_body, robot_1_body_3


def _build_single_link_pendulum_with_tool_site(device):
    """One revolute joint about Y, with gravity on, for a hand-derivable gravity-compensation check.

    The joint rotates the body (and its COM) in the world XZ plane, so
    gravity (along -Z) produces a nonzero torque about the joint axis at
    every angle except the vertical (COM directly below the joint).

    Returns:
        Tuple of (model, state, tool_body, mass, com_distance_from_joint).
    """
    mass = 2.0
    com_distance_from_joint = 0.5
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)

    body = builder.add_link(mass=mass)
    # density=0 so the shape contributes no mass of its own -- the body's
    # mass stays exactly the explicit `mass` above.
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    builder.body_com[body] = wp.vec3(com_distance_from_joint, 0.0, 0.0)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=newton.Axis.Y,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([joint], label="pendulum")
    builder.add_site(body, xform=wp.transform_identity(), label="tool_site")

    model = builder.finalize(device=device)
    state = model.state()
    return model, state, body, mass, com_distance_from_joint


class TestControllerOperationalSpace(unittest.TestCase):
    def test_resolves_single_robot_selection(self):
        """A single-articulation model resolves controlled DOFs, tool body, and link index correctly."""
        device = wp.get_device()
        model, _state, tool_body, coordinate_change_body_from_tool = _build_two_link_arm_with_tool_site(device)

        ctrl = ControllerOperationalSpace(
            model,
            tool="tool_site",
            motion_stiffness=100.0,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_gravity_compensation=False,
        )

        self.assertEqual(ctrl.controlled_robot_count, 1)
        self.assertEqual(ctrl.total_controlled_dofs, 2)
        self.assertEqual(ctrl.max_controlled_dofs, 2)
        np.testing.assert_array_equal(ctrl.q_start.numpy(), [0, 1])
        np.testing.assert_array_equal(ctrl.qd_start.numpy(), [0, 1])
        np.testing.assert_array_equal(ctrl.tool_body.numpy(), [tool_body])
        # The tool site is on the second (and last) joint's child body, so its
        # row-block index within the articulation's Jacobian is 1.
        np.testing.assert_array_equal(ctrl._robot_link_idx.numpy(), [1])
        resolved_transform = ctrl._tool_transform_body.numpy()[0]
        np.testing.assert_allclose(resolved_transform, np.array(coordinate_change_body_from_tool), atol=1e-6)

    def test_resolves_heterogeneous_fleet_selection(self):
        """Two robots with different controlled-DOF counts each resolve their own tool site."""
        device = wp.get_device()
        model, robot_0_body, robot_1_tip_body = _build_heterogeneous_fleet_with_tool_sites(device)

        ctrl = ControllerOperationalSpace(
            model,
            tool="tool_site",
            motion_stiffness=100.0,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_gravity_compensation=False,
        )

        self.assertEqual(ctrl.controlled_robot_count, 2)
        self.assertEqual(ctrl.total_controlled_dofs, 4)
        self.assertEqual(ctrl.max_controlled_dofs, 3)
        np.testing.assert_array_equal(ctrl._controlled_dofs_per_robot.numpy(), [1, 3])
        np.testing.assert_array_equal(ctrl.tool_body.numpy(), [robot_0_body, robot_1_tip_body])
        # Robot 0's tool is on its only (first) joint's child; robot 1's tool
        # is on its third joint's child.
        np.testing.assert_array_equal(ctrl._robot_link_idx.numpy(), [0, 2])

    def test_tool_pattern_matching_nothing_raises(self):
        """A tool pattern that matches no site in the model raises at construction."""
        device = wp.get_device()
        model, _state, _tool_body, _transform = _build_two_link_arm_with_tool_site(device)
        with self.assertRaises(ValueError):
            ControllerOperationalSpace(
                model,
                tool="nonexistent_site",
                motion_stiffness=1.0,
                motion_damping=1.0,
                operational_frame_pose_world=_IDENTITY_TRANSFORM,
                use_gravity_compensation=False,
            )

    def test_tool_pattern_matching_multiple_sites_on_one_robot_raises(self):
        """A tool selection matching more than one site on the same robot raises at construction."""
        device = wp.get_device()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        body = builder.add_link(mass=1.0)
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(
            parent=-1,
            child=body,
            axis=newton.Axis.Z,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([joint], label="robot")
        builder.add_site(body, xform=wp.transform_identity(), label="site_a")
        builder.add_site(body, xform=wp.transform_identity(), label="site_b")
        model = builder.finalize(device=device)

        with self.assertRaises(ValueError):
            ControllerOperationalSpace(
                model,
                tool=["site_a", "site_b"],
                motion_stiffness=1.0,
                motion_damping=1.0,
                operational_frame_pose_world=_IDENTITY_TRANSFORM,
                use_gravity_compensation=False,
            )

    def test_model_without_sites_raises(self):
        """A model with no sites at all raises at construction, rather than resolving zero matches silently."""
        device = wp.get_device()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        body = builder.add_link(mass=1.0)
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(
            parent=-1,
            child=body,
            axis=newton.Axis.Z,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([joint], label="robot")
        model = builder.finalize(device=device)

        with self.assertRaises(ValueError):
            ControllerOperationalSpace(
                model,
                tool="tool_site",
                motion_stiffness=1.0,
                motion_damping=1.0,
                operational_frame_pose_world=_IDENTITY_TRANSFORM,
                use_gravity_compensation=False,
            )

    def test_tool_by_explicit_site_index(self):
        """An explicit site index, rather than a label pattern, resolves the same tool body."""
        device = wp.get_device()
        model, _state, tool_body, _transform = _build_two_link_arm_with_tool_site(device)
        site_index = model.shape_label.index("tool_site")

        ctrl = ControllerOperationalSpace(
            model,
            tool=site_index,
            motion_stiffness=100.0,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_gravity_compensation=False,
        )
        np.testing.assert_array_equal(ctrl.tool_body.numpy(), [tool_body])

    def test_non_model_raises_type_error(self):
        """Passing a non-Model object raises TypeError, not an unrelated AttributeError."""
        with self.assertRaises(TypeError):
            ControllerOperationalSpace(
                "not a model",
                tool="tool_site",
                motion_stiffness=1.0,
                motion_damping=1.0,
                operational_frame_pose_world=_IDENTITY_TRANSFORM,
                use_gravity_compensation=False,
            )

    def test_step_resolves_tool_pose_matching_forward_kinematics(self):
        """step() resolves the tool pose from joint_q, matching a hand-derived planar forward-kinematics formula.

        For the two-link planar arm, joint 1 rotates body 1 (and everything
        downstream of it) about Z by theta1; joint 2 further rotates body 2
        about Z by theta2, with body 2's origin fixed one unit along body 1's
        rotated X axis. So body 2's origin is at
        ``(cos(theta1), sin(theta1), 0)`` and its orientation is
        ``Rot_z(theta1 + theta2)``; the tool site is a further fixed offset
        from body 2's origin, rotated by that same combined angle.
        """
        device = wp.get_device()
        model, _state, _tool_body, coordinate_change_body_from_tool = _build_two_link_arm_with_tool_site(device)

        ctrl = ControllerOperationalSpace(
            model,
            tool="tool_site",
            motion_stiffness=100.0,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_gravity_compensation=False,
        )
        inputs = ctrl.input()
        outputs = ctrl.output()

        theta1, theta2 = 0.3, -0.2
        inputs.joint_q.assign(np.array([theta1, theta2], dtype=np.float32))
        inputs.joint_qd.assign(np.zeros(2, dtype=np.float32))
        inputs.desired_tool_pose_operational.assign(np.zeros((1, 7), dtype=np.float32))
        inputs.desired_twist_operational.assign(np.zeros((1, 6), dtype=np.float32))

        combined_angle = theta1 + theta2
        body2_origin_world = np.array([np.cos(theta1), np.sin(theta1), 0.0])
        cos_c, sin_c = np.cos(combined_angle), np.sin(combined_angle)
        site_local = np.array(coordinate_change_body_from_tool)[:3]
        site_offset_world = np.array(
            [
                cos_c * site_local[0] - sin_c * site_local[1],
                sin_c * site_local[0] + cos_c * site_local[1],
                site_local[2],
            ]
        )
        expected_tool_position_world = body2_origin_world + site_offset_world
        expected_tool_orientation_world = np.array(
            [0.0, 0.0, np.sin(combined_angle / 2.0), np.cos(combined_angle / 2.0)]
        )

        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

        computed_tool_pose_world = ctrl._tool_pose_world.numpy()[0]
        np.testing.assert_allclose(computed_tool_pose_world[:3], expected_tool_position_world, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(computed_tool_pose_world[3:], expected_tool_orientation_world, rtol=1e-4, atol=1e-4)

    def test_step_output_is_zero_when_tool_is_already_at_the_desired_pose_and_still(self):
        """With zero pose error and zero twist error, step() commands zero joint torque.

        Task-space impedance (``use_inertia_decoupling=False``) computes the
        commanded force directly as ``Kp * pose_error + Kd * twist_error``, so
        driving both errors to exactly zero must drive the output to exactly
        zero, independent of the gain values themselves.
        """
        device = wp.get_device()
        model, _state, _tool_body, _transform = _build_two_link_arm_with_tool_site(device)

        ctrl = ControllerOperationalSpace(
            model,
            tool="tool_site",
            motion_stiffness=100.0,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_gravity_compensation=False,
        )
        inputs = ctrl.input()
        outputs = ctrl.output()

        inputs.joint_q.assign(np.array([0.3, -0.2], dtype=np.float32))
        inputs.joint_qd.assign(np.zeros(2, dtype=np.float32))
        inputs.desired_twist_operational.assign(np.zeros((1, 6), dtype=np.float32))

        # First step: read off the current tool pose so it can be fed back in
        # as the desired pose for a second step with exactly zero pose error.
        inputs.desired_tool_pose_operational.assign(np.zeros((1, 7), dtype=np.float32))
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        inputs.desired_tool_pose_operational.assign(ctrl._tool_pose_world.numpy())

        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

        np.testing.assert_allclose(outputs.joint_f.numpy(), np.zeros(2, dtype=np.float32), atol=1e-4)

    def test_step_gravity_compensation_matches_pendulum_formula(self):
        """step() gravity feedforward matches a hand-derived single-pendulum formula.

        With the joint at angle theta, the COM is at world position
        ``(L*cos(theta), 0, -L*sin(theta))`` (rotation about Y), so the
        gravitational potential energy is ``U(theta) = -m * (0, 0, -g) .
        com_world(theta) = -m*g*L*sin(theta)``, and the compensating joint
        torque is ``g(theta) = dU/dtheta = -m*g*L*cos(theta)``.

        Zero motion gains isolate the gravity feedforward term as the only
        contributor to the output torque.
        """
        device = wp.get_device()
        model, _state, _tool_body, mass, com_distance_from_joint = _build_single_link_pendulum_with_tool_site(device)

        ctrl = ControllerOperationalSpace(
            model,
            tool="tool_site",
            motion_stiffness=0.0,
            motion_damping=0.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_gravity_compensation=True,
        )
        inputs = ctrl.input()
        outputs = ctrl.output()

        theta = 0.4
        gravitational_acceleration = 9.81
        inputs.joint_q.assign(np.array([theta], dtype=np.float32))
        inputs.joint_qd.assign(np.zeros(1, dtype=np.float32))
        inputs.desired_twist_operational.assign(np.zeros((1, 6), dtype=np.float32))

        # Desired pose is irrelevant here since motion gains are zero, but
        # every field still has to be a valid transform.
        inputs.desired_tool_pose_operational.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32))

        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

        expected_gravity_torque = -mass * gravitational_acceleration * com_distance_from_joint * np.cos(theta)
        np.testing.assert_allclose(outputs.joint_f.numpy(), [expected_gravity_torque], rtol=1e-4, atol=1e-4)

    def test_step_controlled_mass_matrix_matches_model_mass_matrix(self):
        """step() gathers the controlled-DOF mass matrix correctly from the model's own eval_mass_matrix.

        For this fixture every DOF of the single articulation is controlled,
        in order, so the gathered ``(1, 6, 6)`` block must equal exactly the
        top-left ``6x6`` submatrix of the model's own mass matrix computed
        independently via the public :func:`newton.eval_mass_matrix`.
        """
        device = wp.get_device()
        model, state, _tool_body, _transform = _build_six_dof_arm_with_tool_site(device)

        ctrl = ControllerOperationalSpace(
            model,
            tool="tool_site",
            motion_stiffness=100.0,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=True,
            use_gravity_compensation=False,
        )
        inputs = ctrl.input()
        outputs = ctrl.output()

        joint_q = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.25], dtype=np.float32)
        inputs.joint_q.assign(joint_q)
        inputs.joint_qd.assign(np.zeros(6, dtype=np.float32))
        inputs.desired_twist_operational.assign(np.zeros((1, 6), dtype=np.float32))
        inputs.desired_tool_pose_operational.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32))

        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

        state.joint_q.assign(joint_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        expected_model_mass_matrix = newton.eval_mass_matrix(model, state).numpy()[0, :6, :6]

        np.testing.assert_allclose(
            ctrl._controlled_mass_matrix.numpy()[0], expected_model_mass_matrix, rtol=1e-4, atol=1e-4
        )

    def test_step_wrench_feedforward_matches_jacobian_transpose_force(self):
        """step() forwards inputs.desired_wrench_world through to tau = J^T @ wrench.

        Zero motion gains and an all-axes wrench selection isolate the
        wrench feedforward term as the only contributor, so the output must
        equal the tool Jacobian's transpose (already verified correct by
        the forward-kinematics test above) applied to the fixed wrench.
        """
        device = wp.get_device()
        model, _state, _tool_body, _transform = _build_two_link_arm_with_tool_site(device)

        ctrl = ControllerOperationalSpace(
            model,
            tool="tool_site",
            motion_stiffness=0.0,
            motion_damping=0.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_gravity_compensation=False,
            use_wrench_feedforward=True,
            wrench_selection_axes_tool=wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )
        inputs = ctrl.input()
        outputs = ctrl.output()

        inputs.joint_q.assign(np.array([0.3, -0.2], dtype=np.float32))
        inputs.joint_qd.assign(np.zeros(2, dtype=np.float32))
        inputs.desired_twist_operational.assign(np.zeros((1, 6), dtype=np.float32))
        inputs.desired_tool_pose_operational.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
        desired_wrench_world = np.array([3.0, -1.5, 0.0, 0.0, 0.0, 2.0], dtype=np.float32)
        inputs.desired_wrench_world.assign(np.array([desired_wrench_world]))

        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

        jacobian_tool_world = ctrl._jacobian_tool_world.numpy()[0]
        expected_joint_f = jacobian_tool_world.T @ desired_wrench_world
        np.testing.assert_allclose(outputs.joint_f.numpy(), expected_joint_f, rtol=1e-4, atol=1e-4)

    def test_step_null_space_output_is_zero_when_posture_is_already_at_the_desired_posture_and_still(self):
        """With zero task-space error and zero posture error, step() commands zero joint torque.

        Zero motion gains remove the primary task's contribution entirely.
        The posture PD term (``Kp * posture_error + Kd * posture_twist_error``)
        is then exactly zero because the desired posture equals the current
        one and both velocities are zero -- independent of the null-space
        projector itself, so this checks that ``joint_q_des_null`` is
        forwarded and compared against the correct (compact, controlled-DOF)
        current posture, not the projector math.
        """
        device = wp.get_device()
        model, _state, _tool_body, _transform = _build_seven_dof_arm_with_tool_site(device)

        ctrl = ControllerOperationalSpace(
            model,
            tool="tool_site",
            motion_stiffness=0.0,
            motion_damping=0.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_gravity_compensation=False,
            use_null_space_control=True,
            null_space_stiffness=100.0,
            null_space_damping=10.0,
        )
        inputs = ctrl.input()
        outputs = ctrl.output()

        joint_q = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.25, 0.2], dtype=np.float32)
        inputs.joint_q.assign(joint_q)
        inputs.joint_qd.assign(np.zeros(7, dtype=np.float32))
        inputs.desired_twist_operational.assign(np.zeros((1, 6), dtype=np.float32))
        inputs.desired_tool_pose_operational.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
        # Every joint of this fixture is controlled, in order, so the compact
        # posture target is exactly the same array as the model-space joint_q.
        inputs.joint_q_des_null.assign(joint_q)
        inputs.joint_qd_des_null.assign(np.zeros(7, dtype=np.float32))

        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

        np.testing.assert_allclose(outputs.joint_f.numpy(), np.zeros(7, dtype=np.float32), atol=1e-4)

    def test_step_raises_when_wrench_port_written_but_wrench_control_disabled(self):
        """step() raises, rather than silently ignoring, a wrench port set on a controller built without it."""
        device = wp.get_device()
        model, _state, _tool_body, _transform = _build_two_link_arm_with_tool_site(device)

        ctrl = ControllerOperationalSpace(
            model,
            tool="tool_site",
            motion_stiffness=100.0,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_gravity_compensation=False,
        )
        inputs = ctrl.input()
        inputs.joint_q.assign(np.array([0.3, -0.2], dtype=np.float32))
        inputs.joint_qd.assign(np.zeros(2, dtype=np.float32))
        inputs.desired_twist_operational.assign(np.zeros((1, 6), dtype=np.float32))
        inputs.desired_tool_pose_operational.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
        inputs.desired_wrench_world = wp.zeros(1, dtype=wp.spatial_vector, device=device)

        with self.assertRaises(ValueError):
            ctrl.step(inputs=inputs, outputs=ctrl.output(), dt=0.01)

    def test_step_output_is_superposition_of_each_feature_run_independently(self):
        """With every feature's law computed independently (via _add_term_kernel), the combined output sums.

        Builds four controllers sharing the same joint configuration on the
        redundant 7-DOF arm: motion only, wrench only, null-space-posture
        only, and all three together. Since inertial decoupling is off
        (task-space impedance) and the null-space projector depends only on
        the (identical, across all four) Jacobian, each term's law doesn't
        depend on whether the others are enabled, so the combined controller's
        output must equal the exact sum of the three isolated ones.
        """
        device = wp.get_device()
        model, _state, _tool_body, _transform = _build_seven_dof_arm_with_tool_site(device)

        joint_q = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.25, 0.2], dtype=np.float32)
        joint_qd = np.zeros(7, dtype=np.float32)
        desired_tool_pose_operational = np.array([[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        desired_twist_operational = np.zeros((1, 6), dtype=np.float32)
        desired_wrench_world = np.array([[3.0, -1.5, 0.0, 0.0, 0.0, 2.0]], dtype=np.float32)
        joint_q_des_null = np.array([0.0, 0.1, -0.1, 0.2, -0.2, 0.0, 0.15], dtype=np.float32)
        joint_qd_des_null = np.zeros(7, dtype=np.float32)
        full_selection = wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        def run(*, use_motion, use_wrench, use_null_space):
            ctrl = ControllerOperationalSpace(
                model,
                tool="tool_site",
                motion_stiffness=100.0 if use_motion else 0.0,
                motion_damping=10.0 if use_motion else 0.0,
                operational_frame_pose_world=_IDENTITY_TRANSFORM,
                use_inertia_decoupling=False,
                use_gravity_compensation=False,
                use_wrench_feedforward=use_wrench,
                wrench_selection_axes_tool=full_selection if use_wrench else None,
                use_null_space_control=use_null_space,
                null_space_stiffness=100.0 if use_null_space else None,
                null_space_damping=10.0 if use_null_space else None,
            )
            inputs = ctrl.input()
            outputs = ctrl.output()
            inputs.joint_q.assign(joint_q)
            inputs.joint_qd.assign(joint_qd)
            inputs.desired_tool_pose_operational.assign(desired_tool_pose_operational)
            inputs.desired_twist_operational.assign(desired_twist_operational)
            if use_wrench:
                inputs.desired_wrench_world.assign(desired_wrench_world)
            if use_null_space:
                inputs.joint_q_des_null.assign(joint_q_des_null)
                inputs.joint_qd_des_null.assign(joint_qd_des_null)
            ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
            return outputs.joint_f.numpy()

        tau_motion = run(use_motion=True, use_wrench=False, use_null_space=False)
        tau_wrench = run(use_motion=False, use_wrench=True, use_null_space=False)
        tau_null = run(use_motion=False, use_wrench=False, use_null_space=True)
        tau_all = run(use_motion=True, use_wrench=True, use_null_space=True)

        np.testing.assert_allclose(tau_all, tau_motion + tau_wrench + tau_null, rtol=1e-4, atol=1e-4)

    def test_step_partial_wrench_selection_is_rotated_by_the_resolved_tool_orientation(self):
        """A partial (non-full) wrench selection is rotated using step()'s own FK-resolved tool orientation.

        Only the tool-local X axis is force-controlled here; every other
        axis defaults to motion-controlled. With zero pose and twist error,
        the motion term is exactly zero everywhere regardless of selection,
        isolating the wrench term as the only contributor. Since the arm's
        joints all rotate about world Z, the tool's world orientation is a
        pure Z rotation by ``theta1 + theta2`` -- already verified correct by
        ``test_step_resolves_tool_pose_matching_forward_kinematics`` -- which
        this test reuses directly to build the expected rotated selection
        matrix, independent of :func:`_rotate_selection_matrix_kernel`.
        """
        device = wp.get_device()
        model, _state, _tool_body, _transform = _build_two_link_arm_with_tool_site(device)

        ctrl = ControllerOperationalSpace(
            model,
            tool="tool_site",
            motion_stiffness=100.0,
            motion_damping=10.0,
            operational_frame_pose_world=_IDENTITY_TRANSFORM,
            use_inertia_decoupling=False,
            use_gravity_compensation=False,
            use_wrench_feedforward=True,
            wrench_selection_axes_tool=wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        inputs = ctrl.input()
        outputs = ctrl.output()

        theta1, theta2 = 0.3, -0.2
        inputs.joint_q.assign(np.array([theta1, theta2], dtype=np.float32))
        inputs.joint_qd.assign(np.zeros(2, dtype=np.float32))
        inputs.desired_twist_operational.assign(np.zeros((1, 6), dtype=np.float32))
        desired_wrench_world = np.array([3.0, -1.5, 0.4, 0.0, 0.0, 2.0], dtype=np.float32)
        inputs.desired_wrench_world.assign(np.array([desired_wrench_world]))

        # First step to read off the current tool pose, so the second step
        # (the one actually measured) has exactly zero pose error.
        inputs.desired_tool_pose_operational.assign(np.zeros((1, 7), dtype=np.float32))
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        inputs.desired_tool_pose_operational.assign(ctrl._tool_pose_world.numpy())

        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

        combined_angle = theta1 + theta2
        cos_c, sin_c = np.cos(combined_angle), np.sin(combined_angle)
        world_from_tool_rotation = np.array([[cos_c, -sin_c, 0.0], [sin_c, cos_c, 0.0], [0.0, 0.0, 1.0]])
        # Local weight is 1 on linear X only, 0 everywhere else, so the
        # angular block of the rotated selection matrix is exactly zero.
        local_linear_selection = np.diag([1.0, 0.0, 0.0])
        world_linear_selection = world_from_tool_rotation @ local_linear_selection @ world_from_tool_rotation.T

        selected_wrench_world = np.zeros(6, dtype=np.float32)
        selected_wrench_world[:3] = world_linear_selection @ desired_wrench_world[:3]

        jacobian_tool_world = ctrl._jacobian_tool_world.numpy()[0]
        expected_joint_f = jacobian_tool_world.T @ selected_wrench_world
        np.testing.assert_allclose(outputs.joint_f.numpy(), expected_joint_f, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
