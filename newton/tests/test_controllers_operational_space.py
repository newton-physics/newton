# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the operational-space controller family.

Math kernels are tested standalone first, independent of any Controller class,
following the pattern in ``test_jacobian_mass_matrix.py``. Controller-level
tests are added once the surrounding ``Controller`` classes exist.

Each test launches the two kernels directly, with no shared launch helper, so
what's being computed and checked is visible in the test itself rather than
behind an indirection.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.controllers.impl.operational_space._common import (
    _invert_spd_block_kernel,
    _operational_space_mass_matrix_inverse_kernel,
    _shift_jacobian_to_tool_kernel,
    _tool_pose_and_twist_kernel,
)
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

    # Shift the Jacobian to the tool point (the Chunk 1 kernel).
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
    coordinate_change_world_from_tool = wp.zeros(1, dtype=wp.transform, device=device)
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
        outputs=[coordinate_change_world_from_tool, tool_twist_world],
        device=device,
    )

    # Expected: compose the body's FK pose with the tool site's fixed offset by hand.
    coordinate_change_world_from_body = wp.transform(*state.body_q.numpy()[tool_body])
    expected = coordinate_change_world_from_body * coordinate_change_body_from_tool

    np.testing.assert_allclose(coordinate_change_world_from_tool.numpy()[0], np.array(expected), atol=1e-6)


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
    coordinate_change_world_from_tool = wp.zeros(1, dtype=wp.transform, device=device)
    tool_twist_world = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(
        _tool_pose_and_twist_kernel,
        dim=1,
        inputs=[state.body_q, state.body_qd, model.body_com, tool_body_arr, coordinate_change_body_from_tool_arr],
        outputs=[coordinate_change_world_from_tool, tool_twist_world],
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
    coordinate_change_world_from_tool = wp.zeros(1, dtype=wp.transform, device=device)
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
        outputs=[coordinate_change_world_from_tool, tool_twist_world],
        device=device,
    )

    # Compare: the tool twist's angular part (last 3 components) should be unchanged.
    np.testing.assert_allclose(tool_twist_world.numpy()[0][3:], body_twist_com_world[3:], atol=1e-6)


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


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
