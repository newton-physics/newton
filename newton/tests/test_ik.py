# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from newton._src.sim.ik import _eval_fk_batched
from newton.tests.unittest_utils import (
    add_function_test,
    assert_np_equal,
    get_selected_cuda_test_devices,
    get_test_devices,
)

# ----------------------------------------------------------------------------
# helpers: planar 2-revolute baseline
# ----------------------------------------------------------------------------


def _build_two_link_planar(device) -> newton.Model:
    """Returns a singleton model with one 2-DOF planar arm."""
    builder = newton.ModelBuilder()

    link1 = builder.add_link(
        xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
        mass=1.0,
    )
    joint1 = builder.add_joint_revolute(
        parent=-1,
        child=link1,
        parent_xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
        child_xform=wp.transform([-0.5, 0.0, 0.0], wp.quat_identity()),
        axis=[0.0, 0.0, 1.0],
    )

    link2 = builder.add_link(
        xform=wp.transform([1.5, 0.0, 0.0], wp.quat_identity()),
        mass=1.0,
    )
    joint2 = builder.add_joint_revolute(
        parent=link1,
        child=link2,
        parent_xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
        child_xform=wp.transform([-0.5, 0.0, 0.0], wp.quat_identity()),
        axis=[0.0, 0.0, 1.0],
    )
    
    # Create articulation from joints
    builder.add_articulation([joint1, joint2])

    model = builder.finalize(device=device, requires_grad=True)
    return model


# ----------------------------------------------------------------------------
# helpers - FREE-REV
# ----------------------------------------------------------------------------


def _build_free_plus_revolute(device) -> newton.Model:
    """
    Returns a model whose root link is attached with a FREE joint
    followed by one REV link.
    """
    builder = newton.ModelBuilder()

    link1 = builder.add_link(
        xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
        mass=1.0,
    )
    joint1 = builder.add_joint_free(
        parent=-1,
        child=link1,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )

    link2 = builder.add_link(
        xform=wp.transform([1.0, 0.0, 0.0], wp.quat_identity()),
        mass=1.0,
    )
    joint2 = builder.add_joint_revolute(
        parent=link1,
        child=link2,
        parent_xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
        child_xform=wp.transform([-0.5, 0.0, 0.0], wp.quat_identity()),
        axis=[0.0, 0.0, 1.0],
    )
    
    # Create articulation from joints
    builder.add_articulation([joint1, joint2])

    model = builder.finalize(device=device, requires_grad=True)
    return model


# ----------------------------------------------------------------------------
# helpers - D6
# ----------------------------------------------------------------------------


def _build_single_d6(device) -> newton.Model:
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.JointDofConfig
    link = builder.add_link(xform=wp.transform_identity(), mass=1.0)
    joint = builder.add_joint_d6(
        parent=-1,
        child=link,
        linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y), cfg(axis=newton.Axis.Z)],
        angular_axes=[cfg(axis=[1, 0, 0]), cfg(axis=[0, 1, 0]), cfg(axis=[0, 0, 1])],
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    # Create articulation from the joint
    builder.add_articulation([joint])
    return builder.finalize(device=device, requires_grad=True)


# ----------------------------------------------------------------------------
# common FK utility
# ----------------------------------------------------------------------------


def _fk_end_effector_positions(
    model: newton.Model, body_q_2d: wp.array, n_problems: int, ee_link_index: int, ee_offset: wp.vec3
) -> np.ndarray:
    """Returns an (N,3) array with end-effector world positions for every problem."""
    positions = np.zeros((n_problems, 3), dtype=np.float32)
    body_q_np = body_q_2d.numpy()  # shape: [n_problems, model.body_count]

    for prob in range(n_problems):
        body_tf = body_q_np[prob, ee_link_index]
        pos = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
        rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
        ee_world = wp.transform_point(wp.transform(pos, rot), ee_offset)
        positions[prob] = [ee_world[0], ee_world[1], ee_world[2]]
    return positions


# ----------------------------------------------------------------------------
# 1.  Convergence tests
# ----------------------------------------------------------------------------


def _convergence_test_planar(test, device, mode: ik.IKJacobianMode):
    with wp.ScopedDevice(device):
        n_problems = 3
        model = _build_two_link_planar(device)

        # Create 2D joint_q array [n_problems, joint_coord_count]
        requires_grad = mode in [ik.IKJacobianMode.AUTODIFF, ik.IKJacobianMode.MIXED]
        joint_q_2d = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=requires_grad)

        # Create 2D joint_qd array [n_problems, joint_dof_count]
        joint_qd_2d = wp.zeros((n_problems, model.joint_dof_count), dtype=wp.float32)

        # Create 2D body arrays for output
        body_q_2d = wp.zeros((n_problems, model.body_count), dtype=wp.transform)
        body_qd_2d = wp.zeros((n_problems, model.body_count), dtype=wp.spatial_vector)

        # simple reachable XY targets
        targets = wp.array([[1.5, 1.0, 0.0], [1.5, 1.0, 0.0], [1.5, 1.0, 0.0]], dtype=wp.vec3)
        ee_link = 1
        ee_off = wp.vec3(0.5, 0.0, 0.0)

        pos_obj = ik.IKPositionObjective(
            link_index=ee_link,
            link_offset=ee_off,
            target_positions=targets,
            n_problems=n_problems,
            total_residuals=3,
            residual_offset=0,
        )

        solver = ik.IKSolver(model, joint_q_2d, [pos_obj], lambda_initial=1e-3, jacobian_mode=mode)

        # Run initial FK
        _eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        initial = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)

        solver.solve(iterations=40)

        # Run final FK
        _eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        final = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)

        for prob in range(n_problems):
            err0 = np.linalg.norm(initial[prob] - targets.numpy()[prob])
            err1 = np.linalg.norm(final[prob] - targets.numpy()[prob])
            test.assertLess(err1, err0, f"mode {mode} problem {prob} did not improve")
            test.assertLess(err1, 1e-4, f"mode {mode} problem {prob} final error too high ({err1:.3f})")


def test_convergence_autodiff(test, device):
    _convergence_test_planar(test, device, ik.IKJacobianMode.AUTODIFF)


def test_convergence_analytic(test, device):
    _convergence_test_planar(test, device, ik.IKJacobianMode.ANALYTIC)


def test_convergence_mixed(test, device):
    _convergence_test_planar(test, device, ik.IKJacobianMode.MIXED)


def _convergence_test_free(test, device, mode: ik.IKJacobianMode):
    with wp.ScopedDevice(device):
        n_problems = 3
        model = _build_free_plus_revolute(device)

        requires_grad = mode in [ik.IKJacobianMode.AUTODIFF, ik.IKJacobianMode.MIXED]
        joint_q_2d = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=requires_grad)
        joint_qd_2d = wp.zeros((n_problems, model.joint_dof_count), dtype=wp.float32)
        body_q_2d = wp.zeros((n_problems, model.body_count), dtype=wp.transform)
        body_qd_2d = wp.zeros((n_problems, model.body_count), dtype=wp.spatial_vector)

        targets = wp.array([[1.0, 1.0, 0.0]] * n_problems, dtype=wp.vec3)
        ee_link = 1  # second body
        ee_off = wp.vec3(0.5, 0.0, 0.0)

        pos_obj = ik.IKPositionObjective(
            link_index=ee_link,
            link_offset=ee_off,
            target_positions=targets,
            n_problems=n_problems,
            total_residuals=3,
            residual_offset=0,
        )

        solver = ik.IKSolver(model, joint_q_2d, [pos_obj], lambda_initial=1e-3, jacobian_mode=mode)

        _eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        initial = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)

        solver.solve(iterations=60)

        _eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        final = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)

        for prob in range(n_problems):
            err0 = np.linalg.norm(initial[prob] - targets.numpy()[prob])
            err1 = np.linalg.norm(final[prob] - targets.numpy()[prob])
            test.assertLess(err1, err0, f"[FREE] mode {mode} problem {prob} did not improve")
            test.assertLess(err1, 1e-3, f"[FREE] mode {mode} problem {prob} final error too high ({err1:.3f})")


def test_convergence_autodiff_free(test, device):
    _convergence_test_free(test, device, ik.IKJacobianMode.AUTODIFF)


def test_convergence_analytic_free(test, device):
    _convergence_test_free(test, device, ik.IKJacobianMode.ANALYTIC)


def test_convergence_mixed_free(test, device):
    _convergence_test_free(test, device, ik.IKJacobianMode.MIXED)


def _convergence_test_d6(test, device, mode: ik.IKJacobianMode):
    with wp.ScopedDevice(device):
        n_problems = 3
        model = _build_single_d6(device)
        requires_grad = mode in [ik.IKJacobianMode.AUTODIFF, ik.IKJacobianMode.MIXED]
        joint_q_2d = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=requires_grad)
        joint_qd_2d = wp.zeros((n_problems, model.joint_dof_count), dtype=wp.float32)
        body_q_2d = wp.zeros((n_problems, model.body_count), dtype=wp.transform)
        body_qd_2d = wp.zeros((n_problems, model.body_count), dtype=wp.spatial_vector)

        pos_targets = wp.array([[0.2, 0.3, 0.1]] * n_problems, dtype=wp.vec3)
        angles = [math.pi / 6 + prob * math.pi / 8 for prob in range(n_problems)]
        rot_targets = wp.array([[0.0, 0.0, math.sin(a / 2), math.cos(a / 2)] for a in angles], dtype=wp.vec4)

        pos_obj = ik.IKPositionObjective(0, wp.vec3(0.0, 0.0, 0.0), pos_targets, n_problems, 6, 0)
        rot_obj = ik.IKRotationObjective(0, wp.quat_identity(), rot_targets, n_problems, 6, 3)

        solver = ik.IKSolver(model, joint_q_2d, [pos_obj, rot_obj], lambda_initial=1e-3, jacobian_mode=mode)

        _eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        initial = _fk_end_effector_positions(model, body_q_2d, n_problems, 0, wp.vec3(0.0, 0.0, 0.0))

        solver.solve(iterations=80)

        _eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        final = _fk_end_effector_positions(model, body_q_2d, n_problems, 0, wp.vec3(0.0, 0.0, 0.0))

        for prob in range(n_problems):
            err0 = np.linalg.norm(initial[prob] - pos_targets.numpy()[prob])
            err1 = np.linalg.norm(final[prob] - pos_targets.numpy()[prob])
            test.assertLess(err1, err0)
            test.assertLess(err1, 1e-3)


def test_convergence_autodiff_d6(test, device):
    _convergence_test_d6(test, device, ik.IKJacobianMode.AUTODIFF)


def test_convergence_analytic_d6(test, device):
    _convergence_test_d6(test, device, ik.IKJacobianMode.ANALYTIC)


def test_convergence_mixed_d6(test, device):
    _convergence_test_d6(test, device, ik.IKJacobianMode.MIXED)


# ----------------------------------------------------------------------------
# 2.  Jacobian equality helpers
# ----------------------------------------------------------------------------


def _jacobian_compare(test, device, objective_builder):
    """Build autodiff + analytic solvers for the same objective(s) and compare J."""
    with wp.ScopedDevice(device):
        n_problems = 3
        model = _build_two_link_planar(device)

        # Create 2D joint_q array [n_problems, joint_coord_count]
        joint_q_2d = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=True)

        objectives = objective_builder(model, n_problems)

        solver_auto = ik.IKSolver(model, joint_q_2d, objectives, jacobian_mode=ik.IKJacobianMode.AUTODIFF)
        solver_ana = ik.IKSolver(model, joint_q_2d, objectives, jacobian_mode=ik.IKJacobianMode.ANALYTIC)

        solver_auto.compute_residuals()
        solver_ana.compute_residuals()

        J_auto = solver_auto.compute_jacobian().numpy()
        J_ana = solver_ana.compute_jacobian().numpy()

        assert_np_equal(J_auto, J_ana, tol=1e-4)


# ----------------------------------------------------------------------------
# 2a.  Position Jacobian
# ----------------------------------------------------------------------------


def _pos_objective_builder(model, n_problems):
    targets = wp.array([[1.5, 0.8, 0.0] for _ in range(n_problems)], dtype=wp.vec3)
    pos_obj = ik.IKPositionObjective(
        link_index=1,
        link_offset=wp.vec3(0.5, 0.0, 0.0),
        target_positions=targets,
        n_problems=n_problems,
        total_residuals=3,
        residual_offset=0,
    )
    return [pos_obj]


def test_position_jacobian_compare(test, device):
    _jacobian_compare(test, device, _pos_objective_builder)


# ----------------------------------------------------------------------------
# 2b.  Rotation Jacobian
# ----------------------------------------------------------------------------


def _rot_objective_builder(model, n_problems):
    angles = [math.pi / 6 + prob * math.pi / 8 for prob in range(n_problems)]
    quats = [[0.0, 0.0, math.sin(a / 2), math.cos(a / 2)] for a in angles]
    rot_obj = ik.IKRotationObjective(
        link_index=1,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array(quats, dtype=wp.vec4),
        n_problems=n_problems,
        total_residuals=3,
        residual_offset=0,
    )
    return [rot_obj]


def test_rotation_jacobian_compare(test, device):
    _jacobian_compare(test, device, _rot_objective_builder)


# ----------------------------------------------------------------------------
# 2c.  Joint-limit Jacobian
# ----------------------------------------------------------------------------


def _jl_objective_builder(model, n_problems):
    # Joint limits for singleton model
    dof = model.joint_coord_count
    joint_limit_lower = wp.array([-1.0] * dof, dtype=wp.float32)
    joint_limit_upper = wp.array([1.0] * dof, dtype=wp.float32)

    jl_obj = ik.IKJointLimitObjective(
        joint_limit_lower=joint_limit_lower,
        joint_limit_upper=joint_limit_upper,
        n_problems=n_problems,
        total_residuals=dof,
        residual_offset=0,
        weight=0.1,
    )
    return [jl_obj]


def test_joint_limit_jacobian_compare(test, device):
    _jacobian_compare(test, device, _jl_objective_builder)


# ----------------------------------------------------------------------------
# 2d.  D6 jacobian
# ----------------------------------------------------------------------------


def _d6_objective_builder(model, n_problems):
    pos_targets = wp.array([[0.2, 0.3, 0.1]] * n_problems, dtype=wp.vec3)
    angles = [math.pi / 6 + prob * math.pi / 8 for prob in range(n_problems)]
    rot_targets = wp.array([[0.0, 0.0, math.sin(a / 2), math.cos(a / 2)] for a in angles], dtype=wp.vec4)

    pos_obj = ik.IKPositionObjective(0, wp.vec3(0.0, 0.0, 0.0), pos_targets, n_problems, 6, 0)
    rot_obj = ik.IKRotationObjective(0, wp.quat_identity(), rot_targets, n_problems, 6, 3)
    return [pos_obj, rot_obj]


def test_d6_jacobian_compare(test, device):
    _jacobian_compare(test, device, _d6_objective_builder)


# ----------------------------------------------------------------------------
# 3.  Test-class registration per device
# ----------------------------------------------------------------------------

devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()


class TestIKModes(unittest.TestCase):
    pass


# Planar REV-REV convergence
add_function_test(TestIKModes, "test_convergence_autodiff", test_convergence_autodiff, devices)
add_function_test(TestIKModes, "test_convergence_analytic", test_convergence_analytic, devices)
add_function_test(TestIKModes, "test_convergence_mixed", test_convergence_mixed, devices)

# FREE-joint convergence
add_function_test(TestIKModes, "test_convergence_autodiff_free", test_convergence_autodiff_free, devices)
add_function_test(TestIKModes, "test_convergence_analytic_free", test_convergence_analytic_free, devices)
add_function_test(TestIKModes, "test_convergence_mixed_free", test_convergence_mixed_free, devices)

# D6-joint convergence
add_function_test(TestIKModes, "test_convergence_autodiff_d6", test_convergence_autodiff_d6, cuda_devices)
add_function_test(TestIKModes, "test_convergence_analytic_d6", test_convergence_analytic_d6, devices)
add_function_test(TestIKModes, "test_convergence_mixed_d6", test_convergence_mixed_d6, devices)

# Jacobian equality
add_function_test(TestIKModes, "test_position_jacobian_compare", test_position_jacobian_compare, devices)
add_function_test(TestIKModes, "test_rotation_jacobian_compare", test_rotation_jacobian_compare, cuda_devices)
add_function_test(TestIKModes, "test_joint_limit_jacobian_compare", test_joint_limit_jacobian_compare, devices)
add_function_test(TestIKModes, "test_d6_jacobian_compare", test_d6_jacobian_compare, cuda_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
