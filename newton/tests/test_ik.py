# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from newton._src.sim.ik.ik_common import eval_fk_batched
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


def _build_descendant_free_distance(device, joint_type) -> tuple[newton.Model, int]:
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)
    base = builder.add_link(mass=1.0)
    child = builder.add_link(mass=1.0)
    builder.body_com[child] = wp.vec3(0.21, -0.07, 0.16)

    root = builder.add_joint_revolute(
        parent=-1,
        child=base,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(
            wp.vec3(0.2, -0.1, 0.3),
            wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, -0.2, 1.0)), 0.55),
        ),
        child_xform=wp.transform_identity(),
    )
    child_joint = _add_free_distance_joint(
        builder=builder,
        joint_type=joint_type,
        parent=base,
        child=child,
        parent_xform=wp.transform(
            wp.vec3(0.7, -0.2, 0.4),
            wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.2, 1.0, -0.3)), 0.7),
        ),
        child_xform=wp.transform(
            wp.vec3(0.15, -0.05, 0.2),
            wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, -0.2, 0.4)), -0.9),
        ),
    )
    builder.add_articulation([root, child_joint])
    return builder.finalize(device=device, requires_grad=True), child


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


def _convergence_test_planar(test, device, mode: ik.IKJacobianType):
    with wp.ScopedDevice(device):
        n_problems = 3
        model = _build_two_link_planar(device)

        # Create 2D joint_q array [n_problems, joint_coord_count]
        requires_grad = mode in [ik.IKJacobianType.AUTODIFF, ik.IKJacobianType.MIXED]
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

        pos_obj = ik.IKObjectivePosition(
            link_index=ee_link,
            link_offset=ee_off,
            target_positions=targets,
        )

        solver = ik.IKSolver(model, n_problems, [pos_obj], lambda_initial=1e-3, jacobian_mode=mode)

        # Run initial FK
        eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        initial = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)

        solver.step(joint_q_2d, joint_q_2d, iterations=40, step_size=1.0)

        # Run final FK
        eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        final = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)

        for prob in range(n_problems):
            err0 = np.linalg.norm(initial[prob] - targets.numpy()[prob])
            err1 = np.linalg.norm(final[prob] - targets.numpy()[prob])
            test.assertLess(err1, err0, f"mode {mode} problem {prob} did not improve")
            test.assertLess(err1, 1e-4, f"mode {mode} problem {prob} final error too high ({err1:.3f})")


def test_convergence_autodiff(test, device):
    _convergence_test_planar(test, device, ik.IKJacobianType.AUTODIFF)


def test_convergence_analytic(test, device):
    _convergence_test_planar(test, device, ik.IKJacobianType.ANALYTIC)


def test_convergence_mixed(test, device):
    _convergence_test_planar(test, device, ik.IKJacobianType.MIXED)


def _convergence_test_free(test, device, mode: ik.IKJacobianType):
    with wp.ScopedDevice(device):
        n_problems = 3
        model = _build_free_plus_revolute(device)

        requires_grad = mode in [ik.IKJacobianType.AUTODIFF, ik.IKJacobianType.MIXED]
        joint_q_2d = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=requires_grad)
        joint_qd_2d = wp.zeros((n_problems, model.joint_dof_count), dtype=wp.float32)
        body_q_2d = wp.zeros((n_problems, model.body_count), dtype=wp.transform)
        body_qd_2d = wp.zeros((n_problems, model.body_count), dtype=wp.spatial_vector)

        targets = wp.array([[1.0, 1.0, 0.0]] * n_problems, dtype=wp.vec3)
        ee_link = 1  # second body
        ee_off = wp.vec3(0.5, 0.0, 0.0)

        pos_obj = ik.IKObjectivePosition(
            link_index=ee_link,
            link_offset=ee_off,
            target_positions=targets,
        )

        solver = ik.IKSolver(model, n_problems, [pos_obj], lambda_initial=1e-3, jacobian_mode=mode)

        eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        initial = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)

        solver.step(joint_q_2d, joint_q_2d, iterations=60, step_size=1.0)

        eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        final = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)

        for prob in range(n_problems):
            err0 = np.linalg.norm(initial[prob] - targets.numpy()[prob])
            err1 = np.linalg.norm(final[prob] - targets.numpy()[prob])
            test.assertLess(err1, err0, f"[FREE] mode {mode} problem {prob} did not improve")
            test.assertLess(err1, 1e-3, f"[FREE] mode {mode} problem {prob} final error too high ({err1:.3f})")


def test_convergence_autodiff_free(test, device):
    _convergence_test_free(test, device, ik.IKJacobianType.AUTODIFF)


def test_convergence_analytic_free(test, device):
    _convergence_test_free(test, device, ik.IKJacobianType.ANALYTIC)


def test_convergence_mixed_free(test, device):
    _convergence_test_free(test, device, ik.IKJacobianType.MIXED)


def _convergence_test_d6(test, device, mode: ik.IKJacobianType):
    with wp.ScopedDevice(device):
        n_problems = 3
        model = _build_single_d6(device)
        requires_grad = mode in [ik.IKJacobianType.AUTODIFF, ik.IKJacobianType.MIXED]
        joint_q_2d = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=requires_grad)
        joint_qd_2d = wp.zeros((n_problems, model.joint_dof_count), dtype=wp.float32)
        body_q_2d = wp.zeros((n_problems, model.body_count), dtype=wp.transform)
        body_qd_2d = wp.zeros((n_problems, model.body_count), dtype=wp.spatial_vector)

        pos_targets = wp.array([[0.2, 0.3, 0.1]] * n_problems, dtype=wp.vec3)
        angles = [math.pi / 6 + prob * math.pi / 8 for prob in range(n_problems)]
        rot_targets = wp.array([[0.0, 0.0, math.sin(a / 2), math.cos(a / 2)] for a in angles], dtype=wp.vec4)

        pos_obj = ik.IKObjectivePosition(
            link_index=0,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=pos_targets,
        )
        rot_obj = ik.IKObjectiveRotation(
            link_index=0,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=rot_targets,
        )

        solver = ik.IKSolver(model, n_problems, [pos_obj, rot_obj], lambda_initial=1e-3, jacobian_mode=mode)

        eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        initial = _fk_end_effector_positions(model, body_q_2d, n_problems, 0, wp.vec3(0.0, 0.0, 0.0))

        solver.step(joint_q_2d, joint_q_2d, iterations=80, step_size=1.0)

        eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        final = _fk_end_effector_positions(model, body_q_2d, n_problems, 0, wp.vec3(0.0, 0.0, 0.0))

        for prob in range(n_problems):
            err0 = np.linalg.norm(initial[prob] - pos_targets.numpy()[prob])
            err1 = np.linalg.norm(final[prob] - pos_targets.numpy()[prob])
            test.assertLess(err1, err0)
            test.assertLess(err1, 1e-3)


def test_convergence_autodiff_d6(test, device):
    _convergence_test_d6(test, device, ik.IKJacobianType.AUTODIFF)


def test_convergence_analytic_d6(test, device):
    _convergence_test_d6(test, device, ik.IKJacobianType.ANALYTIC)


def test_convergence_mixed_d6(test, device):
    _convergence_test_d6(test, device, ik.IKJacobianType.MIXED)


def test_convergence_analytic_descendant_free_distance(test, device, joint_type):
    with wp.ScopedDevice(device):
        n_problems = 2
        model, ee_link = _build_descendant_free_distance(device, joint_type)
        joint_q_2d = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=False)
        joint_qd_2d = wp.zeros((n_problems, model.joint_dof_count), dtype=wp.float32)
        body_q_2d = wp.zeros((n_problems, model.body_count), dtype=wp.transform)
        body_qd_2d = wp.zeros((n_problems, model.body_count), dtype=wp.spatial_vector)

        q_np = joint_q_2d.numpy()
        q_start = model.joint_q_start.numpy()
        child_start = q_start[1]
        root_angles = [0.35, -0.28]
        child_translations = [
            np.array([0.24, -0.17, 0.12], dtype=np.float32),
            np.array([-0.18, 0.11, 0.16], dtype=np.float32),
        ]
        child_axes = [wp.normalize(wp.vec3(1.0, 0.3, -0.2)), wp.normalize(wp.vec3(-0.4, 0.8, 0.5))]
        child_angles = [0.42, -0.31]
        for prob in range(n_problems):
            q_np[prob, 0] = root_angles[prob]
            q_np[prob, child_start : child_start + 3] = child_translations[prob]
            child_rot = wp.quat_from_axis_angle(child_axes[prob], child_angles[prob])
            q_np[prob, child_start + 3 : child_start + 7] = np.array(
                [child_rot[0], child_rot[1], child_rot[2], child_rot[3]],
                dtype=np.float32,
            )
        joint_q_2d.assign(q_np)

        ee_off = wp.vec3(0.08, -0.04, 0.06)
        eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        initial_pos = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)
        body_q_np = body_q_2d.numpy()

        pos_targets = initial_pos + np.array([[0.16, -0.09, 0.12], [-0.11, 0.08, -0.07]], dtype=np.float32)
        rot_targets = np.zeros((n_problems, 4), dtype=np.float32)
        rot_axes = [wp.normalize(wp.vec3(0.5, -0.1, 0.8)), wp.normalize(wp.vec3(-0.2, 0.9, 0.3))]
        rot_angles = [0.33, -0.27]
        initial_rot = []
        for prob in range(n_problems):
            q_init = wp.quat(*body_q_np[prob, ee_link, 3:7])
            initial_rot.append(np.array([q_init[0], q_init[1], q_init[2], q_init[3]], dtype=np.float32))
            q_target = wp.normalize(wp.quat_from_axis_angle(rot_axes[prob], rot_angles[prob]) * q_init)
            rot_targets[prob] = np.array([q_target[0], q_target[1], q_target[2], q_target[3]], dtype=np.float32)

        pos_obj = ik.IKObjectivePosition(ee_link, ee_off, wp.array(pos_targets, dtype=wp.vec3))
        rot_obj = ik.IKObjectiveRotation(ee_link, wp.quat_identity(), wp.array(rot_targets, dtype=wp.vec4))
        solver = ik.IKSolver(
            model,
            n_problems,
            [pos_obj, rot_obj],
            lambda_initial=1e-3,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        solver.step(joint_q_2d, joint_q_2d, iterations=70, step_size=1.0)

        eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        final_pos = _fk_end_effector_positions(model, body_q_2d, n_problems, ee_link, ee_off)
        final_q_np = body_q_2d.numpy()
        for prob in range(n_problems):
            pos_err_0 = np.linalg.norm(initial_pos[prob] - pos_targets[prob])
            pos_err_1 = np.linalg.norm(final_pos[prob] - pos_targets[prob])
            rot_err_0 = 2.0 * math.acos(np.clip(abs(np.dot(initial_rot[prob], rot_targets[prob])), 0.0, 1.0))
            rot_err_1 = 2.0 * math.acos(
                np.clip(abs(np.dot(final_q_np[prob, ee_link, 3:7], rot_targets[prob])), 0.0, 1.0)
            )
            test.assertLess(pos_err_1, pos_err_0, f"[{joint_type}] problem {prob} position did not improve")
            test.assertLess(rot_err_1, rot_err_0, f"[{joint_type}] problem {prob} rotation did not improve")
            test.assertLess(pos_err_1, 5e-3, f"[{joint_type}] problem {prob} final position error too high")
            test.assertLess(rot_err_1, 2e-2, f"[{joint_type}] problem {prob} final rotation error too high")


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

        objectives_auto = objective_builder(model, n_problems)
        objectives_ana = objective_builder(model, n_problems)

        solver_auto = ik.IKSolver(model, n_problems, objectives_auto, jacobian_mode=ik.IKJacobianType.AUTODIFF)
        solver_ana = ik.IKSolver(model, n_problems, objectives_ana, jacobian_mode=ik.IKJacobianType.ANALYTIC)

        solver_auto._impl._compute_residuals(joint_q_2d)
        solver_ana._impl._compute_residuals(joint_q_2d)

        ctx_auto = solver_auto._impl._ctx_solver(joint_q_2d)
        J_auto = solver_auto._impl._jacobian_at(ctx_auto).numpy()
        ctx_ana = solver_ana._impl._ctx_solver(joint_q_2d)
        J_ana = solver_ana._impl._jacobian_at(ctx_ana).numpy()

        assert_np_equal(J_auto, J_ana, tol=1e-4)


# ----------------------------------------------------------------------------
# 2a.  Position Jacobian
# ----------------------------------------------------------------------------


def _pos_objective_builder(model, n_problems):
    targets = wp.array([[1.5, 0.8, 0.0] for _ in range(n_problems)], dtype=wp.vec3)
    pos_obj = ik.IKObjectivePosition(
        link_index=1,
        link_offset=wp.vec3(0.5, 0.0, 0.0),
        target_positions=targets,
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
    rot_obj = ik.IKObjectiveRotation(
        link_index=1,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array(quats, dtype=wp.vec4),
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

    jl_obj = ik.IKObjectiveJointLimit(
        joint_limit_lower=joint_limit_lower,
        joint_limit_upper=joint_limit_upper,
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

    pos_obj = ik.IKObjectivePosition(0, wp.vec3(0.0, 0.0, 0.0), pos_targets)
    rot_obj = ik.IKObjectiveRotation(0, wp.quat_identity(), rot_targets)
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
for joint_type in (newton.JointType.FREE, newton.JointType.DISTANCE):
    add_function_test(
        TestIKModes,
        f"test_convergence_analytic_descendant_{_joint_type_name(joint_type)}",
        test_convergence_analytic_descendant_free_distance,
        devices,
        joint_type=joint_type,
    )

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
