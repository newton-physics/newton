# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from newton._src.sim.ik.ik_common import eval_fk_batched
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices


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

    builder.add_articulation([joint1, joint2])

    model = builder.finalize(device=device, requires_grad=True)
    return model


def _build_free_plus_revolute(device) -> newton.Model:
    builder = newton.ModelBuilder()

    link1 = builder.add_link(xform=wp.transform_identity(), mass=1.0)
    joint1 = builder.add_joint_free(
        parent=-1,
        child=link1,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )

    link2 = builder.add_link(xform=wp.transform([1.0, 0.0, 0.0], wp.quat_identity()), mass=1.0)
    joint2 = builder.add_joint_revolute(
        parent=link1,
        child=link2,
        parent_xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
        child_xform=wp.transform([-0.5, 0.0, 0.0], wp.quat_identity()),
        axis=[0.0, 0.0, 1.0],
    )

    builder.add_articulation([joint1, joint2])

    return builder.finalize(device=device, requires_grad=True)


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
    builder.add_articulation([joint])

    return builder.finalize(device=device, requires_grad=True)


def _fk_end_effector_positions(
    body_q_2d: wp.array, n_problems: int, ee_link_index: int, ee_offset: wp.vec3
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


def _convergence_test_lbfgs_planar(test, device, mode: ik.IKJacobianType):
    """Test L-BFGS convergence on planar 2-link robot."""
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

        # Reachable XY targets
        targets = wp.array([[1.5, 1.0, 0.0], [1.2, 0.8, 0.0], [1.8, 0.5, 0.0]], dtype=wp.vec3)
        ee_link = 1
        ee_off = wp.vec3(0.5, 0.0, 0.0)

        pos_obj = ik.IKObjectivePosition(
            link_index=ee_link,
            link_offset=ee_off,
            target_positions=targets,
        )

        # Create L-BFGS solver
        lbfgs_solver = ik.IKSolver(
            model,
            n_problems,
            [pos_obj],
            optimizer=ik.IKOptimizer.LBFGS,
            jacobian_mode=mode,
        )

        # Run initial FK
        eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        initial = _fk_end_effector_positions(body_q_2d, n_problems, ee_link, ee_off)

        # Solve with L-BFGS
        lbfgs_solver.step(joint_q_2d, joint_q_2d, iterations=70)

        # Run final FK
        eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        final = _fk_end_effector_positions(body_q_2d, n_problems, ee_link, ee_off)

        # Check convergence
        for prob in range(n_problems):
            err0 = np.linalg.norm(initial[prob] - targets.numpy()[prob])
            err1 = np.linalg.norm(final[prob] - targets.numpy()[prob])
            test.assertLess(err1, err0, f"L-BFGS mode {mode} problem {prob} did not improve")
            test.assertLess(err1, 3e-3, f"L-BFGS mode {mode} problem {prob} final error too high ({err1:.4f})")


def _convergence_test_lbfgs_free(test, device, mode: ik.IKJacobianType):
    with wp.ScopedDevice(device):
        n_problems = 3
        model = _build_free_plus_revolute(device)

        requires_grad = mode in [ik.IKJacobianType.AUTODIFF, ik.IKJacobianType.MIXED]
        joint_q_2d = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=requires_grad)
        joint_q_2d.fill_(1e-3)
        joint_qd_2d = wp.zeros((n_problems, model.joint_dof_count), dtype=wp.float32)
        body_q_2d = wp.zeros((n_problems, model.body_count), dtype=wp.transform)
        body_qd_2d = wp.zeros((n_problems, model.body_count), dtype=wp.spatial_vector)

        targets = wp.array([[1.0, 1.0, 0.0]] * n_problems, dtype=wp.vec3)
        ee_link = 1
        ee_off = wp.vec3(0.5, 0.0, 0.0)

        pos_obj = ik.IKObjectivePosition(ee_link, ee_off, targets)

        solver = ik.IKSolver(
            model,
            n_problems,
            [pos_obj],
            optimizer=ik.IKOptimizer.LBFGS,
            jacobian_mode=mode,
            h0_scale=1.0,
            line_search_alphas=[0.01, 0.1, 0.5, 0.75, 1.0],
            history_len=12,
        )

        eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        initial = _fk_end_effector_positions(body_q_2d, n_problems, ee_link, ee_off)

        solver.step(joint_q_2d, joint_q_2d, iterations=10)

        eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        final = _fk_end_effector_positions(body_q_2d, n_problems, ee_link, ee_off)

        for prob in range(n_problems):
            err0 = np.linalg.norm(initial[prob] - targets.numpy()[prob])
            err1 = np.linalg.norm(final[prob] - targets.numpy()[prob])
            test.assertLess(err1, err0, f"[FREE] L-BFGS mode {mode} problem {prob} did not improve")
            test.assertLess(err1, 5e-3, f"[FREE] L-BFGS mode {mode} problem {prob} final error too high ({err1:.4f})")


def _convergence_test_lbfgs_d6(test, device, mode: ik.IKJacobianType):
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

        pos_obj = ik.IKObjectivePosition(0, wp.vec3(0.0, 0.0, 0.0), pos_targets)
        rot_obj = ik.IKObjectiveRotation(0, wp.quat_identity(), rot_targets)

        solver = ik.IKSolver(
            model,
            n_problems,
            [pos_obj, rot_obj],
            optimizer=ik.IKOptimizer.LBFGS,
            jacobian_mode=mode,
        )

        eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        initial = _fk_end_effector_positions(body_q_2d, n_problems, 0, wp.vec3(0.0, 0.0, 0.0))

        solver.step(joint_q_2d, joint_q_2d, iterations=90)

        eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        final = _fk_end_effector_positions(body_q_2d, n_problems, 0, wp.vec3(0.0, 0.0, 0.0))

        for prob in range(n_problems):
            err0 = np.linalg.norm(initial[prob] - pos_targets.numpy()[prob])
            err1 = np.linalg.norm(final[prob] - pos_targets.numpy()[prob])
            test.assertLess(err1, err0, f"[D6] L-BFGS mode {mode} problem {prob} did not improve")
            test.assertLess(err1, 1e-3, f"[D6] L-BFGS mode {mode} problem {prob} final error too high ({err1:.4f})")


def _comparison_test_lm_vs_lbfgs(test, device, mode: ik.IKJacobianType):
    """Compare L-BFGS vs LM solver performance."""
    with wp.ScopedDevice(device):
        n_problems = 2
        model = _build_two_link_planar(device)

        requires_grad = mode in [ik.IKJacobianType.AUTODIFF, ik.IKJacobianType.MIXED]

        # Create identical initial conditions for both solvers
        joint_q_lm = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=requires_grad)
        joint_q_lbfgs = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=requires_grad)

        # Set challenging initial configuration
        initial_q = np.array([[0.5, -0.8], [0.3, 1.2]], dtype=np.float32)
        joint_q_lm.assign(initial_q)
        joint_q_lbfgs.assign(initial_q)

        joint_qd_2d = wp.zeros((n_problems, model.joint_dof_count), dtype=wp.float32)
        body_q_2d = wp.zeros((n_problems, model.body_count), dtype=wp.transform)
        body_qd_2d = wp.zeros((n_problems, model.body_count), dtype=wp.spatial_vector)

        # Challenging targets
        targets = wp.array([[1.4, 1.2, 0.0], [1.0, 1.5, 0.0]], dtype=wp.vec3)
        ee_link, ee_off = 1, wp.vec3(0.5, 0.0, 0.0)

        # Create objectives
        pos_obj_lm = ik.IKObjectivePosition(ee_link, ee_off, targets)
        pos_obj_lbfgs = ik.IKObjectivePosition(ee_link, ee_off, targets)

        # Create solvers
        lm_solver = ik.IKSolver(model, n_problems, [pos_obj_lm], lambda_initial=1e-3, jacobian_mode=mode)
        lbfgs_solver = ik.IKSolver(
            model,
            n_problems,
            [pos_obj_lbfgs],
            optimizer=ik.IKOptimizer.LBFGS,
            jacobian_mode=mode,
            history_len=8,
        )

        # Get initial errors
        eval_fk_batched(model, joint_q_lm, joint_qd_2d, body_q_2d, body_qd_2d)
        initial_lm = _fk_end_effector_positions(body_q_2d, n_problems, ee_link, ee_off)

        eval_fk_batched(model, joint_q_lbfgs, joint_qd_2d, body_q_2d, body_qd_2d)
        initial_lbfgs = _fk_end_effector_positions(body_q_2d, n_problems, ee_link, ee_off)

        # Solve with both methods
        lm_solver.step(joint_q_lm, joint_q_lm, iterations=25, step_size=1.0)
        lbfgs_solver.step(joint_q_lbfgs, joint_q_lbfgs, iterations=70)

        # Get final errors
        eval_fk_batched(model, joint_q_lm, joint_qd_2d, body_q_2d, body_qd_2d)
        final_lm = _fk_end_effector_positions(body_q_2d, n_problems, ee_link, ee_off)

        eval_fk_batched(model, joint_q_lbfgs, joint_qd_2d, body_q_2d, body_qd_2d)
        final_lbfgs = _fk_end_effector_positions(body_q_2d, n_problems, ee_link, ee_off)

        # Both solvers should converge
        for prob in range(n_problems):
            target = targets.numpy()[prob]

            err_lm_initial = np.linalg.norm(initial_lm[prob] - target)
            err_lm_final = np.linalg.norm(final_lm[prob] - target)

            err_lbfgs_initial = np.linalg.norm(initial_lbfgs[prob] - target)
            err_lbfgs_final = np.linalg.norm(final_lbfgs[prob] - target)

            # Both should improve
            test.assertLess(err_lm_final, err_lm_initial, f"LM problem {prob} did not improve")
            test.assertLess(err_lbfgs_final, err_lbfgs_initial, f"L-BFGS problem {prob} did not improve")

            # Both should achieve good accuracy
            test.assertLess(err_lm_final, 1e-3, f"LM problem {prob} final error too high ({err_lm_final:.4f})")
            test.assertLess(
                err_lbfgs_final, 1e-3, f"L-BFGS problem {prob} final error too high ({err_lbfgs_final:.4f})"
            )


# Test functions
def test_lbfgs_convergence_autodiff(test, device):
    _convergence_test_lbfgs_planar(test, device, ik.IKJacobianType.AUTODIFF)


def test_lbfgs_convergence_analytic(test, device):
    _convergence_test_lbfgs_planar(test, device, ik.IKJacobianType.ANALYTIC)


def test_lbfgs_convergence_mixed(test, device):
    _convergence_test_lbfgs_planar(test, device, ik.IKJacobianType.MIXED)


def test_lbfgs_convergence_autodiff_free(test, device):
    _convergence_test_lbfgs_free(test, device, ik.IKJacobianType.AUTODIFF)


def test_lbfgs_convergence_analytic_free(test, device):
    _convergence_test_lbfgs_free(test, device, ik.IKJacobianType.ANALYTIC)


def test_lbfgs_convergence_mixed_free(test, device):
    _convergence_test_lbfgs_free(test, device, ik.IKJacobianType.MIXED)


def test_lbfgs_convergence_autodiff_d6(test, device):
    _convergence_test_lbfgs_d6(test, device, ik.IKJacobianType.AUTODIFF)


def test_lbfgs_convergence_analytic_d6(test, device):
    _convergence_test_lbfgs_d6(test, device, ik.IKJacobianType.ANALYTIC)


def test_lbfgs_convergence_mixed_d6(test, device):
    _convergence_test_lbfgs_d6(test, device, ik.IKJacobianType.MIXED)


def test_lm_vs_lbfgs_comparison_autodiff(test, device):
    _comparison_test_lm_vs_lbfgs(test, device, ik.IKJacobianType.AUTODIFF)


def test_lm_vs_lbfgs_comparison_analytic(test, device):
    _comparison_test_lm_vs_lbfgs(test, device, ik.IKJacobianType.ANALYTIC)


def test_lm_vs_lbfgs_comparison_mixed(test, device):
    _comparison_test_lm_vs_lbfgs(test, device, ik.IKJacobianType.MIXED)


def test_lbfgs_convergence_descendant_free_distance(test, device, joint_type):
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
        initial_pos = _fk_end_effector_positions(body_q_2d, n_problems, ee_link, ee_off)
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
            optimizer=ik.IKOptimizer.LBFGS,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            h0_scale=1.0,
            line_search_alphas=[0.01, 0.1, 0.5, 0.75, 1.0],
            history_len=12,
        )

        solver.step(joint_q_2d, joint_q_2d, iterations=18)

        eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
        final_pos = _fk_end_effector_positions(body_q_2d, n_problems, ee_link, ee_off)
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
            test.assertLess(rot_err_1, 3e-2, f"[{joint_type}] problem {prob} final rotation error too high")


# Test registration
devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()


class TestLBFGSIK(unittest.TestCase):
    pass


# Register L-BFGS convergence tests
add_function_test(TestLBFGSIK, "test_lbfgs_convergence_autodiff", test_lbfgs_convergence_autodiff, devices)
add_function_test(TestLBFGSIK, "test_lbfgs_convergence_analytic", test_lbfgs_convergence_analytic, devices)
add_function_test(TestLBFGSIK, "test_lbfgs_convergence_mixed", test_lbfgs_convergence_mixed, devices)
add_function_test(
    TestLBFGSIK, "test_lbfgs_convergence_autodiff_free", test_lbfgs_convergence_autodiff_free, cuda_devices
)
add_function_test(TestLBFGSIK, "test_lbfgs_convergence_analytic_free", test_lbfgs_convergence_analytic_free, devices)
add_function_test(TestLBFGSIK, "test_lbfgs_convergence_mixed_free", test_lbfgs_convergence_mixed_free, devices)
add_function_test(TestLBFGSIK, "test_lbfgs_convergence_autodiff_d6", test_lbfgs_convergence_autodiff_d6, cuda_devices)
add_function_test(TestLBFGSIK, "test_lbfgs_convergence_analytic_d6", test_lbfgs_convergence_analytic_d6, devices)
add_function_test(TestLBFGSIK, "test_lbfgs_convergence_mixed_d6", test_lbfgs_convergence_mixed_d6, cuda_devices)
for joint_type in (newton.JointType.FREE, newton.JointType.DISTANCE):
    add_function_test(
        TestLBFGSIK,
        f"test_lbfgs_convergence_descendant_{_joint_type_name(joint_type)}",
        test_lbfgs_convergence_descendant_free_distance,
        devices,
        joint_type=joint_type,
    )

# Register comparison tests
add_function_test(TestLBFGSIK, "test_lm_vs_lbfgs_comparison_autodiff", test_lm_vs_lbfgs_comparison_autodiff, devices)
add_function_test(TestLBFGSIK, "test_lm_vs_lbfgs_comparison_analytic", test_lm_vs_lbfgs_comparison_analytic, devices)
add_function_test(TestLBFGSIK, "test_lm_vs_lbfgs_comparison_mixed", test_lm_vs_lbfgs_comparison_mixed, devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
