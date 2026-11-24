# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, requires

CRANK, COUPLER, ROCKER = 0.4, 1.0, 0.8
GROUND = 1.0  # distance between the crank and rocker pivots
CRANK_LIMIT = 1.0


def _build_four_bar(device) -> tuple[newton.Model, int]:
    """Planar four-bar in the xz-plane: crank, coupler and rocker in a chain; the rocker's pivot
    on the ground is a loop-closing joint to the world. Returns the model and the loop joint."""
    builder = newton.ModelBuilder()
    crank, coupler, rocker = (builder.add_link() for _ in range(3))
    y_axis = (0.0, 1.0, 0.0)
    joints = [
        builder.add_joint_revolute(-1, crank, axis=y_axis, limit_lower=-CRANK_LIMIT, limit_upper=CRANK_LIMIT),
        builder.add_joint_revolute(
            crank, coupler, parent_xform=wp.transform((CRANK, 0.0, 0.0), wp.quat_identity()), axis=y_axis
        ),
        builder.add_joint_revolute(
            coupler, rocker, parent_xform=wp.transform((COUPLER, 0.0, 0.0), wp.quat_identity()), axis=y_axis
        ),
    ]
    builder.add_articulation(joints)
    loop = builder.add_joint_revolute(
        -1,
        rocker,
        parent_xform=wp.transform((GROUND, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform((ROCKER, 0.0, 0.0), wp.quat_identity()),
        axis=y_axis,
    )
    return builder.finalize(device=device), loop


def _coupler_point(crank_angle: float) -> np.ndarray:
    """World position of the coupler's midpoint when the four-bar is closed at ``crank_angle``."""
    # Revolute joints about +y rotate +x towards -z.
    b = np.array([CRANK * math.cos(crank_angle), -CRANK * math.sin(crank_angle)])
    d = np.array([GROUND, 0.0])
    # C is at COUPLER from B and ROCKER from D (circle intersection, elbow up).
    dist = np.linalg.norm(d - b)
    a = (COUPLER**2 - ROCKER**2 + dist**2) / (2.0 * dist)
    h = math.sqrt(COUPLER**2 - a**2)
    mid = b + a * (d - b) / dist
    c = mid + h * np.array([(d - b)[1], -(d - b)[0]]) / dist
    p = 0.5 * (b + c)
    return np.array([p[0], 0.0, p[1]])


def _body_q_rows(model: newton.Model, joint_q: np.ndarray) -> list[np.ndarray]:
    state = model.state()
    rows = []
    for q in joint_q:
        newton.eval_fk(model, wp.array(q, dtype=wp.float32, device=model.device), model.joint_qd, state)
        rows.append(state.body_q.numpy())
    return rows


def _joint_gap(model: newton.Model, body_q: np.ndarray, joint: int) -> float:
    """Distance [m] between the parent and child frames of ``joint``."""
    parent, child = model.joint_parent.numpy()[joint], model.joint_child.numpy()[joint]
    X_p, X_c = model.joint_X_p.numpy()[joint], model.joint_X_c.numpy()[joint]
    parent_pose = wp.transform(*body_q[parent]) if parent >= 0 else wp.transform_identity()
    p = wp.transform_point(parent_pose * wp.transform(*X_p), wp.vec3(0.0, 0.0, 0.0))
    c = wp.transform_point(wp.transform(*body_q[child]) * wp.transform(*X_c), wp.vec3(0.0, 0.0, 0.0))
    return float(wp.length(p - c))


def test_cunls_closes_loop_within_limits(test, device):
    """Keep a loop-closing joint closed and the crank within its limits while reaching targets."""
    model, loop = _build_four_bar(device)
    # The second target needs a crank angle beyond the limit: the crank stops at the limit.
    angles = [0.6, 1.4]
    targets = np.array([_coupler_point(a) for a in angles], dtype=np.float32)
    objectives = [
        ik.IKObjectivePosition(1, wp.vec3(0.5 * COUPLER, 0.0, 0.0), wp.array(targets, dtype=wp.vec3, device=device)),
        ik.IKObjectiveJointLimit(model.joint_limit_lower, model.joint_limit_upper),
    ]
    solver = ik.IKSolver(model, len(angles), objectives, optimizer=ik.IKOptimizer.CUNLS)
    joint_q = wp.array(np.tile(model.joint_q.numpy(), (len(angles), 1)), dtype=wp.float32, device=device)
    solver.step(joint_q, joint_q, iterations=50)

    q = joint_q.numpy()
    body_q = _body_q_rows(model, q)
    for row in range(len(angles)):
        test.assertLess(_joint_gap(model, body_q[row], loop), 1e-4, f"loop open in problem {row}")
    # Problem 0 reaches its target (on the assembly branch that holds it); problem 1 stops at the limit.
    coupler_mid = wp.transform_point(wp.transform(*body_q[0][1]), wp.vec3(0.5 * COUPLER, 0.0, 0.0))
    np.testing.assert_allclose(np.array(coupler_mid), targets[0], atol=1e-4)
    test.assertAlmostEqual(float(q[1, 0]), CRANK_LIMIT, places=5)


def test_cunls_enforces_mimic_joints_with_seeds(test, device):
    """Hold a mimic coupling and keep the best of several sampled seeds per problem."""
    builder = newton.ModelBuilder()
    link1, link2 = builder.add_link(), builder.add_link()
    z_axis = (0.0, 0.0, 1.0)
    joint1 = builder.add_joint_revolute(-1, link1, axis=z_axis, limit_lower=-2.0, limit_upper=2.0)
    joint2 = builder.add_joint_revolute(
        link1, link2, parent_xform=wp.transform((0.5, 0.0, 0.0), wp.quat_identity()), axis=z_axis
    )
    builder.add_articulation([joint1, joint2])
    builder.set_joint_mimic(joint2, joint1, coeffs=(0.1, 2.0))
    model = builder.finalize(device=device)

    q1 = np.array([-0.6, -0.2, 0.3, 0.7])
    q2 = 0.1 + 2.0 * q1
    reach = np.stack([0.5 * np.cos(q1) + 0.5 * np.cos(q1 + q2), 0.5 * np.sin(q1) + 0.5 * np.sin(q1 + q2)], axis=1)
    targets = np.concatenate([reach, np.zeros((len(q1), 1))], axis=1).astype(np.float32)
    objectives = [
        ik.IKObjectivePosition(link2, wp.vec3(0.5, 0.0, 0.0), wp.array(targets, dtype=wp.vec3, device=device)),
        ik.IKObjectiveJointLimit(model.joint_limit_lower, model.joint_limit_upper),
    ]
    solver = ik.IKSolver(
        model, len(q1), objectives, optimizer=ik.IKOptimizer.CUNLS, sampler=ik.IKSampler.ROBERTS, n_seeds=4
    )
    joint_q = wp.zeros((len(q1), model.joint_coord_count), dtype=wp.float32, device=device)
    solver.step(joint_q, joint_q, iterations=50)

    q = joint_q.numpy()
    np.testing.assert_allclose(q[:, 1], 0.1 + 2.0 * q[:, 0], atol=1e-4)
    np.testing.assert_allclose(q[:, 0], q1, atol=1e-3)


def _build_arm(device, base_pose=None) -> newton.Model:
    """Two-link planar arm (links 0.5 m about z). With ``base_pose``, its root joint's parent is a
    body without an incoming joint at that pose (an external root parent), and an unrelated
    unarticulated body follows the arm."""
    builder = newton.ModelBuilder()
    parent = -1 if base_pose is None else builder.add_link(xform=base_pose)
    link1, link2 = builder.add_link(), builder.add_link()
    z_axis = (0.0, 0.0, 1.0)
    joints = [
        builder.add_joint_revolute(parent, link1, axis=z_axis),
        builder.add_joint_revolute(
            link1, link2, parent_xform=wp.transform((0.5, 0.0, 0.0), wp.quat_identity()), axis=z_axis
        ),
    ]
    builder.add_articulation(joints)
    if base_pose is not None:
        builder.add_link(xform=wp.transform((2.0, 0.0, 0.0), wp.quat_identity()))
    return builder.finalize(device=device)


def _tip_positions(model: newton.Model, joint_q: np.ndarray, link: int) -> np.ndarray:
    """World position of the point 0.5 m along x of ``link`` for each row of ``joint_q``."""
    return np.array(
        [wp.transform_point(wp.transform(*b[link]), wp.vec3(0.5, 0.0, 0.0)) for b in _body_q_rows(model, joint_q)]
    )


def test_cunls_holds_external_root_parent(test, device):
    """Solve an arm whose root joint hangs from a body outside the articulation at its body_q pose."""
    base_pose = wp.transform((0.1, -0.2, 0.15), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.4))
    model = _build_arm(device, base_pose)
    tip = 2  # bodies: base 0, link1 1, link2 2, unrelated 3
    q_goal = np.tile(model.joint_q.numpy(), (2, 1))
    q_goal[:, :2] = [[0.3, 0.8], [-0.5, 1.2]]
    targets = _tip_positions(model, q_goal, tip).astype(np.float32)
    objectives = [ik.IKObjectivePosition(tip, wp.vec3(0.5, 0.0, 0.0), wp.array(targets, dtype=wp.vec3, device=device))]
    solver = ik.IKSolver(model, 2, objectives, optimizer=ik.IKOptimizer.CUNLS)
    joint_q = wp.array(np.tile(model.joint_q.numpy(), (2, 1)) + 0.1, dtype=wp.float32, device=device)
    solver.step(joint_q, joint_q, iterations=30)
    np.testing.assert_allclose(_tip_positions(model, joint_q.numpy(), tip), targets, atol=1e-4)

    unrelated = [ik.IKObjectivePosition(3, wp.vec3(), wp.zeros(1, dtype=wp.vec3, device=device))]
    with test.assertRaisesRegex(ValueError, "objective link 3"):
        ik.IKSolver(model, 1, unrelated, optimizer=ik.IKOptimizer.CUNLS)


def test_cunls_costs_use_current_targets(test, device):
    """Evaluate costs against the objectives' current targets, before any step and after a change."""
    model = _build_arm(device)
    objective = ik.IKObjectivePosition(
        1, wp.vec3(0.5, 0.0, 0.0), wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    )
    optimizer = ik.IKOptimizerCuNLS(model, 1, [objective])
    joint_q = wp.array([[0.0, 0.5 * math.pi]], dtype=wp.float32, device=device)  # tip at (0.5, 0.5, 0)
    np.testing.assert_allclose(optimizer.compute_costs(joint_q).numpy(), [0.5**2 + 0.5**2], rtol=1e-5)
    objective.set_target_positions(wp.array([[0.5, 0.5, 0.0]], dtype=wp.vec3, device=device))
    np.testing.assert_allclose(optimizer.compute_costs(joint_q).numpy(), [0.0], atol=1e-10)


def test_cunls_counts_every_rotation_entry(test, device):
    """Give a rotation objective listed twice for one link two orientation terms, as IKOptimizerLM does,
    whether or not a position objective on that link pairs with one of them."""
    model = _build_arm(device)
    angle = 0.3
    target = wp.array([wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)], dtype=wp.vec4, device=device)
    rotation = ik.IKObjectiveRotation(1, wp.quat_identity(), target)
    position = ik.IKObjectivePosition(
        1, wp.vec3(0.5, 0.0, 0.0), wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    )
    joint_q = wp.zeros((1, 2), dtype=wp.float32, device=device)  # tip at (1, 0, 0): only the rotation errs
    for objectives in ([rotation, rotation], [position, rotation, rotation]):
        costs = ik.IKOptimizerCuNLS(model, 1, objectives).compute_costs(joint_q).numpy()
        np.testing.assert_allclose(costs, [2.0 * angle**2], rtol=1e-4)


def test_cunls_steps_on_strided_views(test, device):
    """Read and write joint coordinates through non-contiguous views, and reject an output that
    aliases the input with other strides."""
    model = _build_arm(device)
    targets = np.array([[0.6, 0.5, 0.0]], dtype=np.float32)
    objective = ik.IKObjectivePosition(1, wp.vec3(0.5, 0.0, 0.0), wp.array(targets, dtype=wp.vec3, device=device))
    optimizer = ik.IKOptimizerCuNLS(model, 1, [objective])
    buffer = wp.array([[0.2, 0.4, 7.0]], dtype=wp.float32, device=device)
    joint_q = buffer[:, :2]  # padded rows
    test.assertFalse(joint_q.is_contiguous)
    optimizer.step(joint_q, joint_q, iterations=30)
    np.testing.assert_allclose(_tip_positions(model, joint_q.numpy(), 1), targets, atol=1e-4)
    test.assertEqual(float(buffer.numpy()[0, 2]), 7.0)

    np.testing.assert_allclose(optimizer.compute_costs(joint_q).numpy(), [0.0], atol=1e-8)
    shared = wp.zeros((1, 4), dtype=wp.float32, device=device)
    with test.assertRaisesRegex(ValueError, "different strides"):
        optimizer.step(shared[:, :2], shared[:, ::2], iterations=1)


def test_cunls_step_captured_right_after_construction(test, device):
    """Capture step() in a CUDA graph right after construction (the examples' idiom) and replay it
    for a new target."""
    model = _build_arm(device)
    targets = np.array([[0.6, 0.5, 0.0]], dtype=np.float32)
    objective = ik.IKObjectivePosition(1, wp.vec3(0.5, 0.0, 0.0), wp.array(targets, dtype=wp.vec3, device=device))
    solver = ik.IKSolver(model, 1, [objective], optimizer=ik.IKOptimizer.CUNLS)
    joint_q = wp.array([[0.2, 0.4]], dtype=wp.float32, device=device)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(joint_q, joint_q, iterations=20)
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(_tip_positions(model, joint_q.numpy(), 1), targets, atol=1e-4)
    targets = np.array([[-0.3, 0.7, 0.0]], dtype=np.float32)
    objective.set_target_positions(wp.array(targets, dtype=wp.vec3, device=device))
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(_tip_positions(model, joint_q.numpy(), 1), targets, atol=1e-4)


def test_cunls_rejects_bounded_loop_joint(test, device):
    """Reject joint limits on a loop-closing joint, which cuNLS cannot bound."""
    model, loop = _build_four_bar(device)
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    dof = model.joint_qd_start.numpy()[loop]
    lower[dof], upper[dof] = -0.5, 0.5
    limit = ik.IKObjectiveJointLimit(wp.array(lower, device=device), wp.array(upper, device=device))
    position = ik.IKObjectivePosition(1, wp.vec3(), wp.zeros(1, dtype=wp.vec3, device=device))
    with test.assertRaisesRegex(ValueError, f"loop-closing joint {loop}"):
        ik.IKSolver(model, 1, [position, limit], optimizer=ik.IKOptimizer.CUNLS)


def test_cunls_raises_on_infeasible_constraints(test, device):
    """Raise instead of returning a configuration that violates its mimic constraint."""
    builder = newton.ModelBuilder()
    link1, link2 = builder.add_link(), builder.add_link()
    z_axis = (0.0, 0.0, 1.0)
    joint1 = builder.add_joint_revolute(-1, link1, axis=z_axis)
    joint2 = builder.add_joint_revolute(
        link1, link2, parent_xform=wp.transform((0.5, 0.0, 0.0), wp.quat_identity()), axis=z_axis
    )
    builder.add_articulation([joint1, joint2])
    builder.set_joint_mimic(joint2, joint1, coeffs=(0.1, 2.0))  # q2 = 0.1 + 2 q1
    model = builder.finalize(device=device)
    zeros = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)  # bounds q1 = q2 = 0
    objectives = [
        ik.IKObjectivePosition(link2, wp.vec3(0.5, 0.0, 0.0), wp.zeros(1, dtype=wp.vec3, device=device)),
        ik.IKObjectiveJointLimit(zeros, zeros),
    ]
    solver = ik.IKSolver(model, 1, objectives, optimizer=ik.IKOptimizer.CUNLS)
    joint_q = wp.zeros((1, model.joint_coord_count), dtype=wp.float32, device=device)
    with test.assertRaisesRegex(RuntimeError, "constraints are not satisfied"):
        solver.step(joint_q, joint_q, iterations=20)


def test_cunls_rejects_unsupported_objective(test, device):
    """Reject objectives the cuNLS backend cannot translate."""
    model, _ = _build_four_bar(device)

    class IKObjectiveCustom(ik.IKObjective):
        def residual_dim(self):
            return 1

    with test.assertRaisesRegex(ValueError, "IKObjectiveCustom"):
        ik.IKSolver(model, 1, [IKObjectiveCustom()], optimizer=ik.IKOptimizer.CUNLS)
    rotation = ik.IKObjectiveRotation(
        1, wp.quat_identity(), wp.zeros(1, dtype=wp.vec4, device=device), canonicalize_quat_err=False
    )
    with test.assertRaisesRegex(ValueError, "canonicalize_quat_err=False"):
        ik.IKSolver(model, 1, [rotation], optimizer=ik.IKOptimizer.CUNLS)


cuda_devices = get_selected_cuda_test_devices()


@requires("pycunls")
class TestIKCuNLS(unittest.TestCase):
    """Verify the cuNLS inverse-kinematics backend: loop closures, mimic joints and hard joint limits."""


for test_fn in (
    test_cunls_closes_loop_within_limits,
    test_cunls_enforces_mimic_joints_with_seeds,
    test_cunls_holds_external_root_parent,
    test_cunls_costs_use_current_targets,
    test_cunls_counts_every_rotation_entry,
    test_cunls_steps_on_strided_views,
    test_cunls_step_captured_right_after_construction,
    test_cunls_rejects_bounded_loop_joint,
    test_cunls_raises_on_infeasible_constraints,
    test_cunls_rejects_unsupported_objective,
):
    add_function_test(TestIKCuNLS, test_fn.__name__, test_fn, cuda_devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
