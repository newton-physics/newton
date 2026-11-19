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
    for row, body_q in enumerate(_body_q_rows(model, q)):
        test.assertLess(_joint_gap(model, body_q, loop), 1e-4, f"loop open in problem {row}")
    np.testing.assert_allclose(q[0, 0], angles[0], atol=1e-3)
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


def test_cunls_rejects_unsupported_objective(test, device):
    """Reject objectives the cuNLS backend cannot translate."""
    model, _ = _build_four_bar(device)

    class IKObjectiveCustom(ik.IKObjective):
        def residual_dim(self):
            return 1

    with test.assertRaisesRegex(ValueError, "IKObjectiveCustom"):
        ik.IKSolver(model, 1, [IKObjectiveCustom()], optimizer=ik.IKOptimizer.CUNLS)


cuda_devices = get_selected_cuda_test_devices()


@requires("pycunls")
class TestIKCuNLS(unittest.TestCase):
    """Verify the cuNLS inverse-kinematics backend: loop closures, mimic joints and hard joint limits."""


add_function_test(
    TestIKCuNLS, "test_cunls_closes_loop_within_limits", test_cunls_closes_loop_within_limits, cuda_devices
)
add_function_test(
    TestIKCuNLS,
    "test_cunls_enforces_mimic_joints_with_seeds",
    test_cunls_enforces_mimic_joints_with_seeds,
    cuda_devices,
)
add_function_test(
    TestIKCuNLS, "test_cunls_rejects_unsupported_objective", test_cunls_rejects_unsupported_objective, cuda_devices
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
