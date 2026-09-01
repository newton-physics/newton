# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cross-check SolverMuJoCo's reported body kinematics against mujoco_warp's
internal forward kinematics for D6 joints with multiple angular DOFs.

Steps a D6-three-angular model for a couple of steps at a real dt, refreshes
``mjw_data`` kinematics so it reflects the post-step ``qpos``/``qvel``, and
asserts ``state.body_q`` / ``state.body_qd`` match ``mjw_data.xpos`` /
``xquat`` / ``cvel`` at every step. Also asserts the body actually rotated
over the simulated interval.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_d6_three_angular_model(device):
    """Single body attached to world by a D6 joint with three orthogonal hinge axes."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    child = builder.add_link(
        mass=1.0,
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=wp.mat33(np.eye(3) * 0.1),
    )
    builder.add_shape_sphere(child, radius=0.1)
    cfg = newton.ModelBuilder.JointDofConfig.create_unlimited
    j = builder.add_joint_d6(
        parent=-1,
        child=child,
        angular_axes=[
            cfg(axis=newton.Axis.X),
            cfg(axis=newton.Axis.Y),
            cfg(axis=newton.Axis.Z),
        ],
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j])
    return builder.finalize(device=device), child


def _body_qd_from_mjw_cvel(mjw_model, mjw_data, mjc_body, world=0):
    """Convert ``mjw_data.cvel`` to Newton's ``body_qd`` convention.

    MuJoCo ``cvel`` is a spatial vector (angular, linear) where the linear
    part is expressed at the subtree COM. Newton's ``body_qd`` is
    (linear_at_body_COM, angular) in world frame.
    """
    cvel = mjw_data.cvel.numpy()[world, mjc_body]
    xipos = mjw_data.xipos.numpy()[world, mjc_body]
    root = int(mjw_model.body_rootid.numpy()[mjc_body])
    subtree_com = mjw_data.subtree_com.numpy()[world, root]

    ang = cvel[:3]
    lin_at_subtree = cvel[3:]
    offset = xipos - subtree_com
    lin_at_body_com = lin_at_subtree - np.cross(offset, ang)
    return np.concatenate([lin_at_body_com, ang])


def _mjc_body_index_for(solver, newton_body):
    """Look up the MuJoCo body index that maps to ``newton_body`` in world 0."""
    table = solver.mjc_body_to_newton.numpy()[0]  # [nbody]
    matches = np.where(table == newton_body)[0]
    assert matches.size == 1, f"expected exactly one mjc body for Newton body {newton_body}"
    return int(matches[0])


def test_mujoco_fk_kernel_matches_mujoco_warp_d6_three_angular(test, device):
    model, child = _build_d6_three_angular_model(device)

    state_in = model.state()
    state_out = model.state()
    control = model.control()

    # Pick non-trivial initial coords so prior-axis rotations are non-identity.
    q_start = int(model.joint_q_start.numpy()[0])
    qd_start = int(model.joint_qd_start.numpy()[0])

    q0 = np.array([0.5, -0.4, 0.7], dtype=np.float32)
    qd0 = np.array([0.9, -0.6, 0.3], dtype=np.float32)

    q = state_in.joint_q.numpy()
    qd = state_in.joint_qd.numpy()
    q[q_start : q_start + 3] = q0
    qd[qd_start : qd_start + 3] = qd0
    state_in.joint_q.assign(q)
    state_in.joint_qd.assign(qd)

    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_cpu=False,
        disable_contacts=True,
    )

    mjc_body = _mjc_body_index_for(solver, child)

    sim_dt = 1.0 / 60.0
    steps = 2

    # Snapshot the initial body orientation so we can assert real motion below.
    state_in_initial = model.state()
    state_in_initial.joint_q.assign(state_in.joint_q)
    state_in_initial.joint_qd.assign(state_in.joint_qd)
    newton.eval_fk(model, state_in_initial.joint_q, state_in_initial.joint_qd, state_in_initial)
    quat_initial = state_in_initial.body_q.numpy()[child, 3:].copy()

    for step_idx in range(steps):
        solver.step(state_in, state_out, control, contacts=None, dt=sim_dt)

        # Refresh mujoco_warp kinematics so xpos/xquat/cvel reflect the
        # post-step qpos/qvel (mjw_data.step leaves them at the pre-step snapshot).
        with wp.ScopedDevice(model.device):
            solver._mujoco_warp.forward(solver.mjw_model, solver.mjw_data)

        body_q_solver = state_out.body_q.numpy()[child]
        body_qd_solver = state_out.body_qd.numpy()[child]

        xpos = solver.mjw_data.xpos.numpy()[0, mjc_body]
        xquat_wxyz = solver.mjw_data.xquat.numpy()[0, mjc_body]
        xquat_xyzw = np.array([xquat_wxyz[1], xquat_wxyz[2], xquat_wxyz[3], xquat_wxyz[0]], dtype=np.float32)
        body_qd_mjw = _body_qd_from_mjw_cvel(solver.mjw_model, solver.mjw_data, mjc_body)

        np.testing.assert_allclose(body_q_solver[:3], xpos, atol=1e-5, err_msg=f"position mismatch at step {step_idx}")

        quat_dot = float(abs(np.dot(body_q_solver[3:], xquat_xyzw)))
        test.assertAlmostEqual(
            quat_dot,
            1.0,
            places=4,
            msg=f"orientation mismatch at step {step_idx}: newton={body_q_solver[3:]} mjw={xquat_xyzw}",
        )

        np.testing.assert_allclose(
            body_qd_solver[:3],
            body_qd_mjw[:3],
            atol=1e-5,
            err_msg=f"linear velocity mismatch at step {step_idx}: newton={body_qd_solver[:3]} mjw={body_qd_mjw[:3]}",
        )

        np.testing.assert_allclose(
            body_qd_solver[3:],
            body_qd_mjw[3:],
            atol=1e-5,
            err_msg=f"angular velocity mismatch at step {step_idx}: newton={body_qd_solver[3:]} mjw={body_qd_mjw[3:]}",
        )

        state_in, state_out = state_out, state_in

    # Confirm the body actually rotated over the simulated interval. With
    # |ω| ≈ 1 rad/s and 2 steps at dt = 1/60, expect a few centiradians.
    quat_final = state_in.body_q.numpy()[child, 3:]
    quat_dot = float(abs(np.dot(quat_initial, quat_final)))
    rotation_angle = 2.0 * np.arccos(min(quat_dot, 1.0))
    test.assertGreater(
        rotation_angle,
        0.01,
        msg=f"body barely rotated over {steps} steps at dt={sim_dt}: angle={rotation_angle:.4f} rad",
    )


class TestMuJoCoFKConsistency(unittest.TestCase):
    pass


add_function_test(
    TestMuJoCoFKConsistency,
    "test_mujoco_fk_kernel_matches_mujoco_warp_d6_three_angular",
    test_mujoco_fk_kernel_matches_mujoco_warp_d6_three_angular,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
