# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for passive joint springs and passive damping in SolverFeatherPGS."""

import unittest

import warp as wp

import newton


def _build_hinge(spring_k: float, spring_ref: float, damping: float, limit_upper: float = 10.0):
    """Build a fixed-base link on a revolute Z joint (gravity exerts no torque about the axis).

    The joint has no drive; any motion toward ``spring_ref`` comes from the passive spring
    alone, and any velocity decay beyond numerics comes from the passive damping alone.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    link = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(link, hx=0.1, hy=0.02, hz=0.02)
    j = b.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
        limit_lower=-10.0,
        limit_upper=limit_upper,
    )
    b.add_articulation([j], label="hinge")
    dof = b.joint_qd_start[j]
    b.joint_spring_stiffness[dof] = spring_k
    b.joint_spring_ref[dof] = spring_ref
    b.joint_damping[dof] = damping
    return b


def _run(builder, steps: int = 1200, qd0: float = 0.0, **solver_kwargs):
    model = builder.finalize()
    solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=8, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    if qd0 != 0.0:
        qd = state_0.joint_qd.numpy()
        qd[0] = qd0
        state_0.joint_qd.assign(qd)
    control = model.control()
    dt = 1.0 / 240.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
    return state_0


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSSprings(unittest.TestCase):
    def test_spring_converges_to_reference(self):
        """Verify an undriven joint settles at the passive spring's reference position."""
        state = _run(_build_hinge(spring_k=0.5, spring_ref=0.5, damping=0.05))
        q = state.joint_q.numpy()
        self.assertAlmostEqual(float(q[0]), 0.5, delta=0.02)

    def test_springref_preloads_against_limit(self):
        """Verify a spring reference beyond the joint range preloads the joint at its limit.

        This is the Robotiq spring_link pattern: ``springref`` is unreachable, so the
        spring presses the joint against its upper stop with constant preload torque.
        """
        state = _run(
            _build_hinge(spring_k=0.5, spring_ref=2.62, damping=0.05, limit_upper=0.3),
            enable_joint_limits=True,
        )
        q = state.joint_q.numpy()
        self.assertAlmostEqual(float(q[0]), 0.3, delta=0.03)

    def test_passive_damping_decays_velocity(self):
        """Verify passive joint damping decays velocity that persists when damping is disabled."""
        damped = _run(_build_hinge(spring_k=0.0, spring_ref=0.0, damping=0.02), steps=480, qd0=5.0)
        undamped = _run(_build_hinge(spring_k=0.0, spring_ref=0.0, damping=0.0), steps=480, qd0=5.0)
        qd_damped = abs(float(damped.joint_qd.numpy()[0]))
        qd_undamped = abs(float(undamped.joint_qd.numpy()[0]))
        self.assertGreater(qd_undamped, 4.0, "undamped joint should keep coasting")
        self.assertLess(qd_damped, 0.2 * qd_undamped)


if __name__ == "__main__":
    unittest.main()
