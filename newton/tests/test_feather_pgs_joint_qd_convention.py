# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The FeatherPGS free-joint ``joint_qd`` convention contract.

FeatherPGS reads and writes free-joint ``joint_qd`` as ``(v_com_world, omega_world)``, the public
convention shared with :class:`~newton.solvers.SolverFeatherstone`, so :func:`newton.eval_fk`
refreshes maximal body state from either solver's joint state.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS, SolverFeatherstone
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

# Off-origin center of mass, so the tests distinguish the body COM from the body origin.
COM_LOCAL = (0.05, -0.03, 0.02)
OMEGA = (0.0, 0.0, 3.0)
V_COM = (0.1, -0.2, 0.05)
DT = 1.0 / 240.0
ORIGINS = ((0.0, 0.0, 0.6), (5.0, 0.0, 0.6), (20.0, 0.0, 0.6))


def _build_free_body(origin, device):
    """Build a single gravity-free, contact-free free-root body placed at ``origin``."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(*origin), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(*COM_LOCAL),
        inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.04),
    )
    builder.add_articulation([builder.add_joint_free(parent=-1, child=body)])
    return builder.finalize(device=device)


def _step_once(model, solver):
    """Step ``solver`` once from the seeded public free-joint twist."""
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    joint_qd[0:3] = V_COM
    joint_qd[3:6] = OMEGA
    state_in.joint_qd.assign(joint_qd)
    solver.step(state_in, state_out, model.control(), None, DT)
    return state_out


def test_public_eval_fk_round_trips_solver_body_qd(test, device, pgs_mode="matrix_free"):
    """Reproduce the solver's published body_qd with eval_fk on its joint state, far from the origin."""
    for origin in ORIGINS:
        with test.subTest(origin=origin):
            model = _build_free_body(origin, device)
            state_out = _step_once(model, SolverFeatherPGS(model, pgs_mode=pgs_mode))

            probe = model.state()
            probe.joint_q.assign(state_out.joint_q)
            probe.joint_qd.assign(state_out.joint_qd)
            newton.eval_fk(model, probe.joint_q, probe.joint_qd, probe)
            np.testing.assert_allclose(probe.body_qd.numpy(), state_out.body_qd.numpy(), rtol=1e-6, atol=1e-6)
            # body_qd's linear part is the COM velocity, independent of the distance from the origin.
            np.testing.assert_allclose(state_out.body_qd.numpy()[0][0:3], V_COM, rtol=0.0, atol=1e-3)


def test_joint_qd_matches_featherstone_far_from_origin(test, device, pgs_mode="matrix_free"):
    """Match SolverFeatherstone's joint_qd and body_qd from the same public twist, far from the origin."""
    for origin in ORIGINS:
        with test.subTest(origin=origin):
            model = _build_free_body(origin, device)
            reference = _step_once(model, SolverFeatherstone(model))
            state_out = _step_once(model, SolverFeatherPGS(model, pgs_mode=pgs_mode))
            # A convention mismatch would differ by omega x x_com_world, about 60 m/s at 20 m.
            np.testing.assert_allclose(state_out.joint_qd.numpy(), reference.joint_qd.numpy(), rtol=0.0, atol=1e-3)
            np.testing.assert_allclose(state_out.body_qd.numpy(), reference.body_qd.numpy(), rtol=0.0, atol=1e-3)


class TestFeatherPGSJointQdConvention(unittest.TestCase):
    pass


for _name, _func in (
    ("test_public_eval_fk_round_trips_solver_body_qd", test_public_eval_fk_round_trips_solver_body_qd),
    ("test_joint_qd_matches_featherstone_far_from_origin", test_joint_qd_matches_featherstone_far_from_origin),
):
    add_function_test(TestFeatherPGSJointQdConvention, _name, _func, devices=get_cuda_test_devices())
    add_function_test(
        TestFeatherPGSJointQdConvention, f"{_name}_split", _func, devices=get_test_devices(), pgs_mode="split"
    )


if __name__ == "__main__":
    unittest.main()
