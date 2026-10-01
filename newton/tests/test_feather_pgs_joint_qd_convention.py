# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The FeatherPGS free-joint ``joint_qd`` convention contract.

FeatherPGS stores free-joint ``joint_qd`` as ``(v_com_world, omega_world)``, so
:func:`newton.eval_fk` refreshes maximal body state from joint state. Integration layers
dispatch on :attr:`SolverFeatherPGS.joint_qd_public_convention` to pick that helper instead of
the vanilla Featherstone solver's velocity-conversion helper.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS, SolverFeatherstone
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

# Off-origin center of mass, so the tests distinguish the body COM from the body origin.
COM_LOCAL = (0.05, -0.03, 0.02)
OMEGA = (0.0, 0.0, 3.0)
V_COM = (0.1, -0.2, 0.05)
DT = 1.0 / 240.0


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


def test_solver_declares_public_joint_qd_convention(test, device):
    """Declare the public convention on FeatherPGS but not on the vanilla Featherstone solver."""
    test.assertTrue(SolverFeatherPGS.joint_qd_public_convention)
    test.assertTrue(SolverFeatherPGS(_build_free_body((0.0, 0.0, 0.6), device)).joint_qd_public_convention)
    test.assertFalse(getattr(SolverFeatherstone, "joint_qd_public_convention", False))


def test_public_eval_fk_round_trips_solver_body_qd(test, device):
    """Reproduce the solver's published body_qd with eval_fk on its joint state, far from the origin."""
    for origin in ((0.0, 0.0, 0.6), (5.0, 0.0, 0.6), (20.0, 0.0, 0.6)):
        with test.subTest(origin=origin):
            model = _build_free_body(origin, device)
            state_in, state_out = model.state(), model.state()
            joint_qd = state_in.joint_qd.numpy()
            joint_qd[0:3] = V_COM
            joint_qd[3:6] = OMEGA
            state_in.joint_qd.assign(joint_qd)
            SolverFeatherPGS(model).step(state_in, state_out, model.control(), None, DT)

            probe = model.state()
            probe.joint_q.assign(state_out.joint_q)
            probe.joint_qd.assign(state_out.joint_qd)
            newton.eval_fk(model, probe.joint_q, probe.joint_qd, probe)
            np.testing.assert_allclose(probe.body_qd.numpy(), state_out.body_qd.numpy(), rtol=1e-6, atol=1e-6)
            # body_qd's linear part is the COM velocity, independent of the distance from the origin.
            np.testing.assert_allclose(state_out.body_qd.numpy()[0][0:3], V_COM, rtol=0.0, atol=1e-3)


class TestFeatherPGSJointQdConvention(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
add_function_test(
    TestFeatherPGSJointQdConvention,
    "test_solver_declares_public_joint_qd_convention",
    test_solver_declares_public_joint_qd_convention,
    devices=devices,
)
add_function_test(
    TestFeatherPGSJointQdConvention,
    "test_public_eval_fk_round_trips_solver_body_qd",
    test_public_eval_fk_round_trips_solver_body_qd,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main()
