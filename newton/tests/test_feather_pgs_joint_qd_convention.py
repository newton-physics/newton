# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The FeatherPGS free-joint ``joint_qd`` convention contract.

FeatherPGS stores free-joint ``joint_qd`` as ``(v_com_world, omega_world)``, so
:func:`newton.eval_fk` refreshes maximal body state from joint state. Integration layers
dispatch on :attr:`SolverFeatherPGS.joint_qd_public_convention` to pick that helper instead of
the vanilla Featherstone solver's velocity-conversion helper.
"""

import itertools
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.featherstone.kernels import eval_fk_with_velocity_conversion
from newton.solvers import SolverFeatherPGS, SolverFeatherstone
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

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


def _step_once(model, pgs_mode):
    """Step FeatherPGS once from a seeded free-joint twist."""
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    joint_qd[0:3] = V_COM
    joint_qd[3:6] = OMEGA
    state_in.joint_qd.assign(joint_qd)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode)
    solver.step(state_in, state_out, model.control(), None, DT)
    return solver, state_out


def _refresh_body_qd(model, source, helper):
    """Rebuild ``body_qd`` from ``source``'s joint state using ``helper``, as a reset would."""
    probe = model.state()
    probe.joint_q.assign(source.joint_q)
    probe.joint_qd.assign(source.joint_qd)
    helper(model, probe.joint_q, probe.joint_qd, probe)
    return probe.body_qd.numpy().copy()


def _consumer_refresh(solver, model, source):
    """Mimic an integration layer dispatching FK on the solver's declared convention."""
    if getattr(solver, "joint_qd_public_convention", False):
        return _refresh_body_qd(model, source, newton.eval_fk)
    return _refresh_body_qd(model, source, eval_fk_with_velocity_conversion)


def _com_world(model, state):
    """World-space center of mass of body 0 [m]."""
    body_q = state.body_q.numpy()[0]
    rotated = wp.quat_rotate(wp.quat(*body_q[3:7]), wp.vec3(*model.body_com.numpy()[0]))
    return np.array(body_q[0:3]) + np.array([rotated[0], rotated[1], rotated[2]])


def test_solver_declares_public_joint_qd_convention(test, device, pgs_mode="matrix_free"):
    """Declare the public convention on FeatherPGS but not on the vanilla Featherstone solver."""
    test.assertTrue(SolverFeatherPGS.joint_qd_public_convention)
    test.assertTrue(
        SolverFeatherPGS(_build_free_body((0.0, 0.0, 0.6), device), pgs_mode=pgs_mode).joint_qd_public_convention
    )
    test.assertFalse(getattr(SolverFeatherstone, "joint_qd_public_convention", False))


def test_public_eval_fk_round_trips_solver_body_qd(test, device, pgs_mode="matrix_free"):
    """Reproduce the solver's published body_qd with eval_fk on its joint state, far from the origin."""
    for origin in ((0.0, 0.0, 0.6), (5.0, 0.0, 0.6), (20.0, 0.0, 0.6)):
        with test.subTest(origin=origin):
            model = _build_free_body(origin, device)
            state_in, state_out = model.state(), model.state()
            joint_qd = state_in.joint_qd.numpy()
            joint_qd[0:3] = V_COM
            joint_qd[3:6] = OMEGA
            state_in.joint_qd.assign(joint_qd)
            SolverFeatherPGS(model, pgs_mode=pgs_mode).step(state_in, state_out, model.control(), None, DT)

            probe = model.state()
            probe.joint_q.assign(state_out.joint_q)
            probe.joint_qd.assign(state_out.joint_qd)
            newton.eval_fk(model, probe.joint_q, probe.joint_qd, probe)
            np.testing.assert_allclose(probe.body_qd.numpy(), state_out.body_qd.numpy(), rtol=1e-6, atol=1e-6)
            # body_qd's linear part is the COM velocity, independent of the distance from the origin.
            np.testing.assert_allclose(state_out.body_qd.numpy()[0][0:3], V_COM, rtol=0.0, atol=1e-3)


def test_featherstone_helper_injects_omega_cross_com_phantom_velocity(test, device, pgs_mode="matrix_free"):
    """Re-reference the linear velocity by exactly ``omega x x_com_world`` with the Featherstone helper."""
    model = _build_free_body((5.0, 0.0, 0.6), device)
    _solver, state_out = _step_once(model, pgs_mode)
    public = _refresh_body_qd(model, state_out, newton.eval_fk)
    converted = _refresh_body_qd(model, state_out, eval_fk_with_velocity_conversion)
    expected_phantom = np.cross(public[0][3:6], _com_world(model, state_out))
    np.testing.assert_allclose(converted[0][0:3] - public[0][0:3], expected_phantom, rtol=1e-5, atol=1e-5)
    # Angular velocity is untouched; only the linear part is re-referenced.
    np.testing.assert_allclose(converted[0][3:6], public[0][3:6], rtol=1e-6, atol=1e-6)


def test_phantom_velocity_grows_with_distance_from_world_origin(test, device, pgs_mode="matrix_free"):
    """Grow the Featherstone helper's phantom velocity with the distance from the origin; eval_fk stays exact."""
    magnitudes = []
    for radius in (0.0, 1.0, 5.0, 20.0):
        with test.subTest(radius=radius):
            model = _build_free_body((radius, 0.0, 0.6), device)
            _solver, state_out = _step_once(model, pgs_mode)
            public = _refresh_body_qd(model, state_out, newton.eval_fk)
            converted = _refresh_body_qd(model, state_out, eval_fk_with_velocity_conversion)
            np.testing.assert_allclose(public, state_out.body_qd.numpy(), rtol=1e-6, atol=1e-6)
            phantom = converted[0][0:3] - public[0][0:3]
            np.testing.assert_allclose(
                phantom, np.cross(public[0][3:6], _com_world(model, state_out)), rtol=1e-5, atol=1e-5
            )
            magnitudes.append(float(np.linalg.norm(phantom)))
    for near, far in itertools.pairwise(magnitudes):
        test.assertGreater(far, near)
    # At a 20 m offset the phantom velocity dwarfs the true COM velocity.
    test.assertGreater(magnitudes[-1], 100.0 * float(np.linalg.norm(V_COM)))


def test_consumer_dispatch_selects_correct_helper(test, device, pgs_mode="matrix_free"):
    """Refresh body state correctly when dispatching on the attribute, and wrongly without it."""
    model = _build_free_body((20.0, 0.0, 0.6), device)
    solver, state_out = _step_once(model, pgs_mode)
    solver_body_qd = state_out.body_qd.numpy().copy()
    np.testing.assert_allclose(_consumer_refresh(solver, model, state_out), solver_body_qd, rtol=1e-6, atol=1e-6)
    # Without the declaration a consumer falls back to the Featherstone-internal helper.
    with mock.patch.object(SolverFeatherPGS, "joint_qd_public_convention", False):
        without = _consumer_refresh(solver, model, state_out)
    phantom = without[0][0:3] - solver_body_qd[0][0:3]
    np.testing.assert_allclose(
        phantom, np.cross(solver_body_qd[0][3:6], _com_world(model, state_out)), rtol=1e-5, atol=1e-5
    )
    test.assertGreater(float(np.linalg.norm(phantom)), 50.0)


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
for _name, _func in (
    ("test_solver_declares_public_joint_qd_convention", test_solver_declares_public_joint_qd_convention),
    ("test_public_eval_fk_round_trips_solver_body_qd", test_public_eval_fk_round_trips_solver_body_qd),
    (
        "test_featherstone_helper_injects_omega_cross_com_phantom_velocity",
        test_featherstone_helper_injects_omega_cross_com_phantom_velocity,
    ),
    (
        "test_phantom_velocity_grows_with_distance_from_world_origin",
        test_phantom_velocity_grows_with_distance_from_world_origin,
    ),
    ("test_consumer_dispatch_selects_correct_helper", test_consumer_dispatch_selects_correct_helper),
):
    add_function_test(
        TestFeatherPGSJointQdConvention, f"{_name}_split", _func, devices=get_test_devices(), pgs_mode="split"
    )


if __name__ == "__main__":
    unittest.main()
