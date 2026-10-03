# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FeatherPGS free-joint ``joint_qd`` convention contract.

FeatherPGS stores free-joint ``joint_qd`` as ``(v_com_world, omega_world)``, so
:func:`newton.eval_fk` is the correct helper for refreshing maximal body state from joint
state. The vanilla Featherstone solver uses a world-origin-referenced spatial twist and needs
``eval_fk_with_velocity_conversion`` instead. Integration layers dispatch on
:attr:`SolverFeatherPGS.joint_qd_public_convention` to pick the matching helper; without the
declaration they fall back to the Featherstone helper and inject ``omega x x_com_world`` of
phantom linear velocity that grows with the body's distance from the world origin.
"""

import itertools
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.featherstone.kernels import eval_fk_with_velocity_conversion
from newton.solvers import SolverFeatherPGS, SolverFeatherstone

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
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _step_once(model):
    """Step FeatherPGS once from a seeded free-joint twist."""
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    joint_qd[0:3] = V_COM
    joint_qd[3:6] = OMEGA
    state_in.joint_qd.assign(joint_qd)

    solver = SolverFeatherPGS(model)
    solver.step(state_in, state_out, control=model.control(), contacts=None, dt=DT)
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


class TestFeatherPGSJointQdConvention(unittest.TestCase):
    def test_solver_declares_public_joint_qd_convention(self):
        """The dispatch key is declared on FeatherPGS and absent on vanilla Featherstone."""
        self.assertTrue(SolverFeatherPGS.joint_qd_public_convention)

        model = _build_free_body((0.0, 0.0, 0.6), "cpu")
        self.assertTrue(SolverFeatherPGS(model).joint_qd_public_convention)

        # Featherstone must NOT opt in: its free-joint qd is an internal world-origin twist,
        # so a consumer's getattr(..., False) has to keep routing it to the conversion helper.
        self.assertFalse(getattr(SolverFeatherstone, "joint_qd_public_convention", False))

    def test_public_eval_fk_round_trips_solver_body_qd(self):
        """eval_fk on FeatherPGS joint state reproduces the solver's own body_qd."""
        model = _build_free_body((5.0, 0.0, 0.6), "cpu")
        _solver, state_out = _step_once(model)

        solver_body_qd = state_out.body_qd.numpy().copy()
        refreshed = _refresh_body_qd(model, state_out, newton.eval_fk)

        np.testing.assert_allclose(refreshed, solver_body_qd, rtol=1e-6, atol=1e-6)

    def test_featherstone_helper_injects_omega_cross_com_phantom_velocity(self):
        """The conversion helper's error is exactly omega x x_com_world."""
        model = _build_free_body((5.0, 0.0, 0.6), "cpu")
        _solver, state_out = _step_once(model)

        public = _refresh_body_qd(model, state_out, newton.eval_fk)
        converted = _refresh_body_qd(model, state_out, eval_fk_with_velocity_conversion)

        omega = public[0][3:6]
        expected_phantom = np.cross(omega, _com_world(model, state_out))
        actual_phantom = converted[0][0:3] - public[0][0:3]

        np.testing.assert_allclose(actual_phantom, expected_phantom, rtol=1e-5, atol=1e-5)
        # Angular velocity is untouched; only the linear part is re-referenced.
        np.testing.assert_allclose(converted[0][3:6], public[0][3:6], rtol=1e-6, atol=1e-6)

    def test_phantom_velocity_grows_with_distance_from_world_origin(self):
        """The phantom velocity scales with env-grid radius while eval_fk stays exact."""
        radii = (0.0, 1.0, 5.0, 20.0)
        magnitudes = []

        for radius in radii:
            with self.subTest(radius=radius):
                model = _build_free_body((radius, 0.0, 0.6), "cpu")
                _solver, state_out = _step_once(model)
                solver_body_qd = state_out.body_qd.numpy().copy()

                public = _refresh_body_qd(model, state_out, newton.eval_fk)
                converted = _refresh_body_qd(model, state_out, eval_fk_with_velocity_conversion)

                # The correct helper is radius-independent.
                np.testing.assert_allclose(public, solver_body_qd, rtol=1e-6, atol=1e-6)

                phantom = converted[0][0:3] - public[0][0:3]
                np.testing.assert_allclose(
                    phantom,
                    np.cross(public[0][3:6], _com_world(model, state_out)),
                    rtol=1e-5,
                    atol=1e-5,
                )
                magnitudes.append(float(np.linalg.norm(phantom)))

        for near, far in itertools.pairwise(magnitudes):
            self.assertGreater(far, near)

        # At a 20 m grid offset the phantom velocity dwarfs the true COM velocity.
        self.assertGreater(magnitudes[-1], 100.0 * float(np.linalg.norm(V_COM)))

    def test_consumer_dispatch_selects_correct_helper(self):
        """A consumer dispatching on the attribute is correct with it and wrong without it."""
        model = _build_free_body((20.0, 0.0, 0.6), "cpu")
        solver, state_out = _step_once(model)
        solver_body_qd = state_out.body_qd.numpy().copy()

        with_fix = _consumer_refresh(solver, model, state_out)
        np.testing.assert_allclose(with_fix, solver_body_qd, rtol=1e-6, atol=1e-6)

        # Pre-fix state: the solver never declared the convention, so getattr(..., False)
        # routed FeatherPGS through the Featherstone-internal helper.
        with mock.patch.object(SolverFeatherPGS, "joint_qd_public_convention", False):
            without_fix = _consumer_refresh(solver, model, state_out)

        phantom = without_fix[0][0:3] - solver_body_qd[0][0:3]
        np.testing.assert_allclose(
            phantom,
            np.cross(solver_body_qd[0][3:6], _com_world(model, state_out)),
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertGreater(float(np.linalg.norm(phantom)), 50.0)


if __name__ == "__main__":
    unittest.main()
