# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check public activation and backend constraints for physical hydro contacts."""

import unittest

import numpy as np

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.test_mujoco_contact_partition import _fill_contacts, _make_fixture


class TestMuJoCoHydroelasticResponse(unittest.TestCase):
    def test_nonzero_margin_preserves_physical_hydro_response(self):
        """Use margin-relative penetration even when MuJoCo's stored distance is positive."""
        model, solver, contacts, state_in, state_out, shapes = _make_fixture(device="cpu")
        outputs = []
        for margin in (0.0, 0.002):
            with self.subTest(shape_margin=margin):
                model.shape_margin.fill_(margin)
                solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
                _fill_contacts(contacts, shapes, [1.0], stiffness=1.0e4, damping=20.0, gap=-0.001)
                solver.step(state_in, state_out, model.control(), contacts, 0.001)
                distance = float(solver.mjw_data.contact.dist.numpy()[0])
                include_margin = float(solver.mjw_data.contact.includemargin.numpy()[0])
                self.assertAlmostEqual(distance - include_margin, -0.001, delta=1.0e-9)
                if margin:
                    self.assertGreater(distance, 0.0)
                velocity = state_out.body_qd.numpy()[0]
                # Independent scalar BE momentum balance: m=1, v=0, r=-0.001.
                expected_velocity = 0.001 * 1.0e4 * 0.001 / (1.0 + 0.001 * 20.0 + 0.001**2 * 1.0e4)
                np.testing.assert_allclose(
                    velocity, [0.0, 0.0, expected_velocity, 0.0, 0.0, 0.0], rtol=3.0e-6, atol=1.0e-8
                )
                outputs.append(velocity)
        np.testing.assert_allclose(outputs[0], outputs[1], rtol=3.0e-6, atol=1.0e-8)

    def test_unsupported_backend_and_rk4_are_rejected(self):
        """Reject native contacts, the native CPU backend, and RK4 explicitly."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        unsupported = (
            {"use_mujoco_contacts": True},
            {"use_mujoco_contacts": False, "use_mujoco_cpu": True},
            {"use_mujoco_contacts": False, "integrator": "rk4"},
        )
        for options in unsupported:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "Hydroelastic force response"):
                SolverMuJoCo(model, use_hydroelastic_force_response=True, **options)


if __name__ == "__main__":
    unittest.main()
