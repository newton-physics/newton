# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import unittest

import numpy as np

import newton
from newton._src.solvers.feather_pgs import SolverFeatherPGS


class TestFeatherPGS(unittest.TestCase):
    def _make_model(self):
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=np.eye(3, dtype=np.float32), label="body")
        return builder.finalize(device="cpu")

    def test_constructor_signature_is_matrix_free_only(self):
        params = inspect.signature(SolverFeatherPGS).parameters

        self.assertNotIn("pgs_mode", params)
        self.assertNotIn("dense_contact_compliance", params)
        self.assertNotIn("dense_max_constraints", params)
        self.assertNotIn("cholesky_kernel", params)
        self.assertNotIn("trisolve_kernel", params)
        self.assertNotIn("hinv_jt_kernel", params)
        self.assertNotIn("delassus_kernel", params)
        self.assertNotIn("pgs_kernel", params)
        self.assertIn("contact_compliance", params)
        self.assertIn("max_constraints", params)

    def test_step_smoke_runs_matrix_free_solver(self):
        model = self._make_model()
        solver = SolverFeatherPGS(model, pgs_iterations=1)

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.contacts()

        solver.step(state_in, state_out, control, contacts, 1.0e-3)

        self.assertIsNone(solver.C)
        self.assertIsNotNone(solver.J_world)
        self.assertIsNotNone(solver.Y_world)


if __name__ == "__main__":
    unittest.main(verbosity=2)
