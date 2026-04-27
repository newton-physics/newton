# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import unittest

import newton._src.solvers as solvers
from newton._src.solvers.feather_pgs import SolverFeatherPGS


class TestFeatherPGSPrivateAPI(unittest.TestCase):
    def test_import_stays_in_private_package(self):
        self.assertNotIn("SolverFeatherPGS", solvers.__all__)
        self.assertFalse(hasattr(solvers, "SolverFeatherPGS"))

    def test_constructor_signature_is_single_path(self):
        params = inspect.signature(SolverFeatherPGS).parameters

        removed = {
            "pgs_mode",
            "pgs_kernel",
            "friction_mode",
            "effort_limit_mode",
            "dense_contact_compliance",
            "dense_max_constraints",
            "cholesky_kernel",
            "trisolve_kernel",
            "hinv_jt_kernel",
            "delassus_kernel",
            "delassus_chunk_size",
            "pgs_chunk_size",
            "small_dof_threshold",
            "pgs_debug",
        }
        for name in removed:
            self.assertNotIn(name, params)

        self.assertIn("contact_compliance", params)
        self.assertIn("max_constraints", params)
        self.assertIn("enable_joint_velocity_limits", params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
