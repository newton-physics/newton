# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import newton
from newton.solvers import SolverWCSPH, sph


class TestSolverWCSPHAPI(unittest.TestCase):
    def test_sph_root_namespace_matches_solver_pattern(self):
        self.assertIs(newton.solvers.SolverWCSPH, SolverWCSPH)
        self.assertIs(newton.solvers.sph, sph)
        self.assertEqual(SolverWCSPH.Config.__name__, "Config")

    def test_sph_public_helper_surface_is_scoped_to_wcsph_setup(self):
        self.assertEqual({"SPHMaterial", "add_sph_particle_grid"}, set(sph.__all__))

    def test_solver_exposes_newton_coupling_contract(self):
        expected_members = (
            "collect_collider_impulses",
            "collider_body_index",
            "notify_model_changed",
            "project_outside",
            "register_custom_attributes",
            "setup_collider",
            "step",
        )
        for name in expected_members:
            self.assertTrue(hasattr(SolverWCSPH, name), name)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
