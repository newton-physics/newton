# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import fields

import newton
from newton.solvers import SolverWCSPH, sph


class TestSolverWCSPHAPI(unittest.TestCase):
    def test_sph_root_namespace_matches_solver_pattern(self):
        self.assertIs(newton.solvers.SolverWCSPH, SolverWCSPH)
        self.assertIs(newton.solvers.sph, sph)
        self.assertEqual(SolverWCSPH.Config.__name__, "Config")

    def test_sph_public_helper_surface_is_scoped_to_wcsph_setup(self):
        expected = {
            "SPHMaterial",
            "SPHRole",
            "add_sph_boundary_from_shape",
            "add_sph_boundary_points",
            "add_sph_particle_grid",
        }

        self.assertEqual(expected, set(sph.__all__))

    def test_solver_public_surface_matches_compact_solver_pattern(self):
        expected_members = {
            "Config",
            "collect_collider_impulses",
            "collider_body_index",
            "notify_model_changed",
            "project_outside",
            "register_custom_attributes",
            "setup_collider",
            "step",
        }
        public_members = {name for name in SolverWCSPH.__dict__ if not name.startswith("_")}
        self.assertEqual(expected_members, public_members)

        config_fields = {field.name for field in fields(SolverWCSPH.Config)}
        supported_fields = {
            "kernel",
            "smoothing_length",
            "rest_density",
            "sound_speed",
            "stiffness",
            "pressure_exponent",
            "viscosity",
            "xsph",
            "enable_surface_tension",
            "surface_tension_normal_threshold",
            "enable_shape_boundaries",
            "boundary_margin",
            "boundary_friction",
            "collider_velocity_mode",
            "enable_boundary_adhesion",
            "enable_boundary_wetting",
        }
        self.assertEqual(supported_fields, config_fields)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
