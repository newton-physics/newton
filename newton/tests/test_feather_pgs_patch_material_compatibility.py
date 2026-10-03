# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve explicit material-law opt-ins alongside default patch friction."""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_contact_compliance import run_fixture
from newton.tests.test_feather_pgs_contact_torsion import fixture


class TestPatchMaterialCompatibility(unittest.TestCase):
    def test_disabled_material_options_keep_default_patches(self):
        """Keep ordinary patch defaults when neither experimental material law is enabled."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_sphere(body, radius=0.05)
        model = builder.finalize(device="cpu")
        solver = SolverFeatherPGS(model, contact_compliance=False, contact_torsion_radius=0.0)
        self.assertEqual(solver.friction_anchor_beta, 0.2)
        self.assertTrue(solver._friction_anchors_enabled)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_torsion_opt_in_keeps_default_patches(self):
        """Resolve omitted patch gain to default patch friction alongside torsion, without a warning."""
        options = {"center_only": True, "contact_shared_anchor": False, "contact_friction_shared_anchor": False}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual, solver, *_ = fixture(0.01, friction_anchor_beta=None, **options)
        self.assertEqual([str(w.message) for w in caught if "friction" in str(w.message)], [])
        reference, explicit, *_ = fixture(0.01, friction_anchor_beta=0.2, **options)
        self.assertTrue(solver._friction_anchors_enabled)
        self.assertTrue(explicit._friction_anchors_enabled)
        self.assertAlmostEqual(solver.friction_anchor_beta, 0.2)
        self.assertGreater(solver._torsion_stats["rows"], 0)
        for name in reference:
            np.testing.assert_array_equal(actual[name], reference[name], err_msg=name)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_compliance_opt_in_preserves_material_law(self):
        """Keep dense and MF material responses identical to explicit point friction."""
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                options = {"articulated": articulated, "enabled": True, "steps": 20, "friction_scale": 0.4}
                with self.assertWarnsRegex(UserWarning, "velocity-only point friction"):
                    actual, paths, solver, _ = run_fixture(**options, solver_options={"friction_anchor_beta": None})
                reference, reference_paths, _, _ = run_fixture(**options, solver_options={"friction_anchor_beta": 0.0})
                np.testing.assert_array_equal(actual, reference)
                self.assertEqual(paths, reference_paths)
                self.assertGreater(solver.compliance_contact_count, 0)
                self.assertFalse(solver._friction_anchors_enabled)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_explicit_patch_compliance_conflict_fails_before_step(self):
        """Reject incompatible explicit requests before any simulation state can advance."""
        with patch.object(SolverFeatherPGS, "step", side_effect=AssertionError("step must not run")):
            with self.assertRaisesRegex(ValueError, "contact_compliance.*friction_anchor_beta"):
                run_fixture(articulated=False, enabled=True, steps=1, solver_options={"friction_anchor_beta": 0.2})
