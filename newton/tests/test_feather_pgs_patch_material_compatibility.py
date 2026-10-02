# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve explicit material-law opt-ins alongside default patch friction."""

import unittest
import warnings

import numpy as np

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_contact_torsion import fixture
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def test_disabled_material_options_keep_default_patches(test, device):
    """Keep ordinary patch defaults when the experimental material law is not enabled."""
    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.05)
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, contact_torsion_radius=0.0)
    test.assertEqual(solver.friction_anchor_beta, 0.2)
    test.assertTrue(solver._friction_anchors_enabled)
    test.assertFalse(solver._contact_torsion_enabled)


def test_torsion_opt_in_keeps_default_patches(test, device):
    """Resolve omitted patch gain to default patch friction alongside torsion, without a warning."""
    options = {"device": device, "center_only": True}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual, solver, *_ = fixture(0.01, friction_anchor_beta=None, **options)
    test.assertEqual([str(w.message) for w in caught if "friction" in str(w.message)], [])
    reference, explicit, *_ = fixture(0.01, friction_anchor_beta=0.2, **options)
    test.assertTrue(solver._friction_anchors_enabled)
    test.assertTrue(explicit._friction_anchors_enabled)
    test.assertAlmostEqual(solver.friction_anchor_beta, 0.2)
    test.assertGreater(solver._torsion_stats["rows"], 0)
    for name in reference:
        np.testing.assert_array_equal(actual[name], reference[name], err_msg=name)


class TestPatchMaterialCompatibility(unittest.TestCase):
    pass


for _fn in (test_disabled_material_options_keep_default_patches, test_torsion_opt_in_keeps_default_patches):
    add_function_test(TestPatchMaterialCompatibility, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
