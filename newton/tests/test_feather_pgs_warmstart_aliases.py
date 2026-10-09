# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deprecated ``mf_warmstart`` aliases of the SolverFeatherPGS warm start."""

import unittest
import warnings

import numpy as np
import warp as wp

from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_contact_compliance import run_fixture as run_compliance_fixture
from newton.tests.test_feather_pgs_warmstart import _build_press, _run_press
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _solver(device, **options):
    """Construct a solver and return it with the warnings its construction emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = SolverFeatherPGS(_build_press(device), **options)
    return solver, [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_aliases_map_onto_the_warm_start_options(test, device):
    """Enable warm start through either flag and take the alias decay only when the alias alone enables it."""
    for options, expected in (
        ({"mf_warmstart": True}, (True, 1.0)),
        ({"mf_warmstart": True, "mf_warmstart_decay": 0.25}, (True, 0.25)),
        ({"mf_warmstart": True, "mf_warmstart_decay": 0.25, "pgs_warmstart": True}, (True, 1.0)),
        ({"mf_warmstart": True, "pgs_warmstart": True, "pgs_warmstart_decay": 0.5}, (True, 0.5)),
        ({"mf_warmstart": False, "mf_warmstart_decay": 0.25}, (False, 1.0)),
        ({"mf_warmstart_decay": 0.25, "pgs_warmstart": True}, (True, 1.0)),
    ):
        with test.subTest(options=options):
            solver, deprecations = _solver(device, **options)
            test.assertEqual((solver.pgs_warmstart, solver.pgs_warmstart_decay), expected)
            test.assertEqual(len(deprecations), 1)
            test.assertIn("pgs_warmstart", str(deprecations[0].message))
    _, deprecations = _solver(device, pgs_warmstart=True)
    test.assertEqual(deprecations, [])


def test_alias_solve_matches_the_warm_start_solve(test, device):
    """Reproduce the pgs_warmstart trajectory bitwise through the deprecated aliases."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        alias = _run_press(device, 60, {"mf_warmstart": True, "mf_warmstart_decay": 0.5})
    reference = _run_press(device, 60, {"pgs_warmstart": True, "pgs_warmstart_decay": 0.5})
    np.testing.assert_array_equal(alias[0], reference[0])
    np.testing.assert_array_equal(alias[2].joint_q.numpy(), reference[2].joint_q.numpy())


def test_aliases_validate_and_follow_the_warm_start_rules(test, device):
    """Reject an invalid alias decay and every combination that rejects warm start."""
    with test.assertRaisesRegex(ValueError, "mf_warmstart_decay"), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        SolverFeatherPGS(_build_press(device), mf_warmstart=True, mf_warmstart_decay=-1.0)
    with test.assertRaisesRegex(ValueError, "pgs_warmstart"), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        run_compliance_fixture(
            device=device, articulated=True, enabled=True, steps=1, solver_options={"mf_warmstart": True}
        )


class TestFeatherPGSWarmstartAliases(unittest.TestCase):
    pass


for _fn in (
    test_aliases_map_onto_the_warm_start_options,
    test_alias_solve_matches_the_warm_start_solve,
    test_aliases_validate_and_follow_the_warm_start_rules,
):
    add_function_test(TestFeatherPGSWarmstartAliases, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
