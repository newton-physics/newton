# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA quality smoke gate for Kamino DVI Dr Legs regressions."""

import unittest

from newton._src.solvers.kamino.tests.test_solvers_dvi import TestDVISolver

_DVI_CUDA_QUALITY_TESTS = (
    "test_08b_dr_legs_contact_capacity_scales_with_world_count",
    "test_09_dr_legs_dvi_first_contact_remains_finite",
    "test_11_dr_legs_dvi_contact_force_balances_weight",
)


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    """Load focused CUDA Dr Legs checks into the main test suite."""
    del loader, tests, pattern
    return unittest.TestSuite(TestDVISolver(name) for name in _DVI_CUDA_QUALITY_TESTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
