# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for linear algebra matrix analysis utilities"""

import unittest

import numpy as np

import newton._src.solvers.kamino._src.linalg as linalg
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Tests
###


class TestUtilsLinAlgMatrix(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for verbose output
        if self.verbose:
            msg.set_log_level(msg.LogLevel.DEBUG)

    def tearDown(self):
        if self.verbose:
            msg.reset_log_level()

    def test_01_spd_matrix_properties(self):
        A = linalg.utils.rand.random_spd_matrix(dim=10, dtype=np.float32, scale=4.0, seed=42)
        A_props = linalg.utils.matrix.SquareSymmetricMatrixProperties(A)
        msg.debug(f"A (shape: {A.shape}, dtype: {A.dtype}):\n{A}\n") if self.verbose else None
        msg.debug(f"A properties:\n{A_props}\n") if self.verbose else None
        msg.debug(f"cond(A): {np.linalg.cond(A)}\n") if self.verbose else None
        msg.debug(f"det(A): {np.linalg.det(A)}\n") if self.verbose else None
        self.assertAlmostEqual(A_props.cond, np.linalg.cond(A), places=6)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
