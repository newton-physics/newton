# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph-capture quality gate for the Kamino LOX solver."""

import unittest

from newton._src.solvers.kamino.tests.test_solver_kamino_lox import TestSolverKaminoLOX


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    """Load the focused CUDA graph-capture check into the main test suite."""
    del loader, tests, pattern
    return unittest.TestSuite((TestSolverKaminoLOX("test_cuda_graph_capture_uses_conditional_loop"),))


if __name__ == "__main__":
    unittest.main(verbosity=2)
