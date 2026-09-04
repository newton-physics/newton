# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform quality gates for the Kamino LOX solver."""

import unittest

from newton._src.solvers.kamino.tests.test_solver_kamino_lox import TestSolverKaminoLOX

_LOX_QUALITY_TESTS = (
    "test_box_on_plane_projects_detected_contact",
    "test_relaxed_joint_proximal_preserves_nonlinear_fixed_point",
    "test_cartpole_projects_detected_joint_limit",
    "test_cartpole_sustained_joint_force_remains_bounded",
    "test_free_fall_advances_projected_velocity_and_pose",
    "test_joint_damping_is_implicit_in_the_smooth_row",
    "test_product_space_structural_split_hinged_contact",
)


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    """Load a focused cross-platform subset into the main test suite."""
    del loader, tests, pattern
    return unittest.TestSuite(TestSolverKaminoLOX(name) for name in _LOX_QUALITY_TESTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
