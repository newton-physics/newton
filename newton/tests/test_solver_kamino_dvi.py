# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform quality gates for the Kamino DVI solver."""

import unittest

from newton._src.solvers.kamino.tests.test_solvers_dvi import TestDVISolver

_DVI_QUALITY_TESTS = (
    "test_00_config_selection",
    "test_00a_multiworld_status_reduction_requires_all_worlds_converged",
    "test_01_dvi_solve_dense_dual_problem",
    "test_02_public_solver_step_with_dvi",
    "test_03_dvi_solve_single_contact",
    "test_03a_sparse_dvi_filtered_matvec_matches_full_rows",
    "test_03b_dvi_contact_block_preconditioner_smoke",
    "test_03d2_dvi_direct_block_finishes_with_bilateral_solve",
    "test_03e_dvi_direct_block_no_unilateral_rows_reports_single_iteration",
    "test_03f_dvi_bilateral_only_solve_resets_stale_status",
    "test_03g_dvi_contact_coloring_separates_dynamic_conflicts",
    "test_03h_dvi_canonical_contact_solution_metrics",
    "test_03i_dvi_coldstart_is_repeatable",
    "test_04_dvi_solve_active_joint_limit",
    "test_05_dvi_solve_multi_world_contacts",
    "test_06_dvi_warmstart_modes",
    "test_06a_dvi_masked_reset_preserves_unselected_worlds",
    "test_07_dvi_singular_limit_rows_remain_finite",
    "test_08_public_solver_short_rollout_with_dvi",
    "test_08a_public_solver_heterogeneous_contact_rollout_with_dvi",
    "test_12_dvi_opening_contact_releases_warmstarted_force",
)


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    """Load a focused cross-platform subset into the main test suite."""
    del loader, tests, pattern
    return unittest.TestSuite(TestDVISolver(name) for name in _DVI_QUALITY_TESTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
