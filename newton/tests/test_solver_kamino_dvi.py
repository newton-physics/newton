# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform quality gates for the Kamino DVI solver."""

import unittest

from newton._src.solvers.kamino.tests.test_solvers_dvi import TestDVISolver

_DVI_QUALITY_TESTS = (
    "test_00_config_selection",
    "test_00a_multiworld_status_reduction_requires_all_worlds_converged",
    "test_00b_bilateral_solver_selection",
    "test_01_dvi_solve_dense_dual_problem",
    "test_01a_dvi_requires_contact_topology_at_allocation",
    "test_02_public_solver_step_with_dvi",
    "test_02a_public_solver_reset_accepts_global_mask_slot",
    "test_03_dvi_solve_single_contact",
    "test_03a_sparse_dvi_filtered_matvec_matches_full_rows",
    "test_03d_dvi_direct_block_honors_per_world_iteration_counts",
    "test_03d2_dvi_direct_block_finishes_with_bilateral_solve",
    "test_03e_dvi_direct_block_no_unilateral_rows_reports_single_iteration",
    "test_03f_dvi_bilateral_only_solve_resets_stale_status",
    "test_03g_dvi_inequality_coloring_separates_dynamic_conflicts",
    "test_03g1_dvi_inequality_coloring_keeps_worlds_independent",
    "test_03g2_dvi_inequality_coloring_handles_more_than_64_colors",
    "test_03h_dvi_canonical_contact_solution_metrics",
    "test_03i_dvi_coldstart_is_repeatable",
    "test_03j_dvi_omega_scales_projected_updates_without_moving_the_solution",
    "test_03k_dvi_inequality_only_status_reports_the_sweep_budget",
    "test_03l_sparse_dvi_skips_inequalities_without_mapped_topology",
    "test_04_dvi_solve_active_joint_limit",
    "test_05_dvi_solve_multi_world_contacts",
    "test_05a_dvi_maps_packed_multiworld_contacts",
    "test_05b_dvi_five_box_stack_converges_within_budget",
    "test_05c_dvi_high_mass_ratio_stack_supports_weight",
    "test_05d_dvi_colors_contacts_with_joint_limits",
    "test_06_dvi_warmstart_modes",
    "test_06a_dvi_masked_reset_preserves_unselected_worlds",
    "test_07_dvi_singular_limit_rows_remain_finite",
    "test_08_public_solver_short_rollout_with_dvi",
    "test_08a_public_solver_heterogeneous_contact_rollout_with_dvi",
    "test_08c_dvi_zero_friction_preserves_tangent_momentum",
    "test_08d_dvi_kinetic_friction_matches_coulomb_deceleration",
    "test_08e_dvi_sliding_sphere_settles_into_analytic_rolling",
    "test_12_dvi_opening_contact_releases_warmstarted_force",
)


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    """Load a focused cross-platform subset into the main test suite."""
    del loader, tests, pattern
    return unittest.TestSuite(TestDVISolver(name) for name in _DVI_QUALITY_TESTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
