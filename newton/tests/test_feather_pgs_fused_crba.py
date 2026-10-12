# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fused mass-matrix assembly and factorization of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs._testing import kernel_overrides
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs._fixtures import _trajectory, _tree
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def test_fused_assembly_matches_the_separate_pipeline(test, device, solve_mode="matrix_free"):
    """Match separate assembly and factorization with the warp-per-articulation and tiled kernels."""
    for links, branched, kernel in ((3, False, "warp"), (6, True, "warp"), (14, False, "tiled"), (15, True, "tiled")):
        with test.subTest(links=links, branched=branched):
            model = _tree(device, links, branched=branched)
            fused, q, qd = _trajectory(model, solve_mode=solve_mode, use_parallel_streams=True)
            reference, q_ref, qd_ref = _trajectory(model, solve_mode=solve_mode)
            size = fused._size_groups[0]
            selected = {
                "warp": fused._crba_cholesky_warp_kernels_by_size[size],
                "tiled": fused._crba_cholesky_kernels_by_size[size],
            }
            test.assertIsNotNone(selected[kernel])
            test.assertIsNone(reference._crba_cholesky_warp_kernels_by_size[size])
            test.assertIsNone(reference._crba_cholesky_kernels_by_size[size])
            np.testing.assert_allclose(fused._L_by_size[size].numpy(), reference._L_by_size[size].numpy(), atol=2.0e-5)
            np.testing.assert_allclose(q, q_ref, rtol=0.0, atol=2.0e-5)
            np.testing.assert_allclose(qd, qd_ref, rtol=0.0, atol=2.0e-4)


def test_fused_assembly_covers_articulations_wider_than_the_block(test, device):
    """Match separate assembly when the tiled block has fewer threads than the articulation has DOFs."""
    for links in (33, 64):
        model = _tree(device, links, worlds=1)
        with kernel_overrides({"sparse_mass_matrix": False}):
            reference, q_ref, qd_ref = _trajectory(model, steps=2)
        size = reference._size_groups[0]
        test.assertIsNone(reference._crba_cholesky_kernels_by_size[size])
        for tile_threads in (32, 64):
            with test.subTest(links=links, tile_threads=tile_threads):
                overrides = {"tile_threads": tile_threads, "sparse_mass_matrix": False}
                with kernel_overrides(overrides):
                    fused, q, qd = _trajectory(model, steps=2, use_parallel_streams=True)
                test.assertIsNotNone(fused._crba_cholesky_kernels_by_size[size])
                np.testing.assert_allclose(
                    fused._L_by_size[size].numpy(), reference._L_by_size[size].numpy(), atol=2.0e-5
                )
                np.testing.assert_allclose(q, q_ref, rtol=1.0e-5, atol=2.0e-5)
                np.testing.assert_allclose(qd, qd_ref, rtol=1.0e-5, atol=2.0e-4)


def test_fused_assembly_follows_the_mass_update_interval(test, device):
    """Refresh the fused factors on the interval and after an inertia notification only."""
    model = _tree(device, 3)
    results = []
    for streams in (True, False):
        solver, q, _ = _trajectory(model, steps=7, use_parallel_streams=streams, update_mass_matrix_interval=3)
        body_mass = model.body_mass.numpy()
        model.body_mass.assign(body_mass * 1.5)
        model.body_inv_mass.assign(1.0 / (body_mass * 1.5))
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
        model.body_mass.assign(body_mass)
        model.body_inv_mass.assign(1.0 / body_mass)
        results.append((q, solver._L_by_size[solver._size_groups[0]].numpy()))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=0.0, atol=2.0e-5)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=0.0, atol=2.0e-5)


def test_fused_assembly_is_selected_only_where_supported(test, device):
    """Keep the separate pipeline without streams, with loop joints, PGS drive rows or on CPU."""
    is_cuda = wp.get_device(device).is_cuda
    solve_mode = "matrix_free" if is_cuda else "split"
    solver = SolverFeatherPGS(_tree(device, 3), solve_mode=solve_mode, use_parallel_streams=True)
    test.assertEqual(bool(solver._crba_source_dof_by_size), is_cuda)
    if not is_cuda:
        return
    for model, options in (
        (_tree(device, 3), {}),
        (_tree(device, 3), {"use_parallel_streams": True, "drive_mode": "rows"}),
        (_tree(device, 3, loop=True), {"use_parallel_streams": True}),
    ):
        with test.subTest(**options):
            solver = SolverFeatherPGS(model, **options)
            test.assertFalse(solver._crba_source_dof_by_size)
            test.assertIsNone(solver._crba_cholesky_warp_kernels_by_size[solver._size_groups[0]])


class TestFeatherPGSFusedCrba(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()
add_function_test(
    TestFeatherPGSFusedCrba,
    "test_fused_assembly_matches_the_separate_pipeline",
    test_fused_assembly_matches_the_separate_pipeline,
    devices=cuda_devices,
)
add_function_test(
    TestFeatherPGSFusedCrba,
    "test_fused_assembly_matches_the_separate_pipeline_split",
    test_fused_assembly_matches_the_separate_pipeline,
    devices=cuda_devices,
    solve_mode="split",
)
add_function_test(
    TestFeatherPGSFusedCrba,
    "test_fused_assembly_covers_articulations_wider_than_the_block",
    test_fused_assembly_covers_articulations_wider_than_the_block,
    devices=cuda_devices,
)
add_function_test(
    TestFeatherPGSFusedCrba,
    "test_fused_assembly_follows_the_mass_update_interval",
    test_fused_assembly_follows_the_mass_update_interval,
    devices=cuda_devices,
)
add_function_test(
    TestFeatherPGSFusedCrba,
    "test_fused_assembly_is_selected_only_where_supported",
    test_fused_assembly_is_selected_only_where_supported,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
