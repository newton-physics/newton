# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA tests for the batched Kamino DVI bilateral-response solve."""

from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.kamino._src.solvers.dvi import cublas as cublas_module
from newton._src.solvers.kamino._src.solvers.dvi import solver as dvi_solver_module
from newton._src.solvers.kamino._src.solvers.dvi.cublas import is_batched_trsm_available, solve_llt_batched
from newton._src.solvers.kamino._src.solvers.dvi.kernels import (
    _pack_batched_bilateral_response,
    _scatter_batched_bilateral_response,
    _solve_bilateral_contact_response,
    _solve_bilateral_unilateral_response_cooperative,
)
from newton.tests.kamino import setup_tests, test_context
from newton.viewer import ViewerNull


class TestCublasLibraryDiscovery(unittest.TestCase):
    def test_windows_candidates_use_cuda_bin_and_versioned_fallbacks(self):
        """Discover versioned cuBLAS DLLs from configured CUDA directories."""
        with tempfile.TemporaryDirectory() as root:
            bin_path = Path(root) / "bin"
            bin_path.mkdir()
            for name in ("cublas64_12.dll", "cublas64_13.dll"):
                (bin_path / name).touch()
            with (
                mock.patch.dict(os.environ, {"CUDA_PATH": root, "CUDA_HOME": "", "PATH": ""}),
                mock.patch.object(cublas_module.ctypes.util, "find_library", return_value=None),
            ):
                directories, candidates = cublas_module._windows_cublas_candidates()

        self.assertEqual(directories, [bin_path])
        self.assertEqual([Path(value).name for value in candidates[:2]], ["cublas64_13.dll", "cublas64_12.dll"])
        self.assertIn("cublas64_20.dll", candidates)
        self.assertIn("cublas64_10.dll", candidates)

    def test_windows_loader_uses_stdcall_and_retains_dll_directory(self):
        """Load cuBLAS with the Windows ABI and retain dependency paths."""
        directory = mock.Mock()
        directory_handle = mock.Mock()
        library = mock.Mock()
        with (
            mock.patch.object(cublas_module.os, "name", "nt"),
            mock.patch.object(
                cublas_module, "_windows_cublas_candidates", return_value=([directory], ["cublas64_13.dll"])
            ),
            mock.patch.object(
                cublas_module.os, "add_dll_directory", return_value=directory_handle, create=True
            ) as add_directory,
            mock.patch.object(cublas_module.ctypes, "WinDLL", return_value=library, create=True) as load_library,
        ):
            actual_library, handles = cublas_module._load_cublas_library()

        add_directory.assert_called_once_with(str(directory))
        load_library.assert_called_once_with("cublas64_13.dll")
        self.assertIs(actual_library, library)
        self.assertEqual(handles, [directory_handle])

    def test_thread_handles_are_destroyed_once(self):
        """Destroy each thread-owned cuBLAS handle exactly once."""
        library = mock.Mock()
        device = mock.Mock()
        owner = cublas_module._ThreadHandles(library)
        handle = cublas_module.ctypes.c_void_p(7)
        owner.handles[0] = (device, handle)

        with mock.patch.object(cublas_module.wp, "ScopedDevice"):
            owner.close()
            owner.close()

        library.cublasDestroy_v2.assert_called_once_with(handle)
        self.assertEqual(owner.handles, {})


class TestDVICublas(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_g1_production_path_is_enabled_and_finite(self):
        """Enable the batched G1 response path and keep its state finite."""
        if not self.device.is_cuda or not is_batched_trsm_available(self.device):
            self.skipTest("cuBLAS batched DVI response test requires CUDA")

        module = importlib.import_module(newton.examples.get_examples()["robot_g1"])
        parser = (
            module.Example.create_parser()
            if hasattr(module.Example, "create_parser")
            else newton.examples.create_parser()
        )
        args = newton.examples.default_args(parser)
        args.solver = "kamino"
        args.world_count = 16
        args.num_frames = 20
        args.quiet = True
        example = module.Example(ViewerNull(num_frames=args.num_frames), args)
        sparse_path = example.solver._solver_kamino.solver_fd._sparse_path
        self.assertIsNotNone(sparse_path.batched_response_factor_ptrs)

        for _ in range(args.num_frames):
            example.step()
        self.assertTrue(np.all(np.isfinite(example.state_0.body_q.numpy())))
        self.assertTrue(np.all(np.isfinite(example.state_0.body_qd.numpy())))
        self.assertTrue(np.all(np.isfinite(example.solver._solver_kamino.solver_fd.data.solution.lambdas.numpy())))

    def test_batched_response_matches_reference_with_graph_and_tail(self):
        """Match the batched sparse response to its captured reference."""
        if not self.device.is_cuda or not is_batched_trsm_available(self.device):
            self.skipTest("cuBLAS batched DVI response test requires CUDA")

        rng = np.random.default_rng(90210)
        worlds = 16
        rows = 17
        rhs_count = 64
        stride = 70
        active_rhs = np.array(([0, 1, 7, 63, 64, 67, 2, 65] * 2), dtype=np.int32)
        mio = np.arange(worlds, dtype=np.int32) * rows * rows
        vio = np.arange(worlds, dtype=np.int32) * rows
        response_mio = np.arange(worlds, dtype=np.int32) * rows * stride

        factors = np.empty((worlds, rows, rows), dtype=np.float32)
        permutations = np.empty((worlds, rows), dtype=np.int32)
        preconditioner = rng.uniform(0.5, 1.5, size=(worlds, rows)).astype(np.float32)
        coupling = rng.normal(size=(worlds, rows, stride)).astype(np.float32)
        expected = np.zeros_like(coupling)
        for wid in range(worlds):
            matrix = rng.normal(size=(rows, rows)).astype(np.float32)
            factor = np.linalg.cholesky(matrix @ matrix.T + rows * np.eye(rows, dtype=np.float32)).astype(np.float32)
            permutation = rng.permutation(rows).astype(np.int32)
            factors[wid] = factor
            permutations[wid] = permutation
            for unilateral in range(int(active_rhs[wid])):
                rhs = preconditioner[wid, permutation] * coupling[wid, permutation, unilateral]
                solved = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
                expected[wid, permutation, unilateral] = preconditioner[wid, permutation] * solved

        problem_njc = wp.array(np.full(worlds, rows, dtype=np.int32), dtype=wp.int32, device=self.device)
        problem_dim = wp.array(rows + active_rhs, dtype=wp.int32, device=self.device)
        bilateral_mio = wp.array(mio, dtype=wp.int32, device=self.device)
        bilateral_vio = wp.array(vio, dtype=wp.int32, device=self.device)
        bilateral_p = wp.array(preconditioner.reshape(-1), dtype=wp.float32, device=self.device)
        bilateral_l = wp.array(factors.reshape(-1), dtype=wp.float32, device=self.device)
        permutation = wp.array(permutations.reshape(-1), dtype=wp.int32, device=self.device)
        response_offsets = wp.array(response_mio, dtype=wp.int32, device=self.device)
        response_stride = wp.array(np.full(worlds, stride, dtype=np.int32), dtype=wp.int32, device=self.device)
        coupling_array = wp.array(coupling.reshape(-1), dtype=wp.float32, device=self.device)
        response_factor = wp.zeros(worlds * rows * stride, dtype=wp.float32, device=self.device)
        response = wp.zeros(worlds * rows * stride, dtype=wp.float32, device=self.device)
        scalar_bytes = wp.types.type_size_in_bytes(wp.float32)
        factor_ptrs = wp.array(
            [bilateral_l.ptr + scalar_bytes * int(offset) for offset in mio], dtype=wp.uint64, device=self.device
        )
        rhs_ptrs = wp.array(
            [response_factor.ptr + scalar_bytes * int(offset) for offset in response_mio],
            dtype=wp.uint64,
            device=self.device,
        )

        def launch_response():
            wp.launch(
                _pack_batched_bilateral_response,
                dim=(worlds, rhs_count, rows),
                inputs=[
                    problem_dim,
                    problem_njc,
                    bilateral_vio,
                    bilateral_p,
                    permutation,
                    True,
                    response_offsets,
                    response_stride,
                    False,
                    response_offsets,
                    coupling_array,
                    response_factor,
                ],
                device=self.device,
            )
            solve_llt_batched(factor_ptrs, rhs_ptrs, rows, rhs_count, worlds)
            wp.launch(
                _scatter_batched_bilateral_response,
                dim=(worlds, rhs_count, rows),
                inputs=[
                    problem_dim,
                    problem_njc,
                    bilateral_vio,
                    bilateral_p,
                    permutation,
                    True,
                    response_offsets,
                    response_offsets,
                    response_stride,
                    False,
                    True,
                    response_factor,
                    response,
                ],
                device=self.device,
            )
            wp.launch(
                dim=worlds * 8 * 32,
                kernel=_solve_bilateral_unilateral_response_cooperative,
                inputs=[
                    problem_dim,
                    problem_njc,
                    bilateral_mio,
                    bilateral_vio,
                    bilateral_p,
                    bilateral_l,
                    permutation,
                    True,
                    response_offsets,
                    response_stride,
                    coupling_array,
                    response_factor,
                    response,
                    rhs_count,
                    8,
                ],
                device=self.device,
                block_dim=32,
            )

        launch_response()
        wp.synchronize_device(self.device)
        response.zero_()
        with wp.ScopedCapture(self.device) as capture:
            launch_response()
        wp.capture_launch(capture.graph)

        actual = response.numpy().reshape(worlds, rows, stride)
        for wid in range(worlds):
            nu = int(active_rhs[wid])
            np.testing.assert_allclose(actual[wid, :, :nu], expected[wid, :, :nu], rtol=3.0e-4, atol=3.0e-4)
            np.testing.assert_array_equal(actual[wid, :, nu:], 0.0)

    def test_basic_urdf_dense_production_path_matches_scalar_response(self):
        """Match the dense basic URDF production path to the scalar response."""
        if not self.device.is_cuda or not is_batched_trsm_available(self.device):
            self.skipTest("cuBLAS batched DVI response test requires CUDA")

        module = importlib.import_module(newton.examples.get_examples()["basic_urdf"])
        parser = (
            module.Example.create_parser()
            if hasattr(module.Example, "create_parser")
            else newton.examples.create_parser()
        )

        def run(min_worlds: int):
            args = newton.examples.default_args(parser)
            args.solver = "kamino"
            args.num_frames = 1
            args.quiet = True
            with mock.patch.object(dvi_solver_module, "_BATCHED_DENSE_RESPONSE_MIN_WORLDS", min_worlds):
                example = module.Example(ViewerNull(num_frames=1), args)
            solver = example.solver._solver_kamino.solver_fd
            enabled = solver._dense_response_factor_ptrs is not None
            example.step()
            return (
                enabled,
                example.state_0.body_q.numpy(),
                example.state_0.body_qd.numpy(),
                solver.data.solution.lambdas.numpy(),
            )

        scalar = run(10_000)
        batched = run(16)
        self.assertFalse(scalar[0])
        self.assertTrue(batched[0])
        for scalar_value, batched_value in zip(scalar[1:], batched[1:], strict=True):
            self.assertTrue(np.all(np.isfinite(batched_value)))
            np.testing.assert_allclose(batched_value, scalar_value, rtol=5.0e-4, atol=5.0e-5)

    def test_dense_batched_response_matches_scalar_layout_with_tail_and_graph(self):
        """Match captured dense batched responses including inactive tail rows."""
        if not self.device.is_cuda or not is_batched_trsm_available(self.device):
            self.skipTest("cuBLAS batched DVI response test requires CUDA")

        rng = np.random.default_rng(51173)
        worlds = 16
        rows = 9
        batched_rhs = 64
        total_rhs = 70
        dim = rows + total_rhs
        matrix_size = dim * dim

        problem_mio = np.arange(worlds, dtype=np.int32) * matrix_size
        bilateral_mio = np.arange(worlds, dtype=np.int32) * rows * rows
        bilateral_vio = np.arange(worlds, dtype=np.int32) * rows
        factor_mio = np.arange(worlds, dtype=np.int32) * rows * batched_rhs
        factors = np.empty((worlds, rows, rows), dtype=np.float32)
        permutations = np.empty((worlds, rows), dtype=np.int32)
        preconditioner = rng.uniform(0.5, 1.5, size=(worlds, rows)).astype(np.float32)
        dense_operator = np.zeros((worlds, dim, dim), dtype=np.float32)
        expected = np.zeros_like(dense_operator)
        for wid in range(worlds):
            matrix = rng.normal(size=(rows, rows)).astype(np.float32)
            factor = np.linalg.cholesky(matrix @ matrix.T + rows * np.eye(rows, dtype=np.float32)).astype(np.float32)
            permutation = rng.permutation(rows).astype(np.int32)
            coupling = rng.normal(size=(rows, total_rhs)).astype(np.float32)
            factors[wid] = factor
            permutations[wid] = permutation
            dense_operator[wid, :rows, rows:] = coupling
            for unilateral in range(total_rhs):
                rhs = preconditioner[wid, permutation] * coupling[permutation, unilateral]
                solved = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
                expected[wid, :rows, rows + unilateral] = solved

        problem_dim = wp.array(np.full(worlds, dim, dtype=np.int32), dtype=wp.int32, device=self.device)
        problem_njc = wp.array(np.full(worlds, rows, dtype=np.int32), dtype=wp.int32, device=self.device)
        problem_offsets = wp.array(problem_mio, dtype=wp.int32, device=self.device)
        bilateral_offsets = wp.array(bilateral_mio, dtype=wp.int32, device=self.device)
        bilateral_vector_offsets = wp.array(bilateral_vio, dtype=wp.int32, device=self.device)
        factor_offsets = wp.array(factor_mio, dtype=wp.int32, device=self.device)
        preconditioner_array = wp.array(preconditioner.reshape(-1), dtype=wp.float32, device=self.device)
        factor_array = wp.array(factors.reshape(-1), dtype=wp.float32, device=self.device)
        permutation_array = wp.array(permutations.reshape(-1), dtype=wp.int32, device=self.device)
        dense_operator_array = wp.array(dense_operator.reshape(-1), dtype=wp.float32, device=self.device)
        response_factor = wp.zeros(worlds * rows * batched_rhs, dtype=wp.float32, device=self.device)
        projected = wp.zeros(worlds * matrix_size, dtype=wp.float32, device=self.device)
        scalar_bytes = wp.types.type_size_in_bytes(wp.float32)
        factor_ptrs = wp.array(
            [factor_array.ptr + scalar_bytes * int(offset) for offset in bilateral_mio],
            dtype=wp.uint64,
            device=self.device,
        )
        rhs_ptrs = wp.array(
            [response_factor.ptr + scalar_bytes * int(offset) for offset in factor_mio],
            dtype=wp.uint64,
            device=self.device,
        )

        def launch_response():
            wp.launch(
                _pack_batched_bilateral_response,
                dim=(worlds, batched_rhs, rows),
                inputs=[
                    problem_dim,
                    problem_njc,
                    bilateral_vector_offsets,
                    preconditioner_array,
                    permutation_array,
                    True,
                    problem_offsets,
                    problem_dim,
                    True,
                    factor_offsets,
                    dense_operator_array,
                    response_factor,
                ],
                device=self.device,
            )
            solve_llt_batched(factor_ptrs, rhs_ptrs, rows, batched_rhs, worlds)
            wp.launch(
                _scatter_batched_bilateral_response,
                dim=(worlds, batched_rhs, rows),
                inputs=[
                    problem_dim,
                    problem_njc,
                    bilateral_vector_offsets,
                    preconditioner_array,
                    permutation_array,
                    True,
                    factor_offsets,
                    problem_offsets,
                    problem_dim,
                    True,
                    False,
                    response_factor,
                    projected,
                ],
                device=self.device,
            )
            wp.launch(
                _solve_bilateral_contact_response,
                dim=(worlds, total_rhs - batched_rhs),
                inputs=[
                    problem_dim,
                    problem_offsets,
                    problem_njc,
                    bilateral_offsets,
                    bilateral_vector_offsets,
                    preconditioner_array,
                    problem_offsets,
                    dense_operator_array,
                    factor_array,
                    permutation_array,
                    True,
                    batched_rhs,
                    projected,
                ],
                device=self.device,
            )

        launch_response()
        wp.synchronize_device(self.device)
        projected.zero_()
        with wp.ScopedCapture(self.device) as capture:
            launch_response()
        wp.capture_launch(capture.graph)

        actual = projected.numpy().reshape(worlds, dim, dim)
        np.testing.assert_allclose(actual[:, :rows, rows:], expected[:, :rows, rows:], rtol=3.0e-4, atol=3.0e-4)
        np.testing.assert_array_equal(actual[:, rows:, :], 0.0)


if __name__ == "__main__":
    unittest.main()
