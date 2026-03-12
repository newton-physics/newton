# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the CGSolver class from linalg/conjugate.py"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import float32
from newton._src.solvers.kamino._src.linalg.conjugate import BatchedLinearOperator, CGSolver, CRSolver
from newton._src.solvers.kamino._src.linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino._src.linalg.linear import ConjugateGradientSolver
from newton._src.solvers.kamino._src.linalg.sparse_matrix import (
    BlockDType,
    BlockSparseMatrices,
    allocate_block_sparse_from_dense,
    dense_to_block_sparse_copy_values,
)
from newton._src.solvers.kamino._src.linalg.utils.rand import random_spd_matrix
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.extract import get_vector_block
from newton._src.solvers.kamino.tests.utils.print import print_error_stats
from newton._src.solvers.kamino.tests.utils.rand import RandomProblemLLT


class TestLinalgConjugate(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for verbose output
        self.seed = 42

    def tearDown(self):
        pass

    def _test_solve(self, solver_cls, problem_params, device):
        problem = RandomProblemLLT(
            **problem_params,
            seed=self.seed,
            np_dtype=np.float32,
            wp_dtype=float32,
            device=device,
        )

        n_worlds = problem.num_blocks
        maxdim = int(problem.maxdims[0])

        b_2d = problem.b_wp.reshape((n_worlds, maxdim))
        x_wp = wp.zeros_like(b_2d, device=device)

        world_active = wp.full(n_worlds, 1, dtype=wp.int32, device=device)

        # Create operator - use maxdim for allocation, then set actual dims
        info = DenseSquareMultiLinearInfo()
        info.finalize(dimensions=[maxdim] * n_worlds, dtype=float32, device=device)
        info.dim = problem.dim_wp  # Override with actual active dimensions
        operator = DenseLinearOperatorData(info=info, mat=problem.A_wp)
        A = BatchedLinearOperator.from_dense(operator)

        atol = wp.full(n_worlds, 1.0e-4, dtype=problem.wp_dtype, device=device)
        rtol = wp.full(n_worlds, 1.0e-5, dtype=problem.wp_dtype, device=device)
        maxiter = wp.full(n_worlds, max(3 * maxdim, 50), dtype=int, device=device)
        solver = solver_cls(
            A=A,
            world_active=world_active,
            atol=atol,
            rtol=rtol,
            maxiter=maxiter,
            Mi=None,
            callback=None,
            use_cuda_graph=False,
        )
        cur_iter, r_norm_sq, atol_sq = solver.solve(b_2d, x_wp)

        x_wp_np = x_wp.numpy().reshape(-1)

        if self.verbose:
            pass
        for block_idx, block_act in enumerate(problem.dims):
            x_found = get_vector_block(block_idx, x_wp_np, problem.dims, problem.maxdims)[:block_act]
            is_x_close = np.allclose(x_found, problem.x_np[block_idx][:block_act], rtol=1e-5, atol=1e-4)
            if self.verbose:
                print(f"Cur iter: {cur_iter}")
                print(f"R norm sq {r_norm_sq}")
                print(f"Atol sq: {atol_sq}")
                if sum(problem.dims) < 20:
                    print("x:")
                    print(x_found)
                    print("x_goal:")
                    print(problem.x_np[block_idx])
                print_error_stats("x", x_found, problem.x_np[block_idx], problem.dims[block_idx])
            self.assertTrue(is_x_close)

    @classmethod
    def _problem_params(cls):
        problems = {
            "small_full": {"maxdims": 7, "dims": [4, 7]},
            "small_partial": {"maxdims": 23, "dims": [14, 11]},
            "large_partial": {"maxdims": 1024, "dims": [11, 51, 101, 376, 999]},
        }
        return problems

    def test_solve_cg_cpu(self):
        device = "cpu"
        solver_cls = CGSolver
        for problem_name, problem_params in self._problem_params().items():
            with self.subTest(problem=problem_name, solver=solver_cls.__name__):
                self._test_solve(solver_cls, problem_params, device)

    def test_solve_cr_cpu(self):
        device = "cpu"
        solver_cls = CRSolver
        for problem_name, problem_params in self._problem_params().items():
            with self.subTest(problem=problem_name, solver=solver_cls.__name__):
                self._test_solve(solver_cls, problem_params, device)

    def test_solve_cg_cuda(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA devices found")
        device = wp.get_cuda_device()
        solver_cls = CGSolver
        for problem_name, problem_params in self._problem_params().items():
            with self.subTest(problem=problem_name, solver=solver_cls.__name__):
                self._test_solve(solver_cls, problem_params, device)

    def test_solve_cr_cuda(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA devices found")
        device = wp.get_cuda_device()
        solver_cls = CRSolver
        for problem_name, problem_params in self._problem_params().items():
            with self.subTest(problem=problem_name, solver=solver_cls.__name__):
                self._test_solve(solver_cls, problem_params, device)

    def _test_sparse_solve(self, solver_cls, n_worlds, dim, block_size, device):
        """Test CG/CR with sparse matrices built from random SPD matrices."""
        rng = np.random.default_rng(self.seed)

        # Pad to block-aligned size
        n_blocks_per_dim = (dim + block_size - 1) // block_size
        padded_dim = n_blocks_per_dim * block_size
        total_blocks = n_blocks_per_dim * n_blocks_per_dim

        # Generate random SPD matrices and RHS vectors
        A_list, A_padded_list, b_list, x_ref_list = [], [], [], []
        for i in range(n_worlds):
            A = random_spd_matrix(dim=dim, seed=self.seed + i, dtype=np.float32)
            A_padded = np.zeros((padded_dim, padded_dim), dtype=np.float32)
            A_padded[:dim, :dim] = A
            b = rng.standard_normal(dim).astype(np.float32)
            A_list.append(A)
            A_padded_list.append(A_padded)
            b_list.append(b)
            x_ref_list.append(np.linalg.solve(A, b))

        # Block coordinates (all blocks, row-major) - same for all worlds
        coords = [
            (bi * block_size, bj * block_size) for bi in range(n_blocks_per_dim) for bj in range(n_blocks_per_dim)
        ]
        all_coords = np.array(coords * n_worlds, dtype=np.int32)

        # Build BlockSparseMatrices
        bsm = BlockSparseMatrices()
        bsm.finalize(
            max_dims=[(padded_dim, padded_dim)] * n_worlds,
            capacities=[total_blocks] * n_worlds,
            nzb_dtype=BlockDType(float32, (block_size, block_size)),
            device=device,
        )
        bsm.dims.assign(np.array([[padded_dim, padded_dim]] * n_worlds, dtype=np.int32))
        bsm.num_nzb.assign(np.array([total_blocks] * n_worlds, dtype=np.int32))
        bsm.nzb_coords.assign(all_coords)
        bsm.assign(A_padded_list)

        # Build dense operator for comparison
        A_dense = np.array([A.flatten() for A in A_padded_list], dtype=np.float32)
        A_wp = wp.array(A_dense, dtype=float32, device=device)
        active_dims = wp.array([dim] * n_worlds, dtype=wp.int32, device=device)

        info = DenseSquareMultiLinearInfo()
        info.finalize(dimensions=[padded_dim] * n_worlds, dtype=float32, device=device)
        info.dim = active_dims
        dense_op = BatchedLinearOperator.from_dense(DenseLinearOperatorData(info=info, mat=A_wp))
        sparse_op = BatchedLinearOperator.from_block_sparse(bsm, active_dims)

        # Prepare RHS
        b_2d = np.zeros((n_worlds, padded_dim), dtype=np.float32)
        for m, b in enumerate(b_list):
            b_2d[m, :dim] = b
        b_wp = wp.array(b_2d, dtype=float32, device=device)

        world_active = wp.full(n_worlds, 1, dtype=wp.int32, device=device)
        atol = wp.full(n_worlds, 1.0e-6, dtype=float32, device=device)
        rtol = wp.full(n_worlds, 1.0e-6, dtype=float32, device=device)

        # Solve with dense operator
        x_dense = wp.zeros((n_worlds, padded_dim), dtype=float32, device=device)
        solver_dense = solver_cls(
            A=dense_op,
            world_active=world_active,
            atol=atol,
            rtol=rtol,
            maxiter=None,
            Mi=None,
            callback=None,
            use_cuda_graph=False,
        )
        solver_dense.solve(b_wp, x_dense)

        # Solve with sparse operator
        x_sparse = wp.zeros((n_worlds, padded_dim), dtype=float32, device=device)
        solver_sparse = solver_cls(
            A=sparse_op,
            world_active=world_active,
            atol=atol,
            rtol=rtol,
            maxiter=None,
            Mi=None,
            callback=None,
            use_cuda_graph=False,
        )
        solver_sparse.solve(b_wp, x_sparse)

        # Compare results
        x_dense_np = x_dense.numpy()
        x_sparse_np = x_sparse.numpy()
        for m in range(n_worlds):
            x_d = x_dense_np[m, :dim]
            x_s = x_sparse_np[m, :dim]
            x_ref = x_ref_list[m]

            if self.verbose:
                print(f"World {m}:")
                print_error_stats("x_dense vs ref", x_d, x_ref, dim)
                print_error_stats("x_sparse vs ref", x_s, x_ref, dim)
                print_error_stats("x_dense vs x_sparse", x_d, x_s, dim)

            self.assertTrue(np.allclose(x_d, x_ref, rtol=1e-3, atol=1e-4), "Dense solution differs from reference")
            self.assertTrue(np.allclose(x_s, x_ref, rtol=1e-3, atol=1e-4), "Sparse solution differs from reference")
            self.assertTrue(np.allclose(x_d, x_s, rtol=1e-5, atol=1e-6), "Dense and sparse solutions differ")

    @classmethod
    def _sparse_problem_params(cls):
        return {
            "small_4x4_blocks": {"n_worlds": 2, "dim": 16, "block_size": 4},
            "medium_6x6_blocks": {"n_worlds": 3, "dim": 48, "block_size": 6},
        }

    def test_sparse_solve_cg_cuda(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA devices found")
        device = wp.get_cuda_device()
        for problem_name, params in self._sparse_problem_params().items():
            with self.subTest(problem=problem_name, solver="CGSolver"):
                self._test_sparse_solve(CGSolver, device=device, **params)

    def test_sparse_solve_cr_cuda(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA devices found")
        device = wp.get_cuda_device()
        for problem_name, params in self._sparse_problem_params().items():
            with self.subTest(problem=problem_name, solver="CRSolver"):
                self._test_sparse_solve(CRSolver, device=device, **params)

    def _build_sparse_operator(self, A: np.ndarray, block_size: int, device):
        """Helper to build a sparse operator from a dense matrix."""
        dim = A.shape[0]
        n_blocks = dim // block_size
        total_blocks = n_blocks * n_blocks

        # Set up block coordinates (all blocks, row-major order)
        coords = [(bi * block_size, bj * block_size) for bi in range(n_blocks) for bj in range(n_blocks)]

        bsm = BlockSparseMatrices()
        bsm.finalize(
            max_dims=[(dim, dim)],
            capacities=[total_blocks],
            nzb_dtype=BlockDType(float32, (block_size, block_size)),
            device=device,
        )
        bsm.dims.assign(np.array([[dim, dim]], dtype=np.int32))
        bsm.num_nzb.assign(np.array([total_blocks], dtype=np.int32))
        bsm.nzb_coords.assign(np.array(coords, dtype=np.int32))
        bsm.assign([A])

        active_dims = wp.array([dim], dtype=wp.int32, device=device)
        return BatchedLinearOperator.from_block_sparse(bsm, active_dims)

    def test_sparse_cg_solve_simple(self):
        """Test CG solve with sparse operator on a 16x16 system with 4x4 blocks."""
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA devices found")
        device = wp.get_cuda_device()

        dim, block_size = 16, 4
        A = random_spd_matrix(dim=dim, seed=self.seed, dtype=np.float32)
        b = np.random.default_rng(self.seed).standard_normal(dim).astype(np.float32)
        x_ref = np.linalg.solve(A, b)

        sparse_op = self._build_sparse_operator(A, block_size, device)

        b_wp = wp.array(b.reshape(1, -1), dtype=float32, device=device)
        x_wp = wp.zeros((1, dim), dtype=float32, device=device)
        world_active = wp.full(1, 1, dtype=wp.int32, device=device)
        atol = wp.full(1, 1e-6, dtype=float32, device=device)
        rtol = wp.full(1, 1e-6, dtype=float32, device=device)

        solver = CGSolver(
            A=sparse_op,
            world_active=world_active,
            atol=atol,
            rtol=rtol,
            maxiter=None,
            Mi=None,
            use_cuda_graph=False,
        )
        solver.solve(b_wp, x_wp)

        x_result = x_wp.numpy().flatten()
        self.assertTrue(
            np.allclose(x_result, x_ref, rtol=1e-3, atol=1e-4),
            f"CG solve failed: {x_result} vs {x_ref}, error={np.abs(x_result - x_ref).max():.2e}",
        )

    def test_dense_to_block_sparse_conversion(self):
        """Test conversion from DenseLinearOperatorData to BlockSparseMatrices and back."""
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA devices found")
        device = wp.get_cuda_device()

        rng = np.random.default_rng(self.seed)
        n_worlds = 4
        block_size = 4
        dims = [12, 16, 8, 20]  # Different dimensions per world

        # Create block-sparse matrices in numpy (some blocks are zero)
        original_matrices = []
        for dim in dims:
            n_blocks = (dim + block_size - 1) // block_size
            matrix = np.zeros((dim, dim), dtype=np.float32)

            # Fill some blocks with random values, leave others as zero
            for bi in range(n_blocks):
                for bj in range(n_blocks):
                    # ~60% chance of non-zero block
                    if rng.random() < 0.6:
                        row_start = bi * block_size
                        col_start = bj * block_size
                        row_end = min(row_start + block_size, dim)
                        col_end = min(col_start + block_size, dim)
                        block_rows = row_end - row_start
                        block_cols = col_end - col_start
                        matrix[row_start:row_end, col_start:col_end] = rng.standard_normal(
                            (block_rows, block_cols)
                        ).astype(np.float32)

            original_matrices.append(matrix)

        # Create DenseLinearOperatorData using canonical compact storage:
        # - Offsets based on maxdim^2 (each world gets maxdim^2 slots)
        # - Within each world, only dim*dim elements stored with stride=dim
        max_dim = max(dims)

        # Allocate with maxdim^2 per world, but only store dim*dim elements compactly
        A_flat = np.full(n_worlds * max_dim * max_dim, np.inf, dtype=np.float32)
        for w, (dim, matrix) in enumerate(zip(dims, original_matrices, strict=False)):
            offset = w * max_dim * max_dim
            # Store compactly with dim as stride (canonical format)
            A_flat[offset : offset + dim * dim] = matrix.flatten()
        A_wp = wp.array(A_flat, dtype=float32, device=device)

        info = DenseSquareMultiLinearInfo()
        info.finalize(dimensions=[max_dim] * n_worlds, dtype=float32, device=device)
        info.dim = wp.array(dims, dtype=wp.int32, device=device)
        dense_op = DenseLinearOperatorData(info=info, mat=A_wp)

        # Allocate BSM with threshold (allow for all blocks)
        bsm = allocate_block_sparse_from_dense(
            dense_op=dense_op,
            block_size=block_size,
            sparsity_threshold=1.0,
            device=device,
        )

        # Convert dense to block sparse
        dense_to_block_sparse_copy_values(
            dense_op=dense_op,
            bsm=bsm,
            block_size=block_size,
        )
        wp.synchronize()

        # Convert back to numpy and compare
        recovered_matrices = bsm.numpy()

        for w, (orig, recovered) in enumerate(zip(original_matrices, recovered_matrices, strict=False)):
            dim = dims[w]
            orig_trimmed = orig[:dim, :dim].astype(np.float32)
            recovered_trimmed = recovered[:dim, :dim].astype(np.float32)

            if self.verbose:
                print(f"World {w} (dim={dim}):")
                print(f"  Original non-zeros: {np.count_nonzero(orig_trimmed)}")
                print(f"  Recovered non-zeros: {np.count_nonzero(recovered_trimmed)}")
                max_diff = np.abs(orig_trimmed - recovered_trimmed).max()
                print(f"  Max abs diff: {max_diff:.2e}")

            self.assertTrue(
                np.allclose(orig_trimmed, recovered_trimmed, rtol=1e-5, atol=1e-6),
                f"World {w}: matrices don't match, max diff={np.abs(orig_trimmed - recovered_trimmed).max():.2e}",
            )

    def test_cg_solver_discover_sparse(self):
        """Test ConjugateGradientSolver with discover_sparse=True."""
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA devices found")
        device = wp.get_cuda_device()

        rng = np.random.default_rng(self.seed)
        n_worlds = 3
        block_size = 6
        maxdim = 24  # Multiple of block_size for clean blocks

        # Generate SPD matrices and RHS
        A_list, b_list, x_ref_list = [], [], []
        for i in range(n_worlds):
            A = random_spd_matrix(dim=maxdim, seed=self.seed + i, dtype=np.float32)
            b = rng.standard_normal(maxdim).astype(np.float32)
            A_list.append(A)
            b_list.append(b)
            x_ref_list.append(np.linalg.solve(A, b))

        # Create dense storage (compact format: dim*dim per world, with maxdim^2 spacing)
        A_flat = np.zeros(n_worlds * maxdim * maxdim, dtype=np.float32)
        for w, A in enumerate(A_list):
            offset = w * maxdim * maxdim
            A_flat[offset : offset + maxdim * maxdim] = A.flatten()
        A_wp = wp.array(A_flat, dtype=float32, device=device)

        # Create DenseLinearOperatorData
        info = DenseSquareMultiLinearInfo()
        info.finalize(dimensions=[maxdim] * n_worlds, dtype=float32, device=device)
        dense_op = DenseLinearOperatorData(info=info, mat=A_wp)

        # Create b and x arrays
        b_2d = np.array(b_list, dtype=np.float32)
        b_wp = wp.array(b_2d.flatten(), dtype=float32, device=device)
        x_wp = wp.zeros(n_worlds * maxdim, dtype=float32, device=device)

        # Solve with discover_sparse=True
        solver = ConjugateGradientSolver(
            discover_sparse=True, sparse_block_size=block_size, sparse_threshold=1.0, device=device
        )
        solver.finalize(dense_op)
        solver.compute(A_wp)
        solver.solve(b_wp, x_wp)

        # Check results
        x_np = x_wp.numpy().reshape(n_worlds, maxdim)
        for w in range(n_worlds):
            x_found = x_np[w]
            x_ref = x_ref_list[w]
            if self.verbose:
                print(f"World {w}: max error = {np.abs(x_found - x_ref).max():.2e}")
            self.assertTrue(
                np.allclose(x_found, x_ref, rtol=1e-3, atol=1e-4),
                f"World {w}: solve failed, max error={np.abs(x_found - x_ref).max():.2e}",
            )

        # Also solve with discover_sparse=False and compare
        x_dense_wp = wp.zeros(n_worlds * maxdim, dtype=float32, device=device)
        solver_dense = ConjugateGradientSolver(discover_sparse=False, device=device)
        solver_dense.finalize(dense_op)
        solver_dense.compute(A_wp)
        solver_dense.solve(b_wp, x_dense_wp)

        x_sparse = x_wp.numpy()
        x_dense = x_dense_wp.numpy()
        if self.verbose:
            print(f"Sparse vs dense max diff: {np.abs(x_sparse - x_dense).max():.2e}")
        self.assertTrue(
            np.allclose(x_sparse, x_dense, rtol=1e-5, atol=1e-6),
            f"Sparse and dense solutions differ: max diff={np.abs(x_sparse - x_dense).max():.2e}",
        )


if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
