# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the base classes in linalg/sparse.py"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.sparse_matrix import BlockDType, BlockSparseMatrices
from newton._src.solvers.kamino._src.linalg.sparse_operator import BlockSparseLinearOperators
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.sparse import sparseplot
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Tests
###


class TestBlockDType(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.seed = 42
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    ###
    # Construction Tests
    ###

    def test_00_make_block_dtype(self):
        # Default construction should fail
        self.assertRaises(TypeError, BlockDType)

        # Scalar block type, shape should be `()` to match numpy scalar behavior
        scalar_block_type_0 = BlockDType(dtype=wp.float32)
        self.assertEqual(scalar_block_type_0.dtype, wp.float32)
        self.assertEqual(scalar_block_type_0.shape, ())

        scalar_block_type_1 = BlockDType(shape=1, dtype=wp.float32)
        self.assertEqual(scalar_block_type_1.dtype, wp.float32)
        self.assertEqual(scalar_block_type_1.shape, ())

        # Vector block types
        vector_block_type_0 = BlockDType(shape=2, dtype=wp.float32)
        self.assertEqual(vector_block_type_0.dtype, wp.float32)
        self.assertEqual(vector_block_type_0.shape, (2,))

        vector_block_type_1 = BlockDType(shape=(3,), dtype=wp.float32)
        self.assertEqual(vector_block_type_1.dtype, wp.float32)
        self.assertEqual(vector_block_type_1.shape, (3,))

        # Matrix block types
        matrix_block_type_0 = BlockDType(shape=(2, 4), dtype=wp.float32)
        self.assertEqual(matrix_block_type_0.dtype, wp.float32)
        self.assertEqual(matrix_block_type_0.shape, (2, 4))

        # Invalid shape specifications should fail
        self.assertRaises(ValueError, BlockDType, shape=0, dtype=wp.float32)
        self.assertRaises(ValueError, BlockDType, shape=(-2,), dtype=wp.float32)
        self.assertRaises(ValueError, BlockDType, shape=(3, -4), dtype=wp.float32)
        self.assertRaises(ValueError, BlockDType, shape=(1, 2, 3), dtype=wp.float32)

        # Invalid dtype specifications should fail
        self.assertRaises(TypeError, BlockDType, shape=2, dtype=None)
        self.assertRaises(TypeError, BlockDType, shape=(2, 2), dtype=str)

    def test_01_block_dtype_size(self):
        # Scalar block type
        scalar_block_type = BlockDType(dtype=wp.float32)
        self.assertEqual(scalar_block_type.size, 1)

        # Vector block type
        vector_block_type = BlockDType(shape=4, dtype=wp.float32)
        self.assertEqual(vector_block_type.size, 4)

        # Matrix block type
        matrix_block_type = BlockDType(shape=(3, 5), dtype=wp.float32)
        self.assertEqual(matrix_block_type.size, 15)

    def test_02_block_dtype_warp_type(self):
        # Scalar block type
        scalar_block_type = BlockDType(dtype=wp.float32)
        warp_scalar_type = scalar_block_type.warp_type
        self.assertEqual(warp_scalar_type, wp.float32)

        # Vector block type
        vector_block_type = BlockDType(shape=4, dtype=wp.float32)
        warp_vector_type = vector_block_type.warp_type
        self.assertEqual(warp_vector_type._length_, 4)
        self.assertEqual(warp_vector_type._wp_scalar_type_, wp.float32)

        # Matrix block type
        matrix_block_type = BlockDType(shape=(3, 5), dtype=wp.float32)
        warp_matrix_type = matrix_block_type.warp_type
        self.assertEqual(warp_matrix_type._shape_, (3, 5))
        self.assertEqual(warp_matrix_type._length_, 15)
        self.assertEqual(warp_matrix_type._wp_scalar_type_, wp.float32)


class TestBlockSparseMatrices(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.seed = 42
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output
        self.plot = test_context.verbose  # Set to True to plot sparse matrices

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    ###
    # Construction Tests
    ###

    def test_00_make_default(self):
        bsm = BlockSparseMatrices()
        self.assertIsInstance(bsm, BlockSparseMatrices)

        # Host-side meta-data should be default-initialized
        self.assertIsNone(bsm.device)
        self.assertEqual(bsm.num_matrices, 0)
        self.assertEqual(bsm.sum_of_num_nzb, 0)
        self.assertEqual(bsm.max_of_num_nzb, 0)
        self.assertEqual(bsm.max_of_max_dims, (0, 0))
        self.assertIsNone(bsm.nzb_dtype)
        self.assertIs(bsm.index_dtype, wp.int32)

        # On-device data should be None
        self.assertIsNone(bsm.max_dims)
        self.assertIsNone(bsm.dims)
        self.assertIsNone(bsm.row_start)
        self.assertIsNone(bsm.col_start)
        self.assertIsNone(bsm.max_nzb)
        self.assertIsNone(bsm.num_nzb)
        self.assertIsNone(bsm.nzb_start)
        self.assertIsNone(bsm.nzb_coords)
        self.assertIsNone(bsm.nzb_values)

        # Finalization should fail since the block size `nzb_size` is not set
        self.assertRaises(RuntimeError, bsm.finalize, max_dims=[(0, 0)], capacities=[0])

    def test_01_make_single_scalar_block_sparse_matrix(self):
        bsm = BlockSparseMatrices(num_matrices=1, nzb_dtype=BlockDType(dtype=wp.float32), device=self.default_device)
        bsm.finalize(max_dims=[(1, 1)], capacities=[1])

        # Check meta-data
        self.assertEqual(bsm.num_matrices, 1)
        self.assertEqual(bsm.sum_of_num_nzb, 1)
        self.assertEqual(bsm.max_of_num_nzb, 1)
        self.assertEqual(bsm.max_of_max_dims, (1, 1))
        self.assertEqual(bsm.nzb_dtype.dtype, wp.float32)
        self.assertEqual(bsm.nzb_dtype.shape, ())
        self.assertIs(bsm.index_dtype, wp.int32)
        self.assertEqual(bsm.device, self.default_device)

        # Check on-device data shapes
        self.assertEqual(bsm.max_dims.shape, (1, 2))
        self.assertEqual(bsm.dims.shape, (1, 2))
        self.assertEqual(bsm.row_start.shape, (1,))
        self.assertEqual(bsm.col_start.shape, (1,))
        self.assertEqual(bsm.max_nzb.shape, (1,))
        self.assertEqual(bsm.num_nzb.shape, (1,))
        self.assertEqual(bsm.nzb_start.shape, (1,))
        self.assertEqual(bsm.nzb_coords.shape, (1, 2))
        self.assertEqual(bsm.nzb_values.shape, (1,))
        self.assertEqual(bsm.nzb_values.size, 1)
        self.assertEqual(bsm.nzb_values.view(dtype=wp.float32).size, 1)

    def test_02_make_single_vector_block_sparse_matrix(self):
        bsm = BlockSparseMatrices(num_matrices=1, nzb_dtype=BlockDType(shape=(6,), dtype=wp.float32))
        bsm.finalize(max_dims=[(6, 1)], capacities=[1], device=self.default_device)

        # Check meta-data
        self.assertEqual(bsm.num_matrices, 1)
        self.assertEqual(bsm.sum_of_num_nzb, 1)
        self.assertEqual(bsm.max_of_num_nzb, 1)
        self.assertEqual(bsm.max_of_max_dims, (6, 1))
        self.assertEqual(bsm.nzb_dtype.dtype, wp.float32)
        self.assertEqual(bsm.nzb_dtype.shape, (6,))
        self.assertIs(bsm.index_dtype, wp.int32)
        self.assertEqual(bsm.device, self.default_device)

        # Check on-device data shapes
        self.assertEqual(bsm.max_dims.shape, (1, 2))
        self.assertEqual(bsm.dims.shape, (1, 2))
        self.assertEqual(bsm.row_start.shape, (1,))
        self.assertEqual(bsm.col_start.shape, (1,))
        self.assertEqual(bsm.max_nzb.shape, (1,))
        self.assertEqual(bsm.num_nzb.shape, (1,))
        self.assertEqual(bsm.nzb_start.shape, (1,))
        self.assertEqual(bsm.nzb_coords.shape, (1, 2))
        self.assertEqual(bsm.nzb_values.shape, (1,))
        self.assertEqual(bsm.nzb_values.size, 1)
        self.assertEqual(bsm.nzb_values.view(dtype=wp.float32).size, 6)

    def test_03_make_single_matrix_block_sparse_matrix(self):
        bsm = BlockSparseMatrices(num_matrices=1, nzb_dtype=BlockDType(shape=(6, 5), dtype=wp.float32))
        bsm.finalize(max_dims=[(6, 5)], capacities=[1], device=self.default_device)

        # Check meta-data
        self.assertEqual(bsm.num_matrices, 1)
        self.assertEqual(bsm.sum_of_num_nzb, 1)
        self.assertEqual(bsm.max_of_num_nzb, 1)
        self.assertEqual(bsm.max_of_max_dims, (6, 5))
        self.assertEqual(bsm.nzb_dtype.dtype, wp.float32)
        self.assertEqual(bsm.nzb_dtype.shape, (6, 5))
        self.assertIs(bsm.index_dtype, wp.int32)
        self.assertEqual(bsm.device, self.default_device)

        # Check on-device data shapes
        self.assertEqual(bsm.max_dims.shape, (1, 2))
        self.assertEqual(bsm.dims.shape, (1, 2))
        self.assertEqual(bsm.row_start.shape, (1,))
        self.assertEqual(bsm.col_start.shape, (1,))
        self.assertEqual(bsm.max_nzb.shape, (1,))
        self.assertEqual(bsm.num_nzb.shape, (1,))
        self.assertEqual(bsm.nzb_start.shape, (1,))
        self.assertEqual(bsm.nzb_coords.shape, (1, 2))
        self.assertEqual(bsm.nzb_values.shape, (1,))
        self.assertEqual(bsm.nzb_values.size, 1)
        self.assertEqual(bsm.nzb_values.view(dtype=wp.float32).size, 30)

    def test_04_build_multiple_vector_block_matrices(self):
        bsm = BlockSparseMatrices(num_matrices=1, nzb_dtype=BlockDType(shape=(6,), dtype=wp.float32))
        bsm.finalize(max_dims=[(6, 1), (12, 2), (6, 4)], capacities=[3, 4, 5], device=self.default_device)

        # Check meta-data
        self.assertEqual(bsm.num_matrices, 3)
        self.assertEqual(bsm.sum_of_num_nzb, 12)
        self.assertEqual(bsm.max_of_num_nzb, 5)
        self.assertEqual(bsm.max_of_max_dims, (12, 4))
        self.assertEqual(bsm.nzb_dtype.dtype, wp.float32)
        self.assertEqual(bsm.nzb_dtype.shape, (6,))
        self.assertIs(bsm.index_dtype, wp.int32)
        self.assertEqual(bsm.device, self.default_device)

        # Check on-device data shapes
        self.assertEqual(bsm.max_dims.shape, (3, 2))
        self.assertEqual(bsm.dims.shape, (3, 2))
        self.assertEqual(bsm.row_start.shape, (3,))
        self.assertEqual(bsm.col_start.shape, (3,))
        self.assertEqual(bsm.max_nzb.shape, (3,))
        self.assertEqual(bsm.num_nzb.shape, (3,))
        self.assertEqual(bsm.nzb_start.shape, (3,))
        self.assertEqual(bsm.nzb_coords.shape, (12, 2))
        self.assertEqual(bsm.nzb_values.shape, (12,))
        self.assertEqual(bsm.nzb_values.size, 12)
        self.assertEqual(bsm.nzb_values.view(dtype=wp.float32).size, 72)

    ###
    # Building Tests
    ###

    def test_10_build_multiple_vector_block_sparse_matrices_full(self):
        """
        Tests building two fully-filled block-sparse matrices with vector-shaped blocks and same overall shape.
        """
        bsm = BlockSparseMatrices(num_matrices=2, nzb_dtype=BlockDType(shape=(6,), dtype=wp.float32))
        bsm.finalize(max_dims=[(2, 12), (2, 12)], capacities=[2, 3], device=self.default_device)

        # Check meta-data
        self.assertEqual(bsm.num_matrices, 2)
        self.assertEqual(bsm.sum_of_num_nzb, 5)
        self.assertEqual(bsm.max_of_num_nzb, 3)
        self.assertEqual(bsm.max_of_max_dims, (2, 12))
        self.assertEqual(bsm.nzb_dtype.dtype, wp.float32)
        self.assertEqual(bsm.nzb_dtype.shape, (6,))
        self.assertIs(bsm.index_dtype, wp.int32)
        self.assertEqual(bsm.device, self.default_device)

        # Check on-device data shapes
        self.assertEqual(bsm.max_dims.shape, (bsm.num_matrices, 2))
        self.assertEqual(bsm.dims.shape, (bsm.num_matrices, 2))
        self.assertEqual(bsm.row_start.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.col_start.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.max_nzb.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.num_nzb.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.nzb_start.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.nzb_coords.shape, (bsm.sum_of_num_nzb, 2))
        self.assertEqual(bsm.nzb_values.shape, (bsm.sum_of_num_nzb,))
        self.assertEqual(bsm.nzb_values.size, bsm.sum_of_num_nzb)
        self.assertEqual(bsm.nzb_values.view(dtype=wp.float32).size, bsm.sum_of_num_nzb * bsm.nzb_dtype.size)

        # Build each matrix as follows:
        # Matrix 0: 2x12 block0diagonal with 2 non-zero blocks at on diagonals (0,0) and (1,6)
        # Matrix 1: 2x12 upper-block-triangular with 3 non-zero blocks at on at (0,0), (0,6), and (1,6)
        nzb_dims_np = np.array([[2, 12], [2, 12]], dtype=np.int32)
        num_nzb_np = np.array([[2], [3]], dtype=np.int32)
        nzb_coords_np = np.array([[0, 0], [1, 6], [0, 0], [0, 6], [1, 6]], dtype=np.int32)
        nzb_values_np = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            ],
            dtype=np.float32,
        )
        bsm.dims.assign(nzb_dims_np)
        bsm.num_nzb.assign(num_nzb_np)
        bsm.nzb_coords.assign(nzb_coords_np)
        bsm.nzb_values.view(dtype=wp.float32).assign(nzb_values_np)
        msg.info("bsm.max_of_max_dims:\n%s", bsm.max_of_max_dims)
        msg.info("bsm.max_dims:\n%s", bsm.max_dims)
        msg.info("bsm.dims:\n%s", bsm.dims)
        msg.info("bsm.max_nzb:\n%s", bsm.max_nzb)
        msg.info("bsm.num_nzb:\n%s", bsm.num_nzb)
        msg.info("bsm.nzb_start:\n%s", bsm.nzb_start)
        msg.info("bsm.nzb_coords:\n%s", bsm.nzb_coords)
        msg.info("bsm.nzb_values:\n%s", bsm.nzb_values)

        # Check host device data
        self.assertEqual(bsm.max_of_max_dims, (2, 12))

        # Check on-device data shapes again to ensure nothing changed during building
        self.assertEqual(bsm.max_dims.shape, (bsm.num_matrices, 2))
        self.assertEqual(bsm.dims.shape, (bsm.num_matrices, 2))
        self.assertEqual(bsm.row_start.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.col_start.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.max_nzb.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.num_nzb.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.nzb_start.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.nzb_coords.shape, (bsm.sum_of_num_nzb, 2))
        self.assertEqual(bsm.nzb_values.shape, (bsm.sum_of_num_nzb,))
        self.assertEqual(bsm.nzb_values.size, bsm.sum_of_num_nzb)
        self.assertEqual(bsm.nzb_values.view(dtype=wp.float32).size, bsm.sum_of_num_nzb * bsm.nzb_dtype.size)

        # Convert to list of numpy arrays for easier verification
        bsm_np = bsm.numpy()
        for i in range(bsm.num_matrices):
            msg.info("bsm_np[%d]:\n%s", i, bsm_np[i])
            if self.plot:
                sparseplot(bsm_np[i], title=f"bsm_np[{i}]")

        # Assign new values to the dense numpy arrays and set them back to the block-sparse matrices
        for i in range(bsm.num_matrices):
            bsm_np[i] += 1.0 * (i + 1)
        bsm.assign(bsm_np)

        # Convert again to list of numpy arrays for easier verification
        bsm_np = bsm.numpy()
        for i in range(bsm.num_matrices):
            msg.info("bsm_np[%d]:\n%s", i, bsm_np[i])
            if self.plot:
                sparseplot(bsm_np[i], title=f"bsm_np[{i}]")


class TestBlockSparseMatrixOperations(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.epsilon = 1e-5  # Threshold for matvec product test
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output
        self.plot = test_context.verbose  # Set to True to plot sparse matrices

        # Random number generation.
        self.seed = 42
        self.rng = None

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    ###
    # Test Helpers
    ###

    def _matvec_product_check(self, ops: BlockSparseLinearOperators):
        """Tests the regular matrix-vector product and the generalized matrix-vector product for the
        given operator, both in regular and transposed versions."""
        bsm = ops.bsm
        num_matrices = bsm.num_matrices
        matrix_max_dims = bsm.max_dims.numpy()
        matrix_dims = bsm.dims.numpy()
        row_start_np = bsm.row_start.numpy()
        col_start_np = bsm.col_start.numpy()

        matrix_max_dims_sum = np.sum(matrix_max_dims, axis=0)

        def product_check(transpose: bool, mask_matrices: bool):
            input_dim, output_dim = (0, 1) if transpose else (1, 0)
            size_input = matrix_max_dims_sum[input_dim]
            size_output = matrix_max_dims_sum[output_dim]
            input_start, output_start = (row_start_np, col_start_np) if transpose else (col_start_np, row_start_np)

            if mask_matrices:
                mask_np = np.ones((num_matrices,), dtype=np.int32)
                mask_np[::2] = 0
                matrix_mask = wp.from_numpy(mask_np, dtype=wp.int32, device=self.default_device)
            else:
                matrix_mask = wp.ones((num_matrices,), dtype=wp.int32, device=self.default_device)

            # Create vectors for matrix-vector multiplications.
            alpha = float(self.rng.standard_normal((1,))[0])
            beta = float(self.rng.standard_normal((1,))[0])
            input_vectors = [self.rng.standard_normal((shape[input_dim],)) for shape in matrix_dims]
            offset_vectors = [self.rng.standard_normal((shape[output_dim],)) for shape in matrix_dims]
            input_vec_np = np.zeros((size_input,), dtype=np.float32)
            offset_vec_np = np.zeros((size_output,), dtype=np.float32)
            for mat_id in range(num_matrices):
                input_vec_np[input_start[mat_id] : input_start[mat_id] + matrix_dims[mat_id, input_dim]] = (
                    input_vectors[mat_id]
                )
                offset_vec_np[output_start[mat_id] : output_start[mat_id] + matrix_dims[mat_id, output_dim]] = (
                    offset_vectors[mat_id]
                )

            # Compute matrix-vector product.
            input_vec = wp.from_numpy(input_vec_np, dtype=wp.float32, device=self.default_device)
            output_vec_matmul = wp.zeros((size_output,), dtype=wp.float32, device=self.default_device)
            output_vec_gemv = wp.from_numpy(offset_vec_np, dtype=wp.float32, device=self.default_device)

            if transpose:
                ops.matvec_transpose(input_vec, output_vec_matmul, matrix_mask)
                ops.gemv_transpose(input_vec, output_vec_gemv, matrix_mask, alpha, beta)
            else:
                ops.matvec(input_vec, output_vec_matmul, matrix_mask)
                ops.gemv(input_vec, output_vec_gemv, matrix_mask, alpha, beta)

            # Compare result to dense matrix-vector product.
            matrices_np = bsm.numpy()
            output_vec_matmul_np = output_vec_matmul.numpy()
            output_vec_gemv_np = output_vec_gemv.numpy()
            matrix_mask_np = matrix_mask.numpy()
            for mat_id in range(num_matrices):
                if matrix_mask_np[mat_id] == 0:
                    output_matmul = output_vec_matmul_np[
                        output_start[mat_id] : output_start[mat_id] + matrix_dims[mat_id, output_dim]
                    ]
                    self.assertEqual(np.max(np.abs(output_matmul)), 0.0)
                    diff_gemv = (
                        offset_vec_np[output_start[mat_id] : output_start[mat_id] + matrix_dims[mat_id, output_dim]]
                        - output_vec_gemv_np[
                            output_start[mat_id] : output_start[mat_id] + matrix_dims[mat_id, output_dim]
                        ]
                    )
                    self.assertEqual(np.max(np.abs(diff_gemv)), 0.0)
                else:
                    if transpose:
                        output_vec_matmul_ref = matrices_np[mat_id].T @ input_vectors[mat_id]
                    else:
                        output_vec_matmul_ref = matrices_np[mat_id] @ input_vectors[mat_id]
                    output_vec_gemv_ref = alpha * output_vec_matmul_ref + beta * offset_vectors[mat_id]

                    diff_matmul = (
                        output_vec_matmul_ref
                        - output_vec_matmul_np[
                            output_start[mat_id] : output_start[mat_id] + matrix_dims[mat_id, output_dim]
                        ]
                    )
                    self.assertLess(np.max(np.abs(diff_matmul)), self.epsilon)
                    diff_gemv = (
                        output_vec_gemv_ref
                        - output_vec_gemv_np[
                            output_start[mat_id] : output_start[mat_id] + matrix_dims[mat_id, output_dim]
                        ]
                    )
                    self.assertLess(np.max(np.abs(diff_gemv)), self.epsilon)

        product_check(transpose=False, mask_matrices=False)
        product_check(transpose=False, mask_matrices=True)
        product_check(transpose=True, mask_matrices=False)
        product_check(transpose=True, mask_matrices=True)

    ###
    # Matrix-Vector Product Tests
    ###

    def test_00_sparse_matrix_vector_product_full(self):
        """
        Tests multiplication of a random dense block matrix with a random vector.
        """

        # Test dimensions.
        blocks_per_dim = np.array([[4, 4], [6, 4], [4, 6]], dtype=np.int32)
        block_dims_array = [(1,), (3,), (2, 2)]

        self.rng = np.random.default_rng(seed=self.seed)

        for block_dims_short in block_dims_array:
            num_matrices = len(blocks_per_dim)
            block_dims = (1, block_dims_short[0]) if len(block_dims_short) == 1 else block_dims_short

            # Add offsets for max dimensions (in terms of blocks).
            max_blocks_per_dim = blocks_per_dim.copy()
            for mat_id in range(num_matrices):
                max_blocks_per_dim[mat_id, :] += [2 * mat_id + 3, 2 * mat_id + 4]

            # Compute matrix dimensions in terms of entries.
            matrix_dims = [(int(shape[0] * block_dims[0]), int(shape[1] * block_dims[1])) for shape in blocks_per_dim]
            matrix_max_dims = [
                (int(shape[0] * block_dims[0]), int(shape[1] * block_dims[1])) for shape in max_blocks_per_dim
            ]

            # Generate random matrix and vector.
            matrices = [self.rng.standard_normal((shape[0], shape[1])) for shape in matrix_dims]

            bsm = BlockSparseMatrices(
                num_matrices=num_matrices, nzb_dtype=BlockDType(shape=block_dims_short, dtype=wp.float32)
            )
            capacities = np.asarray([shape[0] * shape[1] for shape in max_blocks_per_dim], dtype=np.int32)
            num_nzb_np = np.asarray([shape[0] * shape[1] for shape in blocks_per_dim], dtype=np.int32)
            bsm.finalize(max_dims=matrix_max_dims, capacities=[int(c) for c in capacities], device=self.default_device)

            # Fill in sparse matrix data structure.
            nzb_start_np = bsm.nzb_start.numpy()
            nzb_coords_np = np.zeros((bsm.sum_of_num_nzb, 2), dtype=np.int32)
            nzb_values_np = np.zeros((bsm.sum_of_num_nzb, block_dims[0], block_dims[1]), dtype=np.float32)
            for mat_id in range(num_matrices):
                for outer_row_id in range(blocks_per_dim[mat_id, 0]):
                    row_id = outer_row_id * block_dims[0]
                    for outer_col_id in range(blocks_per_dim[mat_id, 1]):
                        col_id = outer_col_id * block_dims[1]
                        global_idx = nzb_start_np[mat_id] + outer_row_id * blocks_per_dim[mat_id, 1] + outer_col_id
                        nzb_coords_np[global_idx, :] = [row_id, col_id]
                        nzb_values_np[global_idx, :, :] = matrices[mat_id][
                            row_id : row_id + block_dims[0], col_id : col_id + block_dims[1]
                        ]

            bsm.dims.assign(matrix_dims)
            bsm.num_nzb.assign(num_nzb_np)
            bsm.nzb_coords.assign(nzb_coords_np)
            bsm.nzb_values.view(dtype=wp.float32).assign(nzb_values_np)

            # Build operator.
            ops = BlockSparseLinearOperators(bsm)

            # Run multiplication operator checks.
            self._matvec_product_check(ops)

    def test_01_sparse_matrix_vector_product_partial(self):
        """
        Tests multiplication of a random block sparse matrix with a random vector.
        """

        # Test dimensions.
        blocks_per_dim = np.array([[7, 7], [17, 11], [13, 19]], dtype=np.int32)
        block_dims_array = [(1,), (3,), (2, 2)]
        sparse_block_offset = 5  # Every i-th block will be filled.

        self.rng = np.random.default_rng(seed=self.seed)

        for block_dims_short in block_dims_array:
            num_matrices = len(blocks_per_dim)
            block_dims = (1, block_dims_short[0]) if len(block_dims_short) == 1 else block_dims_short

            # Add offsets for max dimensions (in terms of blocks).
            max_blocks_per_dim = blocks_per_dim.copy()
            for mat_id in range(num_matrices):
                max_blocks_per_dim[mat_id, :] += [2 * mat_id + 3, 2 * mat_id + 4]

            # Compute matrix dimensions in terms of entries.
            matrix_dims = [(int(shape[0] * block_dims[0]), int(shape[1] * block_dims[1])) for shape in blocks_per_dim]
            matrix_max_dims = [
                (int(shape[0] * block_dims[0]), int(shape[1] * block_dims[1])) for shape in max_blocks_per_dim
            ]

            # Create sparse matrices by randomly selecting which blocks to populate.
            num_nzb_np = np.zeros((num_matrices,), dtype=np.int32)
            max_nzb_np = np.zeros((num_matrices,), dtype=np.int32)
            nzb_coords_list = []
            nzb_values_list = []
            for mat_id in range(num_matrices):
                # Randomly select blocks to be non-zero.
                all_block_indices = [
                    [row * block_dims[0], col * block_dims[1]]
                    for col in range(blocks_per_dim[mat_id, 1])
                    for row in range(blocks_per_dim[mat_id, 0])
                ]
                nzb_coords = all_block_indices[mat_id::sparse_block_offset]
                num_nzb_np[mat_id] = len(nzb_coords)

                # Create non-zero blocks for matrix.
                for _ in nzb_coords:
                    nzb_values_list.append(self.rng.standard_normal(block_dims, dtype=np.float32))

                # Add empty entries.
                for _ in range(5):
                    nzb_coords.append([0, 0])
                    nzb_values_list.append(np.zeros(block_dims, dtype=np.float32))

                max_nzb_np[mat_id] = len(nzb_coords)
                nzb_coords_list.extend(nzb_coords)

            nzb_coords_np = np.asarray(nzb_coords_list, dtype=np.float32)
            nzb_values_np = np.asarray(nzb_values_list, dtype=np.float32)

            bsm = BlockSparseMatrices(
                num_matrices=num_matrices, nzb_dtype=BlockDType(shape=block_dims_short, dtype=wp.float32)
            )
            bsm.finalize(max_dims=matrix_max_dims, capacities=[int(c) for c in max_nzb_np], device=self.default_device)

            # Fill in sparse matrix data structure.
            bsm.dims.assign(matrix_dims)
            bsm.num_nzb.assign(num_nzb_np)
            bsm.nzb_coords.assign(nzb_coords_np)
            bsm.nzb_values.view(dtype=wp.float32).assign(nzb_values_np)

            # Build operator.
            ops = BlockSparseLinearOperators(bsm)

            # Run multiplication operator checks.
            self._matvec_product_check(ops)

            if self.plot:
                matrices = bsm.numpy()
                for i in range(num_matrices):
                    sparseplot(matrices[i], title=f"Matrix {i}")

    def test_02_sparse_matrix_vector_product_jacobian(self):
        """
        Tests multiplication of a block sparse matrix with a random vector, where the sparse matrix has a Jacobian-like
        structure.
        """

        # Problem size as number of rigid bodies and number of constraints.
        problem_sizes = [(10, 10), (20, 22), (100, 105)]
        num_matrices = 5

        block_dims_short = (6,)
        block_dims = (1, 6)

        self.rng = np.random.default_rng(seed=self.seed)

        for problem_size in problem_sizes:
            num_rigid_bodies = problem_size[0]
            num_constraints = problem_size[1]
            max_num_contacts = 10
            max_num_limits = num_constraints // 2

            num_eqs_per_constraint = self.rng.integers(3, 6, num_constraints, endpoint=True, dtype=int)
            num_constraint_eqs = np.sum(num_eqs_per_constraint)

            # Compute matrix dimensions.
            max_blocks_per_dim = (int(num_constraint_eqs) + 3 * max_num_contacts + max_num_limits, num_rigid_bodies)
            matrix_max_dims = [
                (max_blocks_per_dim[0] * block_dims[0], max_blocks_per_dim[1] * block_dims[1])
            ] * num_matrices

            # Create sparse Jacobian-like matrices by randomly selecting which blocks to populate.
            num_nzb_np = np.zeros((num_matrices,), dtype=np.int32)
            max_nzb_np = np.zeros((num_matrices,), dtype=np.int32)
            nzb_coords_list = []
            nzb_values_list = []
            matrix_dims = []
            for mat_id in range(num_matrices):
                row_idx = 0
                nzb_coords = []
                # Add binary constraints.
                for ct_id in range(num_constraints):
                    rb_id_A, rb_id_B = self.rng.choice(num_rigid_bodies, 2, replace=False)
                    for i in range(num_eqs_per_constraint[ct_id]):
                        nzb_coords.append((row_idx + i, block_dims[1] * rb_id_A))
                    for i in range(num_eqs_per_constraint[ct_id]):
                        nzb_coords.append((row_idx + i, block_dims[1] * rb_id_B))
                    row_idx += num_eqs_per_constraint[ct_id]
                # Add some numbers of contacts.
                for _ in range(max_num_contacts // 3):
                    rb_id = self.rng.choice(num_rigid_bodies, 1)[0]
                    for i in range(3):
                        nzb_coords.append((row_idx + i, block_dims[1] * rb_id))
                    row_idx += 3
                # Add some number of limits.
                for _ in range(num_constraints // 3):
                    rb_id_A, rb_id_B = self.rng.choice(num_rigid_bodies, 2, replace=False)
                    nzb_coords.append((row_idx, block_dims[1] * rb_id_A))
                    nzb_coords.append((row_idx, block_dims[1] * rb_id_B))
                    row_idx += 1

                num_nzb_np[mat_id] = len(nzb_coords)
                dims = (row_idx, block_dims[1] * num_rigid_bodies)
                matrix_dims.append(list(dims))

                # Create non-zero blocks for matrix.
                for _ in nzb_coords:
                    nzb_values_list.append(self.rng.standard_normal(block_dims, dtype=np.float32))

                # Add empty entries.
                for _ in range(5):
                    nzb_coords.append([0, 0])
                    nzb_values_list.append(np.zeros(block_dims, dtype=np.float32))

                max_nzb_np[mat_id] = len(nzb_coords)
                nzb_coords_list.extend(nzb_coords)

            matrix_dims = np.asarray(matrix_dims)
            nzb_coords_np = np.asarray(nzb_coords_list, dtype=np.float32)
            nzb_values_np = np.asarray(nzb_values_list, dtype=np.float32)

            bsm = BlockSparseMatrices(
                num_matrices=num_matrices, nzb_dtype=BlockDType(shape=block_dims_short, dtype=wp.float32)
            )
            bsm.finalize(max_dims=matrix_max_dims, capacities=[int(c) for c in max_nzb_np], device=self.default_device)

            # Fill in sparse matrix data structure.
            bsm.dims.assign(matrix_dims)
            bsm.num_nzb.assign(num_nzb_np)
            bsm.nzb_coords.assign(nzb_coords_np)
            bsm.nzb_values.view(dtype=wp.float32).assign(nzb_values_np)

            # Build operator.
            ops = BlockSparseLinearOperators(bsm)

            # Run multiplication operator checks.
            self._matvec_product_check(ops)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
