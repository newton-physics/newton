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

"""Unit tests for the base classes in linalg/core.py"""

import unittest

import numpy as np
import warp as wp

import newton._src.solvers.kamino.utils.logger as msg
from newton._src.solvers.kamino.linalg.core import (
    DenseRectangularMultiLinearInfo,
    DenseSquareMultiLinearInfo,
    make_dtype_tolerance,
)

###
# Tests
###


class TestLinAlgCoreMakeTolerance(unittest.TestCase):
    def setUp(self):
        # Configs
        self.seed = 42
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_tolerance_from_defaults(self):
        tol = make_dtype_tolerance()
        msg.debug(f"tol = {tol} (type: {type(tol)})")
        self.assertIsInstance(tol, wp.float32)
        self.assertAlmostEqual(tol, wp.float32(np.finfo(np.float32).eps), places=7)

    def test_01_make_tolerance_default_for_wp_float64(self):
        tol = make_dtype_tolerance(dtype=wp.float64)
        msg.debug(f"tol = {tol} (type: {type(tol)})")
        self.assertIsInstance(tol, wp.float64)
        self.assertAlmostEqual(tol, wp.float64(np.finfo(np.float64).eps), places=23)

    def test_02_make_tolerance_default_for_wp_float32(self):
        tol = make_dtype_tolerance(dtype=wp.float32)
        msg.debug(f"tol = {tol} (type: {type(tol)})")
        self.assertIsInstance(tol, wp.float32)
        self.assertAlmostEqual(tol, wp.float32(np.finfo(np.float32).eps), places=7)

    def test_03_make_tolerance_from_np_float64_to_wp_float64(self):
        tol = make_dtype_tolerance(tol=np.float64(1e-5), dtype=wp.float64)
        msg.debug(f"tol = {tol} (type: {type(tol)})")
        self.assertIsInstance(tol, wp.float64)
        self.assertAlmostEqual(tol, wp.float64(np.float64(1e-5)), places=23)

    def test_04_make_tolerance_from_np_float64_to_wp_float32(self):
        tol = make_dtype_tolerance(tol=np.float64(1e-5), dtype=wp.float32)
        msg.debug(f"tol = {tol} (type: {type(tol)})")
        self.assertIsInstance(tol, wp.float32)
        self.assertAlmostEqual(tol, wp.float32(np.float64(1e-5)), places=7)

    def test_05_make_tolerance_from_float_to_wp_float32(self):
        tol = make_dtype_tolerance(tol=1e-6, dtype=wp.float32)
        msg.debug(f"tol = {tol} (type: {type(tol)})")
        self.assertIsInstance(tol, wp.float32)
        self.assertAlmostEqual(tol, wp.float32(1e-6), places=7)

    def test_06_make_tolerance_smaller_than_eps(self):
        tol = make_dtype_tolerance(tol=1e-10, dtype=wp.float32)
        msg.debug(f"tol = {tol} (type: {type(tol)})")
        self.assertIsInstance(tol, wp.float32)
        self.assertAlmostEqual(tol, wp.float32(np.finfo(np.float32).eps), places=7)


class TestLinAlgCoreDenseMultiLinearRectangularInfo(unittest.TestCase):
    def setUp(self):
        # Configs
        self.seed = 42
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_single_rectangular_default(self):
        dims = (3, 4)
        info = DenseRectangularMultiLinearInfo()
        info.allocate(dimensions=dims)
        msg.debug("info:\n%s", info)
        msg.debug("info.maxdim: %s", info.maxdim)
        msg.debug("info.dim: %s", info.dim)
        msg.debug("info.mio: %s", info.mio)
        msg.debug("info.rvio: %s", info.rvio)
        msg.debug("info.ivio: %s", info.ivio)
        self.assertEqual(info.num_blocks, 1)
        self.assertEqual(info.dimensions[0], dims)
        self.assertEqual(info.max_dimensions, dims)
        self.assertEqual(info.total_mat_size, dims[0] * dims[1])
        self.assertEqual(info.total_rhs_size, dims[0])
        self.assertEqual(info.total_inp_size, dims[1])
        self.assertEqual(info.dtype, wp.float32)
        self.assertEqual(info.itype, wp.int32)
        self.assertEqual(info.maxdim.shape, (1,))
        self.assertEqual(info.dim.shape, (1,))
        self.assertEqual(info.mio.shape, (1,))
        self.assertEqual(info.rvio.shape, (1,))
        self.assertEqual(info.ivio.shape, (1,))
        self.assertEqual(info.maxdim.numpy()[0][0], 3)
        self.assertEqual(info.maxdim.numpy()[0][1], 4)
        self.assertEqual(info.dim.numpy()[0][0], 3)
        self.assertEqual(info.dim.numpy()[0][1], 4)
        self.assertEqual(info.mio.numpy()[0], 0)
        self.assertEqual(info.rvio.numpy()[0], 0)
        self.assertEqual(info.ivio.numpy()[0], 0)

    def test_01_make_single_rectangular_wp_float64_int64(self):
        dims = (3, 4)
        info = DenseRectangularMultiLinearInfo()
        info.allocate(dimensions=dims, dtype=wp.float64, itype=wp.int64)
        msg.debug("info:\n%s", info)
        msg.debug("info.maxdim: %s", info.maxdim)
        msg.debug("info.dim: %s", info.dim)
        msg.debug("info.mio: %s", info.mio)
        msg.debug("info.rvio: %s", info.rvio)
        msg.debug("info.ivio: %s", info.ivio)
        self.assertEqual(info.num_blocks, 1)
        self.assertEqual(info.dimensions[0], dims)
        self.assertEqual(info.max_dimensions, dims)
        self.assertEqual(info.total_mat_size, dims[0] * dims[1])
        self.assertEqual(info.total_rhs_size, dims[0])
        self.assertEqual(info.total_inp_size, dims[1])
        self.assertEqual(info.dtype, wp.float64)
        self.assertEqual(info.itype, wp.int64)
        self.assertEqual(info.maxdim.shape, (1,))
        self.assertEqual(info.dim.shape, (1,))
        self.assertEqual(info.mio.shape, (1,))
        self.assertEqual(info.rvio.shape, (1,))
        self.assertEqual(info.ivio.shape, (1,))
        self.assertEqual(info.maxdim.numpy()[0][0], 3)
        self.assertEqual(info.maxdim.numpy()[0][1], 4)
        self.assertEqual(info.dim.numpy()[0][0], 3)
        self.assertEqual(info.dim.numpy()[0][1], 4)
        self.assertEqual(info.mio.numpy()[0], 0)
        self.assertEqual(info.rvio.numpy()[0], 0)
        self.assertEqual(info.ivio.numpy()[0], 0)

    def test_02_make_multiple_rectangular(self):
        dims = [(3, 4), (2, 5), (4, 3)]
        info = DenseRectangularMultiLinearInfo()
        info.allocate(dimensions=dims)
        msg.debug("info:\n%s", info)
        msg.debug("info.maxdim: %s", info.maxdim)
        msg.debug("info.dim: %s", info.dim)
        msg.debug("info.mio: %s", info.mio)
        msg.debug("info.rvio: %s", info.rvio)
        msg.debug("info.ivio: %s", info.ivio)
        self.assertEqual(info.num_blocks, len(dims))
        self.assertEqual(info.dimensions, dims)
        self.assertEqual(info.max_dimensions, (4, 5))
        self.assertEqual(info.total_mat_size, sum(m * n for m, n in dims))
        self.assertEqual(info.total_rhs_size, sum(m for m, _ in dims))
        self.assertEqual(info.total_inp_size, sum(n for _, n in dims))
        self.assertEqual(info.dtype, wp.float32)
        self.assertEqual(info.itype, wp.int32)
        self.assertEqual(info.maxdim.shape, (len(dims),))
        self.assertEqual(info.dim.shape, (len(dims),))
        self.assertEqual(info.mio.shape, (len(dims),))
        self.assertEqual(info.rvio.shape, (len(dims),))
        self.assertEqual(info.ivio.shape, (len(dims),))
        for i, (m, n) in enumerate(dims):
            self.assertEqual(info.maxdim.numpy()[i][0], m)
            self.assertEqual(info.maxdim.numpy()[i][1], n)
            self.assertEqual(info.dim.numpy()[i][0], m)
            self.assertEqual(info.dim.numpy()[i][1], n)
            self.assertEqual(info.mio.numpy()[i], sum(d[0] * d[1] for d in dims[:i]))
            self.assertEqual(info.rvio.numpy()[i], sum(d[0] for d in dims[:i]))
            self.assertEqual(info.ivio.numpy()[i], sum(d[1] for d in dims[:i]))


class TestLinAlgCoreDenseMultiLinearSquareInfo(unittest.TestCase):
    def setUp(self):
        # Configs
        self.seed = 42
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_single_square_default(self):
        dims = 4
        info = DenseSquareMultiLinearInfo()
        info.allocate(dimensions=dims)
        msg.debug("info:\n%s", info)
        msg.debug("info.maxdim: %s", info.maxdim)
        msg.debug("info.dim: %s", info.dim)
        msg.debug("info.mio: %s", info.mio)
        msg.debug("info.vio: %s", info.vio)
        self.assertEqual(info.num_blocks, 1)
        self.assertEqual(info.dimensions[0], dims)
        self.assertEqual(info.max_dimension, dims)
        self.assertEqual(info.total_mat_size, dims * dims)
        self.assertEqual(info.total_vec_size, dims)
        self.assertEqual(info.dtype, wp.float32)
        self.assertEqual(info.itype, wp.int32)
        self.assertEqual(info.maxdim.shape, (1,))
        self.assertEqual(info.dim.shape, (1,))
        self.assertEqual(info.mio.shape, (1,))
        self.assertEqual(info.vio.shape, (1,))
        self.assertEqual(info.maxdim.numpy()[0], dims)
        self.assertEqual(info.dim.numpy()[0], dims)
        self.assertEqual(info.mio.numpy()[0], 0)
        self.assertEqual(info.vio.numpy()[0], 0)

    def test_01_make_single_square_wp_float64_int64(self):
        dims = 13
        info = DenseSquareMultiLinearInfo()
        info.allocate(dimensions=dims, dtype=wp.float64, itype=wp.int64)
        msg.debug("info:\n%s", info)
        msg.debug("info.maxdim: %s", info.maxdim)
        msg.debug("info.dim: %s", info.dim)
        msg.debug("info.mio: %s", info.mio)
        msg.debug("info.vio: %s", info.vio)
        self.assertEqual(info.num_blocks, 1)
        self.assertEqual(info.dimensions[0], dims)
        self.assertEqual(info.max_dimension, dims)
        self.assertEqual(info.total_mat_size, dims * dims)
        self.assertEqual(info.total_vec_size, dims)
        self.assertEqual(info.dtype, wp.float64)
        self.assertEqual(info.itype, wp.int64)
        self.assertEqual(info.maxdim.shape, (1,))
        self.assertEqual(info.dim.shape, (1,))
        self.assertEqual(info.mio.shape, (1,))
        self.assertEqual(info.vio.shape, (1,))
        self.assertEqual(info.maxdim.numpy()[0], dims)
        self.assertEqual(info.dim.numpy()[0], dims)
        self.assertEqual(info.mio.numpy()[0], 0)
        self.assertEqual(info.vio.numpy()[0], 0)

    def test_02_make_multiple_square(self):
        dims = [4, 10, 5, 12]
        info = DenseSquareMultiLinearInfo()
        info.allocate(dimensions=dims)
        msg.debug("info:\n%s", info)
        msg.debug("info.maxdim: %s", info.maxdim)
        msg.debug("info.dim: %s", info.dim)
        msg.debug("info.mio: %s", info.mio)
        msg.debug("info.vio: %s", info.vio)
        self.assertEqual(info.num_blocks, len(dims))
        self.assertEqual(info.dimensions, dims)
        self.assertEqual(info.max_dimension, max(dims))
        self.assertEqual(info.total_mat_size, sum(n * n for n in dims))
        self.assertEqual(info.total_vec_size, sum(n for n in dims))
        self.assertEqual(info.dtype, wp.float32)
        self.assertEqual(info.itype, wp.int32)
        self.assertEqual(info.maxdim.shape, (len(dims),))
        self.assertEqual(info.dim.shape, (len(dims),))
        self.assertEqual(info.mio.shape, (len(dims),))
        self.assertEqual(info.vio.shape, (len(dims),))
        for i, n in enumerate(dims):
            self.assertEqual(info.maxdim.numpy()[i], n)
            self.assertEqual(info.dim.numpy()[i], n)
            self.assertEqual(info.mio.numpy()[i], sum(d * d for d in dims[:i]))
            self.assertEqual(info.vio.numpy()[i], sum(d for d in dims[:i]))


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=10000, precision=10, threshold=10000, suppress=True)  # Suppress scientific notation

    # Initialize Warp
    wp.init()

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False

    # Clear Warp caches
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
