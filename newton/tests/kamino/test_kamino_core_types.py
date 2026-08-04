# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: CORE: TYPES

Tests for the ``to_warp_int32_array`` and ``assign_to_warp_int32_array``
helpers in :mod:`newton._src.solvers.kamino._src.core.types`.

These helpers exist to catch silent overflow when converting Python or NumPy
integer data to a Warp ``wp.int32`` array. Direct ``wp.array(np_int64_arr,
dtype=wp.int32)`` and ``wp_int32_array.assign(np_int64_arr)`` silently
truncate; the helpers raise :class:`OverflowError` instead.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import (
    assign_to_warp_int32_array,
    to_warp_int32_array,
)
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Constants
###

INT32_MAX = np.iinfo(np.int32).max
INT32_MIN = np.iinfo(np.int32).min


###
# Tests
###


class TestToWarpInt32Array(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose

        if self.verbose:
            print("\n")
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    # --- Happy paths ------------------------------------------------------

    def test_from_python_list(self):
        result = to_warp_int32_array([1, 2, 3], device=self.default_device)
        np.testing.assert_array_equal(result.numpy(), [1, 2, 3])
        self.assertIs(result.dtype, wp.int32)

    def test_from_tuple(self):
        result = to_warp_int32_array((4, 5, 6), device=self.default_device)
        np.testing.assert_array_equal(result.numpy(), [4, 5, 6])
        self.assertIs(result.dtype, wp.int32)

    def test_from_numpy_int32(self):
        src = np.array([1, 2, 3], dtype=np.int32)
        result = to_warp_int32_array(src, device=self.default_device)
        np.testing.assert_array_equal(result.numpy(), [1, 2, 3])
        self.assertIs(result.dtype, wp.int32)

    def test_from_numpy_int64_within_range(self):
        src = np.array([1, 2, 3], dtype=np.int64)
        result = to_warp_int32_array(src, device=self.default_device)
        np.testing.assert_array_equal(result.numpy(), [1, 2, 3])
        self.assertIs(result.dtype, wp.int32)

    def test_from_mixed_python_and_numpy_ints(self):
        result = to_warp_int32_array([np.int32(10), 20, np.int64(-5)], device=self.default_device)
        np.testing.assert_array_equal(result.numpy(), [10, 20, -5])
        self.assertIs(result.dtype, wp.int32)

    def test_empty_input(self):
        result = to_warp_int32_array([], device=self.default_device)
        self.assertEqual(result.size, 0)
        self.assertIs(result.dtype, wp.int32)

    def test_2d_input_preserves_shape(self):
        src = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
        result = to_warp_int32_array(src, device=self.default_device)
        np.testing.assert_array_equal(result.numpy(), src)
        self.assertEqual(result.shape, (3, 2))
        self.assertIs(result.dtype, wp.int32)

    def test_device_defaults_to_scoped_device(self):
        with wp.ScopedDevice(self.default_device):
            result = to_warp_int32_array([1, 2, 3])
        self.assertEqual(result.device, self.default_device)
        self.assertIs(result.dtype, wp.int32)

    # --- Boundary conditions ---------------------------------------------

    def test_int32_max_value_accepted(self):
        result = to_warp_int32_array([INT32_MAX], device=self.default_device)
        self.assertEqual(int(result.numpy()[0]), INT32_MAX)

    def test_int32_min_value_accepted(self):
        result = to_warp_int32_array([INT32_MIN], device=self.default_device)
        self.assertEqual(int(result.numpy()[0]), INT32_MIN)

    # --- Overflow guards --------------------------------------------------

    def test_overflow_python_int_above_max(self):
        with self.assertRaises(OverflowError) as cm:
            to_warp_int32_array([1, 2, INT32_MAX + 1], device=self.default_device)
        self.assertIn("int32 overflow", str(cm.exception))

    def test_overflow_python_int_below_min(self):
        with self.assertRaises(OverflowError) as cm:
            to_warp_int32_array([INT32_MIN - 1, 0, 1], device=self.default_device)
        self.assertIn("int32 overflow", str(cm.exception))

    def test_overflow_numpy_int64_above_max(self):
        src = np.array([1, 2, 2**33], dtype=np.int64)
        with self.assertRaises(OverflowError) as cm:
            to_warp_int32_array(src, device=self.default_device)
        self.assertIn("int32 overflow", str(cm.exception))

    def test_overflow_numpy_int64_below_min(self):
        src = np.array([INT32_MIN - 1], dtype=np.int64)
        with self.assertRaises(OverflowError) as cm:
            to_warp_int32_array(src, device=self.default_device)
        self.assertIn("int32 overflow", str(cm.exception))

    # --- Type guards ------------------------------------------------------

    def test_reject_python_float_list(self):
        with self.assertRaises(TypeError) as cm:
            to_warp_int32_array([1.0, 2.0, 3.0], device=self.default_device)
        self.assertIn("integer data", str(cm.exception))

    def test_reject_numpy_float64_array(self):
        src = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with self.assertRaises(TypeError):
            to_warp_int32_array(src, device=self.default_device)

    def test_reject_numpy_float32_array(self):
        src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with self.assertRaises(TypeError):
            to_warp_int32_array(src, device=self.default_device)


class TestAssignToWarpInt32Array(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose

        if self.verbose:
            print("\n")
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def _make_dst(self, size: int, dtype=wp.int32) -> wp.array:
        return wp.zeros(size, dtype=dtype, device=self.default_device)

    # --- Happy paths ------------------------------------------------------

    def test_assign_from_python_list(self):
        dst = self._make_dst(3)
        assign_to_warp_int32_array(dst, [10, 20, 30])
        np.testing.assert_array_equal(dst.numpy(), [10, 20, 30])

    def test_assign_from_numpy_int64_within_range(self):
        dst = self._make_dst(3)
        src = np.array([10, 20, 30], dtype=np.int64)
        assign_to_warp_int32_array(dst, src)
        np.testing.assert_array_equal(dst.numpy(), [10, 20, 30])

    def test_assign_from_numpy_int32(self):
        dst = self._make_dst(3)
        src = np.array([10, 20, 30], dtype=np.int32)
        assign_to_warp_int32_array(dst, src)
        np.testing.assert_array_equal(dst.numpy(), [10, 20, 30])

    def test_assign_to_2d_destination(self):
        dst = wp.zeros((3, 2), dtype=wp.int32, device=self.default_device)
        src = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32).flatten()
        assign_to_warp_int32_array(dst, src)
        np.testing.assert_array_equal(dst.numpy(), [[1, 2], [3, 4], [5, 6]])

    def test_assign_empty(self):
        dst = self._make_dst(0)
        assign_to_warp_int32_array(dst, [])
        self.assertEqual(dst.size, 0)

    # --- Boundary conditions ---------------------------------------------

    def test_assign_int32_max(self):
        dst = self._make_dst(1)
        assign_to_warp_int32_array(dst, [INT32_MAX])
        self.assertEqual(int(dst.numpy()[0]), INT32_MAX)

    def test_assign_int32_min(self):
        dst = self._make_dst(1)
        assign_to_warp_int32_array(dst, [INT32_MIN])
        self.assertEqual(int(dst.numpy()[0]), INT32_MIN)

    # --- Overflow guards --------------------------------------------------

    def test_overflow_python_int_above_max(self):
        dst = self._make_dst(3)
        with self.assertRaises(OverflowError) as cm:
            assign_to_warp_int32_array(dst, [1, 2, INT32_MAX + 1])
        self.assertIn("int32 overflow", str(cm.exception))

    def test_overflow_python_int_below_min(self):
        dst = self._make_dst(3)
        with self.assertRaises(OverflowError) as cm:
            assign_to_warp_int32_array(dst, [INT32_MIN - 1, 0, 1])
        self.assertIn("int32 overflow", str(cm.exception))

    def test_overflow_numpy_int64_above_max(self):
        dst = self._make_dst(3)
        src = np.array([1, 2, 2**33], dtype=np.int64)
        with self.assertRaises(OverflowError) as cm:
            assign_to_warp_int32_array(dst, src)
        self.assertIn("int32 overflow", str(cm.exception))

    def test_overflow_numpy_int64_below_min(self):
        dst = self._make_dst(1)
        src = np.array([INT32_MIN - 1], dtype=np.int64)
        with self.assertRaises(OverflowError) as cm:
            assign_to_warp_int32_array(dst, src)
        self.assertIn("int32 overflow", str(cm.exception))

    # --- Destination dtype guards ----------------------------------------

    def test_reject_float32_destination(self):
        dst = wp.zeros(3, dtype=wp.float32, device=self.default_device)
        with self.assertRaises(TypeError) as cm:
            assign_to_warp_int32_array(dst, [1, 2, 3])
        self.assertIn("dtype wp.int32", str(cm.exception))

    def test_reject_int64_destination(self):
        dst = wp.zeros(3, dtype=wp.int64, device=self.default_device)
        with self.assertRaises(TypeError) as cm:
            assign_to_warp_int32_array(dst, [1, 2, 3])
        self.assertIn("dtype wp.int32", str(cm.exception))

    def test_reject_vec2i_destination(self):
        dst = wp.zeros(3, dtype=wp.vec2i, device=self.default_device)
        with self.assertRaises(TypeError) as cm:
            assign_to_warp_int32_array(dst, [1, 2, 3])
        self.assertIn("dtype wp.int32", str(cm.exception))

    # --- Source type guards ----------------------------------------------

    def test_reject_float_source(self):
        dst = self._make_dst(3)
        with self.assertRaises(TypeError) as cm:
            assign_to_warp_int32_array(dst, [1.0, 2.0, 3.0])
        self.assertIn("integer data", str(cm.exception))

    def test_reject_numpy_float64_source(self):
        dst = self._make_dst(3)
        src = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with self.assertRaises(TypeError):
            assign_to_warp_int32_array(dst, src)


###
# Test execution
###

if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
