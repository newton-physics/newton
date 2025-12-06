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

"""Unit tests for `geometry/keying.py`"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.types import int32, uint32, uint64
from newton._src.solvers.kamino.geometry.keying import (
    KeySorter,
    binary_search_find_pair,
    binary_search_find_range_start,
    build_pair_key2,
    make_bitmask,
    make_build_pair_key3_func,
)
from newton._src.solvers.kamino.utils import logger as msg

###
# Helper functions
###


###
# Tests
###


class TestPairKeyOps(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for detailed output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

        # Global numpy configurations
        np.set_printoptions(linewidth=10000, precision=10, threshold=10000, suppress=True)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_bitmask(self):
        """Test make_bitmask function for various bit sizes."""

        # Test make_bitmask for out-of-bounds inputs
        self.assertRaises(ValueError, make_bitmask, 0)
        self.assertRaises(ValueError, make_bitmask, 65)

        # Test make_bitmask for valid inputs
        bitmask_1_Bits = make_bitmask(1)
        msg.info(f"bitmask_1_Bits: {bitmask_1_Bits:#0{4}x} ({bitmask_1_Bits})")
        self.assertEqual(bitmask_1_Bits, 0x1)

        bitmask_3_Bits = make_bitmask(3)
        msg.info(f"bitmask_3_Bits: {bitmask_3_Bits:#0{4}x} ({bitmask_3_Bits})")
        self.assertEqual(bitmask_3_Bits, 0x7)

        bitmask_7_Bits = make_bitmask(7)
        msg.info(f"bitmask_7_Bits: {bitmask_7_Bits:#0{4}x} ({bitmask_7_Bits})")
        self.assertEqual(bitmask_7_Bits, 0x7F)

        bitmask_8_Bits = make_bitmask(8)
        msg.info(f"bitmask_8_Bits: {bitmask_8_Bits:#0{4}x} ({bitmask_8_Bits})")
        self.assertEqual(bitmask_8_Bits, 0xFF)

        bitmask_23_Bits = make_bitmask(23)
        msg.info(f"bitmask_23_Bits: {bitmask_23_Bits:#0{8}x} ({bitmask_23_Bits})")
        self.assertEqual(bitmask_23_Bits, 0x7FFFFF)

        bitmask_32_Bits = make_bitmask(32)
        msg.info(f"bitmask_32_Bits: {bitmask_32_Bits:#0{10}x} ({bitmask_32_Bits})")
        self.assertEqual(bitmask_32_Bits, 0xFFFFFFFF)

        bitmask_64_Bits = make_bitmask(64)
        msg.info(f"bitmask_64_Bits: {bitmask_64_Bits:#0{18}x} ({bitmask_64_Bits})")
        self.assertEqual(bitmask_64_Bits, 0xFFFFFFFFFFFFFFFF)

    def test_01_build_pair_key2(self):
        """Test build_pair_key2 function for various index pairs."""

        # Define a Warp kernel to test build_pair_key2
        @wp.kernel
        def _test_kernel_build_pair_key2(
            index_A: wp.array(dtype=uint32), index_B: wp.array(dtype=uint32), key: wp.array(dtype=uint64)
        ):
            tid = wp.tid()
            key[tid] = build_pair_key2(index_A[tid], index_B[tid])

        # Define test cases for index pairs
        test_cases = [
            (0, 0),
            (1, 1),
            (2, 23),
            (12345, 67890),
            (0x7FFFFFFF, 0xFFFFFFFF),
            (0x12345678, 0x9ABCDEF0),
        ]
        num_test_cases = len(test_cases)
        msg.info("num_test_cases: %d", num_test_cases)
        msg.info("test_cases: %s", test_cases)

        # Create Warp arrays for inputs and outputs
        index_A = wp.array([index_A for index_A, _ in test_cases], dtype=uint32)
        index_B = wp.array([index_B for _, index_B in test_cases], dtype=uint32)
        keys = wp.zeros(num_test_cases, dtype=uint64)
        msg.info("Inputs: index_A: %s", index_A)
        msg.info("Inputs: index_B: %s", index_B)
        msg.info("Inputs: keys: %s", keys)
        self.assertEqual(index_A.size, num_test_cases)
        self.assertEqual(index_B.size, num_test_cases)
        self.assertEqual(keys.size, num_test_cases)

        # Launch the test kernel to generate keys for the given index pairs
        wp.launch(_test_kernel_build_pair_key2, dim=num_test_cases, inputs=[index_A, index_B, keys])

        # Verify the generated keys against expected values
        keys_np = keys.numpy()
        msg.info("Output: keys: %s", keys_np)
        for i, (index_A, index_B) in enumerate(test_cases):
            expected_key = (index_A << 32) | index_B
            msg.info(f"build_pair_key2({index_A}, {index_B}): {keys_np[i]} (expected: {expected_key})")
            self.assertEqual(keys_np[i], expected_key)

    def test_02_build_pair_key3(self):
        """Test build_pair_key3 function for various index pairs with auxiliary index."""

        # Define a Warp kernel to test build_pair_key3
        def make_test_kernel_build_pair_key3(main_key_bits: int, aux_key_bits: int):
            # Generate the build_pair_key3 function with specified bit widths
            build_pair_key3 = make_build_pair_key3_func(main_key_bits, aux_key_bits)

            # Generate the test kernel for the specified build_pair_key3 function
            @wp.kernel
            def _test_kernel_build_pair_key3(
                index_A: wp.array(dtype=uint32),
                index_B: wp.array(dtype=uint32),
                index_C: wp.array(dtype=uint32),
                key: wp.array(dtype=uint64),
            ):
                tid = wp.tid()
                key[tid] = build_pair_key3(index_A[tid], index_B[tid], index_C[tid])

            return _test_kernel_build_pair_key3

        # Define test cases for index pairs
        test_cases = [
            (0, 0, 0),
            (1, 1, 1),
            (2, 23, 3),
            (12345, 67890, 4),
            (0xFFFFFFF, 0xFFFFFFF, 5),
            (0x2345678, 0xABCDEF0, 6),
        ]
        num_test_cases = len(test_cases)
        msg.info("num_test_cases: %d", num_test_cases)
        msg.info("test_cases: %s", test_cases)

        # Create Warp arrays for inputs and outputs
        index_A = wp.array([index_A for index_A, _, _ in test_cases], dtype=uint32)
        index_B = wp.array([index_B for _, index_B, _ in test_cases], dtype=uint32)
        index_C = wp.array([index_C for *_, index_C in test_cases], dtype=uint32)
        keys = wp.zeros(num_test_cases, dtype=uint64)
        msg.info("Inputs: index_A: %s", index_A)
        msg.info("Inputs: index_B: %s", index_B)
        msg.info("Inputs: index_C: %s", index_C)
        msg.info("Inputs: keys: %s", keys)
        self.assertEqual(index_A.size, num_test_cases)
        self.assertEqual(index_B.size, num_test_cases)
        self.assertEqual(index_C.size, num_test_cases)
        self.assertEqual(keys.size, num_test_cases)

        # Test various bit size configurations
        # NOTE: main_key_bits * 2 + aux_key_bits must equal 63
        test_valid_bitsizes = [(21, 21), (20, 23), (22, 19), (18, 27)]
        for main_key_bits, aux_key_bits in test_valid_bitsizes:
            msg.info("Testing build_pair_key3 with main_key_bits=%d, aux_key_bits=%d", main_key_bits, aux_key_bits)
            _test_kernel_build_pair_key2 = make_test_kernel_build_pair_key3(main_key_bits, aux_key_bits)

            # Launch the test kernel to generate keys for the given index pairs
            wp.launch(_test_kernel_build_pair_key2, dim=num_test_cases, inputs=[index_A, index_B, index_C, keys])
            keys_np = keys.numpy()
            msg.info("Output: keys: %s", keys_np)

            # Generate bitmasks for expected key computation
            MAIN_BITMASK = make_bitmask(main_key_bits)
            AUX_BITMASK = make_bitmask(aux_key_bits)

            # Verify the generated keys against expected values
            for i, (index_A_val, index_B_val, index_C_val) in enumerate(test_cases):
                expected_key = (
                    ((index_A_val & MAIN_BITMASK) << (main_key_bits + aux_key_bits))
                    | ((index_B_val & MAIN_BITMASK) << aux_key_bits)
                    | (index_C_val & AUX_BITMASK)
                )
                msg.info(f"expected_key: {expected_key:#0{10}x}")
                msg.info(
                    f"build_pair_key3({index_A_val:#0{10}x}, {index_B_val:#0{10}x}, {index_C_val:#0{10}x}): "
                    f"{keys_np[i]:#0{10}x}"
                )
                self.assertEqual(keys_np[i], expected_key)


class TestBinarySearchOps(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for detailed output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

        # Global numpy configurations
        np.set_printoptions(linewidth=10000, precision=10, threshold=10000, suppress=True)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_binary_search_find_pair(self):
        """Test binary_search_find_pair function."""

        # Define a Warp kernel to test binary_search_find_pair
        @wp.kernel
        def _test_kernel_binary_search_find_pair(
            # Inputs:
            num_active_pairs: wp.array(dtype=int32),
            all_pairs: wp.array(dtype=wp.vec2i),
            target_pair: wp.array(dtype=wp.vec2i),
            # Output:
            target_index: wp.array(dtype=wp.int32),
        ):
            tid = wp.tid()
            target_index[tid] = binary_search_find_pair(num_active_pairs[0], target_pair[tid], all_pairs)

        # Define sorted array of unique integer pairs with some inactive dummy pairs at the end
        pairs_list = [(1, 1), (1, 2), (3, 1), (5, 6), (5, 5), (7, 8), (9, 10), (0, 0), (0, 0)]
        num_all_pairs = len(pairs_list)
        pairs = wp.array(pairs_list, dtype=wp.vec2i)
        num_active_pairs = wp.array([7], dtype=wp.int32)  # Only first 7 pairs are active
        msg.info("pairs:\n%s", pairs)
        msg.info("num_active_pairs: %s", num_active_pairs)
        msg.info("num_all_pairs: %s", num_all_pairs)

        # Define target pairs to search for
        target_pairs_list = [(1, 1), (3, 1), (7, 8), (0, 0), (9, 10), (11, 12)]
        expected_idxs = [0, 2, 5, -1, 6, -1]  # Expected indices or -1 if not found
        num_target_pairs = len(target_pairs_list)
        target_pairs = wp.array(target_pairs_list, dtype=wp.vec2i)
        target_idxs = wp.zeros(num_target_pairs, dtype=wp.int32)
        msg.info("target_pairs:\n%s", target_pairs)
        msg.info("expected_idxs: %s", expected_idxs)

        # Launch the test kernel
        wp.launch(
            _test_kernel_binary_search_find_pair,
            dim=num_target_pairs,
            inputs=[num_active_pairs, pairs, target_pairs, target_idxs],
        )

        # Verify results
        target_idxs_np = target_idxs.numpy()
        msg.info("target_idxs: %s", target_idxs_np)
        for i, expected_index in enumerate(expected_idxs):
            msg.info(f"target {target_pairs_list[i]} at index: {target_idxs_np[i]} (expected: {expected_index})")
            self.assertEqual(target_idxs_np[i], expected_index)

    def test_02_binary_search_find_range_start(self):
        """Test binary_search_find_range_start function."""

        # Define a Warp kernel to test binary_search_find_range_start
        @wp.kernel
        def _test_kernel_binary_search_find_range_start(
            # Inputs:
            num_active_keys: wp.array(dtype=int32),
            all_keys: wp.array(dtype=uint64),
            target_key: wp.array(dtype=uint64),
            # Output:
            target_start: wp.array(dtype=int32),
        ):
            tid = wp.tid()
            target_start[tid] = binary_search_find_range_start(int32(0), num_active_keys[0], target_key[tid], all_keys)

        # Define sorted array of unique integer keys with some inactive dummy keys at the end
        keys_list = [0, 1, 1, 3, 5, 5, 9, 11, 11, 0, 0, 0, 0]
        num_all_keys = len(keys_list)
        keys = wp.array(keys_list, dtype=uint64)
        num_active_keys = wp.array([9], dtype=int32)  # Only first 9 keys are active
        msg.info("keys:\n%s", keys)
        msg.info("num_active_keys: %s", num_active_keys)
        msg.info("num_all_keys: %s", num_all_keys)

        # Define target keys to search for
        target_keys_list = [1, 5, 11, 2, 9, 12]
        expected_range_start_idxs = [1, 4, 7, -1, 6, -1]  # Expected start indices or -1 if not found
        num_target_elements = len(target_keys_list)
        target_keys = wp.array(target_keys_list, dtype=uint64)
        target_start_idxs = wp.zeros(num_target_elements, dtype=int32)
        msg.info("target_keys:\n%s", target_keys)
        msg.info("expected_range_start_idxs: %s", expected_range_start_idxs)

        # Launch the test kernel
        wp.launch(
            _test_kernel_binary_search_find_range_start,
            dim=num_target_elements,
            inputs=[num_active_keys, keys, target_keys, target_start_idxs],
        )

        # Verify results
        target_start_idxs_np = target_start_idxs.numpy()
        msg.info("target_start_idxs: %s", target_start_idxs_np)
        for i, expected in enumerate(expected_range_start_idxs):
            msg.info(f"target {target_keys_list[i]} start index: {target_start_idxs_np[i]} (expected: {expected})")
            self.assertEqual(target_start_idxs_np[i], expected)


class TestKeySorter(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for detailed output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

        # Global numpy configurations
        np.set_printoptions(linewidth=10000, precision=10, threshold=10000, suppress=True)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_key_sorter(self):
        """Test creation of KeySorter instance and relevant memory allocations."""
        max_num_keys = 10
        sorter = KeySorter(max_num_keys=max_num_keys, device=self.default_device)
        msg.info("sorter.sorted_keys: %s", sorter.sorted_keys)
        msg.info("sorter.sorted_keys_int64: %s", sorter.sorted_keys_int64)
        msg.info("sorter.sorted_to_unsorted_map: %s", sorter.sorted_to_unsorted_map)
        self.assertEqual(sorter.sorted_keys.size, 2 * max_num_keys)  # Factor of 2 for sort algorithm
        self.assertEqual(sorter.sorted_keys_int64.size, 2 * max_num_keys)  # Factor of 2 for sort algorithm
        self.assertEqual(sorter.sorted_to_unsorted_map.size, 2 * max_num_keys)  # Factor of 2 for sort algorithm

    def test_01_sort_fixed_keys(self):
        """Test sorting of fixed keys and verification of sorted results."""
        max_num_keys = 10
        sorter = KeySorter(max_num_keys=max_num_keys, device=self.default_device)
        msg.info("sorter.sorted_keys: %s", sorter.sorted_keys)
        msg.info("sorter.sorted_keys_int64: %s", sorter.sorted_keys_int64)
        msg.info("sorter.sorted_to_unsorted_map: %s", sorter.sorted_to_unsorted_map)
        self.assertEqual(sorter.sorted_keys.size, 2 * max_num_keys)  # Factor of 2 for sort algorithm
        self.assertEqual(sorter.sorted_keys_int64.size, 2 * max_num_keys)  # Factor of 2 for sort algorithm
        self.assertEqual(sorter.sorted_to_unsorted_map.size, 2 * max_num_keys)  # Factor of 2 for sort algorithm

        # Generate random keys
        sentinel = make_bitmask(64)
        num_active_keys_const = 8
        num_active_keys = wp.array([num_active_keys_const], dtype=int32)
        keys_list = [5, 3, 9, 1, 7, 3, 5, 11, sentinel, sentinel]
        keys = wp.array(keys_list, dtype=uint64)
        msg.info("num_active_keys: %s", num_active_keys)
        msg.info("keys: %s", keys)

        # Compute expected results using numpy
        expected_sorted_keys = [1, 3, 3, 5, 5, 7, 9, 11]
        expected_sorted_to_unsorted_map = [3, 1, 5, 0, 6, 4, 2, 7]
        msg.info("expected_sorted_keys: %s", expected_sorted_keys)
        msg.info("expected_sorted_to_unsorted_map: %s", expected_sorted_to_unsorted_map)

        # Launch sorter to sort the random keys
        sorter.sort(num_active_keys=num_active_keys, keys=keys)

        # Verify results
        sorted_keys_np = sorter.sorted_keys.numpy()[:num_active_keys_const]
        sorted_to_unsorted_map_np = sorter.sorted_to_unsorted_map.numpy()[:num_active_keys_const]
        msg.info("sorter.sorted_keys: %s", sorted_keys_np)
        msg.info("sorter.sorted_to_unsorted_map: %s", sorted_to_unsorted_map_np)
        for i in range(num_active_keys_const):
            msg.info(f"sorted_keys[{i}]: {sorted_keys_np[i]} (expected: {expected_sorted_keys[i]})")
            msg.info(
                f"sorted_to_unsorted_map[{i}]: {sorted_to_unsorted_map_np[i]} "
                f"(expected: {expected_sorted_to_unsorted_map[i]})"
            )
            self.assertEqual(sorted_keys_np[i], expected_sorted_keys[i])
            self.assertEqual(sorted_to_unsorted_map_np[i], expected_sorted_to_unsorted_map[i])

    def test_02_sort_random_keys(self):
        """Test sorting of random keys and verification of sorted results."""
        max_num_keys = 10
        sorter = KeySorter(max_num_keys=max_num_keys, device=self.default_device)
        msg.info("sorter.sorted_keys: %s", sorter.sorted_keys)
        msg.info("sorter.sorted_keys_int64: %s", sorter.sorted_keys_int64)
        msg.info("sorter.sorted_to_unsorted_map: %s", sorter.sorted_to_unsorted_map)
        self.assertEqual(sorter.sorted_keys.size, 2 * max_num_keys)  # Factor of 2 for sort algorithm
        self.assertEqual(sorter.sorted_keys_int64.size, 2 * max_num_keys)  # Factor of 2 for sort algorithm
        self.assertEqual(sorter.sorted_to_unsorted_map.size, 2 * max_num_keys)  # Factor of 2 for sort algorithm

        # Set up random seed for reproducibility
        rng = np.random.default_rng(seed=42)

        # Generate random keys
        num_active_keys_const = 8
        num_active_keys = wp.array([num_active_keys_const], dtype=int32)
        random_keys_np = rng.integers(low=0, high=100, size=max_num_keys, dtype=np.uint64)
        random_keys_np[-2:] = make_bitmask(63)  # Set last two keys to sentinel value
        random_keys = wp.array(random_keys_np, dtype=uint64)
        msg.info("num_active_keys: %s", num_active_keys)
        msg.info("random_keys: %s", random_keys)

        # Compute expected results using numpy
        expected_sorted_keys_np = np.sort(random_keys_np, stable=True)[:num_active_keys_const]
        expected_sorted_to_unsorted_map_np = np.argsort(random_keys_np, stable=True)[:num_active_keys_const]
        msg.info("expected_sorted_keys: %s", expected_sorted_keys_np.tolist())
        msg.info("expected_sorted_to_unsorted_map: %s", expected_sorted_to_unsorted_map_np.tolist())

        # Launch sorter to sort the random keys
        sorter.sort(num_active_keys=num_active_keys, keys=random_keys)

        # Verify results
        sorted_keys_np = sorter.sorted_keys.numpy()[:num_active_keys_const]
        sorted_to_unsorted_map_np = sorter.sorted_to_unsorted_map.numpy()[:num_active_keys_const]
        msg.info("sorter.sorted_keys: %s", sorted_keys_np.tolist())
        msg.info("sorter.sorted_to_unsorted_map: %s", sorted_to_unsorted_map_np.tolist())
        for i in range(num_active_keys_const):
            msg.info(f"sorted_keys[{i}]: {sorted_keys_np[i]} (expected: {expected_sorted_keys_np[i]})")
            msg.info(
                f"sorted_to_unsorted_map[{i}]: {sorted_to_unsorted_map_np[i]} "
                f"(expected: {expected_sorted_to_unsorted_map_np[i]})"
            )
            self.assertEqual(sorted_keys_np[i], expected_sorted_keys_np[i])
            self.assertEqual(sorted_to_unsorted_map_np[i], expected_sorted_to_unsorted_map_np[i])


###
# Test execution
###

if __name__ == "__main__":
    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
