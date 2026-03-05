# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Tests for match_labels utility."""

import unittest

from newton._src.utils.selection import match_labels


class TestMatchLabels(unittest.TestCase):
    """Unit tests for match_labels."""

    def test_str_exact_match(self):
        labels = ["alpha", "beta", "gamma"]
        self.assertEqual(match_labels(labels, "beta"), [1])

    def test_str_wildcard(self):
        labels = ["arm_left", "arm_right", "leg_left"]
        self.assertEqual(match_labels(labels, "arm_*"), [0, 1])

    def test_str_no_match(self):
        labels = ["alpha", "beta", "gamma"]
        self.assertEqual(match_labels(labels, "delta"), [])

    def test_str_star_matches_all(self):
        labels = ["a", "b", "c"]
        self.assertEqual(match_labels(labels, "*"), [0, 1, 2])

    def test_list_str_union(self):
        labels = ["alpha", "beta", "gamma", "delta"]
        self.assertEqual(match_labels(labels, ["alpha", "gamma"]), [0, 2])

    def test_list_int_passthrough(self):
        labels = ["a", "b", "c"]
        self.assertEqual(match_labels(labels, [2, 0]), [2, 0])

    def test_list_str_wildcard_union(self):
        labels = ["arm_left", "arm_right", "leg_left", "leg_right"]
        result = match_labels(labels, ["arm_*", "leg_left"])
        self.assertEqual(result, [0, 1, 2])

    def test_empty_list_returns_empty(self):
        labels = ["a", "b", "c"]
        self.assertEqual(match_labels(labels, []), [])

    def test_type_error_on_invalid_element(self):
        labels = ["a", "b"]
        with self.assertRaises(TypeError):
            match_labels(labels, [1.5])

    def test_type_error_on_none_element(self):
        labels = ["a", "b"]
        with self.assertRaises(TypeError):
            match_labels(labels, [None])

    def test_int_out_of_bounds_passthrough(self):
        """int indices are passed through without bounds checking."""
        labels = ["a", "b"]
        result = match_labels(labels, [99])
        self.assertEqual(result, [99])

    def test_list_str_overlapping_patterns_deduplicates(self):
        """Overlapping glob patterns should not produce duplicate indices."""
        labels = ["arm_left", "arm_right", "leg_left", "leg_right"]
        result = match_labels(labels, ["arm_*", "*_left"])
        self.assertEqual(result, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
