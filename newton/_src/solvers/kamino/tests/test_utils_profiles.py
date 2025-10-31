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

"""Kamino: Tests for performance-profiles utilities"""

import unittest
from pathlib import Path

import numpy as np

import newton._src.solvers.kamino.utils.logger as msg
import newton._src.solvers.kamino.utils.profiles as profiles

###
# Utilities
###


def _load_csv(path: Path) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=",", dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


###
# Tests
###


class TestUtilsLinAlgProfiles(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True for verbose output
        self.plots = False  # Set to True to generate plots

        # Configure logger
        if self.verbose:
            msg.set_log_level(msg.LogLevel.INFO)

        # Data directory (contains perfprof.csv)
        self.data_dir = Path(__file__).parent / "data"

    def tearDown(self):
        if self.verbose:
            msg.reset_log_level()

    def test_01_perfprof_minimal_data(self):
        # ns = 2 solvers, np = 1 problem
        ns, np_ = 2, 1
        data = np.zeros((ns, np_), dtype=float)
        data[0, :] = [1.0]  # Solver A
        data[1, :] = [2.0]  # Solver B

        # Create a performance profile (taumax = 1.0)
        pp = profiles.PerformanceProfile(data, taumax=1.0)
        self.assertTrue(pp.is_valid)

        # Optional plot
        if self.plots:
            pp.plot(["Solver A", "Solver B"])  # visual sanity check

    def test_02_perfprof_tmigot_ex2(self):
        # Example from https://tmigot.github.io/posts/2024/06/teaching/
        ns, np_ = 2, 8
        data = np.zeros((ns, np_), dtype=float)
        data[0, :] = [1.0, 1.0, 1.0, 5.0, 7.0, 6.0, np.inf, np.inf]  # Solver A
        data[1, :] = [5.0, 10.0, 20.0, 10.0, 15.0, 5.0, 20.0, 20.0]  # Solver B

        pp = profiles.PerformanceProfile(data, taumax=np.inf)
        self.assertTrue(pp.is_valid)

        if self.plots:
            pp.plot(["Solver A", "Solver B"])  # visual sanity check

    def test_03_perfprof_tmigot_ex3(self):
        # Example from https://tmigot.github.io/posts/2024/06/teaching/
        ns, np_ = 2, 5
        data = np.zeros((ns, np_), dtype=float)
        data[0, :] = [1.0, 1.0, 1.0, 1.0, 1.0]  # Solver A
        data[1, :] = [1.0003, 1.0003, 1.0003, 1.0003, 1.0003]  # Solver B

        pp = profiles.PerformanceProfile(data, taumax=1.0005)
        self.assertTrue(pp.is_valid)

        if self.plots:
            pp.plot(["Solver A", "Solver B"])  # visual sanity check

    def test_04_perfprof_tmigot_ex4(self):
        # Example from https://tmigot.github.io/posts/2024/06/teaching/
        ns, np_ = 3, 5
        data = np.zeros((ns, np_), dtype=float)
        data[0, :] = [2.0, 1.0, 1.0, 1.0, 2.0]  # Solver A
        data[1, :] = [1.5, 1.2, 4.0, 5.0, 5.0]  # Solver B
        data[2, :] = [1.0, 2.0, 2.0, 20.0, 20.0]  # Solver C

        pp = profiles.PerformanceProfile(data, taumax=np.inf)
        self.assertTrue(pp.is_valid)

        if self.plots:
            pp.plot(["Solver A", "Solver B", "Solver C"])  # visual sanity check

    def test_05_perfprof_example_large(self):
        # CSV from tests/data/perfprof.csv (matches C++ test path)
        csv_path = self.data_dir / "perfprof.csv"
        if not csv_path.exists():
            self.skipTest("perfprof.csv not found under tests/data")

        mat = _load_csv(csv_path)
        data = mat.T  # match C++ test transpose

        pp = profiles.PerformanceProfile(data, taumax=np.inf)
        self.assertTrue(pp.is_valid)

        if self.plots:
            pp.plot(["Alg1", "Alg2", "Alg3", "Alg4"])  # visual sanity check


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=2000, precision=10, threshold=10000, suppress=True)  # Suppress scientific notation

    # Run all tests
    unittest.main(verbosity=2)
