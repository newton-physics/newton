# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kamino: Tests for performance-profiles utilities"""

import unittest

import numpy as np

import newton._src.solvers.kamino._src.utils.profiles as profiles
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Tests
###


class TestUtilsPerformanceProfiles(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for verbose output
        self.plots = test_context.verbose  # Set to True to generate plots

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

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
        # From perfprof.csv
        data = [
            [32.0, 15.0, 7.0, 9.0],
            [547.0, 338.0, 196.0, 1082.0],
            [11.0, 11.0, 18.0, 112.0],
            [93.0, 102.0, 20.0, 3730.0],
            [40.0, 38.0, 91.0, 74.0],
            [1599.0, 1638.0, 3202.0, 2700.0],
            [30.0, 56.0, 274.0, 75.0],
            [30.0, 56.0, 274.0, 75.0],
            [384.0, 361.0, 574.0, 843.0],
            [19.0, 18.0, 18.0, 18.0],
            [91.0, 87.0, 227.0, 374.0],
            [65339.0, 49665.0, 58191.0, np.inf],
            [np.inf, 68103.0, np.inf, np.inf],
            [12.0, 12.0, 18.0, 13.0],
            [12.0, 12.0, 15.0, 12.0],
            [13.0, 13.0, 16.0, 14.0],
            [15.0, 15.0, 19.0, 15.0],
            [158.0, 167.0, 545.0, 448.0],
            [133.0, 128.0, 280.0, 403.0],
            [133.0, 127.0, 279.0, 356.0],
            [130.0, 126.0, 250.0, 331.0],
            [332.0, 286.0, 1185.0, 794.0],
            [76.0, 64.0, 105.0, 130.0],
            [67.0, 64.0, 125.0, 131.0],
            [64.0, 57.0, 146.0, 151.0],
            [313.0, 261.0, 616.0, 584.0],
            [119.0, 101.0, 248.0, 388.0],
            [103.0, 94.0, 250.0, 304.0],
            [99.0, 88.0, 253.0, 264.0],
            [1432.0, 2188.0, 1615.0, 1856.0],
            [22.0, 21.0, 15.0, 205.0],
            [51.0, 51.0, 76.0, 51.0],
            [37.0, 40.0, 51.0, 63.0],
            [5.0, 5.0, 5.0, 5.0],
            [11552.0, 12992.0, 91294.0, 92516.0],
            [2709.0, 3761.0, 3875.0, 5026.0],
            [8639.0, 9820.0, 48442.0, 30701.0],
            [19.0, 19.0, 31.0, 20.0],
            [488.0, 489.0, 10311.0, 924.0],
            [47.0, 50.0, 162.0, 135.0],
            [5650.0, 5871.0, 15317.0, 9714.0],
            [233.0, 225.0, 602.0, 828.0],
            [322.0, 302.0, 1076.0, 1014.0],
            [33.0, 31.0, 43.0, 194.0],
            [7949.0, 10545.0, 9069.0, 7873.0],
            [np.inf, np.inf, 374.0, np.inf],
            [602.0, 617.0, 1835.0, 1865.0],
            [26.0, 27.0, 42.0, 181.0],
            [386.0, 398.0, 442.0, 938.0],
            [12.0, 11.0, 13.0, 12.0],
            [1438.0, 1368.0, 1462.0, 2218.0],
            [1177.0, 1144.0, 1535.0, 2310.0],
            [306.0, 257.0, 245.0, 915.0],
            [1223.0, 1316.0, 23646.0, 1393.0],
            [8093.0, 5603.0, 40011.0, 26782.0],
            [89403.0, np.inf, np.inf, 62244.0],
            [19.0, 22.0, 14.0, 87.0],
            [404.0, 300.0, 308.0, 590.0],
            [68.0, 68.0, 132.0, 96.0],
            [48.0, 48.0, 48.0, 48.0],
            [45.0, 38.0, 47.0, 65.0],
            [357.0, 351.0, 1054.0, 840.0],
            [51.0, 51.0, 76.0, 51.0],
            [41.0, 40.0, 97.0, 57.0],
            [40.0, 52.0, 29.0, 223.0],
            [5112.0, 5119.0, np.inf, 15677.0],
            [39.0, 29.0, 66.0, 60.0],
            [154.0, 151.0, 351.0, 460.0],
            [18.0, 17.0, 49.0, 37.0],
            [8820.0, 6500.0, 53977.0, 74755.0],
            [3791.0, 8193.0, 47028.0, 37111.0],
            [20.0, 20.0, 28.0, 20.0],
            [26.0, 24.0, 4820.0, 76.0],
            [1543.0, 1254.0, 4309.0, 6017.0],
            [135.0, 137.0, 171.0, 342.0],
            [24.0, 23.0, 29.0, 41.0],
            [30.0, 31.0, np.inf, 41.0],
        ]
        data = np.array(data, dtype=float).T

        pp = profiles.PerformanceProfile(data, taumax=np.inf)
        self.assertTrue(pp.is_valid)

        if self.plots:
            pp.plot(["Alg1", "Alg2", "Alg3", "Alg4"])  # visual sanity check


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
