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

import argparse
import os
import unittest

from newton._src.solvers.kamino.tests import setup_tests

###
# Utilities
###


# Overload of TextTestResult printing a header for each new test module
class ModuleHeaderTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_module = None

    def startTest(self, test):
        module = test.__class__.__module__
        if module != self._current_module:
            self._current_module = module
            filename = module.replace(".", "/") + ".py"

            # Print spacing + header
            self.stream.write("\n\n")
            self.stream.write(f"=== Running tests in: {filename} ===\n")
            self.stream.write("\n")
            self.stream.flush()

        super().startTest(test)


# Overload of TextTestRunner printing a header for each new test module
class ModuleHeaderTestRunner(unittest.TextTestRunner):
    resultclass = ModuleHeaderTestResult


###
# Test execution
###

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Runs all unit tests in Kamino.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",  # Edit to change device (if not running in command line)
        help="The compute device to use.",
    )
    parser.add_argument(
        "--clear-cache",
        default=True,  # Edit to enable/disable cache clear (if not running in command line)
        action=argparse.BooleanOptionalAction,
        help="Whether to clear the warp cache before running tests.",
    )
    parser.add_argument(
        "--verbose",
        default=False,  # Edit to change verbosity (if not running in command line)
        action=argparse.BooleanOptionalAction,
        help="Whether to print detailed information during tests execution.",
    )
    args = parser.parse_args()

    # Perform global setup
    setup_tests(verbose=args.verbose, device=args.device, clear_cache=args.clear_cache)

    # Detect all unit tests
    test_folder = os.path.dirname(os.path.abspath(__file__))
    tests = unittest.defaultTestLoader.discover(test_folder, pattern="test_*.py")

    # Run tests
    ModuleHeaderTestRunner(verbosity=2).run(tests)
