# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import argparse
import unittest
from pathlib import Path

import newton
from newton._src.solvers.kamino.tests import setup_tests as setup_tests_internal
from newton.tests.kamino import setup_tests as setup_tests_newton

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
        default=False,  # Edit to enable/disable cache clear (if not running in command line)
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

    # Perform global setup (internal + public unit tests)
    setup_tests_internal(verbose=args.verbose, device=args.device, clear_cache=args.clear_cache)
    setup_tests_newton(verbose=args.verbose, device=args.device, clear_cache=args.clear_cache)

    # Kamino tests live in two locations: internal (private helpers) and under newton/tests/kamino
    # (public-facing / integration).
    this_file = Path(__file__).resolve()
    repo_root = Path(newton.__file__).resolve().parent
    test_folder_internal = this_file.parent
    test_folder_newton = repo_root / "tests" / "kamino"

    # Discover unit tests from both folders
    # Note: use repo root as top_level_dir as discovery doesn't allow moving up in the folder hierarchy
    tests_internal = unittest.TestLoader().discover(
        start_dir=str(test_folder_internal),
        pattern="test_*.py",
        top_level_dir=str(repo_root),
    )
    tests_newton = unittest.TestLoader().discover(
        start_dir=str(test_folder_newton),
        pattern="test_*.py",
        top_level_dir=str(repo_root),
    )

    # Run tests
    suite = unittest.TestSuite([tests_internal, tests_newton])
    ModuleHeaderTestRunner(verbosity=2).run(suite)
