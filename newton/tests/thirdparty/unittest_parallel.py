# Licensed under the MIT License
# https://github.com/craigahobbs/unittest-parallel/blob/main/LICENSE

# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
unittest-parallel command-line script main module
"""

import argparse
import concurrent.futures  # NVIDIA Modification
import multiprocessing
import os
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from io import StringIO

# Work around a known OpenUSD thread-safety crash in
# UsdPhysics.LoadUsdPhysicsFromRange for collider-dense assets. OpenUSD reads
# this once when pxr initializes, so set it before test modules import pxr and
# preserve any caller-provided override.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

from newton.tests.unittest_utils import (  # NVIDIA modification
    ParallelJunitTestResult,
    write_junit_results,
)

try:
    import coverage

    COVERAGE_AVAILABLE = True  # NVIDIA Modification
except ImportError:
    COVERAGE_AVAILABLE = False  # NVIDIA Modification


# The following variables are NVIDIA Modifications
START_DIRECTORY = os.path.dirname(__file__)  # The directory to start test discovery


def main(argv=None):
    """
    unittest-parallel command-line script main entry point
    """

    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="unittest-parallel",
        # NVIDIA Modifications follow:
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Example usage:
        python -m newton.tests -p 'test_c*.py'
        python -m newton.tests -k 'mgpu' -k 'cuda'
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_const", const=2, default=1, help="Verbose output")
    parser.add_argument("-q", "--quiet", dest="verbose", action="store_const", const=0, default=1, help="Quiet output")
    parser.add_argument("-f", "--failfast", action="store_true", default=False, help="Stop on first fail or error")
    parser.add_argument(
        "-b", "--buffer", action="store_true", default=False, help="Buffer stdout and stderr during tests"
    )
    parser.add_argument(
        "-k",
        dest="testNamePatterns",
        action="append",
        type=_convert_select_pattern,
        help="Only run tests which match the given substring",
    )
    parser.add_argument(
        "-s",
        "--start-directory",
        metavar="START",
        default=os.path.join(os.path.dirname(__file__), ".."),
        help="Directory to start discovery ('.' default)",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        metavar="PATTERN",
        default="test*.py",
        help="'autodetect' suite only: Pattern to match tests ('test*.py' default)",  # NVIDIA Modification
    )
    parser.add_argument(
        "-t",
        "--top-level-directory",
        metavar="TOP",
        help="Top level directory of project (defaults to start directory)",
    )
    parser.add_argument(
        "--junit-report-xml", metavar="FILE", help="Generate JUnit report format XML file"
    )  # NVIDIA Modification
    group_parallel = parser.add_argument_group("parallelization options")
    group_parallel.add_argument(
        "-j",
        "--jobs",
        metavar="COUNT",
        type=int,
        default=0,
        help="The number of test processes (default is 0, all cores)",
    )
    group_parallel.add_argument(
        "-m",
        "--maxjobs",
        metavar="MAXCOUNT",
        type=int,
        default=8,
        help="The maximum number of test processes (default is 8)",
    )  # NVIDIA Modification
    group_parallel.add_argument(
        "--level",
        choices=["module", "class", "test"],
        default="class",
        help="Set the test parallelism level (default is 'class')",
    )
    group_parallel.add_argument(
        "--disable-process-pooling",
        action="store_true",
        default=False,
        help="Do not reuse processes used to run test suites (max_tasks_per_child=1). "
        "For the concurrent.futures backend, this is also enabled automatically when "
        "multiple CUDA devices are detected.",
    )
    group_parallel.add_argument(
        "--disable-concurrent-futures",
        action="store_true",
        default=False,
        help="Use multiprocessing instead of concurrent.futures.",
    )  # NVIDIA Modification
    group_parallel.add_argument(
        "--serial-fallback",
        action="store_true",
        default=False,
        help="Run in a single-process (no spawning) mode without multiprocessing or concurrent.futures.",
    )  # NVIDIA Modification
    group_coverage = parser.add_argument_group("coverage options")
    group_coverage.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    group_coverage.add_argument("--coverage-branch", action="store_true", help="Run tests with branch coverage")
    group_coverage.add_argument(
        "--coverage-html",
        metavar="DIR",
        help="Generate coverage HTML report",
        default=os.path.join(START_DIRECTORY, "..", "..", "htmlcov"),
    )
    group_coverage.add_argument("--coverage-xml", metavar="FILE", help="Generate coverage XML report")
    group_coverage.add_argument(
        "--coverage-fail-under", metavar="MIN", type=float, help="Fail if coverage percentage under min"
    )
    group_warp = parser.add_argument_group("NVIDIA Warp options")  # NVIDIA Modification
    group_warp.add_argument(
        "--no-shared-cache", action="store_true", help="Use a separate kernel cache per test process."
    )
    group_warp.add_argument(
        "--no-cache-clear",
        action="store_true",
        help="Skip clearing the Warp kernel cache before running tests. "
        "Useful for faster iteration and avoiding interference with parallel sessions.",
    )
    args = parser.parse_args(args=argv)

    if args.coverage_branch:
        args.coverage = args.coverage_branch

    if args.coverage and not COVERAGE_AVAILABLE:
        parser.exit(
            status=2, message="--coverage was used, but coverage was not found. Is it installed?\n"
        )  # NVIDIA Modification

    process_count = max(0, args.jobs)
    if process_count == 0:
        process_count = multiprocessing.cpu_count()
    process_count = min(process_count, args.maxjobs)  # NVIDIA Modification

    import warp as wp  # noqa: PLC0415 NVIDIA Modification

    # Clear the Warp cache (NVIDIA Modification)
    if not args.no_cache_clear:
        wp.clear_lto_cache()
        wp.clear_kernel_cache()
        print("Cleared Warp kernel cache")

    # Create the temporary directory (for coverage files)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Discover tests
        with _coverage(args, temp_dir):
            test_loader = unittest.TestLoader()
            if args.testNamePatterns:
                test_loader.testNamePatterns = args.testNamePatterns
            discover_suite = test_loader.discover(
                args.start_directory, pattern=args.pattern, top_level_dir=args.top_level_directory
            )

        # Get the parallelizable test suites
        if args.level == "test":
            test_suites = list(_iter_test_cases(discover_suite))
        elif args.level == "class":
            test_suites = list(_iter_class_suites(discover_suite))
        else:  # args.level == 'module'
            test_suites = list(_iter_module_suites(discover_suite))

        # Don't use more processes than test suites
        process_count = max(1, min(len(test_suites), process_count))

        if not args.serial_fallback:
            # Report test suites and processes
            print(
                f"Running {len(test_suites)} test suites ({discover_suite.countTestCases()} total tests) across {process_count} processes",
                file=sys.stderr,
            )
            if args.verbose > 1:
                print(file=sys.stderr)

            # Create the shared index object used in Warp caches (NVIDIA Modification)
            with multiprocessing.Manager() as manager:
                shared_index = manager.Value("i", -1)

                # Run the tests in parallel
                start_time = time.perf_counter()

                if args.disable_concurrent_futures:
                    multiprocessing_context = multiprocessing.get_context(method="spawn")
                    maxtasksperchild = 1 if args.disable_process_pooling else None
                    with multiprocessing_context.Pool(
                        process_count,
                        maxtasksperchild=maxtasksperchild,
                        initializer=initialize_test_process,
                        initargs=(manager.Lock(), shared_index, args, temp_dir),
                    ) as pool:
                        test_manager = ParallelTestManager(manager, args, temp_dir)
                        results = pool.map(test_manager.run_tests, test_suites)
                else:
                    # NVIDIA Modification added concurrent.futures
                    executor_kwargs = {
                        "max_workers": process_count,
                        "mp_context": multiprocessing.get_context(method="spawn"),
                        "initializer": initialize_test_process,
                        "initargs": (manager.Lock(), shared_index, args, temp_dir),
                    }
                    if sys.version_info >= (3, 11) and (args.disable_process_pooling or wp.get_cuda_device_count() > 1):
                        executor_kwargs["max_tasks_per_child"] = 1
                    with concurrent.futures.ProcessPoolExecutor(**executor_kwargs) as executor:
                        test_manager = ParallelTestManager(manager, args, temp_dir)
                        results = list(executor.map(test_manager.run_tests, test_suites, timeout=2400))
        else:
            # This entire path is an NVIDIA Modification

            # Report test suites and processes
            print(f"Running {discover_suite.countTestCases()} total tests (serial fallback)", file=sys.stderr)
            if args.verbose > 1:
                print(file=sys.stderr)

            # Run the tests in serial
            start_time = time.perf_counter()

            with multiprocessing.Manager() as manager:
                test_manager = ParallelTestManager(manager, args, temp_dir)
                results = [test_manager.run_tests(discover_suite)]

        stop_time = time.perf_counter()
        test_duration = stop_time - start_time

        # Aggregate parallel test run results
        tests_run = 0
        errors = []
        failures = []
        skipped = 0
        expected_failures = 0
        unexpected_successes = 0
        test_records = []  # NVIDIA Modification
        for result in results:
            tests_run += result[0]
            errors.extend(result[1])
            failures.extend(result[2])
            skipped += result[3]
            expected_failures += result[4]
            unexpected_successes += result[5]
            test_records += result[6]  # NVIDIA Modification
        is_success = not (errors or failures or unexpected_successes)

        # Compute test info
        infos = []
        if failures:
            infos.append(f"failures={len(failures)}")
        if errors:
            infos.append(f"errors={len(errors)}")
        if skipped:
            infos.append(f"skipped={skipped}")
        if expected_failures:
            infos.append(f"expected failures={expected_failures}")
        if unexpected_successes:
            infos.append(f"unexpected successes={unexpected_successes}")

        # Report test errors
        if errors or failures:
            print(file=sys.stderr)
            for error in errors:
                print(error, file=sys.stderr)
            for failure in failures:
                print(failure, file=sys.stderr)
        elif args.verbose > 0:
            print(file=sys.stderr)

        # Test report
        print(unittest.TextTestResult.separator2, file=sys.stderr)
        print(f"Ran {tests_run} {'tests' if tests_run > 1 else 'test'} in {test_duration:.3f}s", file=sys.stderr)
        print(file=sys.stderr)
        print(f"{'OK' if is_success else 'FAILED'}{' (' + ', '.join(infos) + ')' if infos else ''}", file=sys.stderr)

        if test_records and args.junit_report_xml:
            # NVIDIA modification to report results in Junit XML format
            write_junit_results(
                args.junit_report_xml,
                test_records,
                tests_run,
                len(failures) + unexpected_successes,
                len(errors),
                skipped,
                test_duration,
            )

        # Return an error status on failure
        if not is_success:
            parser.exit(status=len(errors) + len(failures) + unexpected_successes)

        # Coverage?
        if args.coverage:
            # Combine the coverage files
            cov_options = {}
            cov_options["config_file"] = True  # Grab configuration from pyproject.toml (must install coverage[toml])
            cov = coverage.Coverage(**cov_options)
            cov.combine(data_paths=[os.path.join(temp_dir, x) for x in os.listdir(temp_dir)])

            # Coverage report
            print(file=sys.stderr)
            percent_covered = cov.report(ignore_errors=True, file=sys.stderr)
            print(f"Total coverage is {percent_covered:.2f}%", file=sys.stderr)

            # HTML coverage report
            if args.coverage_html:
                cov.html_report(directory=args.coverage_html, ignore_errors=True)

            # XML coverage report
            if args.coverage_xml:
                cov.xml_report(outfile=args.coverage_xml, ignore_errors=True)

            # Fail under
            if args.coverage_fail_under and percent_covered < args.coverage_fail_under:
                parser.exit(status=2)


def _convert_select_pattern(pattern):
    if "*" not in pattern:
        return f"*{pattern}*"
    return pattern


@contextmanager
def _coverage(args, temp_dir):
    # Running tests with coverage?
    if args.coverage:
        # Generate a random coverage data file name - file is deleted along with containing directory
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as coverage_file:
            pass

        # Create the coverage object
        cov_options = {
            "branch": args.coverage_branch,
            "data_file": coverage_file.name,
            # NVIDIA Modification removed unneeded options
        }
        cov_options["config_file"] = True  # Grab configuration from pyproject.toml (must install coverage[toml])
        cov = coverage.Coverage(**cov_options)
        try:
            # Start measuring code coverage
            cov.start()

            # Yield for unit test running
            yield cov
        finally:
            # Stop measuring code coverage
            cov.stop()

            # Save the collected coverage data to the data file
            cov.save()
    else:
        # Not running tests with coverage - yield for unit test running
        yield None


# Iterate module-level test suites - all top-level test suites returned from TestLoader.discover
def _iter_module_suites(test_suite):
    for module_suite in test_suite:
        if module_suite.countTestCases():
            yield module_suite


# Iterate class-level test suites - test suites that contains test cases
def _iter_class_suites(test_suite):
    has_cases = any(isinstance(suite, unittest.TestCase) for suite in test_suite)
    if has_cases:
        yield test_suite
    else:
        for suite in test_suite:
            yield from _iter_class_suites(suite)


# Iterate test cases (methods)
def _iter_test_cases(test_suite):
    if isinstance(test_suite, unittest.TestCase):
        yield test_suite
    else:
        for suite in test_suite:
            yield from _iter_test_cases(suite)


class ParallelTestManager:
    # Manager proxy calls can fail with ConnectionError, TypeError, or OSError
    # due to a TOCTOU race in Connection.send() where GC can close the
    # connection handle between the closed-check and the write call
    # (see https://github.com/python/cpython/issues/84582). Since failfast
    # is a best-effort optimization, we log a warning and continue.
    _PROXY_ERRORS = (ConnectionError, TypeError, OSError)

    def __init__(self, manager, args, temp_dir):
        self.args = args
        self.temp_dir = temp_dir
        self.failfast = manager.Event()

    def run_tests(self, test_suite):
        # Fail fast?
        try:
            if self.failfast.is_set():
                return [0, [], [], 0, 0, 0, []]  # NVIDIA Modification
        except self._PROXY_ERRORS as exc:
            print(
                f"Warning: failfast proxy is_set() failed ({type(exc).__name__}), continuing test execution",
                file=sys.stderr,
            )

        # NVIDIA Modification for GitLab
        import newton.tests.unittest_utils  # noqa: PLC0415

        newton.tests.unittest_utils.coverage_enabled = self.args.coverage
        newton.tests.unittest_utils.coverage_temp_dir = self.temp_dir
        newton.tests.unittest_utils.coverage_branch = self.args.coverage_branch

        if self.args.junit_report_xml:
            resultclass = ParallelJunitTestResult
        else:
            resultclass = ParallelTextTestResult

        # Run unit tests
        with _coverage(self.args, self.temp_dir):
            runner = unittest.TextTestRunner(
                stream=StringIO(),
                resultclass=resultclass,  # NVIDIA Modification
                verbosity=self.args.verbose,
                failfast=self.args.failfast,
                buffer=self.args.buffer,
            )
            result = runner.run(test_suite)

            # Set failfast, if necessary
            if result.shouldStop:
                try:
                    self.failfast.set()
                except self._PROXY_ERRORS as exc:
                    print(
                        f"Warning: failfast proxy set() failed ({type(exc).__name__}), "
                        "other workers may not stop early",
                        file=sys.stderr,
                    )

            # Return (test_count, errors, failures, skipped_count, expected_failure_count, unexpected_success_count)
            return (
                result.testsRun,
                [self._format_error(result, error) for error in result.errors],
                [self._format_error(result, failure) for failure in result.failures],
                len(result.skipped),
                len(result.expectedFailures),
                len(result.unexpectedSuccesses),
                result.test_record,  # NVIDIA modification
            )

    @staticmethod
    def _format_error(result, error):
        return "\n".join(
            [
                unittest.TextTestResult.separator1,
                result.getDescription(error[0]),
                unittest.TextTestResult.separator2,
                error[1],
            ]
        )


class ParallelTextTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        stream = type(stream)(sys.stderr)
        super().__init__(stream, descriptions, verbosity)
        self.test_record = []  # NVIDIA modification

    def startTest(self, test):
        if self.showAll:
            self.stream.writeln(f"{self.getDescription(test)} ...")
            self.stream.flush()
        elif self.dots:
            self.stream.writeln(f"{test} ...")
            self.stream.flush()
        super(unittest.TextTestResult, self).startTest(test)

    def stopTest(self, test):
        super().stopTest(test)
        # Force garbage collection of CPU-side allocations and release unused
        # CUDA mempool memory to reduce peak host RSS in parallel test runs
        # (see issue #1881).
        import gc  # noqa: PLC0415

        gc.collect()
        import warp as wp  # noqa: PLC0415

        for device_name in wp.get_cuda_devices():
            if wp.is_mempool_enabled(device_name):
                wp.set_mempool_release_threshold(device_name, 0)

    def _add_helper(self, test, show_all_message):
        if self.showAll:
            self.stream.writeln(f"{self.getDescription(test)} ... {show_all_message}")
        elif self.dots:
            self.stream.writeln(f"{test} ... {show_all_message}")
        self.stream.flush()

    def addSuccess(self, test):
        super(unittest.TextTestResult, self).addSuccess(test)
        self._add_helper(test, "ok")

    def addError(self, test, err):
        super(unittest.TextTestResult, self).addError(test, err)
        self._add_helper(test, "ERROR")

    def addFailure(self, test, err):
        super(unittest.TextTestResult, self).addFailure(test, err)
        self._add_helper(test, "FAIL")

    def addSkip(self, test, reason):
        super(unittest.TextTestResult, self).addSkip(test, reason)
        self._add_helper(test, f"skipped {reason!r}")

    def addExpectedFailure(self, test, err):
        super(unittest.TextTestResult, self).addExpectedFailure(test, err)
        self._add_helper(test, "expected failure")

    def addUnexpectedSuccess(self, test):
        super(unittest.TextTestResult, self).addUnexpectedSuccess(test)
        self._add_helper(test, "unexpected success")

    def printErrors(self):
        pass


def initialize_test_process(lock, shared_index, args, temp_dir):
    """Necessary operations to be executed at the start of every test process.

    Currently this function can be used to set a separate Warp cache. (NVIDIA modification)
    If the environment variable `WARP_CACHE_ROOT` is detected, the cache will be placed in the provided path.

    It also ensures that Warp is initialized prior to running any tests.
    """

    with lock:
        shared_index.value += 1
        worker_index = shared_index.value

    with _coverage(args, temp_dir):
        import warp as wp  # noqa: PLC0415

        if args.no_shared_cache:
            from warp._src.thirdparty import appdirs  # noqa: PLC0415

            if "WARP_CACHE_ROOT" in os.environ:
                cache_root_dir = os.path.join(os.getenv("WARP_CACHE_ROOT"), f"{wp.config.version}-{worker_index:03d}")
            else:
                cache_root_dir = appdirs.user_cache_dir(
                    appname="warp", appauthor="NVIDIA", version=f"{wp.config.version}-{worker_index:03d}"
                )

            wp.config.kernel_cache_dir = cache_root_dir
            os.makedirs(cache_root_dir, exist_ok=True)

            if not args.no_cache_clear:
                wp.clear_lto_cache()
                wp.clear_kernel_cache()
        elif "WARP_CACHE_ROOT" in os.environ:
            # Using a shared cache for all test processes
            wp.config.kernel_cache_dir = os.path.join(os.getenv("WARP_CACHE_ROOT"), wp.config.version)


if __name__ == "__main__":  # pragma: no cover
    main()
