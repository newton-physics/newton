# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run existing MJCF tests against either importer without changing their assertions."""

import argparse
import importlib
import json
import sys
import unittest
from collections import Counter
from contextlib import ExitStack
from unittest import mock

import newton

DEFAULT_MODULES = (
    "newton.tests.test_import_mjcf",
    "newton.tests.test_sites_mjcf_import",
    "newton.tests.test_mujoco_general_actuators",
)


class _ComparisonResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self.fixture_errors = []
        self.fixture_import_calls = 0
        self.current = None

    def startTest(self, test):
        super().startTest(test)
        self.current = {"test": test.id(), "status": "passed", "import_calls": 0, "details": []}

    def stopTest(self, test):
        self.records.append(self.current)
        self.current = None
        super().stopTest(test)

    def _record_error(self, test, err, status):
        if self.current is None:
            self.fixture_errors.append({"fixture": test.id(), "details": self._exc_info_to_string(err, test)})
        else:
            self.current["status"] = status
            self.current["details"].append(self._exc_info_to_string(err, test))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record_error(test, err, "failed")

    def addError(self, test, err):
        super().addError(test, err)
        self._record_error(test, err, "error")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._record_error(subtest, err, "failed" if issubclass(err[0], test.failureException) else "error")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.current is not None:
            self.current["status"] = "skipped"
            self.current["details"].append(reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._record_error(test, err, "expected_failure")

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.current["status"] = "unexpected_success"


def _run(importer, modules, pattern):
    loader = unittest.TestLoader()
    if pattern:
        loader.testNamePatterns = [f"*{pattern}*"]
    suite = unittest.TestSuite(loader.loadTestsFromModule(module) for module in modules)

    def test_ids(tests):
        for test in tests:
            if isinstance(test, unittest.TestSuite):
                yield from test_ids(test)
            else:
                yield test.id()

    selected = list(test_ids(suite))
    runner = unittest.TextTestRunner(verbosity=2, resultclass=_ComparisonResult)
    result = runner._makeResult()
    runner.resultclass = lambda *args, **kwargs: result

    def counted(function):
        def call(*args, **kwargs):
            if result.current is not None:
                result.current["import_calls"] += 1
            else:
                result.fixture_import_calls += 1
            return function(*args, **kwargs)

        return call

    method = getattr(newton.ModelBuilder, "add_mjcf_native" if importer == "native" else "add_mjcf")
    parser = importlib.import_module(
        "newton._src.solvers.mujoco.import_mjcf_native" if importer == "native" else "newton._src.utils.import_mjcf"
    ).parse_mjcf
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(newton.ModelBuilder, "add_mjcf", counted(method)))
        for module in modules:
            if hasattr(module, "parse_mjcf"):
                stack.enter_context(mock.patch.object(module, "parse_mjcf", counted(parser)))
        print(f"\nMJCF importer: {importer}", file=sys.stderr, flush=True)
        runner.run(suite)
    executed = {record["test"] for record in result.records}
    for name in selected:
        if name not in executed:
            result.records.append({"test": name, "status": "not_run", "import_calls": 0, "details": []})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importer", choices=("legacy", "native", "both"), default="both")
    parser.add_argument("--output", required=True, help="JSON results, including tracebacks and import call counts")
    parser.add_argument("-k", default=None, help="Test name substring")
    parser.add_argument("modules", nargs="*", default=DEFAULT_MODULES)
    args = parser.parse_args()
    modules = [importlib.import_module(name) for name in args.modules]
    report = {}
    success = True
    for importer in ("legacy", "native") if args.importer == "both" else (args.importer,):
        result = _run(importer, modules, args.k)
        report[importer] = {
            "summary": dict(Counter(record["status"] for record in result.records)),
            "fixture_errors": result.fixture_errors,
            "fixture_import_calls": result.fixture_import_calls,
            "tests": result.records,
        }
        success = success and result.wasSuccessful()
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
    for importer, results in report.items():
        print(f"{importer}: {results['summary']}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
