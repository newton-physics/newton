# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run named argument variants of one example in a subprocess."""

import gc
import json
import runpy
import sys
import unittest
import warnings
from unittest.mock import patch

import warp as wp

import newton.tests.unittest_utils
from newton.tests.unittest_utils import NewtonTestCase


class ExampleVariant(NewtonTestCase):
    def __init__(self, module, case):
        super().__init__()
        self.module = module
        self.case = case

    def id(self):
        return f"{self.module}[{self.case['label']}]"

    def _callSetUp(self):
        if self.case["check_output"]:
            super()._callSetUp()
        else:
            unittest.TestCase._callSetUp(self)

    def runTest(self):
        """Run the normal entry point with this variant's output contract."""
        self.addCleanup(gc.collect)
        if self.case["check_output"]:
            for required, patterns in ((True, self.case["expected"]), (False, self.case["allowed"])):
                register = self.expectOutputRegex if required else self.allowOutputRegex
                for spec in patterns:
                    pattern, stream = (spec, "any") if isinstance(spec, str) else spec
                    register(pattern, stream=stream)
        with patch.object(sys, "argv", [self.module, *self.case["argv"]]), warnings.catch_warnings():
            if self.case["allow_deprecation_warnings"]:
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("default", category=DeprecationWarning, module=r"newton(\.|$)")
            elif not sys.warnoptions:
                warnings.filterwarnings("default", category=DeprecationWarning, module=r"newton(\.|$)")
            runpy.run_module(self.module, run_name="__main__")


def main():
    with open(sys.argv[1], encoding="utf-8") as stream:
        manifest = json.load(stream)
    wp.config.log_level = max(wp.config.log_level, wp.LOG_WARNING)
    newton.tests.unittest_utils.strict_warnings = manifest["strict_warnings"]
    newton.tests.unittest_utils.allowed_deprecation_warnings = tuple(manifest["allowed_deprecation_warnings"])
    suite = unittest.TestSuite(ExampleVariant(manifest["module"], case) for case in manifest["cases"])
    result = unittest.TestResult()
    suite.run(result)
    for test, error in result.failures + result.errors:
        print(f"{test.id()}\n{error}", file=sys.stderr)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
