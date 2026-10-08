# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from unittest import mock

from newton.tests import example_batch


class TestExampleBatch(unittest.TestCase):
    def _run_cases(self, cases, entry_point):
        defaults = {
            "argv": [],
            "expected": [],
            "allowed": [],
            "check_output": True,
            "allow_deprecation_warnings": False,
        }
        suite = unittest.TestSuite(
            example_batch.ExampleVariant("example_module", {**defaults, **case}) for case in cases
        )
        result = unittest.TestResult()
        with (
            mock.patch.object(example_batch.runpy, "run_module", side_effect=entry_point),
            mock.patch.object(example_batch.wp, "synchronize"),
        ):
            suite.run(result)
        return result

    def test_failure_identifies_variant_and_does_not_stop_batch(self):
        seen = []
        original_argv = sys.argv

        def entry_point(module, *, run_name):
            self.assertEqual((module, run_name), ("example_module", "__main__"))
            seen.append(sys.argv.copy())
            if sys.argv[-1] == "xpbd":
                raise ValueError("example assertion failed")

        result = self._run_cases(
            [{"label": solver, "argv": ["--test", "--solver", solver]} for solver in ("xpbd", "vbd")], entry_point
        )
        self.assertEqual(len(seen), 2)
        self.assertIs(sys.argv, original_argv)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0][0].id(), "example_module[xpbd]")
        self.assertIn("example assertion failed", result.errors[0][1])

    def test_output_allowance_does_not_leak_between_variants(self):
        result = self._run_cases(
            [{"label": "allowed", "allowed": [("backend notice", "stdout")]}, {"label": "unexpected"}],
            lambda *args, **kwargs: print("backend notice"),
        )
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0][0].id(), "example_module[unexpected]")

    def test_required_output_must_appear_in_each_variant(self):
        calls = 0

        def entry_point(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                print("expected result")

        result = self._run_cases(
            [{"label": label, "expected": [("expected result", "stdout")]} for label in ("first", "second")],
            entry_point,
        )
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0][0].id(), "example_module[second]")


if __name__ == "__main__":
    unittest.main()
