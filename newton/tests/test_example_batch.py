# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import unittest
from unittest import mock

from newton.tests import example_batch
from newton.tests.test_examples import _check_example_result
from newton.tests.unittest_utils import NewtonTestCase


class TestExampleBatch(unittest.TestCase):
    def _run_cases(self, cases, entry_point):
        result = unittest.TestResult()

        class Batch(NewtonTestCase):
            def runTest(test):
                """Validate each variant's captured result in the parent."""
                for case in cases:
                    with test.subTest(variant=case["label"]):
                        argv = case.get("argv", [])
                        output = example_batch.run_variant("example_module", argv)
                        command = [sys.executable, "-m", "example_module", *argv]
                        _check_example_result(
                            test, subprocess.CompletedProcess(args=command, **output), is_cuda=False, variant=case
                        )

        with (
            mock.patch.object(example_batch.runpy, "run_module", side_effect=entry_point),
            mock.patch.object(example_batch.wp, "synchronize"),
        ):
            Batch().run(result)
        return result

    def test_failure_identifies_variant_and_does_not_stop_batch(self):
        """Keep later variants running after an identified failure."""
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
        self.assertFalse(result.errors)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("variant='xpbd'", result.failures[0][0].id())
        self.assertIn("example assertion failed", result.failures[0][1])
        self.assertIn("-m example_module --test --solver xpbd", result.failures[0][1])

    def test_output_contract_is_isolated_between_variants(self):
        """Keep allowances and required output local to each variant."""
        output = iter(("expected result\nbackend notice", "backend notice"))
        result = self._run_cases(
            [
                {
                    "label": "first",
                    "expect_output_regexes": ["expected result"],
                    "allow_output_regexes": ["backend notice"],
                },
                {"label": "second", "expect_output_regexes": ["expected result"]},
            ],
            lambda *args, **kwargs: print(next(output)),
        )
        self.assertFalse(result.errors)
        self.assertEqual(len(result.failures), 1)
        test, failure = result.failures[0]
        self.assertIn("variant='second'", test.id())
        self.assertIn("Missing expected output", failure)
        self.assertIn("Unexpected stdout:\nbackend notice", failure)


if __name__ == "__main__":
    unittest.main()
