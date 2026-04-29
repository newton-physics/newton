# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import tempfile
import textwrap
import unittest
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_test_durations.py"
_SPEC = importlib.util.spec_from_file_location("check_test_durations", _SCRIPT_PATH)
check_test_durations = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(check_test_durations)


class TestChangedTestDurationCheck(unittest.TestCase):
    def test_parse_changed_ranges_uses_new_file_line_numbers(self):
        diff = textwrap.dedent(
            """\
            diff --git a/newton/tests/test_example.py b/newton/tests/test_example.py
            index 1234567..89abcde 100644
            --- a/newton/tests/test_example.py
            +++ b/newton/tests/test_example.py
            @@ -10,0 +11,2 @@ class TestExample(unittest.TestCase):
            +    def test_new(self):
            +        self.assertTrue(True)
            @@ -25 +27 @@ class TestExample(unittest.TestCase):
            -        self.assertTrue(False)
            +        self.assertTrue(True)
            """
        )

        ranges = check_test_durations.parse_changed_ranges(diff)

        self.assertEqual(
            ranges,
            {
                "newton/tests/test_example.py": [
                    check_test_durations.LineRange(11, 12),
                    check_test_durations.LineRange(27, 27),
                ]
            },
        )

    def test_changed_test_method_filters_only_matching_junit_record(self):
        source = textwrap.dedent(
            """
            import unittest

            class TestExample(unittest.TestCase):
                def test_changed(self):
                    self.assertTrue(True)

                def test_unchanged(self):
                    self.assertTrue(True)
            """
        )
        ranges = {"newton/tests/test_example.py": [check_test_durations.LineRange(5, 5)]}

        patterns = check_test_durations.collect_affected_tests(
            {"newton/tests/test_example.py": source},
            ranges,
        )
        records = [
            check_test_durations.TestRecord("TestExample", "test_changed", 31.0, "ubuntu-latest"),
            check_test_durations.TestRecord("TestExample", "test_unchanged", 99.0, "ubuntu-latest"),
        ]

        report = check_test_durations.evaluate_records(records, patterns, warn_seconds=10.0, fail_seconds=30.0)

        self.assertEqual([record.name for record in report.failures], ["test_changed"])
        self.assertEqual(report.ignored_count, 1)

    def test_class_level_change_includes_all_tests_in_class(self):
        source = textwrap.dedent(
            """
            import unittest

            class TestExample(unittest.TestCase):
                timeout = 5

                def test_one(self):
                    self.assertTrue(True)

                def test_two(self):
                    self.assertTrue(True)
            """
        )
        ranges = {"newton/tests/test_example.py": [check_test_durations.LineRange(5, 5)]}

        patterns = check_test_durations.collect_affected_tests(
            {"newton/tests/test_example.py": source},
            ranges,
        )

        self.assertTrue(patterns.matches("TestExample", "test_one"))
        self.assertTrue(patterns.matches("TestExample", "test_two"))

    def test_dynamic_registration_change_matches_generated_test_names(self):
        source = textwrap.dedent(
            """
            import unittest
            from newton.tests.unittest_utils import add_function_test

            def check_behavior(test, device):
                test.assertTrue(True)

            class TestGenerated(unittest.TestCase):
                pass

            devices = ["cpu", "cuda:0"]
            add_function_test(TestGenerated, "test_behavior", check_behavior, devices=devices)
            """
        )
        ranges = {"newton/tests/test_generated.py": [check_test_durations.LineRange(5, 6)]}

        patterns = check_test_durations.collect_affected_tests(
            {"newton/tests/test_generated.py": source},
            ranges,
        )

        self.assertTrue(patterns.matches("TestGenerated", "test_behavior_cpu"))
        self.assertTrue(patterns.matches("TestGenerated", "test_behavior_cuda_0"))
        self.assertFalse(patterns.matches("TestGenerated", "test_other_cpu"))

    def test_junit_artifacts_are_parsed_with_artifact_name_as_environment(self):
        xml = textwrap.dedent(
            """\
            <?xml version="1.0" encoding="utf-8"?>
            <testsuite name="Warp Tests" tests="1" failures="0" errors="0" skipped="0" time="31.000">
              <testcase classname="TestExample" name="test_slow" time="31.000" />
            </testsuite>
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test-results-ubuntu-latest" / "rspec.xml"
            path.parent.mkdir()
            path.write_text(xml, encoding="utf-8")

            records = check_test_durations.parse_junit_records(Path(tmpdir))

        self.assertEqual(records, [check_test_durations.TestRecord("TestExample", "test_slow", 31.0, "ubuntu-latest")])


if __name__ == "__main__":
    unittest.main()
