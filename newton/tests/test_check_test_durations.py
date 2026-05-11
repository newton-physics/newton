# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_test_durations.py"
_SPEC = importlib.util.spec_from_file_location("check_test_durations", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load check_test_durations module from {_SCRIPT_PATH}")
check_test_durations = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_test_durations)


class TestChangedTestDurationCheck(unittest.TestCase):
    def test_collect_test_classes_follows_newton_test_class_convention(self):
        source = textwrap.dedent(
            """
            class Helper:
                pass

            class TestExample:
                pass
            """
        )

        self.assertEqual(check_test_durations.collect_test_classes(source), {"TestExample"})

    def test_parses_junit_records_and_filters_changed_classes(self):
        xml = textwrap.dedent(
            """\
            <?xml version="1.0" encoding="utf-8"?>
            <testsuite name="Newton Tests" tests="2" failures="0" errors="0" skipped="0" time="130.000">
              <testcase classname="TestChanged" name="test_slow" time="31.000" />
              <testcase classname="TestOther" name="test_slower" time="99.000" />
            </testsuite>
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test-results-ubuntu-latest" / "rspec.xml"
            path.parent.mkdir()
            path.write_text(xml, encoding="utf-8")

            records = check_test_durations.parse_junit_records(Path(tmpdir))

        failures, warnings = check_test_durations.evaluate_records(records, {"TestChanged"})

        self.assertEqual(failures, [("TestChanged", "test_slow", 31.0, "ubuntu-latest")])
        self.assertEqual(warnings, [])

    def test_intentional_duration_gate_probe(self):
        time.sleep(31.0)


if __name__ == "__main__":
    unittest.main()
