# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import subprocess
import tempfile
import textwrap
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

    def test_collects_changed_classes_from_pr_head_not_merge_commit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            _git(repo, "init", "-b", "main")
            tests_dir = repo / "newton" / "tests"
            tests_dir.mkdir(parents=True)
            (tests_dir / "test_upstream.py").write_text("class TestUpstream:\n    pass\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "Initial tests")

            _git(repo, "switch", "-c", "pr")
            (tests_dir / "test_pr_owned.py").write_text("class TestPrOwned:\n    pass\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "Add PR test")

            _git(repo, "switch", "main")
            (tests_dir / "test_upstream.py").write_text("class TestUpstreamChanged:\n    pass\n", encoding="utf-8")
            _git(repo, "commit", "-am", "Change upstream test")

            _git(repo, "switch", "-c", "merge")
            _git(repo, "merge", "--no-ff", "pr", "-m", "Synthetic PR merge")

            old_cwd = os.getcwd()
            try:
                os.chdir(repo)
                classes = check_test_durations._collect_changed_test_classes("main", "pr")
            finally:
                os.chdir(old_cwd)

        self.assertEqual(classes, {"TestPrOwned"})


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=Newton Test", "-c", "user.email=newton@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    unittest.main()
