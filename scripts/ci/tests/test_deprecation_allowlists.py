# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ci" / "check_deprecation_allowlists.py"

spec = importlib.util.spec_from_file_location("check_deprecation_allowlists", SCRIPT)
assert spec is not None and spec.loader is not None
checker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checker
spec.loader.exec_module(checker)


class TestDeprecationAllowlists(unittest.TestCase):
    def _check(self, dependency: str, allowlist: str) -> list[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pyproject = root / "pyproject.toml"
            pyproject.write_text(
                f'[project]\nname = "fixture"\nversion = "0"\ndependencies = [{dependency!r}]\n',
                encoding="utf-8",
            )
            allowlist_path = root / "warnings.txt"
            allowlist_path.write_text(allowlist, encoding="utf-8")
            return checker.check_allowlists(pyproject, [allowlist_path])

    def test_accepts_minimum_below_removal_threshold(self):
        """Accept an exception while the declared minimum is below its threshold."""
        errors = self._check(
            "warp-lang>=1.17.0",
            "# remove-when-minimum: warp-lang>=1.18\nwarning prefix\n",
        )

        self.assertEqual(errors, [])

    def test_rejects_minimum_at_or_above_removal_threshold(self):
        """Reject an exception once the declared minimum reaches its threshold."""
        for dependency in ("warp-lang>=1.18", "warp-lang>=1.18.0", "warp-lang>=1.19"):
            with self.subTest(dependency=dependency):
                errors = self._check(
                    dependency,
                    "# remove-when-minimum: warp-lang>=1.18\nwarning prefix\n",
                )

                self.assertEqual(len(errors), 1)
                self.assertIn("warning prefix", errors[0])
                self.assertIn("remove this allowlist entry", errors[0])

    def test_ignores_upper_bound_when_comparing_minimum(self):
        """Compare the lower bound when the dependency also has an upper bound."""
        errors = self._check(
            "warp-lang>=1.17,<2",
            "# remove-when-minimum: warp-lang>=1.18\nwarning prefix\n",
        )

        self.assertEqual(errors, [])

    def test_requires_one_removal_condition_per_entry(self):
        """Reject missing, duplicate, and orphaned lifecycle directives."""
        cases = {
            "missing": "warning prefix\n",
            "duplicate": (
                "# remove-when-minimum: warp-lang>=1.18\n# remove-when-minimum: warp-lang>=1.19\nwarning prefix\n"
            ),
            "orphaned": "# remove-when-minimum: warp-lang>=1.18\n",
        }
        for name, allowlist in cases.items():
            with self.subTest(name=name):
                self.assertNotEqual(self._check("warp-lang>=1.17", allowlist), [])

    def test_rejects_unsupported_removal_requirement(self):
        """Reject lifecycle requirements that are not one plain greater-or-equal bound."""
        conditions = (
            "warp-lang~=1.18",
            "warp-lang>=1.18,<2",
            "warp-lang[extra]>=1.18",
            "warp-lang>=1.18; python_version >= '3.12'",
            "warp-lang @ https://example.invalid/warp.whl",
        )
        for condition in conditions:
            with self.subTest(condition=condition):
                errors = self._check(
                    "warp-lang>=1.17",
                    f"# remove-when-minimum: {condition}\nwarning prefix\n",
                )
                self.assertEqual(len(errors), 1)
                self.assertIn("plain requirement with exactly one >= specifier", errors[0])

    def test_rejects_missing_or_unsupported_project_minimum(self):
        """Reject a condition whose direct dependency minimum cannot be determined."""
        cases = {
            "other-package>=1": "is not a direct project dependency",
            "warp-lang~=1.17": "does not declare exactly one >= lower bound",
            "warp-lang>=1.17; python_version >= '3.12'": "uses an environment marker",
        }
        for dependency, expected in cases.items():
            with self.subTest(dependency=dependency):
                errors = self._check(
                    dependency,
                    "# remove-when-minimum: warp-lang>=1.18\nwarning prefix\n",
                )
                self.assertEqual(len(errors), 1)
                self.assertIn(expected, errors[0])

    def test_reports_all_obsolete_entries(self):
        """Report every obsolete entry in one checker invocation."""
        errors = self._check(
            "warp-lang>=1.18",
            (
                "# explanatory comment\n"
                "# remove-when-minimum: warp-lang>=1.18\n"
                "first warning\n\n"
                "# remove-when-minimum: warp-lang>=1.18\n"
                "second warning\n"
            ),
        )

        self.assertEqual(len(errors), 2)
        self.assertIn("first warning", errors[0])
        self.assertIn("second warning", errors[1])
