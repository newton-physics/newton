# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ci" / "check_deprecation_allowlists.py"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CHECK_COMMAND = "uv run --no-project --with packaging scripts/ci/check_deprecation_allowlists.py"

spec = importlib.util.spec_from_file_location("check_deprecation_allowlists", SCRIPT)
assert spec is not None and spec.loader is not None
checker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checker
spec.loader.exec_module(checker)


class TestDeprecationAllowlists(unittest.TestCase):
    def _check(self, dependency: str, allowlist: str) -> str | None:
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

    def test_compares_declared_minimum(self):
        """Retire an exception when the inclusive minimum reaches its threshold."""
        cases = {
            "warp-lang>=1.17": False,
            "warp-lang>=1.17,<2": False,
            "warp-lang>=1.18": True,
            "warp-lang>=1.19": True,
        }
        allowlist = "# remove-when-minimum: warp-lang>=1.18\nwarning prefix\n"
        for dependency, is_obsolete in cases.items():
            with self.subTest(dependency=dependency):
                error = self._check(dependency, allowlist)
                self.assertEqual(error is not None, is_obsolete)

        error = self._check(
            "warp-lang>=1.18",
            allowlist + "# remove-when-minimum: warp-lang>=1.18\nsecond warning\n",
        )
        self.assertIn("warning prefix", error)
        self.assertNotIn("second warning", error)

    def test_rejects_invalid_allowlist_metadata(self):
        """Require one plain removal condition for each entry."""
        cases = {
            "missing": "warning prefix\n",
            "duplicate": (
                "# remove-when-minimum: warp-lang>=1.18\n# remove-when-minimum: warp-lang>=1.19\nwarning prefix\n"
            ),
            "orphaned": "# remove-when-minimum: warp-lang>=1.18\n",
            "unsupported": "# remove-when-minimum: warp-lang~=1.18\nwarning prefix\n",
        }
        for name, allowlist in cases.items():
            with self.subTest(name=name):
                self.assertIsNotNone(self._check("warp-lang>=1.17", allowlist))

    def test_rejects_unknown_project_minimum(self):
        """Reject a direct dependency whose minimum cannot be compared."""
        cases = {
            "other-package>=1": "is not a direct project dependency",
            "warp-lang~=1.17": "does not declare exactly one >= lower bound",
            "warp-lang>=1.17; python_version >= '3.12'": "uses an environment marker",
        }
        allowlist = "# remove-when-minimum: warp-lang>=1.18\nwarning prefix\n"
        for dependency, expected in cases.items():
            with self.subTest(dependency=dependency):
                self.assertIn(expected, self._check(dependency, allowlist))


class TestDeprecationAllowlistWorkflow(unittest.TestCase):
    def test_ci_runs_blocking_checker(self):
        """Run the deprecation allowlist checker as a blocking CPU CI step."""
        workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        marker = "      - name: Check deprecation allowlists\n"
        self.assertIn(marker, workflow)
        start = workflow.index(marker)
        end = workflow.find("\n      - name:", start + len(marker))
        block = workflow[start:] if end == -1 else workflow[start:end]
        self.assertIn(f"        run: {CHECK_COMMAND}\n", block)
        self.assertNotIn("continue-on-error", block)
