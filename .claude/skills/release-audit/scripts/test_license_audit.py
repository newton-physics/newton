#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).with_name("license_audit.py")
SPEC = importlib.util.spec_from_file_location("license_audit", MODULE_PATH)
license_audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = license_audit
SPEC.loader.exec_module(license_audit)


class _UrlopenResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class LicenseAuditTest(unittest.TestCase):
    def test_requirement_name_handles_extras_markers_and_versions(self):
        cases = {
            "warp-lang>=1.13.0,<1.14": "warp-lang",
            "newton[viewer] ; python_version >= '3.11'": "newton",
            "  mujoco-warp ~= 3.8": "mujoco-warp",
            "package_name[extra]>=1; sys_platform == 'linux'": "package_name",
        }
        for spec, expected in cases.items():
            with self.subTest(spec=spec):
                self.assertEqual(license_audit._requirement_name(spec), expected)

    def test_parse_lock_combines_marker_and_resolution_markers(self):
        lock_text = """
[[package]]
name = "example"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
marker = "sys_platform == 'linux'"
resolution-markers = ["python_full_version >= '3.12'"]

[[package]]
name = "forked"
version = "0.1"
source = { git = "https://example.com/fork.git" }
"""
        packages = license_audit._parse_lock(lock_text)

        self.assertEqual(packages[0].markers, ("sys_platform == 'linux'", "python_full_version >= '3.12'"))
        self.assertEqual(packages[0].registry, "https://pypi.org/simple")
        self.assertEqual(packages[1].registry, None)
        self.assertEqual(packages[1].source, "git: https://example.com/fork.git")

    def test_locked_license_does_not_query_pypi_for_non_registry_sources(self):
        package = license_audit.LockedPackage(
            name="forked",
            normalized_name="forked",
            version="0.1",
            registry=None,
            source="git: https://example.com/fork.git",
            markers=(),
        )

        with mock.patch.object(license_audit, "_fetch_pypi_license") as fetch:
            metadata = license_audit._locked_license(package, False, 1.0, {})

        fetch.assert_not_called()
        self.assertEqual(metadata["license"], "not checked (non-PyPI source: git: https://example.com/fork.git)")

    def test_license_review_boundaries_and_skip_pypi(self):
        self.assertFalse(license_audit._metadata_needs_review({"license": "not checked (--skip-pypi)"}))
        self.assertTrue(license_audit._metadata_needs_review({"license": "not checked (URLError)"}))
        self.assertTrue(license_audit._license_needs_review("LGPL-2.1-or-later"))
        self.assertTrue(license_audit._license_needs_review("NVIDIA Proprietary Software"))
        self.assertFalse(license_audit._license_needs_review("BSD-3-Clause"))
        self.assertFalse(license_audit._license_needs_review("descriptive text with agplish as a substring"))

    def test_fetch_pypi_license_prefers_modern_json_fields(self):
        payload = {"info": {"license_expression": "MIT", "license": "", "classifiers": []}}
        with mock.patch.object(license_audit.urllib.request, "urlopen", return_value=_UrlopenResponse(payload)):
            metadata = license_audit._fetch_pypi_license("example", "1.0", 1.0)
        self.assertEqual(metadata["license"], "MIT")

        payload = {
            "info": {
                "license_expression": None,
                "license": "",
                "license_files": ["LICENSE"],
                "classifiers": [],
            }
        }
        with mock.patch.object(license_audit.urllib.request, "urlopen", return_value=_UrlopenResponse(payload)):
            metadata = license_audit._fetch_pypi_license("example", "1.0", 1.0)
        self.assertEqual(metadata["license"], "not declared (license files: LICENSE)")

    def test_skip_pypi_build_audit_defers_review_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            self._git(repo, "init")
            self._git(repo, "config", "user.email", "test@example.com")
            self._git(repo, "config", "user.name", "Test User")
            self._write_release_files(repo, "foo==1.0", "foo", "1.0")
            self._git(repo, "add", "pyproject.toml", "uv.lock", "LICENSE.md")
            self._git(repo, "commit", "-m", "base")
            self._git(repo, "tag", "base")
            self._write_release_files(repo, "bar==1.0", "bar", "1.0")
            self._git(repo, "add", "pyproject.toml", "uv.lock")
            self._git(repo, "commit", "-m", "head")
            self._git(repo, "tag", "head")

            audit = license_audit.build_audit(repo, "base", "head", True, 1.0)

        self.assertIn("- License metadata needing review: not evaluated (--skip-pypi)", audit)
        self.assertNotIn("- License metadata needing review: bar", audit)

    def test_license_file_glob_detects_direct_child_notice_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            self._git(repo, "init")
            self._git(repo, "config", "user.email", "test@example.com")
            self._git(repo, "config", "user.name", "Test User")
            self._write_release_files(repo, "foo==1.0", "foo", "1.0")
            self._git(repo, "add", "pyproject.toml", "uv.lock", "LICENSE.md")
            self._git(repo, "commit", "-m", "base")
            self._git(repo, "tag", "base")

            notice_path = repo / "newton" / "licenses" / "NOTICE.txt"
            notice_path.parent.mkdir(parents=True)
            notice_path.write_text("notice\n", encoding="utf-8")
            self._git(repo, "add", "newton/licenses/NOTICE.txt")
            self._git(repo, "commit", "-m", "add notice")
            self._git(repo, "tag", "head")

            audit = license_audit.build_audit(repo, "base", "head", True, 1.0)

        self.assertIn("- In-tree license notice file changes: 1", audit)
        self.assertIn("| A | newton/licenses/NOTICE.txt |", audit)

    def _write_release_files(self, repo: Path, requirement: str, package: str, version: str) -> None:
        (repo / "pyproject.toml").write_text(
            "\n".join(
                [
                    "[project]",
                    'name = "newton"',
                    'license = "Apache-2.0"',
                    'license-files = ["LICENSE.md", "newton/licenses/**/*.txt"]',
                    f'dependencies = ["{requirement}"]',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (repo / "LICENSE.md").write_text("license\n", encoding="utf-8")
        (repo / "uv.lock").write_text(
            "\n".join(
                [
                    "[[package]]",
                    f'name = "{package}"',
                    f'version = "{version}"',
                    'source = { registry = "https://pypi.org/simple" }',
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _git(self, repo: Path, *args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
