# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test Newton's Towncrier policy and release workflow."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from scripts import changelog_policy

TOWNCRIER_CONFIG = """\
[tool.towncrier]
directory = "changelog.d"
filename = "CHANGELOG.md"
start_string = "<!-- towncrier release notes start -->\\n"
title_format = "## [{version}] - {project_date}"
issue_format = "[#{issue}](https://github.com/newton-physics/newton/issues/{issue})"
issue_pattern = "\\\\d+"
header_prefix = "##"
wrap = false

[[tool.towncrier.type]]
directory = "added"
name = "Added"
showcontent = true

[[tool.towncrier.type]]
directory = "changed"
name = "Changed"
showcontent = true

[[tool.towncrier.type]]
directory = "deprecated"
name = "Deprecated"
showcontent = true

[[tool.towncrier.type]]
directory = "removed"
name = "Removed"
showcontent = true

[[tool.towncrier.type]]
directory = "fixed"
name = "Fixed"
showcontent = true
"""

INITIAL_CHANGELOG = """\
# Changelog

## [Unreleased]

<!-- towncrier release notes start -->

### Added

- Preserve this legacy unreleased entry.

## [1.0.0] - 2026-01-01
"""


class ChangelogPolicyTest(unittest.TestCase):
    def test_accepts_supported_fragment_shapes(self):
        """Accept issue, orphan, counter, multiline, and skip fragments."""
        with tempfile.TemporaryDirectory() as temp_directory:
            fragment_directory = Path(temp_directory)
            (fragment_directory / "README.md").write_text("Help.\n", encoding="utf-8")
            (fragment_directory / "3607.added.md").write_text(
                "Add camera rays with:\n\n  - Pinhole support.\n",
                encoding="utf-8",
            )
            (fragment_directory / "3607.added.1.md").write_text(
                "Add fisheye support.\n",
                encoding="utf-8",
            )
            (fragment_directory / "+camera-rays-a1b2c3d4.fixed.md").write_text(
                "Fix camera ray normalization.\n",
                encoding="utf-8",
            )
            (fragment_directory / "+internal-a1b2c3d4.skip").write_text(
                "Internal test-only change.\n",
                encoding="utf-8",
            )

            self.assertEqual(changelog_policy.validate_directory(fragment_directory), [])

    def test_rejects_invalid_fragment_content(self):
        """Reject invalid names, bullets, punctuation, and skip reasons."""
        with tempfile.TemporaryDirectory() as temp_directory:
            fragment_directory = Path(temp_directory)
            (fragment_directory / "feature.md").write_text("Add feature.\n", encoding="utf-8")
            (fragment_directory / "3607.added.md").write_text("- Add feature\n", encoding="utf-8")
            (fragment_directory / "3608.skip").write_text(
                "Internal change.\nSecond line.\n",
                encoding="utf-8",
            )

            errors = changelog_policy.validate_directory(fragment_directory)

        self.assertEqual(len(errors), 4)
        self.assertTrue(any("use ISSUE.CATEGORY.md" in error for error in errors))
        self.assertTrue(any("omit the leading bullet" in error for error in errors))
        self.assertTrue(any("end with a period" in error for error in errors))
        self.assertTrue(any("one-line reason" in error for error in errors))

    def test_requires_one_logical_fragment_for_main(self):
        """Require one identifier while allowing several categories and counters."""
        accepted = [
            changelog_policy.Change("A", PurePosixPath("changelog.d/3607.added.md")),
            changelog_policy.Change("A", PurePosixPath("changelog.d/3607.fixed.md")),
            changelog_policy.Change("A", PurePosixPath("changelog.d/3607.fixed.1.md")),
        ]
        rejected = [
            *accepted,
            changelog_policy.Change(
                "A",
                PurePosixPath("changelog.d/+another-a1b2c3d4.changed.md"),
            ),
        ]

        self.assertEqual(
            changelog_policy.validate_pr_changes(accepted, target_branch="main", labels=set()),
            [],
        )
        self.assertIn(
            "Pull requests to main must add exactly one logical fragment identifier",
            changelog_policy.validate_pr_changes(
                rejected,
                target_branch="main",
                labels=set(),
            ),
        )

    def test_allows_multiple_backport_fragment_identifiers(self):
        """Allow a release backport to carry user-facing and skip fragments."""
        changes = [
            changelog_policy.Change("A", PurePosixPath("changelog.d/3607.fixed.md")),
            changelog_policy.Change("A", PurePosixPath("changelog.d/3610.fixed.md")),
            changelog_policy.Change(
                "A",
                PurePosixPath("changelog.d/+support-a1b2c3d4.skip"),
            ),
        ]

        self.assertEqual(
            changelog_policy.validate_pr_changes(
                changes,
                target_branch="release-1.5",
                labels=set(),
            ),
            [],
        )

    def test_allows_one_time_policy_bootstrap(self):
        """Allow the workflow introduction to install the changelog marker."""
        changes = [
            changelog_policy.Change("M", PurePosixPath("CHANGELOG.md")),
            changelog_policy.Change(
                "A",
                PurePosixPath("scripts/changelog_policy.py"),
            ),
            changelog_policy.Change(
                "A",
                PurePosixPath("changelog.d/+towncrier-workflow-7d9e3a1c.skip"),
            ),
        ]

        self.assertEqual(
            changelog_policy.validate_pr_changes(
                changes,
                target_branch="main",
                labels=set(),
            ),
            [],
        )

    def test_enforces_release_management_scope(self):
        """Allow release management to update only the changelog and deletions."""
        accepted = [
            changelog_policy.Change("M", PurePosixPath("CHANGELOG.md")),
            changelog_policy.Change("D", PurePosixPath("changelog.d/3607.added.md")),
            changelog_policy.Change("D", PurePosixPath("changelog.d/3608.skip")),
        ]
        rejected = [
            *accepted,
            changelog_policy.Change("M", PurePosixPath("newton/__init__.py")),
        ]

        self.assertEqual(
            changelog_policy.validate_pr_changes(
                accepted,
                target_branch="release-1.5",
                labels={"release-management"},
            ),
            [],
        )
        self.assertTrue(
            changelog_policy.validate_pr_changes(
                rejected,
                target_branch="release-1.5",
                labels={"release-management"},
            )
        )


class TowncrierWorkflowTest(unittest.TestCase):
    def setUp(self):
        """Create an isolated Git repository with the Newton Towncrier config."""
        self.temp_directory = tempfile.TemporaryDirectory()
        self.repository = Path(self.temp_directory.name)
        (self.repository / "changelog.d").mkdir()
        (self.repository / "pyproject.toml").write_text(TOWNCRIER_CONFIG, encoding="utf-8")
        (self.repository / "CHANGELOG.md").write_text(INITIAL_CHANGELOG, encoding="utf-8")
        self._git("init", "-b", "main")
        self._git("config", "user.name", "Newton Test")
        self._git("config", "user.email", "newton@example.invalid")

    def tearDown(self):
        """Remove the isolated test repository."""
        self.temp_directory.cleanup()

    def _run(self, *command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=self.repository,
            check=True,
            capture_output=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1"},
        )

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return self._run("git", *arguments)

    def _towncrier(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        executable = shutil.which("towncrier")
        if executable is None:
            self.skipTest("Towncrier is not installed in the test environment")
        return self._run(executable, *arguments)

    def test_build_preserves_legacy_unreleased_entries(self):
        """Keep legacy unreleased entries beneath the first generated release."""
        (self.repository / "changelog.d" / "3607.added.md").write_text(
            "Add issue-linked fragments.\n",
            encoding="utf-8",
        )
        (self.repository / "changelog.d" / "3607.added.1.md").write_text(
            "Add multiple entries in one category.\n",
            encoding="utf-8",
        )
        (self.repository / "changelog.d" / "3607.deprecated.md").write_text(
            "Deprecate direct changelog edits in favor of fragments with:\n\n"
            "  - Multiple lines.\n"
            "  - Migration guidance.\n",
            encoding="utf-8",
        )
        (self.repository / "changelog.d" / "+orphan-a1b2c3d4.fixed.md").write_text(
            "Fix orphan rendering.\n",
            encoding="utf-8",
        )

        self._towncrier("build", "--yes", "--version", "1.5.0", "--date", "2026-08-01")

        changelog = (self.repository / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertLess(changelog.index("## [Unreleased]"), changelog.index("## [1.5.0]"))
        self.assertLess(changelog.index("## [1.5.0]"), changelog.index("Preserve this legacy"))
        self.assertIn(
            "Add issue-linked fragments. ([#3607](https://github.com/newton-physics/newton/issues/3607))",
            changelog,
        )
        self.assertIn("- Add multiple entries in one category.", changelog)
        self.assertIn("  - Multiple lines.", changelog)
        self.assertIn("  - Migration guidance.", changelog)
        self.assertIn(
            "([#3607](https://github.com/newton-physics/newton/issues/3607))",
            changelog,
        )
        self.assertIn("- Fix orphan rendering.", changelog)
        self.assertFalse((self.repository / "changelog.d" / "3607.added.md").exists())
        self.assertFalse((self.repository / "changelog.d" / "3607.added.1.md").exists())
        self.assertFalse((self.repository / "changelog.d" / "3607.deprecated.md").exists())
        self.assertFalse((self.repository / "changelog.d" / "+orphan-a1b2c3d4.fixed.md").exists())

    def test_cherry_pick_keeps_main_only_fragments(self):
        """Keep post-branch fragments when synchronizing a release build to main."""
        fragment_directory = self.repository / "changelog.d"
        (fragment_directory / "100.added.md").write_text(
            "Add branch-point feature.\n",
            encoding="utf-8",
        )
        self._git("add", ".")
        self._git("commit", "-m", "Add branch-point fragment")
        self._git("branch", "release-1.5")

        (fragment_directory / "101.changed.md").write_text(
            "Change main-only behavior.\n",
            encoding="utf-8",
        )
        self._git("add", ".")
        self._git("commit", "-m", "Add main-only fragment")
        (fragment_directory / "102.fixed.md").write_text(
            "Fix backported behavior.\n",
            encoding="utf-8",
        )
        self._git("add", ".")
        self._git("commit", "-m", "Add backport fragment")
        backport_commit = self._git("rev-parse", "HEAD").stdout.strip()

        self._git("switch", "release-1.5")
        self._git("cherry-pick", backport_commit)
        self._towncrier("build", "--yes", "--version", "1.5.0", "--date", "2026-08-01")
        self._git("add", "-A")
        self._git("commit", "-m", "Build release changelog")
        build_commit = self._git("rev-parse", "HEAD").stdout.strip()

        self._git("switch", "main")
        self._git("cherry-pick", build_commit)

        self.assertTrue((fragment_directory / "101.changed.md").exists())
        self.assertFalse((fragment_directory / "100.added.md").exists())
        self.assertFalse((fragment_directory / "102.fixed.md").exists())
        changelog = (self.repository / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn("Add branch-point feature.", changelog)
        self.assertIn("Fix backported behavior.", changelog)
        self.assertNotIn("Change main-only behavior.", changelog)


if __name__ == "__main__":
    unittest.main()
