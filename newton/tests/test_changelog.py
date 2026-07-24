# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

_TOOL_PATH = Path(__file__).parents[2] / "scripts" / "changelog.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("newton_changelog_tool", _TOOL_PATH)
if _TOOL_SPEC is None or _TOOL_SPEC.loader is None:
    raise RuntimeError(f"Could not load changelog tool from {_TOOL_PATH}")
changelog = importlib.util.module_from_spec(_TOOL_SPEC)
sys.modules[_TOOL_SPEC.name] = changelog
_TOOL_SPEC.loader.exec_module(changelog)


BASE_CHANGELOG = """# Changelog

## [Unreleased]

<!-- changelog releases start -->

## [1.0.0] - 2026-01-01

### Added

- Add the initial release.
"""


def changelog_with_unreleased(category: str, entry: str) -> str:
    """Return the base changelog with one accumulated entry."""
    body = f"### {category}\n\n- {entry}\n\n"
    return BASE_CHANGELOG.replace(changelog.RELEASES_MARKER, f"{body}{changelog.RELEASES_MARKER}")


class TemporaryRepository:
    """Provide a small Git repository for changelog integration tests."""

    def __init__(self, root: Path):
        self.root = root
        self.fragments = root / changelog.FRAGMENTS_DIRECTORY

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.root,
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
        return result.stdout.strip()

    def write(self, relative_path: str, text: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        return path

    def fragment(self, name: str, category: str, entry: str) -> Path:
        return self.write(f"changelog.d/{name}.md", f"### {category}\n\n- {entry}\n")

    def skip(self, name: str, reason: str = "No user-facing change: test-only update.") -> Path:
        return self.write(f"changelog.d/{name}.skip", f"{reason}\n")

    def commit(self, message: str) -> str:
        self.git("add", "-A")
        self.git("commit", "-m", message)
        return self.git("rev-parse", "HEAD")


@contextmanager
def temporary_repository():
    """Create an initialized temporary repository."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        repo = TemporaryRepository(Path(temporary_directory))
        repo.git("init", "--initial-branch=main")
        repo.git("config", "user.name", "Newton Tests")
        repo.git("config", "user.email", "newton-tests@example.com")
        repo.git("config", "commit.gpgsign", "false")
        repo.git("config", "core.autocrlf", "false")
        repo.write(changelog.CHANGELOG_FILENAME, BASE_CHANGELOG)
        repo.write("changelog.d/README.md", "Test fragment directory.\n")
        repo.commit("Initialize repository")
        yield repo


class TestChangelogFragments(unittest.TestCase):
    def test_parse_fragment_with_multiple_categories(self):
        """Parse multiple categories, entries, and continuation lines."""
        sections = changelog.parse_fragment_text(
            """### Added

- Add a first feature.
- Add a second feature with more context
  on an indented continuation line.

### Deprecated

- Deprecate the first feature in favor of its replacement.
- Deprecate the second feature in favor of its replacement.
  Preserve this explanatory continuation.
- Deprecate the third feature in favor of its replacement.
"""
        )

        self.assertEqual(
            sections,
            {
                "Added": (
                    changelog.ChangelogEntry("- Add a first feature."),
                    changelog.ChangelogEntry(
                        "- Add a second feature with more context\n  on an indented continuation line."
                    ),
                ),
                "Deprecated": (
                    changelog.ChangelogEntry("- Deprecate the first feature in favor of its replacement."),
                    changelog.ChangelogEntry(
                        "- Deprecate the second feature in favor of its replacement.\n"
                        "  Preserve this explanatory continuation."
                    ),
                    changelog.ChangelogEntry("- Deprecate the third feature in favor of its replacement."),
                ),
            },
        )

    def test_parse_fragment_provenance_is_tool_owned(self):
        """Read generated provenance while rejecting it in contributor fragments."""
        text = """### Added

<!-- changelog-fragment: feature-change-0123abcd -->
- Add a feature.
"""
        sections = changelog.parse_fragment_text(text, allow_fragment_markers=True)

        self.assertEqual(sections["Added"][0].fragment_ids, ("feature-change-0123abcd",))
        with self.assertRaisesRegex(changelog.ChangelogError, "generated by the tool"):
            changelog.parse_fragment_text(text)

    def test_parse_fragment_rejects_invalid_structures(self):
        """Reject fragment structures that cannot compile predictably."""
        invalid_fragments = {
            "empty": "",
            "unknown heading": "### Security\n\n- Fix a vulnerability.\n",
            "duplicate heading": "### Added\n\n- Add one.\n\n### Added\n\n- Add two.\n",
            "out of order": "### Fixed\n\n- Fix one.\n\n### Added\n\n- Add two.\n",
            "empty heading": "### Added\n",
            "orphan text": "Add a feature.\n",
            "unindented continuation": "### Added\n\n- Add a feature.\nMore detail.\n",
            "empty bullet": "### Added\n\n- \n",
        }

        for name, text in invalid_fragments.items():
            with self.subTest(name=name), self.assertRaises(changelog.ChangelogError):
                changelog.parse_fragment_text(text)

    def test_create_fragment_generates_readable_unique_names(self):
        """Generate readable fragment and skip filenames before PR creation."""
        with temporary_repository() as repo:
            with mock.patch.object(changelog.secrets, "token_hex", side_effect=["0123abcd", "89abcdef"]):
                fragment = changelog.create_fragment(
                    repo.root,
                    name="Fix USD / Mesh Mass",
                    category="fixed",
                    content="Fix explicit mesh mass.",
                    skip_reason=None,
                )
                skip = changelog.create_fragment(
                    repo.root,
                    name="Internal cleanup",
                    category=None,
                    content=None,
                    skip_reason="No user-facing change: reorganize tests.",
                )

            self.assertEqual(fragment.name, "fix-usd-mesh-mass-0123abcd.md")
            self.assertEqual(fragment.read_text(encoding="utf-8"), "### Fixed\n\n- Fix explicit mesh mass.\n")
            self.assertEqual(skip.name, "internal-cleanup-89abcdef.skip")
            self.assertEqual(skip.read_text(encoding="utf-8"), "No user-facing change: reorganize tests.\n")

    def test_validate_rejects_paired_fragment_and_skip(self):
        """Reject contradictory release-note and skip files for one ID."""
        with temporary_repository() as repo:
            repo.fragment("paired-change-0123abcd", "Added", "Add a feature.")
            repo.skip("paired-change-0123abcd")

            with self.assertRaisesRegex(changelog.ChangelogError, "cannot combine"):
                changelog.validate_repository(repo.root)

    def test_validate_requires_unreleased_and_release_marker(self):
        """Require stable accumulator and release boundary lines."""
        invalid_values = {
            "missing Unreleased": BASE_CHANGELOG.replace(f"{changelog.UNRELEASED_HEADING}\n\n", ""),
            "non-standalone marker": BASE_CHANGELOG.replace(
                changelog.RELEASES_MARKER,
                f"prefix {changelog.RELEASES_MARKER}",
            ),
        }
        for name, text in invalid_values.items():
            with self.subTest(name=name), temporary_repository() as repo:
                repo.write(changelog.CHANGELOG_FILENAME, text)
                with self.assertRaises(changelog.ChangelogError):
                    changelog.validate_repository(repo.root)

    def test_check_requires_exactly_one_main_fragment(self):
        """Require one fragment or skip file for an ordinary main PR."""
        with temporary_repository() as repo:
            base = repo.git("rev-parse", "HEAD")
            repo.write("implementation.txt", "change\n")
            repo.commit("Change implementation")

            with self.assertRaisesRegex(changelog.ChangelogError, "exactly one"):
                changelog.check_pull_request(repo.root, base_ref=base, target_branch="main")

        with temporary_repository() as repo:
            base = repo.git("rev-parse", "HEAD")
            repo.fragment("first-change-0123abcd", "Added", "Add one feature.")
            repo.fragment("second-change-89abcdef", "Fixed", "Fix another feature.")
            repo.commit("Add two fragments")

            with self.assertRaisesRegex(changelog.ChangelogError, "exactly one"):
                changelog.check_pull_request(repo.root, base_ref=base, target_branch="main")

        with temporary_repository() as repo:
            base = repo.git("rev-parse", "HEAD")
            expected = repo.fragment("one-change-0123abcd", "Changed", "Change one behavior; use the new option.")
            repo.commit("Add one fragment")

            added = changelog.check_pull_request(repo.root, base_ref=base, target_branch="main")
            self.assertEqual(added, [expected])

    def test_check_accepts_skip_and_rejects_existing_fragment_edits(self):
        """Accept an explicit skip while keeping merged fragments immutable."""
        with temporary_repository() as repo:
            base = repo.git("rev-parse", "HEAD")
            expected = repo.skip("test-cleanup-0123abcd")
            repo.commit("Refactor tests")

            added = changelog.check_pull_request(repo.root, base_ref=base, target_branch="main")
            self.assertEqual(added, [expected])

        with temporary_repository() as repo:
            existing = repo.fragment("existing-change-0123abcd", "Added", "Add existing behavior.")
            base = repo.commit("Add pending fragment")
            existing.write_text("### Added\n\n- Add edited behavior.\n", encoding="utf-8")
            repo.commit("Edit pending fragment")

            with self.assertRaisesRegex(changelog.ChangelogError, "immutable"):
                changelog.check_pull_request(repo.root, base_ref=base, target_branch="main")

    def test_check_rejects_direct_edits_but_accepts_transition_marker(self):
        """Reject direct edits while allowing the one-time workflow marker."""
        with temporary_repository() as repo:
            base = repo.git("rev-parse", "HEAD")
            repo.write(changelog.CHANGELOG_FILENAME, BASE_CHANGELOG + "\nUnexpected edit.\n")
            repo.skip("changelog-edit-0123abcd")
            repo.commit("Edit changelog directly")

            with self.assertRaisesRegex(changelog.ChangelogError, "Do not edit"):
                changelog.check_pull_request(repo.root, base_ref=base, target_branch="main")

        with temporary_repository() as repo:
            legacy = BASE_CHANGELOG.replace(f"{changelog.RELEASES_MARKER}\n\n", "")
            repo.write(changelog.CHANGELOG_FILENAME, legacy)
            base = repo.commit("Restore legacy layout")
            repo.write(changelog.CHANGELOG_FILENAME, BASE_CHANGELOG)
            expected = repo.skip("workflow-transition-0123abcd")
            repo.commit("Add changelog workflow marker")

            added = changelog.check_pull_request(repo.root, base_ref=base, target_branch="main")
            self.assertEqual(added, [expected])

    def test_check_limits_authorized_maintenance_scope(self):
        """Allow only consolidation output and consumed fragments in maintenance."""
        with temporary_repository() as repo:
            repo.fragment("release-entry-0123abcd", "Added", "Add release behavior.")
            base = repo.commit("Add pending release entry")
            changelog.consolidate_changelog(repo.root, dry_run=False)
            repo.commit("Consolidate changelog")

            added = changelog.check_pull_request(
                repo.root,
                base_ref=base,
                target_branch="main",
                allow_changelog_update=True,
            )
            self.assertEqual(added, [])

        with temporary_repository() as repo:
            repo.fragment("release-entry-0123abcd", "Added", "Add release behavior.")
            base = repo.commit("Add pending release entry")
            changelog.consolidate_changelog(repo.root, dry_run=False)
            repo.write("implementation.txt", "unexpected change\n")
            repo.commit("Mix implementation into maintenance")

            with self.assertRaisesRegex(changelog.ChangelogError, "must contain only"):
                changelog.check_pull_request(
                    repo.root,
                    base_ref=base,
                    target_branch="main",
                    allow_changelog_update=True,
                )

        with temporary_repository() as repo:
            repo.skip("internal-only-0123abcd")
            base = repo.commit("Add pending skip")
            changelog.consolidate_changelog(repo.root, dry_run=False)
            repo.commit("Consume pending skip")

            added = changelog.check_pull_request(
                repo.root,
                base_ref=base,
                target_branch="main",
                allow_changelog_update=True,
            )
            self.assertEqual(added, [])

        with temporary_repository() as repo:
            repo.write(changelog.CHANGELOG_FILENAME, changelog_with_unreleased("Added", "Add release behavior."))
            base = repo.commit("Accumulate release behavior")
            changelog.release_changelog(
                repo.root,
                version="1.1.0",
                release_date="2026-02-03",
                dry_run=False,
            )
            repo.commit("Promote changelog release")

            added = changelog.check_pull_request(
                repo.root,
                base_ref=base,
                target_branch="release-1.1",
                allow_changelog_update=True,
            )
            self.assertEqual(added, [])

    def test_build_dry_run_merges_without_changing_files(self):
        """Preview the complete Unreleased accumulator without consuming files."""
        with temporary_repository() as repo:
            repo.write(changelog.CHANGELOG_FILENAME, changelog_with_unreleased("Added", "Add existing behavior."))
            fragment = repo.write(
                "changelog.d/combined-change-0123abcd.md",
                "### Added\n\n- Add existing behavior.\n\n### Fixed\n\n- Fix preview behavior.\n",
            )
            before = (repo.root / changelog.CHANGELOG_FILENAME).read_text(encoding="utf-8")

            preview = changelog.consolidate_changelog(repo.root, dry_run=True)

            self.assertEqual(preview.count("- Add existing behavior."), 1)
            self.assertIn("<!-- changelog-fragment: combined-change-0123abcd -->", preview)
            self.assertIn("- Fix preview behavior.", preview)
            self.assertEqual((repo.root / changelog.CHANGELOG_FILENAME).read_text(encoding="utf-8"), before)
            self.assertTrue(fragment.exists())

    def test_build_repeatedly_accumulates_without_duplicates(self):
        """Consolidate repeatedly while preserving history and prior entries."""
        with temporary_repository() as repo:
            original_history = BASE_CHANGELOG[BASE_CHANGELOG.index(changelog.RELEASES_MARKER) :]
            repo.fragment("feature-a-00000001", "Added", "Add behavior A.")
            changelog.consolidate_changelog(repo.root, dry_run=False)
            repo.fragment("feature-b-00000002", "Fixed", "Fix behavior B.")
            changelog.consolidate_changelog(repo.root, dry_run=False)

            result = (repo.root / changelog.CHANGELOG_FILENAME).read_text(encoding="utf-8")
            self.assertEqual(result.count("- Add behavior A."), 1)
            self.assertEqual(result.count("- Fix behavior B."), 1)
            self.assertIn("<!-- changelog-fragment: feature-a-00000001 -->", result)
            self.assertIn("<!-- changelog-fragment: feature-b-00000002 -->", result)
            self.assertEqual(result[result.index(changelog.RELEASES_MARKER) :], original_history)
            self.assertEqual([path.name for path in repo.fragments.iterdir()], ["README.md"])

    def test_release_promotes_unreleased_and_pending_fragments(self):
        """Promote the complete accumulator while retaining released history."""
        with temporary_repository() as repo:
            repo.write(changelog.CHANGELOG_FILENAME, changelog_with_unreleased("Added", "Add accumulated behavior."))
            fragment = repo.fragment("pending-fix-0123abcd", "Fixed", "Fix pending behavior.")
            repo.skip("internal-change-89abcdef")
            before = (repo.root / changelog.CHANGELOG_FILENAME).read_text(encoding="utf-8")

            preview = changelog.release_changelog(
                repo.root,
                version="1.1.0",
                release_date="2026-02-03",
                dry_run=True,
            )
            self.assertIn("- Add accumulated behavior.", preview)
            self.assertIn("- Fix pending behavior.", preview)
            self.assertNotIn("<!-- changelog-fragment:", preview)
            self.assertEqual((repo.root / changelog.CHANGELOG_FILENAME).read_text(encoding="utf-8"), before)
            self.assertTrue(fragment.exists())

            changelog.release_changelog(
                repo.root,
                version="1.1.0",
                release_date="2026-02-03",
                dry_run=False,
            )
            result = (repo.root / changelog.CHANGELOG_FILENAME).read_text(encoding="utf-8")
            layout = changelog.parse_changelog_layout(result)
            self.assertEqual(layout.unreleased, {})
            self.assertIn("## [1.1.0] - 2026-02-03", result)
            self.assertIn("- Add accumulated behavior.", result)
            self.assertIn("- Fix pending behavior.", result)
            self.assertNotIn("<!-- changelog-fragment:", result)
            self.assertIn("## [1.0.0] - 2026-01-01", result)
            self.assertEqual([path.name for path in repo.fragments.iterdir()], ["README.md"])

            repo.fragment("later-change-fedcba98", "Fixed", "Fix later behavior.")
            with self.assertRaisesRegex(changelog.ChangelogError, "already contains"):
                changelog.release_changelog(
                    repo.root,
                    version="1.1.0",
                    release_date="2026-02-04",
                    dry_run=False,
                )

    def test_build_requires_pending_fragments(self):
        """Preview accumulated entries but reject a mutating no-op build."""
        with temporary_repository() as repo:
            preview = changelog.consolidate_changelog(repo.root, dry_run=True)
            self.assertEqual(preview, f"{changelog.UNRELEASED_HEADING}\n")
            with self.assertRaisesRegex(changelog.ChangelogError, "No pending"):
                changelog.consolidate_changelog(repo.root, dry_run=False)

    def test_release_branch_reconciliation_preserves_next_release(self):
        """Reconcile a release after arbitrary consolidation on both branches."""
        with temporary_repository() as repo:
            legacy_body = "### Added\n\n- Add shipped legacy behavior.\n- Add deferred legacy behavior.\n\n"
            repo.write(
                changelog.CHANGELOG_FILENAME,
                BASE_CHANGELOG.replace(changelog.RELEASES_MARKER, f"{legacy_body}{changelog.RELEASES_MARKER}"),
            )
            repo.commit("Accumulate legacy entries")
            repo.fragment("before-cut-a-00000001", "Added", "Add behavior A.")
            changelog.consolidate_changelog(repo.root, dry_run=False)
            repo.commit("Consolidate behavior A")
            repo.fragment("before-cut-b-00000002", "Fixed", "Fix behavior B.")
            repo.commit("Add behavior B (#102)")
            repo.git("branch", "release-1.1")

            repo.fragment("next-release-c-00000003", "Added", "Add behavior C.")
            changelog.consolidate_changelog(repo.root, dry_run=False)
            repo.commit("Consolidate behavior B and C")
            repo.fragment("backported-d-00000004", "Fixed", "Fix behavior D.")
            backport_commit = repo.commit("Fix behavior D (#104)")

            repo.git("switch", "release-1.1")
            release_base = repo.git("rev-parse", "HEAD")
            repo.git("cherry-pick", "-x", backport_commit)
            added = changelog.check_pull_request(
                repo.root,
                base_ref=release_base,
                target_branch="release-1.1",
            )
            self.assertEqual([path.name for path in added], ["backported-d-00000004.md"])

            changelog.consolidate_changelog(repo.root, dry_run=False)
            repo.commit("Consolidate release branch fragments")
            release_path = repo.root / changelog.CHANGELOG_FILENAME
            release_text = release_path.read_text(encoding="utf-8")
            release_layout = changelog.parse_changelog_layout(release_text)
            release_entries = {
                category: tuple(entry for entry in entries if "deferred legacy" not in entry.text)
                for category, entries in release_layout.unreleased.items()
            }
            repo.write(changelog.CHANGELOG_FILENAME, changelog._replace_unreleased(release_text, release_entries))
            changelog.release_changelog(
                repo.root,
                version="1.1.0",
                release_date="2026-08-20",
                dry_run=False,
            )
            release_commit = repo.commit("Release changelog 1.1.0")

            repo.git("switch", "main")
            before = (repo.root / changelog.CHANGELOG_FILENAME).read_text(encoding="utf-8")
            preview = changelog.reconcile_changelog(
                repo.root,
                source_ref=release_commit,
                version="1.1.0",
                dry_run=True,
            )
            self.assertIn("Add behavior C", preview)
            self.assertEqual((repo.root / changelog.CHANGELOG_FILENAME).read_text(encoding="utf-8"), before)

            changelog.reconcile_changelog(
                repo.root,
                source_ref=release_commit,
                version="1.1.0",
                dry_run=False,
            )
            main_text = (repo.root / changelog.CHANGELOG_FILENAME).read_text(encoding="utf-8")
            layout = changelog.parse_changelog_layout(main_text)
            pending_text = "\n".join(entry.text for entries in layout.unreleased.values() for entry in entries)
            self.assertIn("Add behavior C", pending_text)
            self.assertIn("Add deferred legacy behavior", pending_text)
            self.assertNotIn("Add shipped legacy behavior", pending_text)
            self.assertNotIn("Add behavior A", pending_text)
            self.assertNotIn("Fix behavior B", pending_text)
            self.assertNotIn("Fix behavior D", pending_text)
            release = changelog._extract_release(main_text, "1.1.0", source="main changelog")
            release_text = release.text
            self.assertNotIn("<!-- changelog-fragment:", release_text)
            self.assertIn("Add behavior A", release_text)
            self.assertIn("Fix behavior B", release_text)
            self.assertIn("Fix behavior D", release_text)
            self.assertIn("Add shipped legacy behavior", release_text)
            self.assertNotIn("Add deferred legacy behavior", release_text)
            self.assertNotIn("Add behavior C", release_text)
            self.assertIn("## [1.0.0] - 2026-01-01", main_text)
            self.assertEqual([path.name for path in repo.fragments.iterdir()], ["README.md"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
