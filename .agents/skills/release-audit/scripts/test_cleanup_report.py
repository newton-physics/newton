# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test safe release-audit report cleanup."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cleanup_report import cleanup_report


class CleanupReportTest(unittest.TestCase):
    def test_removes_allowed_report(self):
        """Remove an allowed report from the temporary root."""
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            report = temporary_root / "newton-1.5.0-prerelease-report.md"
            report.write_text("report\n", encoding="utf-8")

            cleanup_report(report, temporary_directory=temporary_root)

            self.assertFalse(report.exists())

    def test_rejects_unexpected_filename(self):
        """Reject a file that is not a Newton release-audit report."""
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            report = temporary_root / "unrelated.md"
            report.write_text("keep\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not an allowed"):
                cleanup_report(report, temporary_directory=temporary_root)

            self.assertTrue(report.exists())

    def test_rejects_nested_path(self):
        """Reject an allowed filename outside the temporary root."""
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            nested = temporary_root / "nested"
            nested.mkdir()
            report = nested / "newton-1.5.0-rc-report.md"
            report.write_text("keep\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "directly beneath"):
                cleanup_report(report, temporary_directory=temporary_root)

            self.assertTrue(report.exists())

    def test_rejects_symlink(self):
        """Reject a report symlink without deleting its target."""
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            target = temporary_root / "newton-1.5.0-rc-report.md"
            target.write_text("keep\n", encoding="utf-8")
            report = temporary_root / "newton-1.5.0-prerelease-report.md"
            try:
                report.symlink_to(target)
            except OSError as error:
                self.skipTest(f"symlinks are unavailable: {error}")

            with self.assertRaisesRegex(ValueError, "symlink"):
                cleanup_report(report, temporary_directory=temporary_root)

            self.assertTrue(report.is_symlink())
            self.assertEqual(target.read_text(encoding="utf-8"), "keep\n")


if __name__ == "__main__":
    unittest.main()
