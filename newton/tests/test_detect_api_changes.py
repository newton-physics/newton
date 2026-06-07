# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "detect_api_changes.py"

spec = importlib.util.spec_from_file_location("detect_api_changes", SCRIPT)
assert spec is not None and spec.loader is not None
detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector)


class TestDetectApiChanges(unittest.TestCase):
    def test_main_reports_skipped_modules_without_false_removals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            head = root / "head"
            (base / "newton").mkdir(parents=True)
            (head / "newton").mkdir(parents=True)

            (base / "newton" / "__init__.py").write_text("from .bad import public\n", encoding="utf-8")
            (head / "newton" / "__init__.py").write_text("from .bad import public\n", encoding="utf-8")
            (base / "newton" / "bad.py").write_text(
                "def public() -> int:\n    return 1\n",
                encoding="utf-8",
            )
            (head / "newton" / "bad.py").write_text("def public(:\n    return 1\n", encoding="utf-8")

            output = io.StringIO()
            warnings = io.StringIO()
            with (
                mock.patch.object(sys, "argv", ["detect_api_changes.py", str(base), str(head)]),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(warnings),
            ):
                self.assertEqual(detector.main(), 0)

            report = json.loads(output.getvalue())
            self.assertIn("has_analysis_warnings", report)
            self.assertIn("skipped_modules", report)
            self.assertTrue(report["has_analysis_warnings"])
            self.assertEqual(report["skipped_modules"]["base"], [])
            self.assertEqual(report["skipped_modules"]["head"][0]["module"], "newton.bad")
            self.assertIn("SyntaxError", report["skipped_modules"]["head"][0]["reason"])
            self.assertEqual(report["diff"]["summary"]["removed_count"], 0)
            self.assertFalse(report["diff"]["has_changes"])
            self.assertIn("Analysis warnings", report["comment"])
            self.assertIn("WARNING: skipping newton/bad.py", warnings.getvalue())

    def test_main_reports_decode_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            head = root / "head"
            (base / "newton").mkdir(parents=True)
            (head / "newton").mkdir(parents=True)

            (base / "newton" / "__init__.py").write_text("", encoding="utf-8")
            (head / "newton" / "__init__.py").write_text("", encoding="utf-8")
            (base / "newton" / "bad.py").write_text("", encoding="utf-8")
            (head / "newton" / "bad.py").write_bytes(b"\xff")

            output = io.StringIO()
            warnings = io.StringIO()
            with (
                mock.patch.object(sys, "argv", ["detect_api_changes.py", str(base), str(head)]),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(warnings),
            ):
                self.assertEqual(detector.main(), 0)

            report = json.loads(output.getvalue())
            self.assertTrue(report["has_analysis_warnings"])
            self.assertEqual(report["skipped_modules"]["head"][0]["module"], "newton.bad")
            self.assertIn("UnicodeDecodeError", report["skipped_modules"]["head"][0]["reason"])
            self.assertIn("WARNING: skipping newton/bad.py", warnings.getvalue())

    def test_format_comment_escapes_markdown_control_characters(self):
        comment = detector.format_comment(
            {
                "has_changes": True,
                "summary": {
                    "added_count": 1,
                    "removed_count": 0,
                    "changed_count": 0,
                    "total": 1,
                },
                "added": [
                    {
                        "path": "newton.public.MALICIOUS",
                        "kind": "constant",
                        "signature": "MALICIOUS: </details><img src=x>",
                        "value": "'closing </details>\\n### injected\\n`tick`'",
                    }
                ],
                "removed": [],
                "changed": [],
            }
        )

        self.assertNotIn("</details>", comment)
        self.assertNotIn("<img", comment)
        self.assertNotIn("\n### injected", comment)
        self.assertIn("&lt;/details&gt;", comment)
        self.assertIn("\\n### injected", comment)
        self.assertIn("`tick`", comment)


if __name__ == "__main__":
    unittest.main(verbosity=2)
