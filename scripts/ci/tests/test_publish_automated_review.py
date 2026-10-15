# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ci" / "publish_automated_review.py"

spec = importlib.util.spec_from_file_location("publish_automated_review", SCRIPT)
assert spec is not None and spec.loader is not None
publish = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = publish
spec.loader.exec_module(publish)

DIFF = """\
diff --git a/newton/a.py b/newton/a.py
index 1111111..2222222 100644
--- a/newton/a.py
+++ b/newton/a.py
@@ -10,4 +10,5 @@ def f():
     x = 1
-    y = 2
+    y = 3
+    z = 4
     return x
diff --git a/newton/old.py b/newton/old.py
deleted file mode 100644
--- a/newton/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-a = 1
-b = 2
diff --git a/newton/new.py b/newton/new.py
new file mode 100644
--- /dev/null
+++ b/newton/new.py
@@ -0,0 +1,3 @@
+--- not a header
+++ not a header either
+value = 1
"""


class TestPublishAutomatedReview(unittest.TestCase):
    def test_commentable_lines_cover_added_and_context_lines(self):
        """Map head-side lines in each hunk, skipping deleted files and header-like content."""
        lines = publish.commentable_lines(DIFF)
        self.assertEqual(lines["newton/a.py"], {10, 11, 12, 13})
        self.assertEqual(lines["newton/new.py"], {1, 2, 3})
        self.assertNotIn("++ not a header either", lines)
        self.assertNotIn("newton/old.py", lines)

    def test_plan_comments_anchors_valid_lines_and_ranges(self):
        """Post findings on diff lines inline, with multi-line ranges only when every line is in the diff."""
        findings = [
            {"path": "newton/a.py", "line": 12, "start_line": 11, "severity": "medium", "title": "Range", "body": "b"},
            {"path": "newton/a.py", "line": 12, "start_line": 5, "severity": "low", "title": "Bad range", "body": "b"},
            {"path": "newton/a.py", "line": 40, "severity": "high", "title": "Outside", "body": "b"},
            {"path": "changelog", "line": None, "severity": "medium", "title": "No fragment", "body": "b"},
        ]
        inline, elsewhere, skipped = publish.plan_comments(findings, publish.commentable_lines(DIFF), set())
        self.assertEqual(skipped, 0)
        self.assertEqual([c["line"] for c in inline], [12, 12])
        self.assertEqual(inline[0]["start_line"], 11)
        self.assertNotIn("start_line", inline[1])
        self.assertEqual([f["title"] for f in elsewhere], ["Outside", "No fragment"])

    def test_plan_comments_skips_findings_posted_earlier(self):
        """Skip a finding whose file and normalized title were already posted."""
        finding = {"path": "newton/a.py", "line": 11, "severity": "low", "title": "Same  Title", "body": "b"}
        posted = {publish.fingerprint({"path": "newton/a.py", "title": "same title"})}
        inline, elsewhere, skipped = publish.plan_comments([finding], publish.commentable_lines(DIFF), posted)
        self.assertEqual((inline, elsewhere, skipped), ([], [], 1))

    def test_plan_comments_caps_inline_comments(self):
        """Move findings beyond the inline limit to the summary."""
        findings = [
            {"path": "newton/a.py", "line": 11, "severity": "nit", "title": f"t{i}", "body": "b"}
            for i in range(publish.MAX_INLINE_COMMENTS + 2)
        ]
        inline, elsewhere, _ = publish.plan_comments(findings, publish.commentable_lines(DIFF), set())
        self.assertEqual(len(inline), publish.MAX_INLINE_COMMENTS)
        self.assertEqual(len(elsewhere), 2)

    def test_comment_body_formats_severity_suggestion_and_marker(self):
        """Render the severity icon, a suggestion block, and the fingerprint marker."""
        finding = {"path": "p", "severity": "high", "title": "T", "body": "B", "suggestion": "y = 2\n"}
        body = publish.comment_body(finding)
        self.assertTrue(body.startswith("🔴 **T**"))
        self.assertIn("```suggestion\ny = 2\n```", body)
        self.assertIn(publish.FINDING_MARKER.format(publish.fingerprint(finding)), body)

    def test_comment_body_drops_suggestion_containing_fences(self):
        """Omit suggestions that would break out of the suggestion block."""
        body = publish.comment_body({"title": "T", "body": "B", "suggestion": "```\nx\n```"})
        self.assertNotIn("```suggestion", body)

    def test_summary_body_strips_duplicate_heading_and_warns_when_incomplete(self):
        """Remove a heading repeated by the reviewer and flag incomplete reviews."""
        output = {"summary": "### Automated review\n\nAll good.", "complete": False}
        elsewhere = [{"path": "changelog", "line": None, "severity": "medium", "title": "Add a fragment", "body": "b"}]
        body = publish.summary_body(output, elsewhere, "abc123")
        self.assertTrue(body.startswith(publish.SUMMARY_MARKER))
        self.assertEqual(body.count("### Automated review"), 1)
        self.assertIn("`changelog`: **Add a fragment**", body)
        self.assertIn("[!WARNING]", body)
        self.assertIn("abc123", body)


if __name__ == "__main__":
    unittest.main()
