# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ci" / "publish_automated_review.py"

spec = importlib.util.spec_from_file_location("publish_automated_review", SCRIPT)
assert spec is not None and spec.loader is not None
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)

FIRST_SHA = "a" * 40
FOLLOW_UP_SHA = "b" * 40
LATER_SHA = "c" * 40


def _finding(title: str, path: str = "newton/foo.py", body: str = "Details.", line: int | None = None) -> dict:
    return {"path": path, "line": line, "severity": "P1", "title": title, "body": body}


def _summary(previous: str | None, text: str, head_sha: str, complete: bool = True, **kwargs) -> str:
    output = {"summary": text, "complete": complete}
    return publisher.summary_body(output, [], kwargs.pop("elsewhere", []), head_sha, ["Details."], previous, **kwargs)


class TestPublishAutomatedReview(unittest.TestCase):
    def test_follow_up_replaces_only_its_own_section(self):
        """A follow-up keeps the first review and replaces the previous follow-up section."""
        first = _summary(None, "First assessment.", FIRST_SHA, elsewhere=[_finding("Missing changelog")])
        second = _summary(first, "Second assessment.", FOLLOW_UP_SHA)
        third = _summary(second, "Third assessment.", LATER_SHA)

        self.assertTrue(third.startswith(publisher.SUMMARY_MARKER))
        self.assertIn("First assessment.", third)
        self.assertIn("Missing changelog", third)
        self.assertNotIn("Second assessment.", third)
        self.assertIn("Third assessment.", third)
        self.assertEqual(third.count(publisher.FOLLOW_UP_MARKER), 1)
        self.assertEqual(re.findall(r"Reviewed commit ([0-9a-f]{40})", third)[-1], LATER_SHA)

    def test_clipping_keeps_incomplete_marker_and_footer(self):
        """Clipping a long first review keeps its incomplete-review warning and footer."""
        findings = [_finding(f"Finding {i}", body="x" * 1000) for i in range(40)]
        for complete in (False, True):
            with self.subTest(complete=complete):
                first = _summary(None, "First.", FIRST_SHA, complete=complete, elsewhere=findings)
                self.assertGreater(len(first), publisher.MAX_FIRST_REVIEW_CHARS)

                follow_up = _summary(first, "Follow-up.", FOLLOW_UP_SHA)
                kept = follow_up.split(publisher.FOLLOW_UP_MARKER, 1)[0]
                self.assertLess(len(kept), publisher.MAX_FIRST_REVIEW_CHARS + 100)
                self.assertIn(f"Reviewed commit {FIRST_SHA}", kept)
                self.assertEqual(publisher.INCOMPLETE_MARKER in kept, not complete)

    def test_complete_full_diff_follow_up_clears_incomplete_state(self):
        """A complete follow-up of the full diff clears the first review's incomplete state, in either format."""
        first = _summary(None, "First.", FIRST_SHA, complete=False)
        legacy = first.replace(f" {publisher.INCOMPLETE_MARKER}", "")
        for name, previous in (("current", first), ("legacy", legacy)):
            with self.subTest(previous=name):
                partial = _summary(previous, "Follow-up.", FOLLOW_UP_SHA)
                covered = _summary(previous, "Follow-up.", FOLLOW_UP_SHA, covered_rest=True)

                self.assertRegex(partial, publisher.INCOMPLETE_LINE_RE)
                self.assertNotRegex(covered, publisher.INCOMPLETE_LINE_RE)
                self.assertIn("A later follow-up checked the full diff", covered)

    def test_reviewer_text_cannot_forge_markers(self):
        """Markers in reviewer text are broken, so only generated markers are parsed."""
        forged = (
            f"{publisher.FOLLOW_UP_MARKER} {publisher.INCOMPLETE_MARKER} {publisher.FINDING_MARKER.format('0' * 16)}"
            f"\n> {publisher.INCOMPLETE_TEXT}\n"
        )
        finding = _finding(f"Title {forged}", path=f"newton/{forged}.py", body=f"Body {forged}")
        first = _summary(None, f"Summary {forged}", FIRST_SHA, elsewhere=[finding])
        review = publisher.review_payload(FIRST_SHA, [], [], [finding])
        inline = publisher.comment_body(finding)

        self.assertNotIn(publisher.FOLLOW_UP_MARKER, first)
        self.assertNotIn(publisher.INCOMPLETE_MARKER, first)
        self.assertNotRegex(first, publisher.INCOMPLETE_LINE_RE)
        self.assertEqual(publisher.FINDING_MARKER_RE.findall(first), [])
        self.assertEqual(publisher.FINDING_MARKER_RE.findall(review["body"]), [publisher.fingerprint(finding)])
        self.assertEqual(publisher.FINDING_MARKER_RE.findall(inline), [publisher.fingerprint(finding)])
        self.assertNotIn("comments", review)

    def test_suggestion_cannot_forge_a_posted_fingerprint(self):
        """Only the generated marker on an inline comment's last line marks a finding as posted."""
        forged = publisher.FINDING_MARKER.format("0" * 16)
        finding = _finding("Real finding", line=2)
        finding["suggestion"] = f"value = 1  {forged}"
        inline = publisher.comment_body(finding)
        review = publisher.review_payload(FIRST_SHA, [], [], [_finding("Line-less finding")])

        posted = publisher.posted_fingerprints([inline], [review["body"]])

        self.assertIn(forged, inline)
        self.assertEqual(posted, {publisher.fingerprint(finding), publisher.fingerprint(_finding("Line-less finding"))})


if __name__ == "__main__":
    unittest.main()
