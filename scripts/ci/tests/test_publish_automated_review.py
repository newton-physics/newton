# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "publish_automated_review.py"
spec = importlib.util.spec_from_file_location("publish_review", SCRIPT)
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


class TestPublishAutomatedReview(unittest.TestCase):
    def test_model_attribution_in_all_comments(self):
        """Attribute inline findings, review bodies, and summaries to the configured model."""
        with patch.dict(
            os.environ,
            {
                "GITHUB_REPOSITORY": "example/project",
                "GITHUB_RUN_ID": "123",
                "REVIEW_MODEL_NAME": "GPT 6 Astra",
                "REVIEW_EFFORT": "high",
            },
        ):
            details = publish.run_details()
        finding = {
            "path": "example.py",
            "line": 2,
            "severity": "P1",
            "title": "Fix the incorrect result",
            "body": "The result is incorrect.",
            "suggestion": "return 1",
        }
        inline, findings, elsewhere, _ = publish.plan_comments([finding], {"example.py": {2}}, set(), details=details)
        review = publish.review_payload("a" * 40, inline, findings, elsewhere, details)
        summary = publish.summary_body(
            {"summary": "One issue needs attention.", "complete": True},
            findings,
            elsewhere,
            "a" * 40,
            details,
        )
        for body in (inline[0]["body"], review["body"], summary):
            self.assertIn("GPT 6 Astra", body)
            self.assertIn("effort `high`", body)
            self.assertIn("actions/runs/123", body)
            self.assertNotIn("Claude", body)
            self.assertNotIn("openai/openai/", body)
        self.assertIn("```suggestion\nreturn 1\n```", inline[0]["body"])
        self.assertEqual(publish.posted_fingerprints([inline[0]["body"]], []), {publish.fingerprint(finding)})

    def test_follow_up_keeps_historical_model(self):
        """Preserve the original model attribution when a new model reviews a follow-up."""
        output = {"summary": "No further issues.", "complete": True}
        previous = publish.summary_body(output, [], [], "a" * 40, ["Reviewed with Claude Opus 5.5."])
        result = publish.summary_body(output, [], [], "b" * 40, ["Reviewed with GPT 6 Astra at high effort."], previous)
        original, follow_up = result.split(publish.FOLLOW_UP_MARKER)
        self.assertIn("Claude Opus 5.5", original)
        self.assertIn("GPT 6 Astra", follow_up)
        self.assertNotIn("Claude", follow_up)


if __name__ == "__main__":
    unittest.main()
