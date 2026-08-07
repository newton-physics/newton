# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "pr_api_changes.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
DETECTOR = ROOT / "scripts" / "ci" / "detect_api_changes.py"


class TestPrApiChangesWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def _step_block(self, name: str) -> str:
        marker = f"      - name: {name}\n"
        start = self.workflow.index(marker)
        end = self.workflow.find("\n      - name:", start + len(marker))
        return self.workflow[start:] if end == -1 else self.workflow[start:end]

    def test_ci_tooling_lives_under_scripts_ci(self):
        """Keep the detector and its tests outside the Newton package."""
        self.assertTrue(DETECTOR.is_file())
        self.assertFalse((ROOT / ".github" / "scripts" / "detect_api_changes.py").exists())
        self.assertFalse((ROOT / "newton" / "tests" / "test_detect_api_changes.py").exists())

    def test_ci_runs_api_detector_tests(self):
        """Run CI-tool regression tests explicitly in the main CI workflow."""
        ci_workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("uv run --no-project -m unittest discover -s scripts/ci/tests", ci_workflow)

    def test_external_prs_do_not_require_manual_approval(self):
        """Run trusted static analysis for external PRs without an approval gate."""
        self.assertNotIn("check-author-membership:", self.workflow)
        self.assertNotIn("require-approval:", self.workflow)
        self.assertIn("github.repository == 'newton-physics/newton'", self.workflow)
        self.assertIn(
            'git -C base show "${{ github.event.pull_request.base.sha }}:scripts/ci/detect_api_changes.py"',
            self.workflow,
        )

    def test_security_invariants_are_documented(self):
        """Document why analyzing an untrusted head is safe."""
        self.assertIn("never execute head code", self.workflow)
        self.assertIn("Security invariant - DO NOT BREAK", DETECTOR.read_text(encoding="utf-8"))

    def test_api_review_tracks_advisory_decision(self):
        """Synchronize one label and bot comment from the advisory decision."""
        block = self._step_block("Sync API review")
        self.assertNotIn("\n        if:", block)
        self.assertIn("NEEDS_REVIEW: ${{ steps.detect.outputs.needs_review }}", block)
        self.assertIn("if (needsReview)", block)
        self.assertIn("github.rest.issues.addLabels", block)
        self.assertIn("labels: ['api-changes']", block)
        self.assertIn("github.rest.issues.removeLabel", block)
        self.assertIn("name: 'api-changes'", block)
        self.assertIn("if (error.status !== 404) throw error", block)
        self.assertEqual(block.count("github.rest.issues.listComments"), 1)
        self.assertIn("github.rest.issues.deleteComment", block)
        self.assertIn("github.rest.issues.updateComment", block)
        self.assertIn("github.rest.issues.createComment", block)


if __name__ == "__main__":
    unittest.main(verbosity=2)
