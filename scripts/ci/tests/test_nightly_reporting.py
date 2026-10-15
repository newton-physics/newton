# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise nightly reporting without dispatching workflows or posting issues."""

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[3] / ".github/workflows/scheduled_nightly.yml"


class TestNightlyReporting(unittest.TestCase):
    def test_aggregate_status(self):
        """Fail the orchestrator for unsuccessful suites while allowing skipped suites."""
        workflow = WORKFLOW.read_text()
        step = workflow.split("      - name: Report nightly test status\n", 1)[1].split("\n  notify-on-failure:", 1)[0]
        script = step.split("        run: |\n", 1)[1]
        script = "\n".join(line[10:] for line in script.splitlines())
        self.assertIn("        if: ${{ !cancelled() }}", step)
        for conclusions, expected in (
            (("success", "success", "success"), 0),
            (("success", "success", ""), 0),
            (("failure", "success", "success"), 1),
            (("success", "timed_out", "success"), 1),
            (("success", "success", "cancelled"), 1),
            (("dispatch_error", "success", ""), 1),
        ):
            with self.subTest(conclusions=conclusions):
                result = subprocess.run(
                    ["bash", "-e", "-c", script],
                    env=dict(
                        os.environ,
                        GPU_TESTS=conclusions[0],
                        MINIMUM_DEPS_TESTS=conclusions[1],
                        WARP_NIGHTLY_TESTS=conclusions[2],
                    ),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required to execute the notification script")
    def test_notification_status_and_history(self):
        """Render distinct unsuccessful statuses and label history from newest to oldest."""
        workflow = WORKFLOW.read_text()
        script = workflow.split("          script: |\n", 1)[1]
        script = "\n".join(line[12:] for line in script.splitlines())
        for conclusion, label in (
            ("failure", "❌ Failed"),
            ("timed_out", "❌ Failed (timed out)"),
            ("cancelled", "⏹️ Cancelled"),
            ("dispatch_error", "❌ Failed (dispatch error)"),
        ):
            with self.subTest(conclusion=conclusion):
                rendered = script
                for suite, value in (
                    ("gpu-tests", "success"),
                    ("minimum-deps-tests", conclusion),
                    ("warp-nightly-tests", ""),
                ):
                    rendered = rendered.replace(
                        "${{ needs.run-nightly-tests.outputs." + suite + "-conclusion }}", value
                    )
                    rendered = rendered.replace(
                        "${{ needs.run-nightly-tests.outputs." + suite + "-url }}", "https://example.com/run"
                    )
                harness = """
const context = {repo: {owner: 'owner', repo: 'repo'}, payload: {}, serverUrl: 'https://example.com', runId: 1};
const core = {warning: () => {}, error: () => {}};
const github = {rest: {
  actions: {listWorkflowRuns: async () => ({data: {workflow_runs: [
    {conclusion: 'cancelled'}, {conclusion: 'timed_out'}, {conclusion: 'failure'},
    {conclusion: 'success'}, {conclusion: 'cancelled'}, {conclusion: 'success'}
  ]}})},
  issues: {
    listForRepo: async () => ({data: [{title: 'Nightly failure', number: 3265}]}),
    createComment: async args => console.log(JSON.stringify(args))
  }
}};
"""
                result = subprocess.run(
                    [
                        "node",
                        "-e",
                        harness
                        + "(async () => {\n"
                        + rendered
                        + "\n})().catch(e => {console.error(e); process.exit(1);});",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                body = json.loads(result.stdout)["body"]
                self.assertIn("Recent (newest → oldest)", body)
                self.assertIn(f"| Minimum deps tests | ⏹️❌❌✅⏹️ | {label} |", body)
                self.assertIn("| Multi-GPU tests | ⏹️❌❌✅⏹️ | ✅ Passed |", body)
                self.assertIn("| Warp nightly tests | ⏹️❌❌✅⏹️ | ⏭️ Skipped |", body)
                self.assertIn("History: ✅ Passed, ❌ Failed, ⏹️ Cancelled.", body)


if __name__ == "__main__":
    unittest.main()
