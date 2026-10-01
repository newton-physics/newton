# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check runner attribution contracts using local workflow source.

These tests catch missing tags and incorrect trigger-category wiring across
duplicated workflow definitions and callers. They do not contact GitHub or
AWS, launch an EC2 instance, or verify that AWS applies the requested tags.
Those behaviors require a live smoke test.
"""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKLOADS = {
    ".github/workflows/aws_gpu_tests.yml": "gpu-unit-tests",
    ".github/workflows/aws_gpu_benchmarks.yml": "gpu-benchmarks",
    ".github/workflows/minimum_deps_tests.yml": "minimum-deps-tests",
    ".github/workflows/warp_nightly_tests.yml": "warp-nightly-tests",
}
DIRECT_DISPATCH_WORKFLOWS = WORKLOADS.keys() - {".github/workflows/aws_gpu_benchmarks.yml"}
REUSABLE_CALLERS = {
    ".github/workflows/pr_target_aws_gpu_tests.yml": ("./.github/workflows/aws_gpu_tests.yml", "pull-request"),
    ".github/workflows/pr_target_aws_gpu_benchmarks.yml": (
        "./.github/workflows/aws_gpu_benchmarks.yml",
        "pull-request",
    ),
    ".github/workflows/merge_queue_aws_gpu.yml": ("./.github/workflows/aws_gpu_tests.yml", "merge-queue"),
    ".github/workflows/push_aws_gpu.yml": ("./.github/workflows/aws_gpu_tests.yml", "push"),
}
SCHEDULED_CALLERS = (
    "aws_gpu_tests.yml",
    "minimum_deps_tests.yml",
    "warp_nightly_tests.yml",
)


class TestRunnerWorkflowContract(unittest.TestCase):
    @staticmethod
    def _event_block(workflow: str, event: str, next_marker: str) -> str:
        start = workflow.index(f"  {event}:")
        end = workflow.index(next_marker, start)
        return workflow[start:end]

    @staticmethod
    def _input_block(event_block: str, name: str) -> str:
        start = event_block.index(f"      {name}:\n")
        lines = event_block[start:].splitlines(keepends=True)
        block = [lines[0]]
        for line in lines[1:]:
            indentation = len(line) - len(line.lstrip())
            if line.strip() and indentation <= 6:
                break
            block.append(line)
        return "".join(block)

    @staticmethod
    def _resource_tag_block(workflow: str) -> str:
        start = workflow.index("          aws-resource-tags: >\n")
        end = workflow.index("\n            ]", start) + len("\n            ]")
        return workflow[start:end]

    @classmethod
    def _parse_resource_tags(cls, workflow: str, trigger_category: str) -> list[dict[str, str]]:
        tags = cls._resource_tag_block(workflow)
        substitutions = {
            "${{ github.repository }}": "newton-physics/newton",
            "${{ toJSON(inputs['trigger-category']) }}": json.dumps(trigger_category),
            "${{ github.run_id }}": "123456",
            "${{ github.run_attempt }}": "2",
        }
        for expression, value in substitutions.items():
            tags = tags.replace(expression, value)
        return json.loads(tags[tags.index("[") :])

    def test_leaf_workflows_define_attribution_contract(self):
        """Require each runner workflow to supply its attribution metadata."""
        for path, workload in WORKLOADS.items():
            with self.subTest(path=path):
                workflow = (ROOT / path).read_text(encoding="utf-8")
                call_end = "  workflow_dispatch:" if path in DIRECT_DISPATCH_WORKFLOWS else "jobs:"
                call = self._event_block(workflow, "workflow_call", call_end)
                call_input = self._input_block(call, "trigger-category")
                self.assertIn("        required: true\n", call_input)
                self.assertIn("        type: string\n", call_input)

                if path in DIRECT_DISPATCH_WORKFLOWS:
                    dispatch = self._event_block(workflow, "workflow_dispatch", "\njobs:")
                    dispatch_input = self._input_block(dispatch, "trigger-category")
                    self.assertIn("        type: choice\n", dispatch_input)
                    self.assertIn(
                        "        options:\n          - manual\n          - scheduled-nightly\n",
                        dispatch_input,
                    )
                    self.assertIn("        default: 'manual'\n", dispatch_input)

                tags = self._resource_tag_block(workflow)
                expected_tags = (
                    '"created-by", "Value": "github-actions-newton-role"',
                    '"GitHub-Repository", "Value": "${{ github.repository }}"',
                    f'"Newton-Workload", "Value": "{workload}"',
                    '"GitHub-Run-ID", "Value": "${{ github.run_id }}"',
                    '"GitHub-Run-Attempt", "Value": "${{ github.run_attempt }}"',
                )
                for expected_tag in expected_tags:
                    self.assertIn(expected_tag, tags)

    def test_trigger_category_is_json_encoded(self):
        """Preserve arbitrary trigger-category strings in resource tag JSON."""
        trigger_category = 'manual "quoted"\ncategory'
        for path in WORKLOADS:
            with self.subTest(path=path):
                workflow = (ROOT / path).read_text(encoding="utf-8")
                tags = self._parse_resource_tags(workflow, trigger_category)
                trigger_tag = next(tag for tag in tags if tag["Key"] == "Newton-Trigger")
                self.assertEqual(trigger_tag["Value"], trigger_category)

    def test_callers_pass_expected_trigger_categories(self):
        """Map each runner caller to its normalized trigger category."""
        for path, (called_workflow, trigger) in REUSABLE_CALLERS.items():
            with self.subTest(path=path):
                workflow = (ROOT / path).read_text(encoding="utf-8")
                start = workflow.index(f"    uses: {called_workflow}\n")
                end = workflow.index("    secrets:", start)
                self.assertIn(f"      trigger-category: {trigger}\n", workflow[start:end])

        scheduled = (ROOT / ".github/workflows/scheduled_nightly.yml").read_text(encoding="utf-8")
        for called_workflow in SCHEDULED_CALLERS:
            with self.subTest(path=".github/workflows/scheduled_nightly.yml", workflow=called_workflow):
                dispatch = next(
                    line
                    for line in scheduled.splitlines()
                    if f"dispatch_workflow_and_wait.py {called_workflow}" in line
                )
                self.assertIn('-f "inputs[trigger-category]=scheduled-nightly"', dispatch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
