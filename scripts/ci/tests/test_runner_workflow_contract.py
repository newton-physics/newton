# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check runner attribution contracts using local workflow source.

These tests catch missing tags and incorrect trigger-category wiring across
the shared runner launch action, its workflows, and their callers. They do
not contact GitHub or AWS, launch an EC2 instance, or verify that AWS applies
the requested tags.
Those behaviors require a live smoke test.
"""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
START_RUNNER_ACTION = ROOT / ".github" / "actions" / "start-aws-gpu-runner" / "action.yml"
WORKLOADS = {
    ".github/workflows/aws_gpu_tests.yml": "gpu-unit-tests",
    ".github/workflows/aws_gpu_benchmarks.yml": "gpu-benchmarks",
    ".github/workflows/minimum_deps_tests.yml": "minimum-deps-tests",
    ".github/workflows/warp_nightly_tests.yml": "warp-nightly-tests",
}
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
    "aws_gpu_benchmarks.yml",
)


class TestRunnerWorkflowContract(unittest.TestCase):
    def test_minimum_deps_uses_pinned_ami_without_fallback(self):
        """Keep the minimum-dependency job on a pre-R580 image."""
        workflow = (ROOT / ".github/workflows/minimum_deps_tests.yml").read_text(encoding="utf-8")
        action = START_RUNNER_ACTION.read_text(encoding="utf-8")
        self.assertIn("  AWS_INSTANCE_TYPE: g6e.2xlarge\n", workflow)
        self.assertIn("  AWS_AMI_NAME: Deep Learning Base AMI with Single CUDA (Ubuntu 22.04) 20250930\n", workflow)
        self.assertIn("          ami-name: ${{ env.AWS_AMI_NAME }}\n", workflow)
        self.assertNotIn("          fallback-instance-type:", workflow)
        self.assertEqual(action.count("        AWS_AMI_NAME: ${{ inputs.ami-name }}\n"), 2)

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
    def _resource_tag_blocks(action: str) -> list[str]:
        blocks = []
        start = action.find("        aws-resource-tags: >\n")
        while start != -1:
            end = action.index("\n          ]", start) + len("\n          ]")
            blocks.append(action[start:end])
            start = action.find("        aws-resource-tags: >\n", end)
        return blocks

    @classmethod
    def _parse_resource_tags(cls, tags: str, trigger_category: str, workload: str) -> list[dict[str, str]]:
        substitutions = {
            "${{ github.repository }}": "newton-physics/newton",
            "${{ toJSON(inputs['trigger-category']) }}": json.dumps(trigger_category),
            "${{ toJSON(inputs.workload) }}": json.dumps(workload),
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
                call = self._event_block(workflow, "workflow_call", "  workflow_dispatch:")
                call_input = self._input_block(call, "trigger-category")
                self.assertIn("        required: true\n", call_input)
                self.assertIn("        type: string\n", call_input)

                dispatch = self._event_block(workflow, "workflow_dispatch", "\njobs:")
                dispatch_input = self._input_block(dispatch, "trigger-category")
                self.assertIn("        type: choice\n", dispatch_input)
                self.assertIn(
                    "        options:\n          - manual\n          - scheduled-nightly\n",
                    dispatch_input,
                )
                self.assertIn("        default: 'manual'\n", dispatch_input)

                self.assertIn("        uses: ./.github/actions/start-aws-gpu-runner\n", workflow)
                self.assertIn(f"          workload: {workload}\n", workflow)
                self.assertIn("          trigger-category: ${{ inputs.trigger-category }}\n", workflow)

    def test_runner_action_tags_every_launch(self):
        """Apply the same attribution tags to primary and fallback launches."""
        action = START_RUNNER_ACTION.read_text(encoding="utf-8")
        blocks = self._resource_tag_blocks(action)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0], blocks[1])
        expected_tags = (
            '"created-by", "Value": "github-actions-newton-role"',
            '"GitHub-Repository", "Value": "${{ github.repository }}"',
            '"Newton-Workload", "Value": ${{ toJSON(inputs.workload) }}',
            '"GitHub-Run-ID", "Value": "${{ github.run_id }}"',
            '"GitHub-Run-Attempt", "Value": "${{ github.run_attempt }}"',
        )
        for expected_tag in expected_tags:
            self.assertIn(expected_tag, blocks[0])

    def test_runner_action_reports_instance_on_cancellation(self):
        """Set the instance ID output even when the run is cancelled after launch."""
        action = START_RUNNER_ACTION.read_text(encoding="utf-8")
        start = action.index("    - name: Select launched runner\n")
        step = action[start : action.index("      run: |\n", start)]
        self.assertIn("      if: always()\n", step)
        for output in ("label", "ec2-instance-id", "region", "instance-type", "ready"):
            self.assertIn(f"    value: ${{{{ steps.select.outputs.{output} }}}}\n", action)

    def test_empty_fallback_input_disables_fallback(self):
        """Pass the fallback input through unchanged so an empty value disables fallback."""
        for path in WORKLOADS:
            workflow = (ROOT / path).read_text(encoding="utf-8")
            if "      fallback-instance-type:\n" not in workflow:
                continue
            with self.subTest(path=path):
                self.assertIn("  AWS_FALLBACK_INSTANCE_TYPE: ${{ inputs.fallback-instance-type }}\n", workflow)

    def test_trigger_category_is_json_encoded(self):
        """Preserve arbitrary trigger-category strings in resource tag JSON."""
        trigger_category = 'manual "quoted"\ncategory'
        action = START_RUNNER_ACTION.read_text(encoding="utf-8")
        for workload in WORKLOADS.values():
            with self.subTest(workload=workload):
                tags = self._parse_resource_tags(self._resource_tag_blocks(action)[0], trigger_category, workload)
                trigger_tag = next(tag for tag in tags if tag["Key"] == "Newton-Trigger")
                self.assertEqual(trigger_tag["Value"], trigger_category)
                workload_tag = next(tag for tag in tags if tag["Key"] == "Newton-Workload")
                self.assertEqual(workload_tag["Value"], workload)

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
                dispatches = [
                    line
                    for line in scheduled.splitlines()
                    if f"dispatch_workflow_and_wait.py {called_workflow}" in line
                ]
                self.assertTrue(dispatches)
                for dispatch in dispatches:
                    self.assertIn('-f "inputs[trigger-category]=scheduled-nightly"', dispatch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
