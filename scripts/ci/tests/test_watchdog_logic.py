# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the watchdog's instance-selection and attribution logic locally.

These tests execute the Lambda source embedded in the CloudFormation template
with a fake EC2 paginator. They catch repository-filter regressions and missing
attribution-tag handling, but do not call AWS or validate CloudFormation, IAM,
or boto3 behavior. Those behaviors require deployment and live smoke checks.
"""

import textwrap
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "scripts" / "ci" / "aws" / "overdue-newton-github-runner-watchdog.yaml"


class FilteringPaginator:
    def __init__(self, instances):
        self.instances = instances

    def paginate(self, Filters):
        matching = self.instances
        for item in Filters:
            name = item["Name"]
            values = item["Values"]
            if name == "instance-state-name":
                matching = [instance for instance in matching if instance["State"]["Name"] in values]
            elif name.startswith("tag:"):
                key = name.removeprefix("tag:")
                matching = [
                    instance
                    for instance in matching
                    if {tag["Key"]: tag["Value"] for tag in instance.get("Tags", [])}.get(key) in values
                ]
        yield {"Reservations": [{"Instances": matching}]}


class FakeEc2:
    def __init__(self, instances):
        self.instances = instances

    def get_paginator(self, operation):
        if operation != "describe_instances":
            raise AssertionError(f"Unexpected operation: {operation}")
        return FilteringPaginator(self.instances)


class TestWatchdogLogic(unittest.TestCase):
    def _lambda_namespace(self) -> dict:
        self.assertTrue(TEMPLATE.is_file(), f"Missing watchdog template: {TEMPLATE}")
        template = TEMPLATE.read_text(encoding="utf-8")
        marker = "        ZipFile: |\n"
        self.assertIn(marker, template)
        source_lines = []
        for line in template[template.index(marker) + len(marker) :].splitlines(keepends=True):
            if line.strip() and not line.startswith("          "):
                break
            source_lines.append(line)
        source = textwrap.dedent("".join(source_lines))
        namespace = {}
        exec(compile(source, str(TEMPLATE), "exec"), namespace)
        return namespace

    @staticmethod
    def _instance(now, *, repository=None, include_attribution=True):
        tags = [{"Key": "created-by", "Value": "github-actions-newton-role"}]
        if repository is not None:
            tags.append({"Key": "GitHub-Repository", "Value": repository})
        if include_attribution:
            tags.extend(
                [
                    {"Key": "Newton-Trigger", "Value": "manual"},
                    {"Key": "Newton-Workload", "Value": "gpu-unit-tests"},
                    {"Key": "GitHub-Run-ID", "Value": "123456"},
                    {"Key": "GitHub-Run-Attempt", "Value": "2"},
                ]
            )
        return {
            "InstanceId": "i-0123456789abcdef0",
            "InstanceType": "g7e.2xlarge",
            "LaunchTime": now - timedelta(minutes=90),
            "State": {"Name": "running"},
            "Tags": tags,
        }

    def test_watchdog_finds_owned_runners_from_any_repository(self):
        """Report an overdue owned runner regardless of repository tag value."""
        namespace = self._lambda_namespace()
        now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
        instance = self._instance(now, repository="example/newton")

        overdue = namespace["find_overdue_instances"](
            ["us-east-2"],
            lambda region: FakeEc2([instance]),
            now,
            60,
        )

        self.assertEqual(
            overdue,
            [
                {
                    "instance_id": "i-0123456789abcdef0",
                    "instance_type": "g7e.2xlarge",
                    "region": "us-east-2",
                    "launch_time": "2026-08-08T10:30:00+00:00",
                    "age_minutes": 90,
                    "repository": "example/newton",
                    "trigger": "manual",
                    "workload": "gpu-unit-tests",
                    "run_id": "123456",
                    "run_attempt": "2",
                }
            ],
        )

    def test_watchdog_excludes_unowned_instances(self):
        """Exclude overdue instances not owned by Newton."""
        namespace = self._lambda_namespace()
        now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
        instance = self._instance(now)
        instance["Tags"][0]["Value"] = "other-runner-role"

        overdue = namespace["find_overdue_instances"](
            ["us-east-2"],
            lambda region: FakeEc2([instance]),
            now,
            60,
        )

        self.assertEqual(overdue, [])

    def test_watchdog_tolerates_missing_attribution_tags(self):
        """Report unknown metadata for runners created before attribution tags."""
        namespace = self._lambda_namespace()
        now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
        instance = self._instance(now, include_attribution=False)

        overdue = namespace["find_overdue_instances"](
            ["us-west-2"],
            lambda region: FakeEc2([instance]),
            now,
            60,
        )

        self.assertEqual(
            {key: overdue[0][key] for key in ("repository", "trigger", "workload", "run_id", "run_attempt")},
            {
                "repository": "unknown",
                "trigger": "unknown",
                "workload": "unknown",
                "run_id": "unknown",
                "run_attempt": "unknown",
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
