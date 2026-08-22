# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the watchdog's embedded Lambda behavior with local AWS fakes.

These tests execute the Python embedded in the CloudFormation template and
fake only its EC2 and CloudWatch boundaries. They do not validate
CloudFormation, IAM, boto3, regional AWS behavior, or actual termination.
Those behaviors require deployment and live smoke checks.
"""

import textwrap
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "scripts" / "ci" / "aws" / "overdue-newton-github-runner-watchdog.yaml"


class FilteringPaginator:
    def __init__(self, instances, error=None):
        self.instances = instances
        self.error = error

    def paginate(self, Filters):
        if self.error is not None:
            raise self.error

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
    def __init__(self, instances, pagination_error=None, termination_error=None):
        self.instances = instances
        self.pagination_error = pagination_error
        self.termination_error = termination_error
        self.termination_calls = []

    def get_paginator(self, operation):
        if operation != "describe_instances":
            raise AssertionError(f"Unexpected operation: {operation}")
        return FilteringPaginator(self.instances, self.pagination_error)

    def terminate_instances(self, InstanceIds):
        self.termination_calls.append(InstanceIds)
        if self.termination_error is not None:
            raise self.termination_error


class FakeCloudWatch:
    def __init__(self, error=None):
        self.metric_calls = []
        self.error = error

    def put_metric_data(self, **kwargs):
        self.metric_calls.append(kwargs)
        if self.error is not None:
            raise self.error


class FakeBoto3:
    def __init__(self, regional_ec2, cloudwatch):
        self.regional_ec2 = regional_ec2
        self.cloudwatch = cloudwatch
        self.client_calls = []

    def client(self, service_name, region_name):
        self.client_calls.append((service_name, region_name))
        if service_name == "ec2":
            return self.regional_ec2[region_name]
        if service_name == "cloudwatch":
            return self.cloudwatch
        raise AssertionError(f"Unexpected service: {service_name}")


class TestWatchdogLogic(unittest.TestCase):
    def _lambda_namespace(self) -> dict:
        self.assertTrue(TEMPLATE.is_file(), f"Missing watchdog template: {TEMPLATE}")
        template = TEMPLATE.read_text(encoding="utf-8")
        marker = "        ZipFile: |\n"
        self.assertIn(marker, template)
        source_lines = []
        # Stop at the YAML indentation boundary so tests execute the exact
        # embedded source without adding a YAML parser dependency.
        for line in template[template.index(marker) + len(marker) :].splitlines(keepends=True):
            if line.strip() and not line.startswith("          "):
                break
            source_lines.append(line)
        source = textwrap.dedent("".join(source_lines))
        namespace = {}
        exec(compile(source, str(TEMPLATE), "exec"), namespace)
        return namespace

    @staticmethod
    def _instance(
        now,
        *,
        age_minutes=90,
        instance_id="i-0123456789abcdef0",
        repository=None,
        include_attribution=True,
        state="running",
    ):
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
            "InstanceId": instance_id,
            "InstanceType": "g7e.2xlarge",
            "LaunchTime": now - timedelta(minutes=age_minutes),
            "State": {"Name": state},
            "Tags": tags,
        }

    def test_watchdog_filters_owned_running_instances_and_applies_thresholds(self):
        """Select owned running runners at the candidate and overdue thresholds."""
        namespace = self._lambda_namespace()
        now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
        instances = [
            self._instance(now, age_minutes=89, instance_id="i-89"),
            self._instance(now, age_minutes=90, instance_id="i-90"),
            self._instance(now, age_minutes=119, instance_id="i-119"),
            self._instance(now, age_minutes=120, instance_id="i-120"),
            self._instance(now, age_minutes=95, instance_id="i-missing", include_attribution=False),
            self._instance(now, age_minutes=130, instance_id="i-unowned"),
            self._instance(now, age_minutes=130, instance_id="i-stopped", state="stopped"),
        ]
        instances[5]["Tags"][0]["Value"] = "another-runner-role"

        scan = namespace["scan_runner_instances"](
            ["us-east-2"],
            lambda region: FakeEc2(instances),
            now,
            90,
            120,
            "",
        )

        self.assertEqual(
            [item["instance_id"] for item in scan["owned"]],
            ["i-120", "i-119", "i-missing", "i-90", "i-89"],
        )
        self.assertEqual(
            [item["instance_id"] for item in scan["candidates"]],
            ["i-120", "i-119", "i-missing", "i-90"],
        )
        self.assertEqual([item["instance_id"] for item in scan["overdue"]], ["i-120"])
        self.assertEqual(
            {key: scan["owned"][2][key] for key in ("repository", "trigger", "workload", "run_id", "run_attempt")},
            {
                "repository": "unknown",
                "trigger": "unknown",
                "workload": "unknown",
                "run_id": "unknown",
                "run_attempt": "unknown",
            },
        )
        self.assertEqual(scan["failures"], [])

    def test_watchdog_requires_an_exact_present_repository_tag(self):
        """Restrict cleanup candidates to an exact repository tag value.

        Keep eligibility based on the raw tag: a missing tag is displayed as
        ``"unknown"`` in diagnostics, but must not match that fallback value.
        """
        namespace = self._lambda_namespace()
        now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
        instances = [
            self._instance(now, instance_id="i-selected", repository="example/selected"),
            self._instance(now, instance_id="i-other", repository="example/other"),
            self._instance(now, instance_id="i-missing"),
            self._instance(now, instance_id="i-tagged-unknown", repository="unknown"),
        ]

        selected_scan = namespace["scan_runner_instances"](
            ["ap-northeast-1"],
            lambda region: FakeEc2(instances),
            now,
            90,
            120,
            "example/selected",
        )
        unknown_scan = namespace["scan_runner_instances"](
            ["ap-northeast-1"],
            lambda region: FakeEc2(instances),
            now,
            90,
            120,
            "unknown",
        )

        self.assertEqual([item["instance_id"] for item in selected_scan["candidates"]], ["i-selected"])
        self.assertEqual(
            [item["repository"] for item in unknown_scan["owned"]],
            ["unknown", "example/other", "example/selected", "unknown"],
        )
        self.assertEqual(
            [item["instance_id"] for item in unknown_scan["candidates"]],
            ["i-tagged-unknown"],
        )

    def test_watchdog_rejects_unrestricted_low_threshold_before_aws_calls(self):
        """Reject enabled low-threshold cleanup without a required repository.

        Validate this safety boundary before creating AWS clients so a
        misconfigured aggressive cleanup cannot inspect or terminate runners.
        """
        namespace = self._lambda_namespace()
        now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
        invalid_boto3 = FakeBoto3({}, FakeCloudWatch())
        unrestricted_environment = {
            "AWS_REGION": "us-east-1",
            "MONITORED_REGIONS": "us-east-1",
            "TERMINATE_AFTER_MINUTES": "10",
            "ALERT_AFTER_MINUTES": "120",
            "TERMINATION_ENABLED": "true",
            "REQUIRED_REPOSITORY": "",
        }

        with (
            patch.dict("os.environ", unrestricted_environment, clear=True),
            patch.dict("sys.modules", {"boto3": SimpleNamespace(client=invalid_boto3.client)}),
            self.assertRaisesRegex(ValueError, "RequiredRepository"),
        ):
            namespace["lambda_handler"]({}, None)

        self.assertEqual(invalid_boto3.client_calls, [])

        accepted_boto3 = FakeBoto3({"us-east-1": FakeEc2([])}, FakeCloudWatch())
        namespace["datetime"] = SimpleNamespace(now=lambda zone: now)
        with (
            patch.dict(
                "os.environ",
                {**unrestricted_environment, "REQUIRED_REPOSITORY": "example/selected"},
                clear=True,
            ),
            patch.dict("sys.modules", {"boto3": SimpleNamespace(client=accepted_boto3.client)}),
            redirect_stdout(StringIO()),
        ):
            summary = namespace["lambda_handler"]({}, None)

        self.assertEqual(summary["required_repository"], "example/selected")
        self.assertEqual(
            accepted_boto3.client_calls,
            [("ec2", "us-east-1"), ("cloudwatch", "us-east-1")],
        )

    def test_watchdog_accepts_fractional_termination_threshold(self):
        """Accept fractional thresholds allowed by CloudFormation."""
        namespace = self._lambda_namespace()
        environment = {
            "MONITORED_REGIONS": "us-east-1",
            "TERMINATE_AFTER_MINUTES": "90.5",
            "ALERT_AFTER_MINUTES": "120",
            "TERMINATION_ENABLED": "false",
            "REQUIRED_REPOSITORY": "",
        }

        try:
            config = namespace["load_config"](environment)
        except ValueError as exc:
            self.fail(f"CloudFormation-valid threshold was rejected: {exc}")

        self.assertEqual(config["terminate_after_minutes"], 90.5)

    def test_watchdog_reports_would_terminate_without_terminating_in_shadow_mode(self):
        """Record shadow-mode cleanup decisions without calling EC2 termination."""
        namespace = self._lambda_namespace()
        candidate = {"instance_id": "i-shadow", "region": "us-west-2", "age_minutes": 90}
        ec2 = FakeEc2([])

        cleanup = namespace["apply_cleanup"]([candidate], lambda region: ec2, False)

        self.assertEqual(
            cleanup["decisions"],
            [
                {
                    "instance_id": "i-shadow",
                    "region": "us-west-2",
                    "age_minutes": 90,
                    "decision": "would_terminate",
                }
            ],
        )
        self.assertEqual(cleanup["terminated"], [])
        self.assertEqual(cleanup["failures"], [])
        self.assertEqual(ec2.termination_calls, [])

    def test_watchdog_completes_healthy_work_before_raising_collected_failures(self):
        """Complete healthy cleanup work before raising collected failures.

        Deliberately fail one regional scan and one termination while a second
        termination remains healthy. The handler must finish all possible work
        and emit its diagnostics before it raises the collected failures.
        """
        namespace = self._lambda_namespace()
        now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
        failed_ec2 = FakeEc2(
            [self._instance(now, instance_id="i-termination-failed")],
            termination_error=RuntimeError("termination denied"),
        )
        successful_ec2 = FakeEc2([self._instance(now, age_minutes=120, instance_id="i-terminated")])
        cloudwatch = FakeCloudWatch()
        boto3 = FakeBoto3(
            {
                "ap-northeast-1": FakeEc2([], pagination_error=RuntimeError("scan unavailable")),
                "us-east-1": failed_ec2,
                "us-west-2": successful_ec2,
            },
            cloudwatch,
        )
        environment = {
            "AWS_REGION": "us-east-1",
            "MONITORED_REGIONS": "ap-northeast-1,us-east-1,us-west-2",
            "TERMINATE_AFTER_MINUTES": "90",
            "ALERT_AFTER_MINUTES": "120",
            "TERMINATION_ENABLED": "true",
            "REQUIRED_REPOSITORY": "",
        }
        namespace["datetime"] = SimpleNamespace(now=lambda zone: now)
        output = StringIO()

        with (
            patch.dict("os.environ", environment, clear=True),
            patch.dict("sys.modules", {"boto3": SimpleNamespace(client=boto3.client)}),
            redirect_stdout(output),
            self.assertRaisesRegex(RuntimeError, "Watchdog encountered cleanup failures"),
        ):
            namespace["lambda_handler"]({}, None)

        summary = namespace["json"].loads(output.getvalue())
        self.assertEqual(summary["scanned_regions"], ["ap-northeast-1", "us-east-1", "us-west-2"])
        self.assertEqual(summary["owned_count"], 2)
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["terminated_count"], 1)
        self.assertEqual(summary["overdue_count"], 1)
        decisions = {item["instance_id"]: item for item in summary["decisions"]}
        self.assertEqual(set(decisions), {"i-terminated", "i-termination-failed"})
        self.assertEqual(decisions["i-terminated"]["decision"], "terminated")
        self.assertNotIn("error", decisions["i-terminated"])
        self.assertEqual(decisions["i-termination-failed"]["decision"], "termination_failed")
        self.assertEqual(decisions["i-termination-failed"]["error"], "termination denied")
        self.assertEqual(
            summary["failures"],
            [
                {"operation": "scan", "region": "ap-northeast-1", "error": "scan unavailable"},
                {
                    "operation": "terminate",
                    "region": "us-east-1",
                    "instance_id": "i-termination-failed",
                    "error": "termination denied",
                },
            ],
        )
        self.assertEqual(failed_ec2.termination_calls, [["i-termination-failed"]])
        self.assertEqual(successful_ec2.termination_calls, [["i-terminated"]])
        self.assertEqual(
            cloudwatch.metric_calls,
            [
                {
                    "Namespace": "Newton/GitHubRunnerWatchdog",
                    "MetricData": [
                        {"MetricName": "OwnedRunnerCount", "Value": 2.0, "Unit": "Count"},
                        {"MetricName": "TerminationCandidateCount", "Value": 2.0, "Unit": "Count"},
                        {"MetricName": "TerminatedRunnerCount", "Value": 1.0, "Unit": "Count"},
                        {"MetricName": "OverdueRunnerCount", "Value": 1.0, "Unit": "Count"},
                        {"MetricName": "OldestRunnerAgeMinutes", "Value": 120.0, "Unit": "Count"},
                    ],
                }
            ],
        )

    def test_watchdog_emits_summary_before_cloudwatch_failure(self):
        """Emit the structured summary before propagating a metrics failure.

        Print the summary first so operators retain cleanup diagnostics when
        CloudWatch publishing fails and the invocation raises.
        """
        namespace = self._lambda_namespace()
        now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
        cloudwatch = FakeCloudWatch(error=RuntimeError("metrics unavailable"))
        boto3 = FakeBoto3(
            {"us-west-2": FakeEc2([self._instance(now, instance_id="i-metric-failure")])},
            cloudwatch,
        )
        environment = {
            "AWS_REGION": "us-east-1",
            "MONITORED_REGIONS": "us-west-2",
            "TERMINATE_AFTER_MINUTES": "90",
            "ALERT_AFTER_MINUTES": "120",
            "TERMINATION_ENABLED": "false",
            "REQUIRED_REPOSITORY": "",
        }
        namespace["datetime"] = SimpleNamespace(now=lambda zone: now)
        output = StringIO()

        with (
            patch.dict("os.environ", environment, clear=True),
            patch.dict("sys.modules", {"boto3": SimpleNamespace(client=boto3.client)}),
            redirect_stdout(output),
            self.assertRaisesRegex(RuntimeError, "metrics unavailable"),
        ):
            namespace["lambda_handler"]({}, None)

        summary = namespace["json"].loads(output.getvalue())
        self.assertEqual(summary["owned_count"], 1)
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["decisions"][0]["decision"], "would_terminate")
        self.assertEqual(len(cloudwatch.metric_calls), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
