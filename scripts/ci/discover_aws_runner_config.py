# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Discover AWS EC2 runner networking for GitHub Actions.

The script is designed to run inside ``aws_gpu_tests.yml`` after AWS
credentials are configured. It writes one GitHub Actions step output:

``availability-zones-config``
    JSON array passed to ``machulav/ec2-github-runner``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

AwsCall = Callable[..., Any]
Warn = Callable[[str], None]
AWS_CLI_TIMEOUT_SECONDS = 120


def warning(message: str) -> None:
    """Emit a GitHub Actions warning annotation."""
    print(f"::warning::{message}", flush=True)


def error(message: str) -> None:
    """Emit a GitHub Actions error annotation."""
    print(f"::error::{message}", flush=True)


def print_aws_cli_output(label: str, output: str | bytes | None) -> None:
    """Print captured AWS CLI output to stderr when available."""
    if not output:
        return
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    print(f"AWS CLI {label}:", file=sys.stderr)
    print(output, file=sys.stderr)


def aws(region: str, *args: str) -> Any | None:
    """Run an AWS EC2 CLI command and parse the JSON response.

    Args:
        region: AWS region name.
        args: Arguments placed after ``aws ec2``.

    Returns:
        Parsed JSON output, or ``None`` when the command fails or returns no
        output.
    """
    cmd = ["aws", "ec2", *args, "--region", region, "--output", "json"]
    command_label = f"aws ec2 {args[0]}" if args else "aws ec2"
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=AWS_CLI_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        warning(f"{region}: AWS CLI command timed out after {AWS_CLI_TIMEOUT_SECONDS}s: {command_label}")
        print_aws_cli_output("stdout before timeout", exc.stdout)
        print_aws_cli_output("stderr before timeout", exc.stderr)
        return None
    except subprocess.CalledProcessError as exc:
        warning(f"{region}: AWS CLI command failed: {command_label}")
        print_aws_cli_output("stdout", exc.stdout)
        print_aws_cli_output("stderr", exc.stderr)
        return None

    output = completed.stdout.strip()
    if not output:
        return None
    try:
        return json.loads(output)
    except ValueError as exc:
        warning(f"{region}: AWS CLI command returned invalid JSON: {command_label}")
        print(f"Invalid JSON output from AWS CLI: {exc}", file=sys.stderr)
        print(f"Output prefix: {output[:1000]!r}", file=sys.stderr)
        return None


def allows_outbound_internet(security_group: Mapping[str, Any]) -> bool:
    """Return whether a security group has IPv4 internet egress."""
    for permission in security_group.get("IpPermissionsEgress", []):
        for ip_range in permission.get("IpRanges", []):
            if ip_range.get("CidrIp") == "0.0.0.0/0":
                return True
    return False


def discover_candidates(
    regions: Sequence[str],
    instance_type: str,
    tag_key: str,
    aws_call: AwsCall = aws,
    warn: Warn = warning,
) -> list[dict[str, str]]:
    """Discover eligible EC2 runner candidates.

    Args:
        regions: Ordered AWS regions to inspect.
        instance_type: EC2 instance type requested by the workflow.
        tag_key: Tag key used to find eligible subnets and security groups.
        aws_call: AWS EC2 call helper.
        warn: Warning callback.

    Returns:
        Candidate objects accepted by ``machulav/ec2-github-runner``.
    """
    candidates: list[dict[str, str]] = []

    for region in regions:
        print(f"Checking {region} for {instance_type}", flush=True)

        offered_zone_ids = aws_call(
            region,
            "describe-instance-type-offerings",
            "--location-type",
            "availability-zone-id",
            "--filters",
            f"Name=instance-type,Values={instance_type}",
            "--query",
            "InstanceTypeOfferings[].Location",
        )
        if not offered_zone_ids:
            warn(f"{region}: {instance_type} is not offered in any availability zone; skipping region")
            continue
        offered_zone_ids = set(offered_zone_ids)
        print(f"{region}: {instance_type} offered AZ IDs: {', '.join(sorted(offered_zone_ids))}", flush=True)

        images = aws_call(
            region,
            "describe-images",
            "--owners",
            "amazon",
            "--filters",
            "Name=name,Values=Deep Learning Base AMI with Single CUDA (Ubuntu 22.04) ????????",
            "Name=state,Values=available",
            "--query",
            "reverse(sort_by(Images, &CreationDate))[:1].ImageId",
        )
        if not images:
            warn(f"{region}: no Deep Learning Base GPU AMI found; skipping region")
            continue
        image_id = images[0]

        subnets = aws_call(
            region,
            "describe-subnets",
            "--filters",
            f"Name=tag:{tag_key},Values=true",
            "--query",
            "Subnets[]",
        )
        if not subnets:
            warn(f"{region}: no tagged subnets found; skipping region")
            continue

        security_groups = aws_call(
            region,
            "describe-security-groups",
            "--filters",
            f"Name=tag:{tag_key},Values=true",
            "--query",
            "SecurityGroups[]",
        )
        groups_by_vpc: dict[str, list[Mapping[str, Any]]] = {}
        for security_group in security_groups or []:
            if not allows_outbound_internet(security_group):
                warn(
                    f"{region}: security group {security_group['GroupId']} "
                    "does not allow outbound traffic to 0.0.0.0/0; skipping security group"
                )
                continue
            groups_by_vpc.setdefault(security_group["VpcId"], []).append(security_group)
        for vpc_groups in groups_by_vpc.values():
            vpc_groups.sort(key=lambda security_group: security_group["GroupId"])

        seen_zone_ids = set()
        eligible_count = 0
        subnets.sort(key=lambda subnet: (subnet.get("AvailabilityZoneId", ""), subnet["SubnetId"]))

        for subnet in subnets:
            subnet_id = subnet["SubnetId"]
            vpc_id = subnet["VpcId"]
            zone_name = subnet.get("AvailabilityZone", "")
            zone_id = subnet.get("AvailabilityZoneId", "")

            if not zone_id:
                warn(f"{region}: subnet {subnet_id} has no AvailabilityZoneId; skipping subnet")
                continue

            if zone_id not in offered_zone_ids:
                warn(
                    f"{region}: {instance_type} is not offered in {zone_name} ({zone_id}); skipping subnet {subnet_id}"
                )
                continue

            if zone_id in seen_zone_ids:
                warn(f"{region}: skipping duplicate subnet {subnet_id} for availability zone ID {zone_id}")
                continue

            vpc_groups = groups_by_vpc.get(vpc_id, [])
            if not vpc_groups:
                warn(f"{region}: subnet {subnet_id} in VPC {vpc_id} has no tagged security group; skipping subnet")
                continue

            security_group_id = vpc_groups[0]["GroupId"]
            seen_zone_ids.add(zone_id)
            eligible_count += 1
            candidates.append(
                {
                    "imageId": image_id,
                    "subnetId": subnet_id,
                    "securityGroupId": security_group_id,
                    "region": region,
                }
            )
            print(
                "Candidate: "
                f"region={region} az={zone_name} az_id={zone_id} "
                f"vpc={vpc_id} subnet={subnet_id} security_group={security_group_id}",
                flush=True,
            )

        if eligible_count == 0:
            warn(f"{region}: no eligible subnet/security-group pairs found")

    return candidates


def set_output(name: str, value: str) -> None:
    """Write a key-value pair to the GitHub Actions step-output file."""
    path = os.environ.get("GITHUB_OUTPUT", "")
    if path:
        with open(path, "a", encoding="utf-8") as output_file:
            output_file.write(f"{name}={value}\n")


def main() -> int:
    """Entry point for the workflow step."""
    regions = os.environ["AWS_REGION_CANDIDATES"].split()
    instance_type = os.environ["AWS_INSTANCE_TYPE"]
    tag_key = os.environ["AWS_RUNNER_RESOURCE_TAG"]

    candidates = discover_candidates(regions, instance_type, tag_key)
    if not candidates:
        error("No eligible EC2 runner candidates were discovered.")
        return 1

    config = json.dumps(candidates, separators=(",", ":"))
    json.loads(config)

    print("Generated availability-zones-config:", flush=True)
    print(json.dumps(candidates, indent=2), flush=True)

    set_output("availability-zones-config", config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
