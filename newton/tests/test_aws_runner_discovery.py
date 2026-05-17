# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "discover_aws_runner_config.py"
_SPEC = importlib.util.spec_from_file_location("discover_aws_runner_config", _SCRIPT_PATH)
discover_aws_runner_config = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(discover_aws_runner_config)


class TestAwsRunnerDiscovery(unittest.TestCase):
    def test_discovers_supported_candidates_and_skips_bad_subnets(self):
        warnings = []

        def aws(region, *args):
            command = args[0]
            self.assertEqual(region, "us-east-1")
            if command == "describe-instance-type-offerings":
                return ["use1-az2"]
            if command == "describe-images":
                return ["ami-123"]
            if command == "describe-subnets":
                return [
                    {
                        "SubnetId": "subnet-supported-b",
                        "VpcId": "vpc-1",
                        "AvailabilityZone": "us-east-1a",
                        "AvailabilityZoneId": "use1-az2",
                    },
                    {
                        "SubnetId": "subnet-supported-a",
                        "VpcId": "vpc-1",
                        "AvailabilityZone": "us-east-1a",
                        "AvailabilityZoneId": "use1-az2",
                    },
                    {
                        "SubnetId": "subnet-unsupported",
                        "VpcId": "vpc-1",
                        "AvailabilityZone": "us-east-1f",
                        "AvailabilityZoneId": "use1-az5",
                    },
                ]
            if command == "describe-security-groups":
                return [
                    {
                        "GroupId": "sg-2",
                        "VpcId": "vpc-1",
                        "IpPermissionsEgress": [{"IpRanges": [{"CidrIp": "0.0.0.0/0"}]}],
                    },
                    {
                        "GroupId": "sg-1",
                        "VpcId": "vpc-1",
                        "IpPermissionsEgress": [{"IpRanges": [{"CidrIp": "0.0.0.0/0"}]}],
                    },
                ]
            raise AssertionError(f"unexpected AWS command: {command}")

        candidates = discover_aws_runner_config.discover_candidates(
            ["us-east-1"],
            "g7e.12xlarge",
            "newton-github-runner",
            aws_call=aws,
            warn=warnings.append,
        )

        self.assertEqual(
            candidates,
            [
                {
                    "imageId": "ami-123",
                    "subnetId": "subnet-supported-a",
                    "securityGroupId": "sg-1",
                    "region": "us-east-1",
                }
            ],
        )
        self.assertTrue(any("use1-az5" in warning for warning in warnings))
        self.assertTrue(any("duplicate subnet subnet-supported-b" in warning for warning in warnings))

    def test_rejects_security_groups_without_internet_egress(self):
        warnings = []

        def aws(region, *args):
            command = args[0]
            if command == "describe-instance-type-offerings":
                return ["use1-az2"]
            if command == "describe-images":
                return ["ami-123"]
            if command == "describe-subnets":
                return [
                    {
                        "SubnetId": "subnet-supported",
                        "VpcId": "vpc-1",
                        "AvailabilityZone": "us-east-1a",
                        "AvailabilityZoneId": "use1-az2",
                    }
                ]
            if command == "describe-security-groups":
                return [
                    {
                        "GroupId": "sg-no-egress",
                        "VpcId": "vpc-1",
                        "IpPermissionsEgress": [{"IpRanges": [{"CidrIp": "10.0.0.0/8"}]}],
                    }
                ]
            raise AssertionError(f"unexpected AWS command: {command}")

        candidates = discover_aws_runner_config.discover_candidates(
            ["us-east-1"],
            "g7e.12xlarge",
            "newton-github-runner",
            aws_call=aws,
            warn=warnings.append,
        )

        self.assertEqual(candidates, [])
        self.assertTrue(any("does not allow outbound traffic" in warning for warning in warnings))
        self.assertTrue(any("no tagged security group" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
