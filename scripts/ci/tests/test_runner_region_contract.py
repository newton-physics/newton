# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that AWS runner launch regions remain covered by the watchdog."""

import importlib.util
import re
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
DISCOVERY_SCRIPT = ROOT / "scripts" / "ci" / "discover_aws_runner_config.py"
WATCHDOG_TEMPLATE = ROOT / "scripts" / "ci" / "aws" / "overdue-newton-github-runner-watchdog.yaml"
RUNNER_WORKFLOWS = (
    ROOT / ".github" / "workflows" / "aws_gpu_tests.yml",
    ROOT / ".github" / "workflows" / "aws_gpu_benchmarks.yml",
    ROOT / ".github" / "workflows" / "minimum_deps_tests.yml",
    ROOT / ".github" / "workflows" / "warp_nightly_tests.yml",
)

spec = importlib.util.spec_from_file_location("discover_aws_runner_config", DISCOVERY_SCRIPT)
assert spec is not None and spec.loader is not None
discovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(discovery)


def _environment_value(path: Path, name: str) -> str:
    prefix = f"{name}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.removeprefix(prefix).strip()
    raise AssertionError(f"Missing {name} in {path}")


def _workflow_default_regions(path: Path) -> tuple[str, ...]:
    value = _environment_value(path, "AWS_REGION_CANDIDATES")
    if value.startswith("${{"):
        match = re.search(r"\|\| '([^']+)'", value)
        if match is None:
            raise AssertionError(f"Missing fallback region list in {path}")
        value = match.group(1)
    return tuple(value.split())


class TestRunnerRegionContract(unittest.TestCase):
    def test_preserves_supported_region_subsets_in_caller_order(self):
        """Preserve caller ordering when every candidate is supported."""
        self.assertEqual(
            discovery.parse_region_candidates("ap-northeast-2 us-east-1 us-west-2"),
            ["ap-northeast-2", "us-east-1", "us-west-2"],
        )

    def test_rejects_runner_regions_outside_the_watchdog_allowlist(self):
        """Reject unsupported regions before making AWS discovery calls."""

        def unexpected_discovery(*args, **kwargs):
            self.fail("AWS discovery was called for an unsupported region")

        environment = {
            "AWS_REGION_CANDIDATES": "us-east-1 eu-west-1",
            "AWS_INSTANCE_TYPE": "g7e.2xlarge",
            "AWS_RUNNER_RESOURCE_TAG": "newton-github-runner",
        }
        output = StringIO()

        with (
            patch.dict("os.environ", environment, clear=True),
            patch.object(discovery, "discover_candidates", unexpected_discovery),
            redirect_stdout(output),
        ):
            result = discovery.main()

        self.assertEqual(result, 1)
        self.assertIn("eu-west-1", output.getvalue())

    def test_keeps_runner_region_allowlists_synchronized(self):
        """Keep every runner workflow aligned with the watchdog allowlist."""
        self.assertTrue(
            hasattr(discovery, "ALLOWED_REGIONS"),
            "The discovery script must expose its enforced region allowlist",
        )
        allowed_regions = tuple(discovery.ALLOWED_REGIONS)
        watchdog_regions = tuple(_environment_value(WATCHDOG_TEMPLATE, "MONITORED_REGIONS").split(","))

        self.assertEqual(watchdog_regions, allowed_regions)
        for workflow in RUNNER_WORKFLOWS:
            with self.subTest(workflow=workflow.name):
                self.assertEqual(_workflow_default_regions(workflow), allowed_regions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
