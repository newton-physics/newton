# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check standalone selection without constructing a physics workload."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[2]
CLI = ROOT / "asv" / "run_collision_fidelity.py"


class TestFidelityCLI(unittest.TestCase):
    def test_list_and_unsupported_case(self):
        """List all cases and report a second-solver recipe without launching it."""
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        listed = subprocess.run(
            [sys.executable, str(CLI), "--list", "--json"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(listed.returncode, 0, listed.stderr)
        cases = json.loads(listed.stdout)
        self.assertEqual(len(cases), 45)
        self.assertEqual(len({case["case_id"] for case in cases}), 45)
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, str(CLI), "--case", "r2_franka_duck", "--out", temporary],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads((Path(temporary) / "r2_franka_duck.json").read_text())
            self.assertEqual(output["execution"]["status"], "unsupported")
            self.assertEqual(output["execution"]["steps"], 0)
            self.assertNotIn("Warp", result.stdout)

    def test_invalid_selection_has_no_output(self):
        """Reject unknown cases and invalid durations before creating results."""
        with tempfile.TemporaryDirectory() as temporary:
            for index, options in enumerate(
                (
                    ("--case", "typo"),
                    ("--all", "--duration", "nan"),
                    ("--case", "r2_franka_duck", "--json"),
                    ("--case", "r2_franka_duck", "--duration", "1e-20"),
                )
            ):
                with self.subTest(options=options):
                    output = Path(temporary) / str(index)
                    result = subprocess.run(
                        [sys.executable, str(CLI), *options, "--out", str(output)],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertFalse(output.exists())

    def test_existing_usd_is_preserved(self):
        """Reject an output namespace containing the requested USD recording."""
        with tempfile.TemporaryDirectory() as temporary:
            usd = Path(temporary) / "r2_franka_duck.usd"
            usd.write_text("previous recording")
            result = subprocess.run(
                [sys.executable, str(CLI), "--case", "r2_franka_duck", "--usd", "--out", temporary],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(usd.read_text(), "previous recording")
            self.assertFalse((Path(temporary) / "summary.json").exists())

    def test_claimed_namespace_is_not_reused_before_results_exist(self):
        """Respect an atomic run claim even before its first result is written."""
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / ".collision-fidelity-run").mkdir()
            result = subprocess.run(
                [sys.executable, str(CLI), "--case", "r2_franka_duck", "--out", temporary],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse((Path(temporary) / "summary.json").exists())

    def test_empty_directory_is_claimed(self):
        """Reserve a fresh namespace before recording an unsupported case."""
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, str(CLI), "--case", "r2_franka_duck", "--out", temporary],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(temporary) / ".collision-fidelity-run").is_dir())


if __name__ == "__main__":
    unittest.main()
