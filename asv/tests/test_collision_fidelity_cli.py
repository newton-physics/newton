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
            output = Path(temporary) / "results"
            for options in (("--case", "typo"), ("--all", "--duration", "nan")):
                with self.subTest(options=options):
                    result = subprocess.run(
                        [sys.executable, str(CLI), *options, "--out", str(output)],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
