# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ci" / "warp_cpu_target.py"

spec = importlib.util.spec_from_file_location("warp_cpu_target", SCRIPT)
assert spec is not None and spec.loader is not None
warp_cpu_target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warp_cpu_target)


class TestWarpCpuTarget(unittest.TestCase):
    def _run(self, cpu_target):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            env = {"GITHUB_OUTPUT": str(output), "GITHUB_STEP_SUMMARY": str(Path(tmp) / "summary")}
            with (
                patch.dict(os.environ, env),
                patch.object(warp_cpu_target, "cpu_target", cpu_target),
                redirect_stdout(StringIO()) as stdout,
            ):
                self.assertEqual(warp_cpu_target.main(), 0)
            return output.read_text(), stdout.getvalue()

    def test_writes_target_output(self):
        output, _ = self._run(lambda: ("8dc95227", "znver3"))
        self.assertEqual(output, "target=8dc95227\n")

    def test_falls_back_to_shared_key_on_error(self):
        def fail():
            raise ImportError("no warp")

        output, stdout = self._run(fail)
        self.assertEqual(output, "target=unknown\n")
        self.assertIn("::warning::", stdout)


if __name__ == "__main__":
    unittest.main()
