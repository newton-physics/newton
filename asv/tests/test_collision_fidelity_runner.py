# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check runner validation and failure reporting without advancing physics."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))

from collision_fidelity import runner
from collision_fidelity.catalog import get_case
from collision_fidelity.protocol import Timeline


class TestFidelityRunner(unittest.TestCase):
    def test_duration_shorter_than_substep_is_rejected(self):
        """Reject a zero-step or overshooting duration before scene construction."""
        dt = Timeline(get_case("r2_franka_duck")).dt
        for duration in (1e-20, 0.5 * dt):
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, "substep"):
                runner.run_case("r2_franka_duck", duration=duration)

    def test_revision_is_optional_when_git_is_not_installed(self):
        """Keep optional source provenance from preventing execution without git."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            module = SimpleNamespace(__file__=str(root / "package" / "__init__.py"))
            with patch.dict(os.environ, {"PATH": temporary}):
                self.assertIsNone(runner._revision(module))

    def test_usd_close_error_keeps_execution_evidence(self):
        """Contain a real USD save failure and preserve a prior physics error."""
        try:
            from pxr import Usd  # noqa: PLC0415, F401

            from newton.viewer import ViewerUSD  # noqa: PLC0415
        except ImportError:
            self.skipTest("USD recording requires the optional usd-core package")
        close = getattr(runner, "_close_viewer", None)
        self.assertTrue(callable(close), "The runner must contain viewer.close failures")
        with tempfile.TemporaryDirectory() as temporary:
            for status in ("completed", "nonfinite"):
                with self.subTest(status=status):
                    viewer = ViewerUSD(str(Path(temporary) / f"{status}.usda"))
                    viewer.stage.GetRootLayer().SetPermissionToSave(False)
                    execution = {"status": status, "completed": status == "completed", "steps": 7}
                    if status == "nonfinite":
                        execution["reason"] = "Nonfinite state observed"
                    close(viewer, execution)
                    self.assertEqual(execution["steps"], 7)
                    self.assertIn("usd_export_error", execution)
                    self.assertEqual(execution["status"], "error" if status == "completed" else "nonfinite")
                    if status == "nonfinite":
                        self.assertEqual(execution["reason"], "Nonfinite state observed")
                    json.dumps(execution, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
