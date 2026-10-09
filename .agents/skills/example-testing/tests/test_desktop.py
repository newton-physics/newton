# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test desktop input cleanup without a display or capture dependencies."""

import runpy
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class TestDesktopCleanup(unittest.TestCase):
    def test_release_inputs_after_command_failure(self):
        """Attempt every input release even when an input command fails."""
        script = Path(__file__).parents[1] / "scripts" / "desktop.py"
        fake_mss = types.ModuleType("mss")
        fake_mss.tools = types.ModuleType("mss.tools")
        for modifier in (None, "Shift_L", "Control_L"):
            for failure in ("keydown", "mousedown", "mouseup", None):
                if modifier is None and failure == "keydown":
                    continue
                with self.subTest(modifier=modifier, failure=failure):
                    commands = []

                    def run(command, *, check, commands=commands, failure=failure):
                        commands.append(command)
                        if command[1] == failure:
                            raise subprocess.CalledProcessError(1, command)
                        return subprocess.CompletedProcess(command, 0)

                    argv = [str(script), "drag", "123", "2", "10", "20", "30", "40"]
                    if modifier:
                        argv.extend(["--modifier", modifier])
                    with (
                        mock.patch.object(sys, "argv", argv),
                        mock.patch.dict(sys.modules, {"mss": fake_mss, "mss.tools": fake_mss.tools}),
                        mock.patch("subprocess.run", side_effect=run),
                        mock.patch("time.sleep"),
                    ):
                        if failure:
                            with self.assertRaises(subprocess.CalledProcessError):
                                runpy.run_path(str(script), run_name="__main__")
                        else:
                            runpy.run_path(str(script), run_name="__main__")

                    self.assertIn(["xdotool", "mouseup", "2"], commands)
                    if modifier:
                        self.assertEqual(commands[-1], ["xdotool", "keyup", modifier])


if __name__ == "__main__":
    unittest.main()
