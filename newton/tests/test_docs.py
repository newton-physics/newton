# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import newton
from newton._src.documentation import _location


class TestDocumentationInfo(unittest.TestCase):
    def test_discovery_and_command(self):
        """Locate documentation consistently through Python and the CLI."""
        output = StringIO()
        newton.info(output=output)
        self.assertIn(newton.__version__, output.getvalue())
        root, mode = _location()
        self.assertIsNotNone(root)
        self.assertIn(str(root), output.getvalue())
        self.assertIn(mode, output.getvalue())
        result = subprocess.run(
            [sys.executable, "-m", "newton.docs", "--path"], capture_output=True, text=True, check=True
        )
        self.assertEqual(Path(result.stdout.strip()), root)
        if mode == "bundled snapshot":
            module_help = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import newton; newton.info(newton.solvers.experimental.coupled)",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("newton_solvers_experimental_coupled.md", module_help.stdout)

    def test_object_and_symbol_help(self):
        """Describe actual objects and resolve public names without simulation."""
        output = StringIO()
        newton.info(newton.ModelBuilder, output=output)
        self.assertIn("ModelBuilder(", output.getvalue())
        self.assertIn("simulation models", output.getvalue())
        module_output = StringIO()
        newton.info(newton.solvers, output=module_output)
        self.assertTrue(module_output.getvalue().startswith("newton.solvers"))
        with redirect_stdout(StringIO()) as output:
            newton.info("ModelBuilder")
        self.assertIn("ModelBuilder", output.getvalue())
        with self.assertRaisesRegex(ValueError, "No documented public symbol"):
            newton.info("definitely_not_a_newton_symbol")

    def test_qualified_members_and_aliases(self):
        """Resolve members and constants without choosing an arbitrary alias."""
        for name in (
            "newton.ModelBuilder.finalize",
            "ModelBuilder.finalize",
            "__version__",
            "newton.solvers.experimental.coupled.SolverCoupled",
            "SolverCoupled",
            "add_actuator_dcmotor",
        ):
            with self.subTest(name=name):
                output = StringIO()
                newton.info(name, output=output)
                self.assertTrue(output.getvalue())
        with self.assertRaisesRegex(ValueError, "Ambiguous documentation name") as ambiguous:
            newton.info("add_cloth_grid", output=StringIO())
        self.assertIn("newton.ModelBuilder.add_cloth_grid", str(ambiguous.exception))
        self.assertIn("newton.solvers.style3d.add_cloth_grid", str(ambiguous.exception))
        output = StringIO()
        method = newton.selection.DeformableVolumeView.set_mask_from_indices
        newton.info(method, output=output)
        _root, mode = _location()
        if mode == "bundled snapshot":
            self.assertIn("Matching public API references:", output.getvalue())
            self.assertIn("DeformableVolumeView.set_mask_from_indices", output.getvalue())
            self.assertIn("DeformableCurveView.set_mask_from_indices", output.getvalue())
            with self.assertRaisesRegex(ValueError, "Ambiguous documentation name"):
                newton.info("set_mask_from_indices", output=output)


if __name__ == "__main__":
    unittest.main()
