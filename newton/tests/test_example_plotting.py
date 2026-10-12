# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from newton.examples.basic.example_basic_plotting import Example


class TestExamplePlotting(unittest.TestCase):
    def test_save_plot_without_gui(self):
        """Save a diagnostic PNG without creating a pyplot figure manager."""
        try:
            import matplotlib
            import matplotlib.image as image
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("Requires matplotlib")

        # Plot recorded data without building a solver or running a simulation.
        example = Example.__new__(Example)
        example.frame_dt = 1.0 / 60.0
        example.log_iterations = [1.0, 2.0, 2.0]
        example.log_energy_kinetic = [0.0, 1.0, 2.0]
        example.log_energy_potential = [3.0, 2.0, 1.0]
        example.log_nefc = [0.0, 6.0, 6.0]
        backend = matplotlib.get_backend(auto_select=False)

        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(io.StringIO()),
            mock.patch.object(
                plt,
                "new_figure_manager",
                side_effect=AssertionError("Saving a PNG must not initialize a GUI backend"),
            ),
        ):
            previous_cwd = os.getcwd()
            try:
                os.chdir(directory)
                example._plot()
            finally:
                os.chdir(previous_cwd)

            pixels = image.imread(Path(directory) / "solver_convergence.png")
            self.assertEqual(pixels.shape[:2], (1200, 1500))
            self.assertLess(pixels.min(), pixels.max())
            self.assertEqual(matplotlib.get_backend(auto_select=False), backend)


if __name__ == "__main__":
    unittest.main()
