# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

import newton
from newton.viewer._standalone import SOLVER_MAP, SimState, _create_parser, load_file

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "assets")


class TestStandaloneViewer(unittest.TestCase):
    """Tests for the standalone viewer file loading and argument parsing."""

    def test_parser_defaults(self):
        """Verify default argument values."""
        args = _create_parser().parse_args([])
        self.assertIsNone(args.file)
        self.assertEqual(args.solver, "mujoco")
        self.assertIsNone(args.device)
        self.assertFalse(args.no_ground)

    def test_parser_with_file(self):
        """Verify file argument is parsed."""
        args = _create_parser().parse_args(["test.usd"])
        self.assertEqual(args.file, "test.usd")

    def test_parser_solver_choices(self):
        """Verify all SOLVER_MAP keys are valid CLI choices."""
        for solver_name in SOLVER_MAP:
            args = _create_parser().parse_args(["--solver", solver_name])
            self.assertEqual(args.solver, solver_name)

    def test_load_file_unsupported_format(self):
        """Verify unsupported formats raise ValueError."""
        with self.assertRaises(ValueError):
            load_file("test.stl")

    def test_load_usd_with_ground(self):
        """Load a USD asset with ground plane and verify SimState."""
        asset_path = os.path.join(_ASSETS_DIR, "cartpole_single_pendulum.usda")
        if not os.path.exists(asset_path):
            self.skipTest(f"Asset not found: {asset_path}")
        sim = load_file(asset_path, solver_name="mujoco", device="cpu", ground=True)
        self.assertIsInstance(sim, SimState)
        self.assertIsNotNone(sim.model)
        self.assertIsNotNone(sim.solver)
        self.assertGreater(sim.model.body_count, 0)
        self.assertGreater(sim.dt, 0.0)
        self.assertIsNone(sim.graph)  # CPU path, no CUDA graph

    def test_load_usd_without_ground(self):
        """Load a USD asset without ground plane."""
        asset_path = os.path.join(_ASSETS_DIR, "cartpole_single_pendulum.usda")
        if not os.path.exists(asset_path):
            self.skipTest(f"Asset not found: {asset_path}")
        sim = load_file(asset_path, solver_name="xpbd", device="cpu", ground=False)
        self.assertIsInstance(sim, SimState)

    def test_solver_map_classes_exist(self):
        """Verify all solver classes in SOLVER_MAP actually exist."""
        for solver_name, cls_name in SOLVER_MAP.items():
            self.assertTrue(
                hasattr(newton.solvers, cls_name),
                f"Solver class {cls_name} for key '{solver_name}' not found in newton.solvers",
            )


if __name__ == "__main__":
    unittest.main()
