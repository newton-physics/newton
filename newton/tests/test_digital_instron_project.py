# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Digital Instron geometry and inputs."""

import json
import unittest
from itertools import pairwise
from pathlib import Path

import numpy as np

from projects.digital_instron_v2.core import Material, metrics, predict
from projects.digital_instron_v2.geometry import build_column_grid, load_mesh, raycast_surface, transform_mesh
from projects.digital_instron_v2.workflow import prepare_trials


class TestDigitalInstronProject(unittest.TestCase):
    def test_project_inputs_exist(self):
        """Resolve every input named by the manifest."""

        base = Path("DigitalInstron")
        config = json.loads((base / "manifest_v2.json").read_text())
        paths = [base / config["midsole_mesh"]]
        paths += [base / trial["averaged_cycle_path"] for trial in config["trials"]]
        paths += [base / trial["indenter"]["path"] for trial in config["trials"] if "path" in trial["indenter"]]
        self.assertTrue(all(path.exists() for path in paths))

    def test_raycast_box_thickness(self):
        """Recover uniform thickness from a box mesh."""

        import trimesh

        mesh = trimesh.creation.box(extents=(0.02, 0.02, 0.01))
        grid = build_column_grid(mesh, 0.005)
        np.testing.assert_allclose(grid.slack_m, 0.01)

    def test_midsole_grid_has_no_internal_holes(self):
        """Create columns continuously across the interior midsole footprint."""

        mesh = load_mesh("DigitalInstron/puma-fast-r-nitro-elite-3-3d-internal-wt-LR.obj", 0.001)
        grid = build_column_grid(mesh, 0.005)
        cells = {tuple(np.rint(point / 0.005).astype(int)) for point in grid.uv_m}
        internal_holes = 0
        for u in range(min(cell[0] for cell in cells), max(cell[0] for cell in cells) + 1):
            for v in range(min(cell[1] for cell in cells), max(cell[1] for cell in cells) + 1):
                if (u, v) in cells:
                    continue
                neighbors = sum((u + du, v + dv) in cells for du, dv in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                internal_holes += neighbors >= 3

        self.assertEqual(internal_holes, 0)

    def test_load_project_mesh(self):
        """Load the source midsole mesh."""

        mesh = load_mesh("DigitalInstron/puma-fast-r-nitro-elite-3-3d-internal-wt-LR.obj", 0.001)
        self.assertGreater(len(mesh.faces), 0)

    def test_condition_fullfoot_fixture(self):
        """Recover the physical shoe-last fixture from the source STL."""

        mesh = load_mesh(
            "DigitalInstron/Instron Shoe Last Size 9 6drop merged attachment 1 left.STL",
            0.001,
            [90.0, 0.0, 0.0],
            0.08,
        )
        np.testing.assert_allclose(mesh.extents, [0.24168859, 0.08105086, 0.04080122], rtol=1.0e-5)
        self.assertEqual(len(mesh.faces), 15853)
        self.assertAlmostEqual(float(np.mean(mesh.bounds[:, 0])), 0.0)

    def test_left_shoe_last_mirrors_source(self):
        """Mirror the source shoe last laterally while preserving its crop and topology."""

        right = load_mesh(
            "DigitalInstron/Instron Shoe Last Size 9 6drop merged attachment 1.STL", 0.001, [90.0, 0.0, 0.0], 0.08
        )
        left = load_mesh(
            "DigitalInstron/Instron Shoe Last Size 9 6drop merged attachment 1 left.STL",
            0.001,
            [90.0, 0.0, 0.0],
            0.08,
        )

        self.assertEqual(len(left.faces), len(right.faces))
        np.testing.assert_allclose(left.extents, right.extents)
        np.testing.assert_allclose(left.bounds[:, 1], -right.bounds[::-1, 1])

    def test_raycast_open_contact_surface(self):
        """Retain single-hit rays regardless of STL winding."""

        import trimesh

        vertices = np.array([[-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
        mesh = trimesh.Trimesh(vertices, [[0, 1, 2], [0, 2, 3]], process=False)
        surface = raycast_surface(mesh, np.array([[0.0, 0.0]]), 2, "near")

        np.testing.assert_allclose(surface, 0.0)

    def test_select_near_side_of_shoe_last(self):
        """Select the sole-facing side rather than the upper shell."""

        base = Path("DigitalInstron")
        config = json.loads((base / "manifest_v2.json").read_text())
        source = next(trial for trial in config["trials"] if trial["fixture"] == "fullfoot_last")
        mesh = load_mesh(
            base / source["indenter"]["path"],
            0.001,
            source["indenter"]["rotation_deg"],
            source["indenter"]["crop_height_m"],
        )
        u = np.linspace(mesh.bounds[0, 0], mesh.bounds[1, 0], 40)
        v = np.linspace(mesh.bounds[0, 1], mesh.bounds[1, 1], 20)
        uu, vv = np.meshgrid(u, v)
        uv = np.column_stack((uu.ravel(), vv.ravel()))
        near = raycast_surface(mesh, uv, 2, "near")
        far = raycast_surface(mesh, uv, 2, "far")
        double_hit = np.isfinite(near) & np.isfinite(far) & (far > near)

        self.assertGreater(np.count_nonzero(double_hit), 100)
        self.assertGreater(float(np.median(far[double_hit] - near[double_hit])), 0.015)

    def test_level_fullfoot_last_against_midsole(self):
        """Remove longitudinal and lateral clearance ramps from the full-foot pose."""

        base = Path("DigitalInstron")
        config = json.loads((base / "manifest_v2.json").read_text())
        source = next(trial for trial in config["trials"] if trial["fixture"] == "fullfoot_last")
        midsole = load_mesh(base / config["midsole_mesh"], 0.001)
        grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])
        last = load_mesh(
            base / source["indenter"]["path"],
            0.001,
            source["indenter"]["rotation_deg"],
            source["indenter"]["crop_height_m"],
        )
        transform_mesh(last, source["indenter"]["pose_rotation_deg"], source["indenter"]["pose_translation_m"])
        surface = raycast_surface(last, grid.uv_m, grid.thickness_axis, source["indenter"]["contact_side"])
        active = np.isfinite(surface)
        clearance = grid.top_m[active] - surface[active]
        plane = np.linalg.lstsq(
            np.column_stack((np.ones(np.count_nonzero(active)), grid.uv_m[active])), clearance, rcond=None
        )[0]

        np.testing.assert_allclose(plane[1:], 0.0, atol=1.0e-3)

    def test_report_perfect_metrics(self):
        """Report zero errors for a perfect force trace."""

        displacement = np.array([0.0, 0.001, 0.002, 0.001, 0.0])
        force = np.array([0.0, 10.0, 20.0, 10.0, 0.0])
        result = metrics(force, force, displacement)
        self.assertEqual(result["force_rmse_relative"], 0.0)
        self.assertEqual(result["peak_force_error"], 0.0)

    def test_shared_pointwise_fit_matches_both_branches(self):
        """Keep loading and unloading RMSE below ten percent for both fixtures."""

        base = Path("DigitalInstron")
        config = json.loads((base / "manifest_v2.json").read_text())
        midsole = load_mesh(base / config["midsole_mesh"], 0.001)
        grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])
        trials, displacement, uv = prepare_trials(base, config, grid, midsole)
        material = Material(19033.8644, 5.1303377966, 0.1055112900, 918.1319626)

        for trial in trials:
            result = metrics(trial.force_n, predict(trial, material), displacement[trial.name])
            self.assertLess(result["loading_rmse_relative"], 0.10, f"{trial.name}: {result}")
            self.assertLess(result["unloading_rmse_relative"], 0.10, f"{trial.name}: {result}")

        fullfoot = next(trial for trial in trials if trial.name == "fullfoot_185ms")
        initial_active = np.count_nonzero(fullfoot.slack_m - fullfoot.lengths_m[0] > 0.0)
        self.assertGreaterEqual(initial_active, 20)
        self.assertLessEqual(initial_active, 40)
        peak_frame = int(np.argmax(fullfoot.displacement_m))
        compression = np.maximum(fullfoot.slack_m - fullfoot.lengths_m[peak_frame], 0.0)
        centroid_x = float(np.average(uv[fullfoot.name][:, 0], weights=compression))
        self.assertLess(abs(centroid_x), 0.01)

        edges = np.linspace(np.min(uv[fullfoot.name][:, 0]), np.max(uv[fullfoot.name][:, 0]), 5)
        quartile_compression = np.asarray(
            [
                np.sum(compression[(uv[fullfoot.name][:, 0] >= lower) & (uv[fullfoot.name][:, 0] < upper)])
                for lower, upper in pairwise(edges)
            ]
        )
        self.assertLess(float(np.max(quartile_compression) / np.min(quartile_compression)), 1.5)


if __name__ == "__main__":
    unittest.main()
