# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.digital_instron_v2 import (
    FoundationMaterial,
    FrameQCError,
    build_raycast_spring_grid,
    condition_midsole_mesh,
    detect_mesh_frame,
    evaluate_foundation,
    evaluate_foundation_lengths,
    fit_foundation_material_autodiff,
    foundation_lengths_loss_gradient,
    infer_frame_config,
    load_manifest,
    load_trial_frame,
    make_cylinder_grid,
    raycast_grid_thickness,
    rearfoot_punch_center_uv,
    validate_frame_config,
)
from projects.digital_instron_v2.foundation import FoundationFitSample, finite_difference_loss_gradient, warp_loss_gradient
from projects.digital_instron_v2.mujoco_adapter import apply_foundation_wrench_to_body_f


def _write_box_obj(path: Path, *, size=(0.12, 0.05, 0.03), omit_top: bool = False) -> None:
    sx, sy, sz = size
    vertices = [
        (-sx / 2, -sy / 2, -sz / 2),
        (sx / 2, -sy / 2, -sz / 2),
        (sx / 2, sy / 2, -sz / 2),
        (-sx / 2, sy / 2, -sz / 2),
        (-sx / 2, -sy / 2, sz / 2),
        (sx / 2, -sy / 2, sz / 2),
        (sx / 2, sy / 2, sz / 2),
        (-sx / 2, sy / 2, sz / 2),
    ]
    faces = [
        (1, 2, 3),
        (1, 3, 4),
        (5, 8, 7),
        (5, 7, 6),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 8),
        (3, 8, 4),
        (4, 8, 5),
        (4, 5, 1),
    ]
    if omit_top:
        faces = faces[:2] + faces[4:]
    lines = [f"v {x} {y} {z}" for x, y, z in vertices]
    lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n")


class TestDigitalInstronV2Geometry(unittest.TestCase):
    def test_cylinder_grid_uses_requested_spacing(self):
        grid = make_cylinder_grid(radius_m=0.01, spacing_m=0.005)

        self.assertEqual(grid.xy_m.shape[1], 2)
        self.assertGreater(len(grid.xy_m), 0)
        self.assertAlmostEqual(grid.cell_area_m2, 0.005 * 0.005)
        self.assertLessEqual(float(np.max(np.linalg.norm(grid.xy_m, axis=1))), 0.010000001)

    def test_midsole_mesh_qc_accepts_watertight_synthetic_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh_path = root / "box.obj"
            _write_box_obj(mesh_path)

            report = condition_midsole_mesh(mesh_path, root / "cache", source_units="m", remesh=False)

            self.assertTrue(report["input_watertight"])
            self.assertTrue(report["repaired_watertight"])
            self.assertTrue(Path(str(report["repaired_mesh"])).exists())
            self.assertAlmostEqual(float(report["thickness_m"]), 0.03)

    def test_midsole_mesh_qc_fails_loudly_on_bad_thickness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh_path = root / "thin.obj"
            _write_box_obj(mesh_path, size=(0.12, 0.05, 0.001))

            with self.assertRaisesRegex(Exception, "thickness"):
                condition_midsole_mesh(mesh_path, root / "cache", source_units="m", remesh=False)

    def test_detect_mesh_frame_and_raycast_thickness(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = Path(tmp) / "box.obj"
            _write_box_obj(mesh_path, size=(0.12, 0.05, 0.03))
            mesh_report = condition_midsole_mesh(
                mesh_path,
                Path(tmp) / "cache",
                source_units="m",
                remesh=False,
            )

            from projects.digital_instron_v2.geometry import _load_obj_mesh

            loaded_vertices, loaded_faces = _load_obj_mesh(Path(str(mesh_report["repaired_mesh"])))
            frame = detect_mesh_frame(loaded_vertices)
            grid = make_cylinder_grid(radius_m=0.01, spacing_m=0.005)
            grid_uv = grid.xy_m + frame.center_m[list(frame.plane_axes)]
            ray = raycast_grid_thickness(loaded_vertices, loaded_faces, grid_uv, frame=frame)
            spring_grid = build_raycast_spring_grid(loaded_vertices, loaded_faces, spacing_m=0.02)

            self.assertEqual(frame.thickness_axis, 2)
            self.assertEqual(int(np.count_nonzero(np.isfinite(ray["thickness_m"]))), len(grid.xy_m))
            np.testing.assert_allclose(ray["thickness_m"], np.full(len(grid.xy_m), 0.03), atol=1.0e-9)
            self.assertGreater(len(spring_grid.xy_m), len(grid.xy_m))
            np.testing.assert_allclose(spring_grid.slack_length_m, np.full(len(spring_grid.xy_m), 0.03), atol=1.0e-9)

    def test_rearfoot_punch_center_uses_heel_side_length_fraction(self):
        vertices = np.asarray(
            [
                [-0.05, -0.01, -0.15],
                [0.05, -0.01, -0.15],
                [0.05, 0.01, 0.15],
                [-0.05, 0.01, 0.15],
            ],
            dtype=np.float64,
        )
        frame = detect_mesh_frame(vertices)

        heel_min = rearfoot_punch_center_uv(vertices, frame, heel_side="min", length_fraction=0.25)
        heel_max = rearfoot_punch_center_uv(vertices, frame, heel_side="max", length_fraction=0.25)

        self.assertEqual(frame.plane_axes, (0, 2))
        np.testing.assert_allclose(heel_min, np.asarray([0.0, -0.075]))
        np.testing.assert_allclose(heel_max, np.asarray([0.0, 0.075]))

    def test_rearfoot_punch_center_uses_local_lateral_slice(self):
        rear_vertices = np.asarray(
            [
                [-0.03, -0.01, 0.00],
                [0.03, -0.01, 0.00],
                [-0.03, 0.01, 0.04],
                [0.03, 0.01, 0.04],
                [-0.08, -0.01, 0.20],
                [0.04, -0.01, 0.20],
                [-0.08, 0.01, 0.24],
                [0.04, 0.01, 0.24],
            ],
            dtype=np.float64,
        )
        frame = detect_mesh_frame(rear_vertices)

        center = rearfoot_punch_center_uv(
            rear_vertices,
            frame,
            heel_side="min",
            length_fraction=0.1,
            lateral_fraction=0.5,
            lateral_band_fraction=0.3,
        )

        np.testing.assert_allclose(center, np.asarray([0.0, 0.024]))


class TestDigitalInstronV2Foundation(unittest.TestCase):
    def test_foundation_force_loss_and_gradients_are_differentiable(self):
        grid = make_cylinder_grid(radius_m=0.01, spacing_m=0.005)
        material = FoundationMaterial(
            stiffness_pa=2.0e6,
            ogden_alpha=2.0,
            lock_strain=0.65,
            damping_pa_s=1.0e4,
        )
        compression = np.full(len(grid.xy_m), 0.004)
        velocity = np.full(len(grid.xy_m), 0.01)

        result = evaluate_foundation(
            grid.xy_m,
            compression,
            velocity,
            cell_area_m2=grid.cell_area_m2,
            thickness_m=0.03,
            material=material,
            measured_force_n=40.0,
        )
        grad = warp_loss_gradient(
            grid.xy_m,
            compression,
            velocity,
            cell_area_m2=grid.cell_area_m2,
            thickness_m=0.03,
            material=material,
            measured_force_n=40.0,
        )
        fd_stiffness = finite_difference_loss_gradient(
            grid.xy_m,
            compression,
            velocity,
            cell_area_m2=grid.cell_area_m2,
            thickness_m=0.03,
            material=material,
            measured_force_n=40.0,
            stiffness_eps=100.0,
        )

        self.assertGreater(result.force_n, 0.0)
        self.assertGreater(result.loss, 0.0)
        self.assertTrue(np.all(np.isfinite(grad)))
        self.assertAlmostEqual(float(grad[0]), fd_stiffness, delta=max(1.0e-3, abs(fd_stiffness) * 0.05))

    def test_foundation_aggregates_net_wrench(self):
        xy = np.asarray([[0.01, 0.0], [-0.01, 0.0]], dtype=np.float64)
        material = FoundationMaterial(1.0e6, 2.0, 0.65, 0.0)

        balanced = evaluate_foundation(
            xy,
            np.asarray([0.003, 0.003]),
            np.zeros(2),
            cell_area_m2=1.0e-4,
            thickness_m=0.03,
            material=material,
        )
        unbalanced = evaluate_foundation(
            xy,
            np.asarray([0.004, 0.002]),
            np.zeros(2),
            cell_area_m2=1.0e-4,
            thickness_m=0.03,
            material=material,
        )

        self.assertAlmostEqual(float(balanced.wrench[4]), 0.0, places=5)
        self.assertNotAlmostEqual(float(unbalanced.wrench[4]), 0.0)

    def test_foundation_uses_raycast_slack_lengths(self):
        xy = np.asarray([[0.0, 0.0], [0.01, 0.0]], dtype=np.float64)
        slack = np.asarray([0.03, 0.06], dtype=np.float64)
        material = FoundationMaterial(1.0e6, 2.0, 0.65, 0.0)

        no_compression = evaluate_foundation_lengths(
            xy,
            slack.copy(),
            slack,
            np.zeros(2),
            cell_area_m2=1.0e-4,
            material=material,
        )
        compressed = evaluate_foundation_lengths(
            xy,
            slack - np.asarray([0.003, 0.006]),
            slack,
            np.zeros(2),
            cell_area_m2=1.0e-4,
            material=material,
        )

        self.assertAlmostEqual(no_compression.force_n, 0.0)
        self.assertGreater(compressed.force_n, 0.0)

    def test_foundation_lengths_gradient_matches_finite_difference(self):
        xy = np.asarray([[0.0, 0.0], [0.01, 0.0]], dtype=np.float64)
        slack = np.asarray([0.03, 0.04], dtype=np.float64)
        current = slack - np.asarray([0.003, 0.004], dtype=np.float64)
        velocity = np.zeros(2, dtype=np.float64)
        material = FoundationMaterial(1.0e6, 2.0, 0.65, 0.0)

        grad = foundation_lengths_loss_gradient(
            xy,
            current,
            slack,
            velocity,
            cell_area_m2=1.0e-4,
            material=material,
            measured_force_n=20.0,
        ).gradient
        eps = 100.0
        plus = FoundationMaterial(material.stiffness_pa + eps, 2.0, 0.65, 0.0)
        minus = FoundationMaterial(material.stiffness_pa - eps, 2.0, 0.65, 0.0)
        loss_plus = evaluate_foundation_lengths(
            xy, current, slack, velocity, cell_area_m2=1.0e-4, material=plus, measured_force_n=20.0
        ).loss
        loss_minus = evaluate_foundation_lengths(
            xy, current, slack, velocity, cell_area_m2=1.0e-4, material=minus, measured_force_n=20.0
        ).loss

        self.assertAlmostEqual(float(grad[0]), (loss_plus - loss_minus) / (2.0 * eps), delta=1.0e-3)

    def test_autodiff_fit_reduces_synthetic_loss(self):
        xy = np.asarray([[0.0, 0.0], [0.01, 0.0]], dtype=np.float64)
        slack = np.asarray([0.03, 0.04], dtype=np.float64)
        velocity = np.zeros(2, dtype=np.float64)
        true_material = FoundationMaterial(2.0e6, 2.0, 0.65, 0.0)
        initial = FoundationMaterial(8.0e5, 2.0, 0.65, 0.0)
        samples = []
        for compression in (0.002, 0.004, 0.006):
            current = slack - compression
            measured = evaluate_foundation_lengths(
                xy, current, slack, velocity, cell_area_m2=1.0e-4, material=true_material
            ).force_n
            samples.append(FoundationFitSample(current, slack, velocity, measured))

        initial_loss = sum(
            evaluate_foundation_lengths(
                xy,
                sample.current_length_m,
                sample.slack_length_m,
                sample.velocity_mps,
                cell_area_m2=1.0e-4,
                material=initial,
                measured_force_n=sample.measured_force_n,
            ).loss
            for sample in samples
        )
        result = fit_foundation_material_autodiff(
            xy,
            samples,
            cell_area_m2=1.0e-4,
            initial_material=initial,
            iterations=12,
            learning_rates=(0.1, 0.0, 0.0, 0.0, 0.0),
        )
        final_loss = sum(
            evaluate_foundation_lengths(
                xy,
                sample.current_length_m,
                sample.slack_length_m,
                sample.velocity_mps,
                cell_area_m2=1.0e-4,
                material=result.material,
                measured_force_n=sample.measured_force_n,
            ).loss
            for sample in samples
        )

        self.assertLess(final_loss, initial_loss)
        self.assertGreater(result.material.stiffness_pa, initial.stiffness_pa)

    def test_mujoco_adapter_adds_wrench_to_body_force(self):
        body_f = np.zeros((2, 6), dtype=np.float64)
        wrench = np.asarray([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

        apply_foundation_wrench_to_body_f(body_f, 1, wrench)

        np.testing.assert_allclose(body_f[1], wrench)
        np.testing.assert_allclose(body_f[0], np.zeros(6))


class TestDigitalInstronV2ManifestAndFrames(unittest.TestCase):
    def test_manifest_parsing_resolves_paths_and_fit_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "trial.csv"
            csv_path.write_text('"Total Time (s)","Position (mm)","Force (N)"\n0,0,0\n1,2,-100\n')
            mesh_path = root / "mesh.obj"
            _write_box_obj(mesh_path)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "midsole_mesh": "mesh.obj",
                        "cache_dir": "cache",
                        "qc": {},
                        "grid": {"coarse_spacing_m": 0.005},
                        "fit": {"model": "locked_ogden_vertical_foundation_v2"},
                        "trials": [
                            {
                                "name": "rearfoot",
                                "csv_path": "trial.csv",
                                "fixture": "rearfoot_punch",
                                "include_in_fit": True,
                                "indenter": {"type": "cylinder", "radius_m": 0.045},
                            }
                        ],
                    }
                )
            )

            manifest = load_manifest(manifest_path)

            self.assertEqual(manifest.trials[0].name, "rearfoot")
            self.assertTrue(manifest.midsole_mesh.is_absolute())
            self.assertTrue(manifest.trials[0].csv_path.exists())

    def test_frame_qc_infers_and_applies_saved_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "trace.csv"
            csv_path.write_text(
                '"Total Time (s)","Position (mm)","Force (N)"\n'
                "0,10,-5\n"
                "0.1,8,-80\n"
                "0.2,6,-160\n"
            )

            frame = infer_frame_config(csv_path, min_force_span_n=50.0, min_position_span_mm=1.0)
            trace = load_trial_frame(csv_path, frame)

            self.assertEqual(frame.force_sign, -1.0)
            self.assertGreater(float(trace["force_n"][-1]), 0.0)
            self.assertGreater(float(trace["displacement_m"][-1]), 0.0)

    def test_frame_qc_rejects_missing_columns_and_bad_saved_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "trace.csv"
            csv_path.write_text('"Time","Load"\n0,1\n1,2\n')

            with self.assertRaises(FrameQCError):
                infer_frame_config(csv_path)
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_frame_config({"time_column": "Time"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
