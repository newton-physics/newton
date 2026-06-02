# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from projects.digital_instron_v2 import (
    FoundationMaterial,
    FoundationResult,
    FrameQCError,
    BakedMidsoleGeometry,
    build_baked_midsole_geometry,
    evaluate_foundation_baked,
    fit_foundation_material_baked_batches_autodiff,
    build_raycast_spring_grid,
    condition_midsole_mesh,
    detect_mesh_frame,
    evaluate_foundation,
    evaluate_foundation_lengths,
    evaluate_foundation_sdf,
    fit_foundation_material_autodiff,
    foundation_lengths_loss_gradient,
    infer_frame_config,
    load_manifest,
    load_trial_frame,
    make_cylinder_grid,
    raycast_grid_thickness,
    rearfoot_punch_center_uv,
    validate_frame_config,
    write_visualization_report,
)
from projects.digital_instron_v2.foundation import (
    FoundationFitSample,
    FoundationTrialBatch,
    _foundation_sdf_kernel,
    evaluate_foundation_lengths_batch,
    finite_difference_loss_gradient,
    fit_foundation_material_batches_autodiff,
    warp_loss_gradient,
)
from projects.digital_instron_v2.geometry import _load_obj_mesh
from projects.digital_instron_v2.mujoco_adapter import (
    apply_foundation_wrench_to_body_f,
    apply_sdf_foundation_wrench,
)
from projects.digital_instron_v2.sdf_utils import build_indenter_sdf
from projects.digital_instron_v2.workflow import (
    _fullfoot_contact_diagnostics,
    _hysteresis_segments,
    _spring_state_for_trial_frame,
    _trial_contact_surface_cache,
    _trial_displacement_split,
    run_fit_autodiff,
)

_cuda_available = wp.is_cuda_available()


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


def _write_sloped_plate_stl(path: Path) -> None:
    vertices = [
        (-0.03, 0.004, -0.03),
        (0.03, 0.024, -0.03),
        (0.03, 0.024, 0.03),
        (-0.03, 0.004, 0.03),
    ]
    faces = [(0, 1, 2), (0, 2, 3)]
    lines = ["solid sloped_plate"]
    for face in faces:
        lines.extend(["facet normal 0 1 0", "outer loop"])
        for index in face:
            x, y, z = vertices[index]
            lines.append(f"vertex {x} {y} {z}")
        lines.extend(["endloop", "endfacet"])
    lines.append("endsolid sloped_plate")
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

    def test_raycast_spring_grid_applies_thickness_axis_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = Path(tmp) / "box.obj"
            _write_box_obj(mesh_path, size=(0.12, 0.05, 0.03))

            vertices, faces = _load_obj_mesh(mesh_path)
            auto_grid = build_raycast_spring_grid(vertices, faces, spacing_m=0.02)
            override_grid = build_raycast_spring_grid(vertices, faces, spacing_m=0.02, thickness_axis=1)

            self.assertEqual(auto_grid.frame.thickness_axis, 2)
            self.assertEqual(override_grid.frame.thickness_axis, 1)
            np.testing.assert_allclose(auto_grid.slack_length_m, np.full(len(auto_grid.slack_length_m), 0.03))
            np.testing.assert_allclose(
                override_grid.slack_length_m,
                np.full(len(override_grid.slack_length_m), 0.05),
            )

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
    def test_hysteresis_segments_split_downsampled_cycle_jumps(self):
        displacement = np.asarray([0.0, 0.001, 0.002, 0.003, 0.018, 0.019, 0.020], dtype=np.float64)

        segments = _hysteresis_segments(displacement)

        self.assertEqual([segment.tolist() for segment in segments], [[0, 1, 2, 3], [4, 5, 6]])

    def _launch_sdf_foundation(
        self,
        *,
        xy: np.ndarray,
        top_z: np.ndarray,
        slack_length_m: np.ndarray | None = None,
        velocity_mps: np.ndarray | None = None,
        cell_area_m2: np.ndarray,
        indenter_sdf,
        material: FoundationMaterial,
        indenter_pos=(0.0, 0.0, 0.0),
    ) -> tuple[float, np.ndarray]:
        device = "cuda:0"
        force_out = wp.zeros(1, dtype=wp.float32, device=device)
        wrench_out = wp.zeros(6, dtype=wp.float32, device=device)
        params = wp.array(
            [
                material.stiffness_pa,
                material.ogden_alpha,
                material.lock_strain,
                material.damping_pa_s,
                material.damping_power,
                material.prony_stiffness_pa,
                material.prony_damping_pa_s,
                material.pasternak_stiffness_n_per_m,
                getattr(material, "spatial_slope", 0.0) + 1.0,
            ],
            dtype=wp.float32,
            device=device,
        )
        from projects.digital_instron_v2.geometry import compute_grid_neighbors
        from projects.digital_instron_v2.foundation import infer_spacing, _infer_longitudinal_axis_and_x_max

        spacing_val = infer_spacing(xy)
        neighbors_val = compute_grid_neighbors(xy, spacing_val)
        wp_neighbors = wp.array(neighbors_val, dtype=wp.int32, device=device)
        longitudinal_axis, x_min, x_max = _infer_longitudinal_axis_and_x_max(xy)
        wp.launch(
            _foundation_sdf_kernel,
            dim=len(xy),
            inputs=[
                wp.array(top_z.astype(np.float32), dtype=wp.float32, device=device),
                wp.array(
                    (
                        np.full(len(xy), 0.02, dtype=np.float32)
                        if slack_length_m is None
                        else slack_length_m.astype(np.float32)
                    ),
                    dtype=wp.float32,
                    device=device,
                ),
                wp.array(
                    (np.zeros(len(xy), dtype=np.float32) if velocity_mps is None else velocity_mps.astype(np.float32)),
                    dtype=wp.float32,
                    device=device,
                ),
                wp.array([wp.vec2(float(x), float(y)) for x, y in xy], dtype=wp.vec2, device=device),
                wp.array(cell_area_m2.astype(np.float32), dtype=wp.float32, device=device),
                indenter_sdf,
                wp.vec3(*indenter_pos),
                wp.quat_identity(),
                params,
                wp_neighbors,
                float(spacing_val),
                int(longitudinal_axis),
                float(x_min),
                float(x_max),
                force_out,
                wrench_out,
            ],
            device=device,
        )
        return float(force_out.numpy()[0]), wrench_out.numpy()

    @unittest.skipIf(not _cuda_available, "Requires CUDA")
    def test_sdf_kernel_flat_plate_matches_analytical(self):
        xy = np.asarray([[0.0, 0.0]], dtype=np.float32)
        top_z = np.asarray([0.0], dtype=np.float32)
        cell_area = np.asarray([1.0e-4], dtype=np.float32)
        material = FoundationMaterial(2.0e6, 2.0, 0.65, 0.0)
        sdf = build_indenter_sdf(
            "flat_plate",
            bounds=((-0.02, -0.02), (0.02, 0.02)),
            target_voxel_size=0.0005,
            margin=0.004,
            narrow_band_range=(-0.004, 0.004),
            device="cuda:0",
        )

        force, wrench = self._launch_sdf_foundation(
            xy=xy,
            top_z=top_z,
            cell_area_m2=cell_area,
            indenter_sdf=sdf,
            material=material,
        )

        thickness_m = 0.02
        penetration_m = 0.001
        strain = penetration_m / thickness_m
        expected_stress = (
            material.stiffness_pa
            * ((1.0 - strain / material.lock_strain) ** (-material.ogden_alpha) - 1.0)
            / material.ogden_alpha
        )
        expected_force = float(cell_area[0] * expected_stress)
        self.assertGreater(force, 0.0)
        self.assertAlmostEqual(force, expected_force, delta=expected_force * 0.25)
        self.assertAlmostEqual(float(wrench[2]), force, delta=max(1.0e-5, abs(force) * 1.0e-5))

        static_damped_force, _ = self._launch_sdf_foundation(
            xy=xy,
            top_z=top_z,
            cell_area_m2=cell_area,
            indenter_sdf=sdf,
            material=FoundationMaterial(2.0e6, 2.0, 0.65, 1.0e6),
        )
        self.assertAlmostEqual(static_damped_force, force, delta=max(1.0e-4, force * 0.05))

        moving_damped_force, _ = self._launch_sdf_foundation(
            xy=xy,
            top_z=top_z,
            velocity_mps=np.asarray([-0.01], dtype=np.float32),
            cell_area_m2=cell_area,
            indenter_sdf=sdf,
            material=FoundationMaterial(2.0e6, 2.0, 0.65, 1.0e6),
        )
        self.assertGreater(moving_damped_force, force)

    @unittest.skipIf(not _cuda_available, "Requires CUDA")
    def test_sdf_kernel_cylinder_penetration(self):
        xy = np.asarray([[0.0, 0.0], [0.03, 0.0]], dtype=np.float32)
        top_z = np.asarray([0.0, 0.0], dtype=np.float32)
        cell_area = np.asarray([1.0e-4, 1.0e-4], dtype=np.float32)
        material = FoundationMaterial(2.0e6, 2.0, 0.65, 0.0)
        sdf = build_indenter_sdf(
            "cylinder",
            radius_m=0.0225,
            height_m=0.004,
            target_voxel_size=0.0005,
            margin=0.004,
            narrow_band_range=(-0.004, 0.004),
            device="cuda:0",
        )

        total_force, total_wrench = self._launch_sdf_foundation(
            xy=xy,
            top_z=top_z,
            cell_area_m2=cell_area,
            indenter_sdf=sdf,
            material=material,
        )
        center_force, _ = self._launch_sdf_foundation(
            xy=xy[:1],
            top_z=top_z[:1],
            cell_area_m2=cell_area[:1],
            indenter_sdf=sdf,
            material=material,
        )

        self.assertGreater(center_force, 0.0)
        self.assertAlmostEqual(total_force, center_force, delta=max(1.0e-4, center_force * 0.05))
        self.assertAlmostEqual(float(total_wrench[4]), 0.0, delta=max(1.0e-4, center_force * 0.05))

    @unittest.skipIf(not _cuda_available, "Requires CUDA")
    def test_sdf_kernel_no_penetration_above_indenter(self):
        xy = np.asarray([[0.0, 0.0], [0.005, 0.0]], dtype=np.float32)
        top_z = np.asarray([0.004, 0.005], dtype=np.float32)
        cell_area = np.asarray([1.0e-4, 1.0e-4], dtype=np.float32)
        material = FoundationMaterial(2.0e6, 2.0, 0.65, 0.0)
        sdf = build_indenter_sdf(
            "flat_plate",
            bounds=((-0.02, -0.02), (0.02, 0.02)),
            target_voxel_size=0.0005,
            margin=0.004,
            narrow_band_range=(-0.004, 0.004),
            device="cuda:0",
        )

        force, wrench = self._launch_sdf_foundation(
            xy=xy,
            top_z=top_z,
            cell_area_m2=cell_area,
            indenter_sdf=sdf,
            material=material,
        )

        self.assertAlmostEqual(force, 0.0, places=6)
        np.testing.assert_allclose(wrench, np.zeros(6), atol=1.0e-6)

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

    def test_batch_fit_updates_shape_and_state_parameters(self):
        xy = np.asarray([[0.0, 0.0], [0.012, 0.0], [0.0, 0.012]], dtype=np.float64)
        slack = np.asarray([0.032, 0.038, 0.044], dtype=np.float64)
        compression = np.asarray([0.001, 0.004, 0.009, 0.014, 0.018, 0.011, 0.005], dtype=np.float64)
        current = slack[None, :] - compression[:, None]
        current_velocity = np.gradient(current, axis=0) / 0.01
        cell_area = np.full(len(slack), 1.5e-4, dtype=np.float64)
        weights = np.full(len(compression), 1.0 / len(compression), dtype=np.float64)
        phase = ("baseline_pre", "loading", "loading", "peak", "peak", "unloading", "baseline_post")
        true_material = FoundationMaterial(
            stiffness_pa=1.8e6,
            ogden_alpha=3.4,
            lock_strain=0.82,
            damping_pa_s=8.0e4,
            damping_power=1.6,
            prony_stiffness_pa=9.0e5,
            prony_damping_pa_s=4.5e4,
            state_warmup_cycles=2,
        )
        initial = FoundationMaterial(
            stiffness_pa=6.0e5,
            ogden_alpha=1.2,
            lock_strain=0.55,
            damping_pa_s=1.0e4,
            damping_power=0.5,
            prony_stiffness_pa=1.2e5,
            prony_damping_pa_s=2.4e3,
            state_warmup_cycles=true_material.state_warmup_cycles,
        )
        template = FoundationTrialBatch(
            name="synthetic_batch",
            current_length_m=current,
            slack_length_m=slack,
            velocity_mps=current_velocity,
            measured_force_n=np.zeros(len(compression), dtype=np.float64),
            sample_weight=weights,
            cell_area_m2=cell_area,
            time_s=np.arange(len(compression), dtype=np.float64) * 0.01,
            dt_s=np.concatenate(([0.0], np.full(len(compression) - 1, 0.01))),
            displacement_m=compression,
            phase=phase,
        )
        measured = evaluate_foundation_lengths_batch(
            xy, template, material=true_material, device="cpu"
        ).predicted_force_n
        batch = FoundationTrialBatch(
            name=template.name,
            current_length_m=template.current_length_m,
            slack_length_m=template.slack_length_m,
            velocity_mps=template.velocity_mps,
            measured_force_n=measured,
            sample_weight=template.sample_weight,
            cell_area_m2=template.cell_area_m2,
            time_s=template.time_s,
            dt_s=template.dt_s,
            displacement_m=template.displacement_m,
            phase=template.phase,
        )

        result = fit_foundation_material_batches_autodiff(
            xy,
            [batch],
            initial_material=initial,
            iterations=4,
            per_cylinder_area=True,
            device="cpu",
        )

        self.assertNotEqual(result.material.ogden_alpha, initial.ogden_alpha)
        self.assertNotEqual(result.material.lock_strain, initial.lock_strain)
        self.assertNotEqual(result.material.damping_power, initial.damping_power)
        self.assertNotEqual(result.material.prony_stiffness_pa, initial.prony_stiffness_pa)
        self.assertNotEqual(result.material.prony_damping_pa_s, initial.prony_damping_pa_s)
        self.assertEqual(result.material.state_warmup_cycles, initial.state_warmup_cycles)
        self.assertIn("grad_ogden_alpha", result.history[0])
        self.assertIn("grad_lock_strain", result.history[0])
        self.assertIn("grad_damping_power", result.history[0])
        self.assertIn("grad_prony_stiffness_pa", result.history[0])
        self.assertIn("grad_prony_damping_pa_s", result.history[0])
        self.assertTrue(np.isfinite(result.history[0]["grad_prony_stiffness_pa"]))
        self.assertTrue(np.isfinite(result.history[0]["grad_prony_damping_pa_s"]))
        self.assertNotEqual(result.history[0]["grad_prony_stiffness_pa"], 0.0)

    def test_fit_with_per_cylinder_area(self):
        xy = np.asarray([[0.0, 0.0], [0.01, 0.0]], dtype=np.float64)
        slack = np.asarray([0.03, 0.04], dtype=np.float64)
        velocity = np.zeros(2, dtype=np.float64)
        per_cell_area = np.asarray([1e-4, 2e-4], dtype=np.float64)
        true_material = FoundationMaterial(2.0e6, 2.0, 0.65, 0.0)
        initial = FoundationMaterial(8.0e5, 2.0, 0.65, 0.0)
        samples = []
        for compression in (0.002, 0.004, 0.006):
            current = slack - compression
            measured = evaluate_foundation_lengths(
                xy, current, slack, velocity, cell_area_m2=per_cell_area, material=true_material
            ).force_n
            samples.append(FoundationFitSample(current, slack, velocity, measured, cell_area_m2=per_cell_area))

        result = fit_foundation_material_autodiff(
            xy,
            samples,
            cell_area_m2=1.0e-4,
            initial_material=initial,
            iterations=15,
            learning_rates=(0.1, 0.0, 0.0, 0.0, 0.0),
        )

        self.assertGreater(result.material.stiffness_pa, initial.stiffness_pa)
        self.assertLess(abs(result.material.stiffness_pa - 2.0e6) / 2.0e6, 0.5)

    def test_fit_backward_compat_uniform_area(self):
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

        result = fit_foundation_material_autodiff(
            xy,
            samples,
            cell_area_m2=1.0e-4,
            initial_material=initial,
            iterations=12,
            learning_rates=(0.1, 0.0, 0.0, 0.0, 0.0),
        )

        self.assertGreater(result.material.stiffness_pa, initial.stiffness_pa)

    def test_fullfoot_stl_contact_uses_surface_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh_path = root / "midsole.obj"
            stl_path = root / "sloped_plate.stl"
            _write_box_obj(mesh_path, size=(0.04, 0.02, 0.04))
            _write_sloped_plate_stl(stl_path)

            vertices, faces = _load_obj_mesh(mesh_path)
            spring_grid = build_raycast_spring_grid(vertices, faces, spacing_m=0.01)
            trial = SimpleNamespace(
                name="fullfoot_stl",
                fixture="fullfoot_last",
                include_in_fit=True,
                indenter={
                    "type": "stl",
                    "path": str(stl_path),
                    "units": "m",
                    "rotation_deg": [0.0, 0.0, 0.0],
                },
            )
            contact_surfaces = _trial_contact_surface_cache(SimpleNamespace(trials=(trial,)), spring_grid)
            current_length, velocity = _spring_state_for_trial_frame(
                spring_grid,
                trial,
                np.zeros(len(spring_grid.xy_m), dtype=bool),
                contact_surfaces,
                displacement_m=0.004,
                displacement_velocity_mps=0.01,
            )
            compression = spring_grid.slack_length_m - current_length

            self.assertGreater(float(np.max(compression)), 0.0)
            self.assertGreater(float(np.max(compression) - np.min(compression)), 0.001)
            self.assertEqual(int(np.count_nonzero(compression > 0.0)), len(compression))
            self.assertEqual(int(np.count_nonzero(velocity < 0.0)), int(np.count_nonzero(compression > 0.0)))

    def test_spring_state_combines_top_and_bottom_compression(self):
        spring_grid = SimpleNamespace(
            slack_length_m=np.asarray([0.02, 0.02, 0.02], dtype=np.float64),
            bottom_m=np.asarray([-0.020, -0.019, -0.016], dtype=np.float64),
        )
        trial = SimpleNamespace(
            name="rearfoot_two_sided",
            fixture="rearfoot_punch",
            indenter={"top_displacement_fraction": 0.5, "bottom_displacement_fraction": 0.5},
        )
        rearfoot_mask = np.asarray([True, True, False])

        current_length, velocity = _spring_state_for_trial_frame(
            spring_grid,
            trial,
            rearfoot_mask,
            {},
            displacement_m=0.004,
            displacement_velocity_mps=0.01,
        )

        compression = spring_grid.slack_length_m - current_length
        np.testing.assert_allclose(compression, [0.004, 0.003, 0.0], atol=1.0e-12)
        np.testing.assert_allclose(velocity, [-0.01, -0.01, 0.0], atol=1.0e-12)

    def test_trial_displacement_split_rejects_top_only_contact(self):
        trial = SimpleNamespace(
            name="rearfoot_top_only",
            fixture="rearfoot_punch",
            indenter={"top_displacement_fraction": 1.0, "bottom_displacement_fraction": 0.0},
        )

        with self.assertRaisesRegex(ValueError, "top and bottom"):
            _trial_displacement_split(trial)

    def test_fullfoot_stl_height_offset_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh_path = root / "midsole.obj"
            stl_path = root / "sloped_plate.stl"
            _write_box_obj(mesh_path, size=(0.04, 0.02, 0.04))
            _write_sloped_plate_stl(stl_path)

            vertices, faces = _load_obj_mesh(mesh_path)
            spring_grid = build_raycast_spring_grid(vertices, faces, spacing_m=0.01)
            base_trial = SimpleNamespace(
                name="fullfoot_base",
                fixture="fullfoot_last",
                include_in_fit=True,
                indenter={"type": "stl", "path": str(stl_path), "units": "m"},
            )
            offset_trial = SimpleNamespace(
                name="fullfoot_offset",
                fixture="fullfoot_last",
                include_in_fit=True,
                indenter={"type": "stl", "path": str(stl_path), "units": "m", "height_offset_m": 0.003},
            )
            contact_surfaces = _trial_contact_surface_cache(
                SimpleNamespace(trials=(base_trial, offset_trial)), spring_grid
            )

            base_surface, base_valid = contact_surfaces["fullfoot_base"]
            offset_surface, offset_valid = contact_surfaces["fullfoot_offset"]
            np.testing.assert_array_equal(base_valid, offset_valid)
            diff = offset_surface[base_valid] - base_surface[base_valid]
            # height_offset_m shifts by exactly 0.003 for non-clamped points;
            # clamped points (pinned at top_m) show 0.0 difference.
            self.assertTrue(
                np.all((diff >= 0.0) & (diff <= 0.003 + 1e-9)),
                f"height_offset shift should be in [0, 0.003], got {diff}",
            )
            self.assertGreater(np.max(diff), 0.001, "at least some points should reflect the height_offset shift")

            diagnostics = _fullfoot_contact_diagnostics(
                spring_grid,
                {"fullfoot_base": contact_surfaces["fullfoot_base"]},
                displacement_m=0.004,
            )["fullfoot_base"]
            clearance_mm = (spring_grid.top_m[base_valid] - (base_surface[base_valid] - 0.004)) * 1000.0
            self.assertEqual(diagnostics["valid_count"], float(np.count_nonzero(base_valid)))
            self.assertEqual(diagnostics["active_count"], float(np.count_nonzero(clearance_mm > 0.0)))
            self.assertAlmostEqual(
                diagnostics["valid_area_mm2"], np.count_nonzero(base_valid) * spring_grid.cell_area_m2 * 1.0e6
            )
            self.assertAlmostEqual(diagnostics["contact_p50_mm"], float(np.percentile(clearance_mm, 50.0)))

    def test_mujoco_adapter_adds_wrench_to_body_force(self):
        body_f = np.zeros((2, 6), dtype=np.float64)
        wrench = np.asarray([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

        apply_foundation_wrench_to_body_f(body_f, 1, wrench)

        np.testing.assert_allclose(body_f[1], wrench)
        np.testing.assert_allclose(body_f[0], np.zeros(6))

    @unittest.skipIf(not _cuda_available, "Requires CUDA")
    def test_evaluate_foundation_sdf_returns_result(self):
        xy = np.asarray([[0.0, 0.0]], dtype=np.float32)
        top_z = np.asarray([0.0], dtype=np.float32)
        cell_area = np.asarray([1.0e-4], dtype=np.float32)
        material = FoundationMaterial(2.0e6, 2.0, 0.65, 0.0)
        sdf = build_indenter_sdf(
            "flat_plate",
            bounds=((-0.02, -0.02), (0.02, 0.02)),
            target_voxel_size=0.0005,
            margin=0.004,
            narrow_band_range=(-0.004, 0.004),
            device="cuda:0",
        )

        result = evaluate_foundation_sdf(
            xy_m=xy,
            top_z_m=top_z,
            slack_length_m=np.full(len(xy), 0.02, dtype=np.float32),
            velocity_mps=np.zeros(len(xy), dtype=np.float32),
            cell_area_m2=cell_area,
            material=material,
            indenter_sdf=sdf,
            indenter_pos=(0.0, 0.0, 0.0),
            device="cuda:0",
        )

        self.assertIsInstance(result, FoundationResult)
        self.assertGreater(result.force_n, 0.0)
        self.assertEqual(result.wrench.shape, (6,))
        self.assertGreaterEqual(result.loss, 0.0)

    @unittest.skipIf(not _cuda_available, "Requires CUDA")
    def test_evaluate_foundation_sdf_force_scales_with_area(self):
        xy = np.asarray([[0.0, 0.0]], dtype=np.float32)
        top_z = np.asarray([0.0], dtype=np.float32)
        material = FoundationMaterial(2.0e6, 2.0, 0.65, 0.0)
        sdf = build_indenter_sdf(
            "flat_plate",
            bounds=((-0.02, -0.02), (0.02, 0.02)),
            target_voxel_size=0.0005,
            margin=0.004,
            narrow_band_range=(-0.004, 0.004),
            device="cuda:0",
        )

        result_small = evaluate_foundation_sdf(
            xy_m=xy,
            top_z_m=top_z,
            slack_length_m=np.full(len(xy), 0.02, dtype=np.float32),
            velocity_mps=np.zeros(len(xy), dtype=np.float32),
            cell_area_m2=1e-4,
            material=material,
            indenter_sdf=sdf,
            indenter_pos=(0.0, 0.0, 0.0),
            device="cuda:0",
        )
        result_large = evaluate_foundation_sdf(
            xy_m=xy,
            top_z_m=top_z,
            slack_length_m=np.full(len(xy), 0.02, dtype=np.float32),
            velocity_mps=np.zeros(len(xy), dtype=np.float32),
            cell_area_m2=1e-3,
            material=material,
            indenter_sdf=sdf,
            indenter_pos=(0.0, 0.0, 0.0),
            device="cuda:0",
        )

        self.assertGreater(result_small.force_n, 0.0)
        self.assertGreater(result_large.force_n, 0.0)
        self.assertAlmostEqual(
            result_large.force_n / result_small.force_n,
            10.0,
            delta=0.5,
        )

    def test_pasternak_shear_coupling(self):
        """Verify that Pasternak shear coupling increases force under uneven compression."""
        x = np.linspace(-0.01, 0.01, 3)
        y = np.linspace(-0.01, 0.01, 3)
        xx, yy = np.meshgrid(x, y)
        xy = np.column_stack([xx.ravel(), yy.ravel()])

        mat_zero_shear = FoundationMaterial(
            stiffness_pa=1.0e6,
            ogden_alpha=2.0,
            lock_strain=0.65,
            damping_pa_s=0.0,
            pasternak_stiffness_n_per_m=0.0,
        )
        mat_with_shear = FoundationMaterial(
            stiffness_pa=1.0e6,
            ogden_alpha=2.0,
            lock_strain=0.65,
            damping_pa_s=0.0,
            pasternak_stiffness_n_per_m=5.0e5,
        )

        compression = np.zeros(9)
        compression[4] = 0.005
        velocity = np.zeros(9)
        cell_area = 1.0e-4

        res_zero = evaluate_foundation(
            xy,
            compression,
            velocity,
            cell_area_m2=cell_area,
            thickness_m=0.03,
            material=mat_zero_shear,
        )

        res_shear = evaluate_foundation(
            xy,
            compression,
            velocity,
            cell_area_m2=cell_area,
            thickness_m=0.03,
            material=mat_with_shear,
        )

        self.assertGreater(res_shear.force_n, res_zero.force_n)

    def test_pasternak_free_boundary_preserves_uniform_compression(self):
        """Uniform compression should not gain artificial edge force from missing neighbors."""
        x = np.linspace(-0.01, 0.01, 3)
        y = np.linspace(-0.01, 0.01, 3)
        xx, yy = np.meshgrid(x, y)
        xy = np.column_stack([xx.ravel(), yy.ravel()])
        compression = np.full(9, 0.005)
        velocity = np.zeros(9)

        mat_zero_pasternak = FoundationMaterial(1.0e6, 2.0, 0.65, 0.0, pasternak_stiffness_n_per_m=0.0)
        mat_with_pasternak = FoundationMaterial(1.0e6, 2.0, 0.65, 0.0, pasternak_stiffness_n_per_m=5.0e5)

        res_zero = evaluate_foundation(
            xy,
            compression,
            velocity,
            cell_area_m2=1.0e-4,
            thickness_m=0.03,
            material=mat_zero_pasternak,
        )
        res_pasternak = evaluate_foundation(
            xy,
            compression,
            velocity,
            cell_area_m2=1.0e-4,
            thickness_m=0.03,
            material=mat_with_pasternak,
        )

        self.assertAlmostEqual(res_pasternak.force_n, res_zero.force_n, delta=abs(res_zero.force_n) * 1.0e-5)

    def test_qlv_prony_relaxation(self):
        """Verify that stateful QLV Prony series relaxation causes force decay under constant strain."""
        xy = np.asarray([[0.0, 0.0]], dtype=np.float64)

        current_len = np.full((10, 1), 0.025)
        slack_len = np.full(1, 0.03)
        velocity = np.zeros((10, 1))
        dt_s = 0.01

        measured_force = np.zeros(10)
        sample_weight = np.ones(10) / 10.0
        cell_area = np.full(1, 1.0e-4)

        mat_relaxing = FoundationMaterial(
            stiffness_pa=1.0e6,
            ogden_alpha=2.0,
            lock_strain=0.65,
            damping_pa_s=0.0,
            damping_power=1.0,
            prony_stiffness_pa=5.0e5,
            prony_damping_pa_s=2.5e4,
            state_warmup_cycles=0,
        )

        from projects.digital_instron_v2.foundation import FoundationTrialBatch, evaluate_foundation_lengths_batch

        batch = FoundationTrialBatch(
            name="step_hold",
            current_length_m=current_len,
            slack_length_m=slack_len,
            velocity_mps=velocity,
            measured_force_n=measured_force,
            sample_weight=sample_weight,
            cell_area_m2=cell_area,
            time_s=np.arange(10) * dt_s,
            dt_s=np.full(10, dt_s),
            displacement_m=np.full(10, 0.005),
            phase=tuple(["hold"] * 10),
        )

        result = evaluate_foundation_lengths_batch(
            xy,
            batch,
            material=mat_relaxing,
            device="cpu",
        )

        forces = result.predicted_force_n
        for i in range(1, len(forces)):
            self.assertLessEqual(forces[i], forces[i - 1])
        self.assertLess(forces[-1], forces[0])

    def test_workflow_fit_autodiff_with_per_cylinder(self):
        """Integration: run_fit_autodiff produces per_cylinder_area in output."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"

            # 1. Create synthetic box mesh (watertight, 120x50x30 mm)
            mesh_path = root / "midsole.obj"
            _write_box_obj(mesh_path, size=(0.12, 0.05, 0.03))

            # 2. Condition mesh (produces repaired mesh in cache_dir)
            mesh_report = condition_midsole_mesh(mesh_path, cache_dir, source_units="m", remesh=False)
            self.assertIn("repaired_mesh", mesh_report)

            # 3. Build SpringSurfaceGrid to verify grid is viable
            vertices, faces = _load_obj_mesh(Path(str(mesh_report["repaired_mesh"])))
            spring_grid = build_raycast_spring_grid(vertices, faces, spacing_m=0.02)
            self.assertGreater(len(spring_grid.xy_m), 0)

            # 4. Create synthetic CSV with time, position (mm), force (N)
            csv_path = root / "rearfoot_punch.csv"
            time_s = np.linspace(0, 1, 100)
            force_n = np.linspace(0, 200, 100)
            position_mm = np.linspace(0, 5, 100)
            header = "Total Time,Position,Force\n"
            data = "\n".join(f"{t},{p},{f}" for t, p, f in zip(time_s, position_mm, force_n, strict=True))
            csv_path.write_text(header + data)
            avg_path = root / "rearfoot_avg.csv"
            avg_time_s = np.linspace(0.0, 0.2, 21)
            avg_displacement_m = np.concatenate((np.linspace(0.0, 0.004, 11), np.linspace(0.0036, 0.0, 10)))
            avg_force_n = np.concatenate((np.linspace(0.0, 200.0, 11), np.linspace(180.0, 0.0, 10)))
            avg_velocity_m_s = np.gradient(avg_displacement_m, avg_time_s)
            avg_rows = [
                "phase,time_s,displacement_m,displacement_mm,force_n,position_mm_raw,force_n_raw,velocity_m_s,cycle_energy_j"
            ]
            for phase, time, displacement, force, velocity in zip(
                np.linspace(0.0, 1.0, len(avg_time_s)),
                avg_time_s,
                avg_displacement_m,
                avg_force_n,
                avg_velocity_m_s,
                strict=True,
            ):
                avg_rows.append(
                    f"{phase},{time},{displacement},{displacement * 1000.0},{force},0,{-force},{velocity},0"
                )
            avg_path.write_text("\n".join(avg_rows) + "\n")

            # 5. Create manifest JSON
            manifest_path = root / "manifest.json"
            manifest = {
                "midsole_mesh": str(mesh_path),
                "cache_dir": str(cache_dir),
                "qc": {"mesh_source_units": "m"},
                "grid": {"coarse_spacing_m": 0.02},
                "fit": {
                    "initial_stiffness_pa": 2.0e6,
                    "initial_ogden_alpha": 2.0,
                    "initial_lock_strain": 0.65,
                    "initial_damping_pa_s": 1.0e4,
                    "initial_damping_power": 1.0,
                    "initial_prony_stiffness_pa": 1.0e6,
                    "initial_prony_damping_pa_s": 5.0e4,
                    "state_warmup_cycles": 1,
                },
                "trials": [
                    {
                        "name": "rearfoot_punch",
                        "csv_path": str(csv_path),
                        "averaged_cycle_path": str(avg_path),
                        "fixture": "rearfoot_punch",
                        "indenter": {"type": "cylinder", "radius_m": 0.0225},
                        "include_in_fit": True,
                    }
                ],
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))

            # 6. Run fit-autodiff workflow step
            args = argparse.Namespace(
                manifest=str(manifest_path),
                output_dir=None,
                step="fit-autodiff",
                autodiff_iterations=2,
                autodiff_sample_count=2,
                autodiff_device="cpu",
                hysteresis_sample_count=12,
            )
            report = run_fit_autodiff(args)

            # 7. Verify output material has per_cylinder_area field
            self.assertIn("material", report)
            self.assertIn("per_cylinder_area", report["material"])
            self.assertTrue(report["material"]["per_cylinder_area"])
            self.assertTrue(0.0 < report["material"]["prony_stiffness_pa"] < 2.0e6)
            self.assertEqual(report["material"]["state_warmup_cycles"], 1)

            # 8. Verify saved JSON includes per_cylinder_area
            output_path = cache_dir / "digital_instron_v2_autodiff_fit.json"
            self.assertTrue(output_path.exists())
            saved = json.loads(output_path.read_text())
            self.assertIn("material", saved)
            self.assertIn("per_cylinder_area", saved["material"])
            self.assertTrue(saved["material"]["per_cylinder_area"])
            self.assertIn("hysteresis", saved)
            self.assertEqual(saved["fit_source"], "averaged_cycle")
            self.assertEqual(saved["sample_count"], len(avg_time_s))
            self.assertIn("acceptance", saved)
            material_path = Path(saved["foundation_material_json"])
            self.assertTrue(material_path.exists())
            material_artifact = json.loads(material_path.read_text())
            self.assertEqual(material_artifact["schema_version"], "digital_instron_v2_foundation_material_1")
            self.assertEqual(material_artifact["contact_model"]["type"], "two_sided_spring_grid")
            self.assertEqual(material_artifact["contact_model"]["compression_components"], "top_plus_bottom")
            trial_contact = material_artifact["contact_model"]["trials"]["rearfoot_punch"]
            self.assertAlmostEqual(trial_contact["top_displacement_fraction"], 0.5)
            self.assertAlmostEqual(trial_contact["bottom_displacement_fraction"], 0.5)
            self.assertIn("calibration_envelope", material_artifact)
            envelope = material_artifact["calibration_envelope"]
            self.assertGreater(envelope["preferred_peak_top_compression_m"], 0.0)
            self.assertGreater(envelope["preferred_peak_bottom_compression_m"], 0.0)
            self.assertGreater(envelope["preferred_peak_stack_compression_m"], 0.0)
            self.assertGreater(envelope["preferred_one_sided_hydro_shoe_stroke_m"], 0.0)
            trial_envelope = material_artifact["calibration_envelope"]["trials"]["rearfoot_punch"]
            self.assertGreater(trial_envelope["peak_displacement_m"], 0.0)
            self.assertGreater(trial_envelope["peak_max_compression_m"], 0.0)
            self.assertGreater(trial_envelope["peak_top_compression_m"], 0.0)
            self.assertGreater(trial_envelope["peak_bottom_compression_m"], 0.0)
            self.assertGreater(trial_envelope["peak_active_area_m2"], 0.0)
            hysteresis_png = Path(saved["hysteresis"]["hysteresis_png"])
            hysteresis_csv = Path(saved["hysteresis"]["hysteresis_csv"])
            hysteresis_trials_csv = Path(saved["hysteresis"]["hysteresis_trials_csv"])
            self.assertTrue(hysteresis_png.exists())
            self.assertGreater(hysteresis_png.stat().st_size, 0)
            self.assertTrue(hysteresis_csv.exists())
            self.assertTrue(hysteresis_trials_csv.exists())
            self.assertGreaterEqual(len(saved["hysteresis"]["trials"]), 1)


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
                                "averaged_cycle_path": "trial.csv",
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
            self.assertTrue(manifest.trials[0].averaged_cycle_path.exists())

    def test_manifest_parses_sdf_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "trail.csv"
            csv_path.write_text('"Total Time (s)","Position (mm)","Force (N)"\n0,0,0\n')
            mesh_path = root / "mesh.obj"
            _write_box_obj(mesh_path)
            stl_path = root / "indenter.stl"
            stl_path.write_text(
                "solid indenter\n"
                "facet normal 0 0 1\n"
                "outer loop\n"
                "vertex 0 0 0\n"
                "vertex 0.001 0 0\n"
                "vertex 0 0.001 0\n"
                "endloop\n"
                "endfacet\n"
                "endsolid indenter\n"
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "midsole_mesh": "mesh.obj",
                        "cache_dir": "cache",
                        "qc": {},
                        "grid": {
                            "coarse_spacing_m": 0.005,
                            "force_thickness_axis": 1,
                        },
                        "fit": {"model": "locked_ogden_vertical_foundation_v2"},
                        "trials": [
                            {
                                "name": "flatplate_trial",
                                "csv_path": "trail.csv",
                                "fixture": "rearfoot_punch",
                                "include_in_fit": True,
                                "indenter": {
                                    "type": "flat_plate",
                                    "plate_height": 0.01,
                                },
                            },
                            {
                                "name": "cylinder_trial",
                                "csv_path": "trail.csv",
                                "fixture": "rearfoot_punch",
                                "include_in_fit": True,
                                "indenter": {
                                    "type": "cylinder",
                                    "radius_m": 0.0225,
                                    "height_m": 0.05,
                                },
                            },
                            {
                                "name": "stl_trial",
                                "csv_path": "trail.csv",
                                "fixture": "rearfoot_punch",
                                "include_in_fit": True,
                                "indenter": {
                                    "type": "stl",
                                    "path": "indenter.stl",
                                    "rotation_deg": [0.0, 0.0, 0.0],
                                    "height_offset_m": -0.002,
                                },
                            },
                        ],
                    }
                )
            )

            manifest = load_manifest(manifest_path)

            # Verify force_thickness_axis in grid
            self.assertEqual(manifest.grid["force_thickness_axis"], 1)

            # Verify each indenter type
            trials_by_name = {t.name: t for t in manifest.trials}

            fp = trials_by_name["flatplate_trial"]
            self.assertEqual(fp.indenter["type"], "flat_plate")
            self.assertEqual(fp.indenter["plate_height"], 0.01)

            cy = trials_by_name["cylinder_trial"]
            self.assertEqual(cy.indenter["type"], "cylinder")
            self.assertEqual(cy.indenter["radius_m"], 0.0225)
            self.assertEqual(cy.indenter["height_m"], 0.05)

            st = trials_by_name["stl_trial"]
            self.assertEqual(st.indenter["type"], "stl")
            self.assertEqual(st.indenter["path"], str(stl_path.resolve()))
            self.assertEqual(st.indenter["height_offset_m"], -0.002)

    def test_manifest_parses_thickness_axis_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "trial.csv"
            csv_path.write_text('"Total Time (s)","Position (mm)","Force (N)"\n0,0,0\n')
            mesh_path = root / "mesh.obj"
            _write_box_obj(mesh_path)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "midsole_mesh": "mesh.obj",
                        "cache_dir": "cache",
                        "qc": {},
                        "grid": {
                            "coarse_spacing_m": 0.005,
                            "force_thickness_axis": 2,
                        },
                        "fit": {"model": "locked_ogden_vertical_foundation_v2"},
                        "trials": [
                            {
                                "name": "test",
                                "csv_path": "trial.csv",
                                "fixture": "rearfoot_punch",
                                "include_in_fit": True,
                                "indenter": {"type": "cylinder", "radius_m": 0.02},
                            }
                        ],
                    }
                )
            )

            manifest = load_manifest(manifest_path)

            self.assertEqual(manifest.grid["force_thickness_axis"], 2)
            self.assertIn("coarse_spacing_m", manifest.grid)

    def test_frame_qc_infers_and_applies_saved_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "trace.csv"
            csv_path.write_text('"Total Time (s)","Position (mm)","Force (N)"\n0,10,-5\n0.1,8,-80\n0.2,6,-160\n')

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


class TestDigitalInstronV2MujocoAdapter(unittest.TestCase):
    """Tests for the MuJoCo adapter convenience wrappers."""

    def test_apply_sdf_wrench_to_body_f(self):
        body_f = np.zeros((3, 6), dtype=np.float64)
        result = FoundationResult(
            force_n=100.0,
            wrench=np.array([0.0, 0.0, 100.0, 5.0, -3.0, 0.0]),
            loss=0.5,
        )
        apply_sdf_foundation_wrench(body_f, 1, result)

        np.testing.assert_array_equal(body_f[1], [0.0, 0.0, 100.0, 5.0, -3.0, 0.0])
        np.testing.assert_array_equal(body_f[0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(body_f[2], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_apply_foundation_wrench_to_body_f_still_works(self):
        body_f = np.zeros((3, 6), dtype=np.float64)
        wrench = np.array([10.0, 20.0, 30.0, -1.0, -2.0, -3.0])
        apply_foundation_wrench_to_body_f(body_f, 0, wrench)
        np.testing.assert_array_equal(body_f[0], [10.0, 20.0, 30.0, -1.0, -2.0, -3.0])
        np.testing.assert_array_equal(body_f[1], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_apply_sdf_wrench_raises_on_bad_body_index(self):
        body_f = np.zeros((2, 6), dtype=np.float64)
        result = FoundationResult(
            force_n=100.0,
            wrench=np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0]),
            loss=0.5,
        )
        with self.assertRaises(IndexError):
            apply_sdf_foundation_wrench(body_f, 5, result)

    def test_apply_sdf_wrench_raises_on_bad_shape(self):
        body_f = np.zeros((3, 4), dtype=np.float64)
        result = FoundationResult(
            force_n=100.0,
            wrench=np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0]),
            loss=0.5,
        )
        with self.assertRaises(ValueError):
            apply_sdf_foundation_wrench(body_f, 1, result)


class TestDigitalInstronV2Visualization(unittest.TestCase):
    """Tests for offline visual diagnostics."""

    def test_visualization_per_cylinder_force_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # 1. Create synthetic box mesh
            mesh_path = root / "midsole.obj"
            _write_box_obj(mesh_path, size=(0.12, 0.05, 0.03))

            # 2. Dummy CSV (required by load_manifest)
            csv_path = root / "trial.csv"
            csv_path.write_text('"Total Time (s)","Position (mm)","Force (N)"\n0,0,0\n1,2,-100\n')

            # 3. Minimal manifest
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "midsole_mesh": str(mesh_path),
                        "cache_dir": "cache",
                        "qc": {
                            "mesh_source_units": "m",
                            "min_midsole_thickness_m": 0.005,
                            "max_midsole_thickness_m": 0.08,
                        },
                        "grid": {"coarse_spacing_m": 0.02},
                        "fit": {
                            "initial_stiffness_pa": 2.0e6,
                            "initial_ogden_alpha": 2.0,
                            "initial_lock_strain": 0.65,
                            "initial_damping_pa_s": 1.0e4,
                            "initial_damping_power": 1.0,
                        },
                        "trials": [
                            {
                                "name": "rearfoot_punch",
                                "csv_path": str(csv_path),
                                "fixture": "rearfoot_punch",
                                "indenter": {"type": "cylinder", "radius_m": 0.0225},
                                "include_in_fit": True,
                            }
                        ],
                    }
                )
            )

            manifest = load_manifest(manifest_path)
            write_visualization_report(manifest, root)

            force_heatmap = root / "digital_instron_v2_force_heatmap.png"
            self.assertTrue(force_heatmap.exists(), msg="force heatmap PNG should exist")
            self.assertGreater(force_heatmap.stat().st_size, 0, msg="force heatmap should be non-empty")


class TestDigitalInstronV2Baked(unittest.TestCase):
    """Tests for the baked (spatially-invariant) texture-sampled calibration logic."""

    def test_baked_geometry_construction_and_evaluation(self):
        # 1. Create a dummy/synthetic BakedMidsoleGeometry
        thickness_map = np.full((10, 10), 0.03, dtype=np.float64)
        top_map = np.full((10, 10), 0.015, dtype=np.float64)
        bottom_map = np.full((10, 10), -0.015, dtype=np.float64)
        mins_uv = np.array([-0.05, -0.05], dtype=np.float64)
        maxs_uv = np.array([0.05, 0.05], dtype=np.float64)
        from projects.digital_instron_v2 import MeshFrame
        frame = MeshFrame(
            plane_axes=(0, 1),
            thickness_axis=2,
            center_m=np.zeros(3),
            extents_m=np.array([0.1, 0.1, 0.03])
        )
        baked_geo = BakedMidsoleGeometry(
            thickness_map=thickness_map,
            top_map=top_map,
            bottom_map=bottom_map,
            mins_uv=mins_uv,
            maxs_uv=maxs_uv,
            frame=frame
        )

        xy = np.asarray([[0.0, 0.0]], dtype=np.float64)
        indenter_map = np.full((10, 10), 0.015, dtype=np.float64)
        indenter_valid_map = np.full((10, 10), 1.0, dtype=np.float64)
        material = FoundationMaterial(2.0e6, 2.0, 0.65, 0.0)

        # 2. Evaluate with compression
        result = evaluate_foundation_baked(
            xy,
            baked_geo,
            indenter_map,
            indenter_valid_map,
            cell_area_m2=1.0e-4,
            material=material,
            displacement_m=0.005,
            displacement_velocity_mps=0.0,
            device="cpu"
        )
        self.assertGreater(result.force_n, 0.0)
        self.assertGreater(result.loss, 0.0)

    def test_baked_batch_fit_reduces_loss(self):
        # 1. Create a dummy/synthetic BakedMidsoleGeometry
        thickness_map = np.full((10, 10), 0.03, dtype=np.float64)
        top_map = np.full((10, 10), 0.015, dtype=np.float64)
        bottom_map = np.full((10, 10), -0.015, dtype=np.float64)
        mins_uv = np.array([-0.05, -0.05], dtype=np.float64)
        maxs_uv = np.array([0.05, 0.05], dtype=np.float64)
        from projects.digital_instron_v2 import MeshFrame
        frame = MeshFrame(
            plane_axes=(0, 1),
            thickness_axis=2,
            center_m=np.zeros(3),
            extents_m=np.array([0.1, 0.1, 0.03])
        )
        baked_geo = BakedMidsoleGeometry(
            thickness_map=thickness_map,
            top_map=top_map,
            bottom_map=bottom_map,
            mins_uv=mins_uv,
            maxs_uv=maxs_uv,
            frame=frame
        )

        xy = np.asarray([[0.0, 0.0]], dtype=np.float64)
        indenter_map = np.full((10, 10), 0.015, dtype=np.float64)
        indenter_valid_map = np.full((10, 10), 1.0, dtype=np.float64)

        true_material = FoundationMaterial(
            stiffness_pa=2.0e6,
            ogden_alpha=2.0,
            lock_strain=0.65,
            damping_pa_s=1.0e4,
            damping_power=1.0
        )
        initial = FoundationMaterial(
            stiffness_pa=8.0e5,
            ogden_alpha=2.0,
            lock_strain=0.65,
            damping_pa_s=1.0e4,
            damping_power=1.0
        )

        # Build a batch
        displacement = np.asarray([0.002, 0.004, 0.006], dtype=np.float64)
        velocity = np.zeros(3, dtype=np.float64)
        dt = np.asarray([0.0, 0.01, 0.01], dtype=np.float64)
        weights = np.full(3, 1.0 / 3.0, dtype=np.float64)

        from projects.digital_instron_v2.foundation import evaluate_foundation_baked_batch

        dummy_batch_template = FoundationTrialBatch(
            name="synthetic_baked_batch",
            current_length_m=None, # Not used in baked path
            slack_length_m=None,   # Not used in baked path
            velocity_mps=velocity,
            measured_force_n=np.zeros(3, dtype=np.float64),
            sample_weight=weights,
            cell_area_m2=np.asarray([1.0e-4], dtype=np.float64),
            time_s=np.array([0.0, 0.01, 0.02], dtype=np.float64),
            dt_s=dt,
            displacement_m=displacement,
            phase=("loading", "loading", "loading")
        )

        # Generate "measured" force using true material
        res_true = evaluate_foundation_baked_batch(
            xy,
            baked_geo,
            indenter_map,
            indenter_valid_map,
            dummy_batch_template,
            material=true_material,
            device="cpu"
        )

        batch = FoundationTrialBatch(
            name="synthetic_baked_batch",
            current_length_m=None,
            slack_length_m=None,
            velocity_mps=velocity,
            measured_force_n=res_true.predicted_force_n,
            sample_weight=weights,
            cell_area_m2=np.asarray([1.0e-4], dtype=np.float64),
            time_s=np.array([0.0, 0.01, 0.02], dtype=np.float64),
            dt_s=dt,
            displacement_m=displacement,
            phase=("loading", "loading", "loading")
        )

        # Initial loss with initial params
        res_initial = evaluate_foundation_baked_batch(
            xy,
            baked_geo,
            indenter_map,
            indenter_valid_map,
            batch,
            material=initial,
            device="cpu"
        )

        # Fit
        indenter_maps_by_trial = {batch.name: (indenter_map, indenter_valid_map)}
        fit_res = fit_foundation_material_baked_batches_autodiff(
            xy,
            baked_geo,
            indenter_maps_by_trial,
            [batch],
            initial_material=initial,
            iterations=5,
            device="cpu"
        )

        self.assertLess(fit_res.history[-1]["loss"], res_initial.loss)
        self.assertGreater(fit_res.material.stiffness_pa, initial.stiffness_pa)

    def test_baked_contact_ignores_legacy_displacement_split(self):
        thickness_map = np.full((10, 10), 0.03, dtype=np.float64)
        top_map = np.full((10, 10), 0.015, dtype=np.float64)
        bottom_map = np.full((10, 10), -0.015, dtype=np.float64)
        mins_uv = np.array([-0.05, -0.05], dtype=np.float64)
        maxs_uv = np.array([0.05, 0.05], dtype=np.float64)
        from projects.digital_instron_v2 import MeshFrame

        frame = MeshFrame(
            plane_axes=(0, 1),
            thickness_axis=2,
            center_m=np.zeros(3),
            extents_m=np.array([0.1, 0.1, 0.03]),
        )
        baked_geo = BakedMidsoleGeometry(
            thickness_map=thickness_map,
            top_map=top_map,
            bottom_map=bottom_map,
            mins_uv=mins_uv,
            maxs_uv=maxs_uv,
            frame=frame,
        )
        xy = np.asarray([[0.0, 0.0]], dtype=np.float64)
        indenter_map = np.full((10, 10), 0.015, dtype=np.float64)
        indenter_valid_map = np.full((10, 10), 1.0, dtype=np.float64)
        material = FoundationMaterial(2.0e6, 2.0, 0.65, 0.0)

        result_a = evaluate_foundation_baked(
            xy,
            baked_geo,
            indenter_map,
            indenter_valid_map,
            cell_area_m2=1.0e-4,
            material=material,
            displacement_m=0.005,
            displacement_velocity_mps=0.0,
            top_fraction=0.1,
            bottom_fraction=0.9,
            device="cpu",
        )
        result_b = evaluate_foundation_baked(
            xy,
            baked_geo,
            indenter_map,
            indenter_valid_map,
            cell_area_m2=1.0e-4,
            material=material,
            displacement_m=0.005,
            displacement_velocity_mps=0.0,
            top_fraction=0.9,
            bottom_fraction=0.1,
            device="cpu",
        )

        self.assertAlmostEqual(result_a.force_n, result_b.force_n, places=7)

    def test_baked_fit_locks_out_pasternak_parameter(self):
        thickness_map = np.full((10, 10), 0.03, dtype=np.float64)
        top_map = np.full((10, 10), 0.015, dtype=np.float64)
        bottom_map = np.full((10, 10), -0.015, dtype=np.float64)
        mins_uv = np.array([-0.05, -0.05], dtype=np.float64)
        maxs_uv = np.array([0.05, 0.05], dtype=np.float64)
        from projects.digital_instron_v2 import MeshFrame

        frame = MeshFrame(
            plane_axes=(0, 1),
            thickness_axis=2,
            center_m=np.zeros(3),
            extents_m=np.array([0.1, 0.1, 0.03]),
        )
        baked_geo = BakedMidsoleGeometry(
            thickness_map=thickness_map,
            top_map=top_map,
            bottom_map=bottom_map,
            mins_uv=mins_uv,
            maxs_uv=maxs_uv,
            frame=frame,
        )
        xy = np.asarray([[0.0, 0.0]], dtype=np.float64)
        indenter_map = np.full((10, 10), 0.015, dtype=np.float64)
        indenter_valid_map = np.full((10, 10), 1.0, dtype=np.float64)
        batch = FoundationTrialBatch(
            name="synthetic_baked_batch",
            current_length_m=None,
            slack_length_m=None,
            velocity_mps=np.zeros(2, dtype=np.float64),
            measured_force_n=np.asarray([1.0, 2.0], dtype=np.float64),
            sample_weight=np.full(2, 0.5, dtype=np.float64),
            cell_area_m2=np.asarray([1.0e-4], dtype=np.float64),
            time_s=np.array([0.0, 0.01], dtype=np.float64),
            dt_s=np.array([0.01, 0.01], dtype=np.float64),
            displacement_m=np.asarray([0.002, 0.004], dtype=np.float64),
            phase=("loading", "loading"),
        )
        initial = FoundationMaterial(
            stiffness_pa=8.0e5,
            ogden_alpha=2.0,
            lock_strain=0.65,
            damping_pa_s=1.0e4,
            damping_power=1.0,
            pasternak_stiffness_n_per_m=5.0e5,
        )

        fit_res = fit_foundation_material_baked_batches_autodiff(
            xy,
            baked_geo,
            {batch.name: (indenter_map, indenter_valid_map)},
            [batch],
            initial_material=initial,
            iterations=1,
            device="cpu",
        )

        self.assertEqual(fit_res.material.pasternak_stiffness_n_per_m, 0.0)
        self.assertEqual(fit_res.history[-1]["pasternak_stiffness_n_per_m"], 0.0)
        self.assertEqual(fit_res.history[-1]["grad_pasternak_stiffness_n_per_m"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
