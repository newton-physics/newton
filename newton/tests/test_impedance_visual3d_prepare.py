"""Tests for complete Visual3D Cartesian preparation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from projects.impedance_instron.cartesian import prepare_visual3d
from projects.impedance_instron.cartesian.data import load as load_reference
from projects.impedance_instron.cartesian.profile import load as load_profile


class Visual3DJointAngleParseTest(unittest.TestCase):
    """Validate Visual3D joint-angle exports with empty optional signals."""

    def test_skip_empty_scalar_without_shifting_virtual_foot(self) -> None:
        """Keep the virtual-foot XYZ components aligned after an empty angle."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion_joint_angles.txt"
            path.write_text(
                "source\n"
                "\tRAnkleAngle\tRAnkleAngle\tRAnkleAngle\tRCalAngle\tRVirtualFootAngle\tRVirtualFootAngle\tRVirtualFootAngle\n"
                "\tLINK_MODEL_BASED\tLINK_MODEL_BASED\tLINK_MODEL_BASED\tLINK_MODEL_BASED\tLINK_MODEL_BASED\tLINK_MODEL_BASED\tLINK_MODEL_BASED\n"
                "\tORIGINAL\tORIGINAL\tORIGINAL\tORIGINAL\tORIGINAL\tORIGINAL\tORIGINAL\n"
                "ITEM\tX\tY\tZ\t0\tX\tY\tZ\n"
                "1\t-25\t1\t2\t\t5\t6\t7\n",
                encoding="utf-8",
            )
            labels, values = prepare_visual3d._joint_angles(path)
            self.assertEqual(labels, ("RAnkleAngle", "RVirtualFootAngle"))
            np.testing.assert_array_equal(values, [[-25, 1, 2, 5, 6, 7]])


def _ascii(path: Path, labels: list[str], values: np.ndarray, kind: str = "TARGET") -> None:
    width = 3 * len(labels)

    def repeated(value: str) -> str:
        return " ".join(value for _ in labels for _ in range(3))

    header = [
        "source",
        " ".join(name for name in labels for _ in range(3)),
        repeated(kind),
        repeated("ORIGINAL"),
        "ITEM " + " ".join(["X", "Y", "Z"] * len(labels)),
    ]
    rows = [f"{i + 1} " + " ".join(f"{x:.8f}" for x in row) for i, row in enumerate(values.reshape(len(values), width))]
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


class Visual3DPreparationTest(unittest.TestCase):
    """Verify a complete synthetic Visual3D export reaches runtime contracts."""

    def test_prepare_writes_loadable_bundle(self):
        """Prepare and validate a complete selected-window reference bundle."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dynamic = root / "dynamic"
            static = root / "static"
            output = root / "prepared"
            dynamic.mkdir()
            static.mkdir()
            manifest = {
                "schema": "visual3d_export_manifest_1",
                "point_rate_hz": 100.0,
                "analog_rate_hz": 200.0,
                "marker_units": "m",
                "force_units": "N",
                "moment_units": "N*m",
                "cop_units": "m",
                "up_axis": "+Z",
                "forward_axis": "-Y",
                "force_side": "left",
            }
            (dynamic / "visual3d_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (static / "visual3d_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            names = ["LHEE1", "LHEE2", "LHEE3", "LTOE", "LMT1H", "LMT5H"]
            static_values = np.tile(
                np.array([[0, 0, 0], [0, 0.01, 0], [0.02, 0, 0], [0.25, 0, 0], [0.2, 0, 0], [0.22, 0, 0]]), (4, 1, 1)
            )
            _ascii(static / "static_all_targets.txt", names, static_values)
            static_centers = np.tile(np.array([[0.0, 0.0, 0.8], [0.0, 0.0, 0.45], [0.0, 0.0, 0.08]]), (4, 1, 1))
            _ascii(static / "static_joint_centers.txt", ["LHIP", "LKNEE", "LANKLE"], static_centers, "LINK_MODEL_BASED")
            count = 11
            t = np.arange(count) / 100.0
            dynamic_values = np.tile(static_values[0], (count, 1, 1))
            dynamic_values[:, :, 0] += t[:, None]
            _ascii(dynamic / "motion_processed_targets.txt", names, dynamic_values)
            _ascii(dynamic / "motion_all_targets.txt", names, dynamic_values)
            dynamic_centers = np.tile(static_centers[0], (count, 1, 1))
            dynamic_centers[:, :, 0] += t[:, None]
            _ascii(
                dynamic / "motion_joint_centers.txt", ["LHIP", "LKNEE", "LANKLE"], dynamic_centers, "LINK_MODEL_BASED"
            )
            trial = SimpleNamespace(
                marker_time_s=t,
                marker_names=tuple(names),
                marker_positions_m=dynamic_values,
                marker_valid=np.ones((count, len(names)), dtype=bool),
                analog_time_s=np.arange(22) / 200.0,
                force_n=np.tile(np.array([[0.0, 0.0, 100.0]]), (22, 1)),
                cop_m=np.zeros((22, 3)),
                manifest=manifest,
            )
            trial.force_n[:4, 2] = 0.0
            trial.force_n[-4:, 2] = 0.0
            trial.cop_m[:4] = np.nan
            trial.cop_m[-4:] = np.nan
            original_loader = prepare_visual3d.load_visual3d_export
            prepare_visual3d.load_visual3d_export = lambda _: trial
            try:
                profile = {
                    "schema": "cartesian_single_leg_1",
                    "masses_kg": [1.0, 1.0, 1.0],
                    "com_local_m": [[0, 0], [0, 0], [0, 0]],
                    "inertias_kg_m2": [1, 1, 1],
                    "hip_stiffness_n_m": [1, 1],
                    "hip_damping_ns_m": [0, 0],
                    "joint_stiffness_nm_rad": [1, 1],
                    "joint_damping_nms_rad": [0, 0],
                    "equilibrium_lower": [-2, -2, -3, -3],
                    "equilibrium_upper": [2, 2, 3, 3],
                    "equilibrium_rate_limit": [10, 10, 10, 10],
                    "equilibrium_acceleration_limit": [10, 10, 10, 10],
                    "joint_lower_rad": [-3, -3],
                    "joint_upper_rad": [-0.01, -0.01],
                    "provenance": {"inertial": "test", "impedance": "test", "limits": "test"},
                }
                profile_path = root / "profile.json"
                profile_path.write_text(json.dumps(profile), encoding="utf-8")
                shoe_path = root / "shoe.json"
                shoe_path.write_text("{}", encoding="utf-8")
                bundle = prepare_visual3d.prepare(
                    dynamic,
                    static,
                    output,
                    profile_path,
                    shoe_path,
                    side="left",
                    start_s=0.01,
                    end_s=0.09,
                    subject_mass_kg=70.0,
                    belt_speed_m_s=0.2,
                )
            finally:
                prepare_visual3d.load_visual3d_export = original_loader
            loaded = load_reference(bundle / "reference.npz")
            loaded_profile = load_profile(bundle / "profile.json")
            self.assertEqual(len(loaded["time_s"]), 9)
            self.assertAlmostEqual(float(loaded["state"][0, 0]), 0.0)
            self.assertEqual(loaded_profile["schema"], "cartesian_single_leg_1")
            metadata = json.loads(str(loaded["metadata_json"]))
            self.assertEqual(metadata["joint_angles_source"], "markers")

    def test_prepare_uses_visual3d_joint_angles(self):
        """Verify that exported Visual3D joint angles are used directly when present."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dynamic = root / "dynamic"
            static = root / "static"
            output = root / "prepared"
            dynamic.mkdir()
            static.mkdir()
            manifest = {
                "schema": "visual3d_export_manifest_1",
                "point_rate_hz": 100.0,
                "analog_rate_hz": 200.0,
                "marker_units": "m",
                "force_units": "N",
                "moment_units": "N*m",
                "cop_units": "m",
                "up_axis": "+Z",
                "forward_axis": "-Y",
                "force_side": "left",
            }
            (dynamic / "visual3d_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (static / "visual3d_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            names = ["LHEE1", "LHEE2", "LHEE3", "LTOE", "LMT1H", "LMT5H"]
            static_values = np.tile(
                np.array([[0, 0, 0], [0, 0.01, 0], [0.02, 0, 0], [0.25, 0, 0], [0.2, 0, 0], [0.22, 0, 0]]), (4, 1, 1)
            )
            _ascii(static / "static_all_targets.txt", names, static_values)
            static_centers = np.tile(np.array([[0.0, 0.0, 0.8], [0.0, 0.0, 0.45], [0.0, 0.0, 0.08]]), (4, 1, 1))
            _ascii(static / "static_joint_centers.txt", ["LHIP", "LKNEE", "LANKLE"], static_centers, "LINK_MODEL_BASED")
            count = 11
            t = np.arange(count) / 100.0
            dynamic_values = np.tile(static_values[0], (count, 1, 1))
            dynamic_values[:, :, 0] += t[:, None]
            _ascii(dynamic / "motion_processed_targets.txt", names, dynamic_values)
            _ascii(dynamic / "motion_all_targets.txt", names, dynamic_values)
            dynamic_centers = np.tile(static_centers[0], (count, 1, 1))
            dynamic_centers[:, :, 0] += t[:, None]
            _ascii(
                dynamic / "motion_joint_centers.txt", ["LHIP", "LKNEE", "LANKLE"], dynamic_centers, "LINK_MODEL_BASED"
            )

            # Add motion_joint_angles.txt with LKneeAngle and LVirtualFootAngle
            angle_names = ["LKneeAngle", "LVirtualFootAngle"]
            # Visual3D knee flexion is positive in FullBuild, virtual foot dorsiflexion is positive
            knee_deg = np.linspace(15.0, 25.0, count)
            foot_deg = np.linspace(2.0, 8.0, count)
            angle_values = np.zeros((count, 2, 3))
            angle_values[:, 0, 0] = knee_deg
            angle_values[:, 1, 0] = foot_deg
            _ascii(dynamic / "motion_joint_angles.txt", angle_names, angle_values, "LINK_MODEL_BASED")

            trial = SimpleNamespace(
                marker_time_s=t,
                marker_names=tuple(names),
                marker_positions_m=dynamic_values,
                marker_valid=np.ones((count, len(names)), dtype=bool),
                analog_time_s=np.arange(22) / 200.0,
                force_n=np.tile(np.array([[0.0, 0.0, 100.0]]), (22, 1)),
                cop_m=np.zeros((22, 3)),
                manifest=manifest,
            )
            original_loader = prepare_visual3d.load_visual3d_export
            prepare_visual3d.load_visual3d_export = lambda _: trial
            try:
                profile = {
                    "schema": "cartesian_single_leg_1",
                    "masses_kg": [1.0, 1.0, 1.0],
                    "com_local_m": [[0, 0], [0, 0], [0, 0]],
                    "inertias_kg_m2": [1, 1, 1],
                    "hip_stiffness_n_m": [1, 1],
                    "hip_damping_ns_m": [0, 0],
                    "joint_stiffness_nm_rad": [1, 1],
                    "joint_damping_nms_rad": [0, 0],
                    "equilibrium_lower": [-2, -2, -3, -3],
                    "equilibrium_upper": [2, 2, 3, 3],
                    "equilibrium_rate_limit": [10, 10, 10, 10],
                    "equilibrium_acceleration_limit": [10, 10, 10, 10],
                    "joint_lower_rad": [-3, -3],
                    "joint_upper_rad": [-0.01, -0.01],
                    "provenance": {"inertial": "test", "impedance": "test", "limits": "test"},
                }
                profile_path = root / "profile.json"
                profile_path.write_text(json.dumps(profile), encoding="utf-8")
                shoe_path = root / "shoe.json"
                shoe_path.write_text("{}", encoding="utf-8")
                bundle = prepare_visual3d.prepare(
                    dynamic,
                    static,
                    output,
                    profile_path,
                    shoe_path,
                    side="left",
                    start_s=0.01,
                    end_s=0.09,
                    subject_mass_kg=70.0,
                    belt_speed_m_s=0.2,
                )
                ground_bundle = prepare_visual3d.prepare(
                    dynamic,
                    static,
                    root / "ground",
                    profile_path,
                    shoe_path,
                    side="left",
                    start_s=0.01,
                    end_s=0.09,
                    subject_mass_kg=70.0,
                    belt_speed_m_s=0.2,
                    virtual_foot_reference="ground",
                    shoe_static_pitch_rad=-0.24446041090480894,
                )
            finally:
                prepare_visual3d.load_visual3d_export = original_loader
            loaded = load_reference(bundle / "reference.npz")
            metadata = json.loads(str(loaded["metadata_json"]))
            self.assertEqual(metadata["joint_angles_source"], "visual3d_virtual_foot")
            # Knee flexion in Newton is negative, so sign should be negated
            expected_knee = -np.deg2rad(knee_deg[1:10])
            expected_foot = np.deg2rad(foot_deg[1:10])
            np.testing.assert_allclose(loaded["joint_target_rad"][:, 0], expected_knee, rtol=1e-5)
            np.testing.assert_allclose(loaded["joint_target_rad"][:, 1], expected_foot, rtol=1e-5)

            ground = load_reference(ground_bundle / "reference.npz")
            q = ground["state"]
            reconstructed = q[:, 2] + q[:, 3] + q[:, 4] + np.pi / 2 - ground["shoe_static_pitch_rad"]
            np.testing.assert_allclose(reconstructed, expected_foot, atol=1e-9)
            np.testing.assert_allclose(ground["foot_ground_target_rad"], expected_foot, atol=1e-9)
            self.assertGreater(np.max(np.abs(q[:, 4] - expected_foot)), 0.01)

    def test_prepare_uses_visual3d_anatomical_ankle_with_static_correction(self):
        """Verify that anatomical ankle angle is zeroed against static baseline."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dynamic = root / "dynamic"
            static = root / "static"
            output = root / "prepared"
            dynamic.mkdir()
            static.mkdir()
            manifest = {
                "schema": "visual3d_export_manifest_1",
                "point_rate_hz": 100.0,
                "analog_rate_hz": 200.0,
                "marker_units": "m",
                "force_units": "N",
                "moment_units": "N*m",
                "cop_units": "m",
                "up_axis": "+Z",
                "forward_axis": "-Y",
                "force_side": "left",
            }
            (dynamic / "visual3d_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (static / "visual3d_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            names = ["LHEE1", "LHEE2", "LHEE3", "LTOE", "LMT1H", "LMT5H"]
            static_values = np.tile(
                np.array([[0, 0, 0], [0, 0.01, 0], [0.02, 0, 0], [0.25, 0, 0], [0.2, 0, 0], [0.22, 0, 0]]), (4, 1, 1)
            )
            _ascii(static / "static_all_targets.txt", names, static_values)
            static_centers = np.tile(np.array([[0.0, 0.0, 0.8], [0.0, 0.0, 0.45], [0.0, 0.0, 0.08]]), (4, 1, 1))
            _ascii(static / "static_joint_centers.txt", ["LHIP", "LKNEE", "LANKLE"], static_centers, "LINK_MODEL_BASED")

            # Static ankle angle baseline has an anatomical ~10 deg offset
            static_angles = np.zeros((4, 2, 3))
            static_angles[:, 0, 0] = 0.0  # knee
            static_angles[:, 1, 0] = 10.0  # LAnkleAngle
            _ascii(static / "static_joint_angles.txt", ["LKneeAngle", "LAnkleAngle"], static_angles, "LINK_MODEL_BASED")

            count = 11
            t = np.arange(count) / 100.0
            dynamic_values = np.tile(static_values[0], (count, 1, 1))
            dynamic_values[:, :, 0] += t[:, None]
            _ascii(dynamic / "motion_processed_targets.txt", names, dynamic_values)
            _ascii(dynamic / "motion_all_targets.txt", names, dynamic_values)
            dynamic_centers = np.tile(static_centers[0], (count, 1, 1))
            dynamic_centers[:, :, 0] += t[:, None]
            _ascii(
                dynamic / "motion_joint_centers.txt", ["LHIP", "LKNEE", "LANKLE"], dynamic_centers, "LINK_MODEL_BASED"
            )

            # Dynamic ankle angle: 10 + 5 deg = 15 deg dorsiflexion raw
            dyn_angles = np.zeros((count, 2, 3))
            dyn_angles[:, 0, 0] = 20.0
            dyn_angles[:, 1, 0] = 15.0
            _ascii(dynamic / "motion_joint_angles.txt", ["LKneeAngle", "LAnkleAngle"], dyn_angles, "LINK_MODEL_BASED")

            trial = SimpleNamespace(
                marker_time_s=t,
                marker_names=tuple(names),
                marker_positions_m=dynamic_values,
                marker_valid=np.ones((count, len(names)), dtype=bool),
                analog_time_s=np.arange(22) / 200.0,
                force_n=np.tile(np.array([[0.0, 0.0, 100.0]]), (22, 1)),
                cop_m=np.zeros((22, 3)),
                manifest=manifest,
            )
            original_loader = prepare_visual3d.load_visual3d_export
            prepare_visual3d.load_visual3d_export = lambda _: trial
            try:
                profile = {
                    "schema": "cartesian_single_leg_1",
                    "masses_kg": [1.0, 1.0, 1.0],
                    "com_local_m": [[0, 0], [0, 0], [0, 0]],
                    "inertias_kg_m2": [1, 1, 1],
                    "hip_stiffness_n_m": [1, 1],
                    "hip_damping_ns_m": [0, 0],
                    "joint_stiffness_nm_rad": [1, 1],
                    "joint_damping_nms_rad": [0, 0],
                    "equilibrium_lower": [-2, -2, -3, -3],
                    "equilibrium_upper": [2, 2, 3, 3],
                    "equilibrium_rate_limit": [10, 10, 10, 10],
                    "equilibrium_acceleration_limit": [10, 10, 10, 10],
                    "joint_lower_rad": [-3, -3],
                    "joint_upper_rad": [-0.01, -0.01],
                    "provenance": {"inertial": "test", "impedance": "test", "limits": "test"},
                }
                profile_path = root / "profile.json"
                profile_path.write_text(json.dumps(profile), encoding="utf-8")
                shoe_path = root / "shoe.json"
                shoe_path.write_text("{}", encoding="utf-8")
                bundle = prepare_visual3d.prepare(
                    dynamic,
                    static,
                    output,
                    profile_path,
                    shoe_path,
                    side="left",
                    start_s=0.01,
                    end_s=0.09,
                    subject_mass_kg=70.0,
                    belt_speed_m_s=0.2,
                )
            finally:
                prepare_visual3d.load_visual3d_export = original_loader
            loaded = load_reference(bundle / "reference.npz")
            metadata = json.loads(str(loaded["metadata_json"]))
            self.assertEqual(metadata["joint_angles_source"], "visual3d_ankle_offset_corrected")
            # 15 deg - 10 deg static offset = 5 deg = 0.087266 rad
            expected_ankle = np.deg2rad(5.0)
            np.testing.assert_allclose(loaded["joint_target_rad"][:, 1], expected_ankle, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
