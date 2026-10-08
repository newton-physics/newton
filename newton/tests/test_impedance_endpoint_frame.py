# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify Visual3D endpoint registration without changing measured kinematics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from projects.impedance_instron.cartesian import prepare_visual3d
from projects.impedance_instron.cartesian.data import load as load_reference
from projects.impedance_instron.cartesian.mechanics import Body

GROUND_REFERENCES = ("ground", "reconstructed_ground", "raw_ground_deprecated", "sole_markers")


def _rotation(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def _write_triplets(path: Path, labels: list[str], values: np.ndarray, kind: str = "TARGET") -> None:
    header = [
        "source",
        " ".join(name for name in labels for _ in range(3)),
        " ".join([kind] * (3 * len(labels))),
        " ".join(["ORIGINAL"] * (3 * len(labels))),
        "ITEM " + " ".join(["X", "Y", "Z"] * len(labels)),
    ]
    rows = [
        f"{i + 1} " + " ".join(f"{value:.17g}" for value in row)
        for i, row in enumerate(values.reshape(len(values), -1))
    ]
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


class TestEndpointFramePreparation(unittest.TestCase):
    """Exercise the real preparation path for every accepted reference mode."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dynamic = self.root / "dynamic"
        self.static = self.root / "static"
        self.dynamic.mkdir()
        self.static.mkdir()
        self.marker_pitch = -0.14
        self.shoe_pitch = -0.24446041090480894
        self.displacement = np.array([0.20, -0.07])
        self.ankle = np.array([0.12, 0.01, 0.11])
        self.mth = self.ankle + np.array([self.displacement[0], 0.0, self.displacement[1]])
        self.ground_pitch = np.array([0.0, 0.10, -0.15, 0.05, -0.20])
        self.time = np.arange(5) / 100.0
        self.belt_speed = 0.3
        self.profile_path = self.root / "profile.json"
        self.profile_path.write_text(
            json.dumps(
                {
                    "schema": "cartesian_single_leg_1",
                    "masses_kg": [1.0, 1.0, 1.0],
                    "com_local_m": [[0.1, 0.0], [0.1, 0.0], [0.05, -0.01]],
                    "inertias_kg_m2": [1.0, 1.0, 1.0],
                    "hip_stiffness_n_m": [1.0, 1.0],
                    "hip_damping_ns_m": [0.0, 0.0],
                    "joint_stiffness_nm_rad": [1.0, 1.0],
                    "joint_damping_nms_rad": [0.0, 0.0],
                    "equilibrium_lower": [-2.0, -2.0, -3.0, -3.0],
                    "equilibrium_upper": [2.0, 2.0, 3.0, 3.0],
                    "equilibrium_rate_limit": [10.0] * 4,
                    "equilibrium_acceleration_limit": [10.0] * 4,
                    "joint_lower_rad": [-3.0, -3.0],
                    "joint_upper_rad": [-0.01, -0.01],
                    "provenance": {"inertial": "test", "impedance": "test", "limits": "test"},
                }
            ),
            encoding="utf-8",
        )
        self.shoe_path = self.root / "shoe.json"
        self.shoe_path.write_text("{}", encoding="utf-8")
        self.bundle_count = 0

    def _prepare(
        self, mode: str, shoe_pitch: float | None, *, side: str = "left", angles: bool = True
    ) -> dict[str, np.ndarray]:
        prefix = "L" if side == "left" else "R"
        manifest = {
            "marker_units": "m",
            "up_axis": "+Z",
            "forward_axis": "+X",
            "force_side": side,
        }
        for root in (self.static, self.dynamic):
            (root / "visual3d_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        heel = self.mth - 0.24 * np.array([np.cos(self.marker_pitch), 0.0, np.sin(self.marker_pitch)])
        markers = np.array(
            [
                heel + np.array([-0.01, -0.01, 0.0]),
                heel + np.array([0.01, -0.01, 0.0]),
                heel + np.array([0.0, 0.02, 0.0]),
                self.mth + np.array([0.04, 0.0, 0.0]),
                self.mth + np.array([0.0, -0.03, 0.0]),
                self.mth + np.array([0.0, 0.03, 0.0]),
            ]
        )
        names = [prefix + suffix for suffix in ("HEE1", "HEE2", "HEE3", "TOE", "MT1H", "MT5H")]
        static_values = np.tile(markers, (3, 1, 1))
        centers = self.ankle + np.array([[0.0, 0.0, 0.8], [0.0, 0.0, 0.4], [0.0, 0.0, 0.0]])
        center_names = [prefix + suffix for suffix in ("HIP", "KNEE", "ANKLE")]
        _write_triplets(self.static / "static_all_targets.txt", names, static_values)
        _write_triplets(
            self.static / "static_joint_centers.txt", center_names, np.tile(centers, (3, 1, 1)), "LINK_MODEL_BASED"
        )
        dynamic_values = np.tile(markers, (len(self.time), 1, 1))
        for i, angle in enumerate(self.ground_pitch):
            dynamic_values[i][:, [0, 2]] = (markers[:, [0, 2]] - self.ankle[[0, 2]]) @ _rotation(angle).T
            dynamic_values[i][:, [0, 2]] += self.ankle[[0, 2]]
        for name in ("motion_all_targets.txt", "motion_processed_targets.txt"):
            _write_triplets(self.dynamic / name, names, dynamic_values)
        _write_triplets(
            self.dynamic / "motion_joint_centers.txt",
            center_names,
            np.tile(centers, (len(self.time), 1, 1)),
            "LINK_MODEL_BASED",
        )
        angles_path = self.dynamic / "motion_joint_angles.txt"
        if angles:
            angle_values = np.zeros((len(self.time), 2, 3))
            angle_values[:, 0, 0] = np.arange(len(self.time)) * 2.0
            angle_values[:, 1, 0] = np.rad2deg(self.ground_pitch)
            _write_triplets(
                angles_path, [prefix + "KneeAngle", prefix + "VirtualFootAngle"], angle_values, "LINK_MODEL_BASED"
            )
        else:
            angles_path.unlink(missing_ok=True)
        trial = SimpleNamespace(
            marker_time_s=self.time,
            marker_names=tuple(names),
            marker_positions_m=dynamic_values,
            analog_time_s=self.time,
            force_n=np.tile([12.0, 0.0, 100.0], (len(self.time), 1)),
            cop_m=np.tile([0.25, 0.0, 0.0], (len(self.time), 1)),
            manifest=manifest,
        )
        self.bundle_count += 1
        with patch.object(prepare_visual3d, "load_visual3d_export", return_value=trial):
            bundle = prepare_visual3d.prepare(
                self.dynamic,
                self.static,
                self.root / f"bundle_{self.bundle_count}",
                self.profile_path,
                self.shoe_path,
                side=side,
                start_s=0.0,
                end_s=float(self.time[-1]),
                subject_mass_kg=70.0,
                belt_speed_m_s=self.belt_speed,
                virtual_foot_reference=mode,
                shoe_static_pitch_rad=shoe_pitch,
            )
        return load_reference(bundle / "reference.npz")

    def test_ground_modes_use_shoe_reference_angle(self) -> None:
        """Reconstruct the measured static endpoint in every ground-referenced mode."""
        for side in ("left", "right"):
            for mode in GROUND_REFERENCES:
                for shoe_pitch in (self.shoe_pitch, 0.0, 0.21, self.marker_pitch):
                    with self.subTest(side=side, mode=mode, shoe_pitch=shoe_pitch):
                        reference = self._prepare(mode, shoe_pitch, side=side)
                        local = reference["endpoint_local_m"]
                        np.testing.assert_allclose(local, _rotation(shoe_pitch).T @ self.displacement, atol=1e-14)
                        np.testing.assert_allclose(_rotation(shoe_pitch) @ local, self.displacement, atol=1e-14)
                        self.assertAlmostEqual(float(reference["static_pitch_rad"]), self.marker_pitch)
                        np.testing.assert_allclose(reference["static_ankle_m"], self.ankle[[0, 2]], atol=1e-14)
                        state = reference["state"]
                        absolute_pitch = state[:, 2] + state[:, 3] + state[:, 4] + np.pi / 2
                        np.testing.assert_allclose(absolute_pitch, self.ground_pitch + shoe_pitch, atol=1e-14)
                        reconstructed = np.array([_rotation(angle) @ local for angle in absolute_pitch])
                        expected = np.array([_rotation(angle) @ self.displacement for angle in self.ground_pitch])
                        np.testing.assert_allclose(reconstructed, expected, atol=1e-14)
                        np.testing.assert_allclose(reference["foot_ground_target_rad"], self.ground_pitch, atol=1e-14)

    def test_shank_and_marker_modes_preserve_legacy_endpoint(self) -> None:
        """Keep measured-pitch registration regardless of an unused shoe-pitch argument."""
        for angles in (True, False):
            for side in ("left", "right"):
                for shoe_pitch in (None, 0.0, self.shoe_pitch, 0.21):
                    with self.subTest(angles=angles, side=side, shoe_pitch=shoe_pitch):
                        reference = self._prepare("shank", shoe_pitch, angles=angles, side=side)
                        np.testing.assert_allclose(
                            reference["endpoint_local_m"],
                            _rotation(self.marker_pitch).T @ self.displacement,
                            atol=1e-14,
                        )
                        self.assertNotIn("foot_ground_target_rad", reference)
                        metadata = json.loads(str(reference["metadata_json"]))
                        self.assertEqual(
                            metadata["joint_angles_source"], "visual3d_virtual_foot" if angles else "markers"
                        )
                        expected_ankle = self.ground_pitch if angles else self.ground_pitch + self.marker_pitch
                        np.testing.assert_allclose(reference["state"][:, 4], expected_ankle, atol=1e-14)

    def test_sole_markers_without_joint_angles_uses_shoe_frame(self) -> None:
        """Select the endpoint frame by reference mode rather than the angle-source fallback."""
        reference = self._prepare("sole_markers", self.shoe_pitch, angles=False)
        np.testing.assert_allclose(
            _rotation(self.shoe_pitch) @ reference["endpoint_local_m"], self.displacement, atol=1e-14
        )
        np.testing.assert_allclose(reference["foot_ground_target_rad"], self.ground_pitch, atol=1e-14)

    def test_static_endpoint_height_regression(self) -> None:
        """Remove the approximately 21 mm static height error without moving the ankle."""
        reference = self._prepare("sole_markers", self.shoe_pitch)
        legacy_local = _rotation(self.marker_pitch).T @ self.displacement
        legacy_error = _rotation(self.shoe_pitch) @ legacy_local - self.displacement
        self.assertAlmostEqual(legacy_error[1], -0.0205, delta=0.001)
        body = Body(reference["lengths_m"], reference["endpoint_local_m"], [1.0] * 3, [[0.0, 0.0]] * 3, [1.0] * 3)
        positions = body.kinematics(reference["state"][0])
        np.testing.assert_allclose(positions[2], self.ankle[[0, 2]], atol=1e-14)
        np.testing.assert_allclose(positions[3], self.mth[[0, 2]], atol=1e-14)

    def test_endpoint_change_preserves_all_other_reference_fields(self) -> None:
        """Keep raw angles, state, ankle, GRF, COP, and metadata independent of the endpoint."""
        for mode in (*GROUND_REFERENCES, "shank"):
            with self.subTest(mode=mode):
                reference = self._prepare(mode, self.shoe_pitch)
                legacy_local = _rotation(self.marker_pitch).T @ self.displacement
                with patch.object(prepare_visual3d, "_endpoint_in_state_frame", return_value=legacy_local):
                    legacy = self._prepare(mode, self.shoe_pitch)
                self.assertEqual(reference.keys(), legacy.keys())
                for name in reference:
                    if name != "endpoint_local_m":
                        np.testing.assert_array_equal(reference[name], legacy[name], err_msg=name)
                np.testing.assert_allclose(reference["raw_virtual_foot_angle_rad"][:, 0], self.ground_pitch, atol=1e-14)
                np.testing.assert_allclose(reference["state"][:, 3], -np.deg2rad(np.arange(5) * 2.0), atol=1e-14)
                np.testing.assert_array_equal(reference["grf_target_n"], np.tile([12.0, 100.0], (5, 1)))
                expected_ankle = np.tile(self.ankle[[0, 2]], (5, 1))
                expected_ankle[:, 0] += self.time * self.belt_speed
                np.testing.assert_allclose(reference["ankle_target_m"], expected_ankle, atol=1e-14)
                np.testing.assert_allclose(reference["cop_target_m"], 0.25 + self.time * self.belt_speed, atol=1e-14)

    def test_ground_modes_still_require_finite_shoe_pitch(self) -> None:
        """Preserve rejection of missing and nonfinite ground-frame registration angles."""
        for mode in GROUND_REFERENCES:
            for shoe_pitch in (None, np.nan, np.inf, -np.inf):
                with self.subTest(mode=mode, shoe_pitch=shoe_pitch):
                    with self.assertRaisesRegex(
                        ValueError, "Ground-referenced input requires the fixed shoe static pitch"
                    ):
                        self._prepare(mode, shoe_pitch)


class TestEndpointFrameTransform(unittest.TestCase):
    """Check the explicit pure transform independently of export IO."""

    def test_reference_angle_round_trip(self) -> None:
        """Preserve signed measured offsets and their norm across reference angles and wraps."""
        for displacement in ([0.2, -0.07], [-0.2, 0.07], [0.0, 0.0], [0.0, -0.1], [0.1, 0.0]):
            measured = np.array(displacement)
            measured.setflags(write=False)
            for pitch in (-2 * np.pi, -np.pi, -np.pi / 2, -0.24446041090480894, 0.0, 0.14, np.pi / 2, np.pi, 2 * np.pi):
                with self.subTest(displacement=displacement, pitch=pitch):
                    local = prepare_visual3d._endpoint_in_state_frame(measured, reference_pitch_rad=pitch)
                    np.testing.assert_allclose(_rotation(pitch) @ local, measured, atol=1e-14)
                    self.assertAlmostEqual(float(np.linalg.norm(local)), float(np.linalg.norm(measured)), places=14)
                    np.testing.assert_array_equal(measured, displacement)
                    self.assertFalse(np.shares_memory(local, measured))

    def test_inverse_rotation_preserves_toe_up_sign(self) -> None:
        """Rotate world +x toward local -z for a positive toe-up reference angle."""
        local = prepare_visual3d._endpoint_in_state_frame(np.array([1.0, 0.0]), reference_pitch_rad=np.pi / 2)
        np.testing.assert_allclose(local, [0.0, -1.0], atol=1e-14)
        zero_pitch = prepare_visual3d._endpoint_in_state_frame(np.array([0.2, -0.07]), reference_pitch_rad=0.0)
        np.testing.assert_array_equal(zero_pitch, [0.2, -0.07])

    def test_explicit_frame_is_required(self) -> None:
        """Reject an omitted or positional frame instead of silently inferring marker pitch."""
        with self.assertRaises(TypeError):
            prepare_visual3d._endpoint_in_state_frame(np.array([0.2, -0.07]))
        with self.assertRaises(TypeError):
            prepare_visual3d._endpoint_in_state_frame(np.array([0.2, -0.07]), 0.0)

    def test_endpoint_registration_leaves_ankle_and_dynamics_unchanged(self) -> None:
        """Keep leg kinematics, mass, bias, and fixed foot-local point geometry unchanged."""
        displacement = np.array([0.2, -0.07])
        endpoints = [
            prepare_visual3d._endpoint_in_state_frame(displacement, reference_pitch_rad=pitch)
            for pitch in (-0.14, -0.24446041090480894)
        ]
        bodies = [
            Body([0.4, 0.45], local, [8.0, 4.0, 1.0], [[0.2, 0.0], [0.2, 0.0], [0.05, -0.01]], [1.0] * 3)
            for local in endpoints
        ]
        velocity = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
        for foot_pitch in (-0.4, 0.0, 0.3):
            with self.subTest(foot_pitch=foot_pitch):
                q = np.array([0.1, 0.9, -1.1, -0.3, foot_pitch + 1.4 - np.pi / 2])
                np.testing.assert_array_equal(bodies[0].kinematics(q)[:3], bodies[1].kinematics(q)[:3])
                for old, new in zip(bodies[0].dynamics(q, velocity), bodies[1].dynamics(q, velocity), strict=True):
                    np.testing.assert_array_equal(old, new)
                for old, new in zip(
                    bodies[0].point(q, 2, [0.1, -0.05], velocity),
                    bodies[1].point(q, 2, [0.1, -0.05], velocity),
                    strict=True,
                ):
                    np.testing.assert_array_equal(old, new)


if __name__ == "__main__":
    unittest.main()
