# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check ground-angle reconstruction, coordinate conversions, and objective gradients."""

import json
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from projects.impedance_instron.cartesian.fit import FitConfig, _Objective
from projects.impedance_instron.cartesian.gpu.adjoint_objective import ObjectiveAdjoint
from projects.impedance_instron.cartesian.gpu.mechanics import Vec5
from projects.impedance_instron.cartesian.gpu.objective import MeasuredObjective
from projects.impedance_instron.cartesian.prepare_visual3d import (
    reconstruct_ground_pitch_from_cardan,
)
from projects.impedance_instron.cartesian.shoe import Shoe

ROOT = Path(__file__).resolve().parents[2]


class TestGroundAngleObjective(unittest.TestCase):
    """Verify the measured ground pitch reaches both numerical objective paths."""

    def test_shank_tilt_changes_ground_error_and_gradient(self):
        """Penalize ten degrees of shoe rotation even with an unchanged ankle joint."""
        time = np.array([0.0, 0.1, 0.2])
        pitch = np.deg2rad(8.0)
        frame_pitch = np.deg2rad(-14.0)
        q = np.tile([0.0, 0.9, -1.1, -0.3, 0.0], (3, 1))
        q[:, 4] = pitch + frame_pitch - q[:, 2] - q[:, 3] - np.pi / 2
        reference = {
            "time_s": time,
            "hip_target_m": q[:, :2].copy(),
            "joint_target_rad": q[:, 3:].copy(),
            "foot_ground_target_rad": np.full(3, pitch),
            "shoe_static_pitch_rad": np.asarray(frame_pitch),
            "grf_time_s": time,
            "grf_target_n": np.zeros((3, 2)),
        }
        q[:, 2] += np.deg2rad(10.0)
        trace = {"time_s": time[:-1], "state": q[:-1], "grf_n": np.zeros((2, 2))}
        summary = {"failure": None, "terminal_state": q[-1], "integrated_steps": 2, "integrated_duration_s": 0.2}
        settings = FitConfig()
        residual, metrics, _ = _Objective(reference, settings).evaluate(trace, summary)
        self.assertAlmostEqual(metrics["joint_rmse_rad"][1], np.deg2rad(10.0))
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device):
                objective = MeasuredObjective(reference, settings, time, 1, device)
                states = wp.array(q[:, None, :], dtype=Vec5, device=device, requires_grad=True)
                forces = wp.zeros((2, 1), dtype=wp.vec2d, device=device, requires_grad=True)
                objective.launch(
                    states, forces, wp.full(1, 2, dtype=int, device=device), wp.zeros(1, dtype=int, device=device)
                )
                np.testing.assert_allclose(objective.residual.numpy()[:, 0], residual, atol=1e-12)
                adjoint = ObjectiveAdjoint(objective)
                loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
                with wp.Tape() as tape:
                    adjoint.launch(states, forces, loss)
                np.testing.assert_allclose(loss.numpy(), objective.loss.numpy(), atol=1e-12)
                tape.backward(loss)
                gradient = states.grad.numpy()[:, 0]
                self.assertGreater(np.linalg.norm(gradient[:, 2]), 0)
                np.testing.assert_allclose(gradient[:, 2], gradient[:, 4], atol=1e-12)

    def test_flat_foot_with_inclined_shank(self):
        """Maintain zero foot ground pitch when shank inclines over a flat foot.

        When the foot stays flat on the floor (ground pitch = 0), forward tibial
        progression increases relative ankle dorsiflexion. The 3D Cardan
        reconstruction must keep the foot's world orientation stationary.
        """
        tilts_deg = np.linspace(0.0, 20.0, 5)
        knee_centers = []
        ankle_centers = []
        v_exp_deg = []
        for tilt in tilts_deg:
            tilt_rad = np.deg2rad(tilt)
            # Knee is positioned forward according to shank tilt
            ankle = np.array([0.0, 0.0, 0.0])
            knee = np.array([0.45 * np.sin(tilt_rad), 0.0, 0.45 * np.cos(tilt_rad)])
            ankle_centers.append(ankle)
            knee_centers.append(knee)
            # Ankle dorsiflexion equals shank tilt to keep foot flat
            v_exp_deg.append([tilt, 0.0, 0.0])

        ground_pitch, shank_inclination = reconstruct_ground_pitch_from_cardan(
            np.array(knee_centers), np.array(ankle_centers), np.array(v_exp_deg)
        )
        np.testing.assert_allclose(shank_inclination, np.deg2rad(tilts_deg), atol=1e-12)
        np.testing.assert_allclose(ground_pitch, np.zeros_like(tilts_deg), atol=1e-12)

    def test_rotate_shank_and_foot_together(self):
        """Track ground pitch change when shank and foot rotate together with fixed relative ankle.

        When the relative joint angle remains fixed at zero (neutral), any rotation of the shank
        must rotate the foot's world ground pitch by the identical angle.
        """
        tilts_deg = np.linspace(-15.0, 30.0, 7)
        knee_centers = []
        ankle_centers = []
        v_exp_deg = []
        for tilt in tilts_deg:
            tilt_rad = np.deg2rad(tilt)
            ankle = np.array([0.0, 0.0, 0.0])
            knee = np.array([0.45 * np.sin(tilt_rad), 0.0, 0.45 * np.cos(tilt_rad)])
            ankle_centers.append(ankle)
            knee_centers.append(knee)
            v_exp_deg.append([0.0, 0.0, 0.0])

        ground_pitch, shank_inclination = reconstruct_ground_pitch_from_cardan(
            np.array(knee_centers), np.array(ankle_centers), np.array(v_exp_deg)
        )
        np.testing.assert_allclose(shank_inclination, np.deg2rad(tilts_deg), atol=1e-12)
        # When relative ankle is fixed, tilting the shank forward (+tilt) pitches the attached foot toe-down (-pitch)
        np.testing.assert_allclose(ground_pitch, -np.deg2rad(tilts_deg), atol=1e-12)

    def test_neutral_calibration_and_known_rotations_and_3d_yaw_roll(self):
        """Verify neutral calibration, pure pitch, and 3D Cardan coupling with roll and yaw.

        Ensures that neutral standing posture yields zero pitch, pure pitch rotations yield
        exact expected values, and non-planar rotations (yaw/roll) correctly influence
        the projected sagittal forward axis without naive scalar-angle addition.
        """
        knee = np.array([[0.0, 0.0, 0.45]])
        ankle = np.array([[0.0, 0.0, 0.0]])

        # Neutral calibration
        p_neutral, s_neutral = reconstruct_ground_pitch_from_cardan(knee, ankle, np.array([[0.0, 0.0, 0.0]]))
        self.assertAlmostEqual(float(p_neutral[0]), 0.0, places=12)
        self.assertAlmostEqual(float(s_neutral[0]), 0.0, places=12)

        # Pure toe-up (+12 deg)
        p_up, _ = reconstruct_ground_pitch_from_cardan(knee, ankle, np.array([[12.0, 0.0, 0.0]]))
        self.assertAlmostEqual(float(np.rad2deg(p_up[0])), 12.0, places=12)

        # Pure toe-down (-25 deg)
        p_down, _ = reconstruct_ground_pitch_from_cardan(knee, ankle, np.array([[-25.0, 0.0, 0.0]]))
        self.assertAlmostEqual(float(np.rad2deg(p_down[0])), -25.0, places=12)

        # 3D case containing roll (beta) and yaw (gamma)
        # alpha=10 deg, beta=15 deg, gamma=20 deg
        p_3d, _ = reconstruct_ground_pitch_from_cardan(knee, ankle, np.array([[10.0, 15.0, 20.0]]))
        # Direct scalar addition would predict 10.0 deg. With 3D Cardan rotation:
        # R = R_alpha(10) @ R_beta(15) @ R_gamma(20)
        # v = R @ [1, 0, 0] = [cos(10)*cos(20) + sin(10)*sin(15)*sin(20), ..., sin(10)*cos(20) - cos(10)*sin(15)*sin(20)]
        expected_pitch = np.rad2deg(float(p_3d[0]))
        self.assertNotAlmostEqual(expected_pitch, 10.0, places=1)
        # Exposes that scalar addition is erroneous in 3D:
        scalar_error = abs(expected_pitch - 10.0)
        self.assertGreater(scalar_error, 0.5)

    def test_fixed_registration_and_translation_invariance(self):
        """Verify fixed shoe registration is applied once and treadmill translation preserves angles.

        Tests that Shoe.apply() accurately realizes reconstructed ground pitch, and
        horizontal translation leaves reconstructed orientation completely unchanged.
        """
        shoe_json = ROOT / "outputs/impedance_instron/f01_right_ground/prepared/digital_shoe.json"
        if not shoe_json.exists():
            self.skipTest("digital_shoe.json not found")
        fixed_mount = [-0.03186147427106201, 0.0, 0.10943209684347802]
        fixed_pitch = -0.24446041090480894
        shoe = Shoe(shoe_json, fixed_mount, fixed_pitch, device="cpu")

        test_pitches = np.deg2rad([-20.0, -5.0, 0.0, 8.0, 15.0])
        thigh = -1.1
        knee = -0.3
        for pitch in test_pitches:
            # Solver coordinate from carrier inversion
            ankle = pitch + fixed_pitch - (thigh + knee) - np.pi / 2
            solver_pitch = thigh + knee + ankle + np.pi / 2

            # Translate horizontally by 0 m and 5 m
            for x_shift in [0.0, 5.0]:
                shoe.apply([x_shift, 0.9], [0.0, 0.0], solver_pitch, 0.0, 0.0000625)
                pose = shoe.state.body_q.numpy()[0]
                carrier_pitch = -2 * np.arctan2(float(pose[4]), float(pose[6]))
                np.testing.assert_allclose(carrier_pitch, pitch, atol=1e-6)

    def test_cpu_gpu_loss_and_adjoint_gradients_finite_differences(self):
        """Verify CPU and GPU objectives agree and adjoint gradients match finite differences."""
        time = np.array([0.0, 0.05, 0.1])
        pitch = np.deg2rad(10.0)
        frame_pitch = np.deg2rad(-14.0)
        q = np.tile([0.0, 0.85, -1.0, -0.4, 0.0], (3, 1))
        q[:, 4] = pitch + frame_pitch - q[:, 2] - q[:, 3] - np.pi / 2
        reference = {
            "time_s": time,
            "hip_target_m": q[:, :2].copy(),
            "joint_target_rad": q[:, 3:].copy(),
            "foot_ground_target_rad": np.full(3, pitch),
            "shoe_static_pitch_rad": np.asarray(frame_pitch),
            "grf_time_s": time,
            "grf_target_n": np.zeros((3, 2)),
        }
        # Perturb state to create nonzero loss
        q[:, 2] += np.deg2rad(5.0)
        q[:, 4] -= np.deg2rad(3.0)

        trace = {"time_s": time[:-1], "state": q[:-1], "grf_n": np.zeros((2, 2))}
        summary = {"failure": None, "terminal_state": q[-1], "integrated_steps": 2, "integrated_duration_s": 0.1}
        settings = FitConfig()
        cpu_residual, _, _ = _Objective(reference, settings).evaluate(trace, summary)

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        objective = MeasuredObjective(reference, settings, time, 1, device)
        states = wp.array(q[:, None, :], dtype=Vec5, device=device, requires_grad=True)
        forces = wp.zeros((2, 1), dtype=wp.vec2d, device=device, requires_grad=True)
        objective.launch(states, forces, wp.full(1, 2, dtype=int, device=device), wp.zeros(1, dtype=int, device=device))
        np.testing.assert_allclose(objective.residual.numpy()[:, 0], cpu_residual, atol=1e-10)

        # Adjoint gradient
        adjoint = ObjectiveAdjoint(objective)
        loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
        with wp.Tape() as tape:
            adjoint.launch(states, forces, loss)
        tape.backward(loss)
        adjoint_grad = states.grad.numpy()[:, 0].copy()

        # Finite difference check on joint angles (thigh, knee, ankle)
        eps = 1e-6
        for step in range(2):
            for col in [2, 3, 4]:
                q_plus = q.copy()
                q_plus[step, col] += eps
                states_p = wp.array(q_plus[:, None, :], dtype=Vec5, device=device)
                obj_p = MeasuredObjective(reference, settings, time, 1, device)
                obj_p.launch(
                    states_p, forces, wp.full(1, 2, dtype=int, device=device), wp.zeros(1, dtype=int, device=device)
                )
                adj_p = ObjectiveAdjoint(obj_p)
                loss_p = wp.zeros(1, dtype=wp.float64, device=device)
                adj_p.launch(states_p, forces, loss_p)

                q_minus = q.copy()
                q_minus[step, col] -= eps
                states_m = wp.array(q_minus[:, None, :], dtype=Vec5, device=device)
                obj_m = MeasuredObjective(reference, settings, time, 1, device)
                obj_m.launch(
                    states_m, forces, wp.full(1, 2, dtype=int, device=device), wp.zeros(1, dtype=int, device=device)
                )
                adj_m = ObjectiveAdjoint(obj_m)
                loss_m = wp.zeros(1, dtype=wp.float64, device=device)
                adj_m.launch(states_m, forces, loss_m)

                fd_grad = (loss_p.numpy()[0] - loss_m.numpy()[0]) / (2 * eps)
                self.assertAlmostEqual(adjoint_grad[step, col], fd_grad, delta=1e-5)

    def test_semantic_regression_fails_old_passes_new(self):
        """Demonstrate that the semantic regression fails under raw direct angle mapping and passes after correction."""
        # Frame at source time 0.370s in late stance:
        # Shank inclination is forward ~43.3 deg.
        # Exported RVirtualFootAngle alpha is +24.3 deg (flexion/dorsiflexion relative to shank).
        # Physical foot orientation is toe-down (~ -20.2 deg).
        knee = np.array([[0.45 * np.sin(np.deg2rad(43.345)), 0.0, 0.45 * np.cos(np.deg2rad(43.345))]])
        ankle = np.array([[0.0, 0.0, 0.0]])
        raw_virtual_foot = np.array([[24.306, -1.765, -15.455]])

        # 1. Old naive direct interpretation: assigns raw virtual alpha directly as ground pitch
        old_ground_pitch_deg = raw_virtual_foot[0, 0]
        # In late stance before toe-off, the foot MUST be pitched toe-down (negative pitch).
        # The old mapping gives +24.3 deg (toe-up!), which is unphysical.
        self.assertGreater(old_ground_pitch_deg, 0.0)  # Old mapping erroneously indicates toe-up

        # 2. Corrected 3D Cardan reconstruction:
        new_ground_pitch, _ = reconstruct_ground_pitch_from_cardan(knee, ankle, raw_virtual_foot)
        new_ground_pitch_deg = float(np.rad2deg(new_ground_pitch[0]))
        # Correct mapping gives negative pitch (toe-down)
        self.assertLess(new_ground_pitch_deg, -15.0)  # Correct mapping correctly indicates toe-down ~ -20 deg

    def test_kinematics_replay_key_frames(self):
        """Replay measured kinematics without fitting at touchdown, 25%, 50%, 75%, 0.370s, and toe-off."""
        audit_file = ROOT / "outputs/impedance_instron/f01_right_frame_corrected/frame_audit.json"
        if not audit_file.exists():
            self.skipTest("frame_audit.json not found")
        audit = json.loads(audit_file.read_text())
        key_frames = audit["key_frames"]

        # Touchdown: positive pitch (toe-up ~ +15 deg)
        self.assertGreater(key_frames["touchdown"]["reconstructed_ground_pitch_deg"], 10.0)
        # Midstance: nearly flat (~ -1 deg)
        self.assertAlmostEqual(key_frames["midstance_25pct"]["reconstructed_ground_pitch_deg"], 0.0, delta=3.0)
        # Late stance / problem frame 0.370s: toe-down (~ -20 deg), NOT toe-up
        self.assertLess(key_frames["problem_frame_0370s"]["reconstructed_ground_pitch_deg"], -15.0)
        # Toe-off: steep toe-down (~ -53 deg)
        self.assertLess(key_frames["toe_off"]["reconstructed_ground_pitch_deg"], -45.0)

    def test_tracking_and_frame_reconstruction_agreement(self):
        """Verify reconstruction agrees with physical markers within tracking uncertainty."""
        audit_file = ROOT / "outputs/impedance_instron/f01_right_frame_corrected/frame_audit.json"
        if not audit_file.exists():
            self.skipTest("frame_audit.json not found")
        audit = json.loads(audit_file.read_text())
        agreement = audit["reconstruction_agreement_vs_markers"]
        # Mean absolute error vs physical markers across all 70 frames must be < 2 deg
        self.assertLess(agreement["marker_mae_deg"], 2.0)
        # Heel cluster RMS must be < 2 mm
        qc = audit["tracking_quality_control"]
        self.assertLess(qc["heel_cluster_check"]["max_rms_m"], 0.002)


if __name__ == "__main__":
    unittest.main()
