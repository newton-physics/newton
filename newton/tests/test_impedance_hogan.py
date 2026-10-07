# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the pelvis-leg impedance pipeline without motion files or a calibrated shoe."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.tests.test_digital_shoe import _tiny_artifact
from projects.impedance_instron.cartesian.mechanics import Body
from projects.impedance_instron.cartesian.shoe import Shoe
from projects.impedance_instron.hogan.adaptation import (
    ShoeAdaptation,
    ShoeDescriptor,
    characterize_shoe,
    leg_stiffness,
)
from projects.impedance_instron.hogan.control import Impedance
from projects.impedance_instron.hogan.mechanics import Chain, RestOfBody, chain_from_profile, from_leg_coordinates
from projects.impedance_instron.hogan.plan import Plan, build_plan
from projects.impedance_instron.hogan.rollout import Config, simulate

_PROFILE = {
    "masses_kg": [8.66, 3.45, 1.46],
    "com_local_m": [[0.17, 0.0], [0.17, 0.0], [0.05, 0.0]],
    "inertias_kg_m2": [0.15, 0.05, 0.008],
}


def _reference(hip_z=0.899, duration_s=0.02, frames=21, grf_z=0.0, cop_x=None):
    """Build a static standing reference with a vertical leg and level foot."""
    time = np.linspace(0.0, duration_s, frames)
    state = np.tile([0.0, hip_z, -0.5 * math.pi, 0.0, 0.0], (frames, 1))
    reference = {
        "time_s": time,
        "state": state,
        "velocity": np.zeros_like(state),
        "hip_target_m": state[:, :2].copy(),
        "joint_target_rad": state[:, 3:5].copy(),
        "lengths_m": np.array([0.4, 0.4]),
        "endpoint_local_m": np.array([0.15, -0.05]),
        "static_pitch_rad": np.asarray(0.0),
        "static_ankle_m": np.zeros(2),
        "static_heel_m": np.zeros(2),
        "subject_mass_kg": np.asarray(70.0),
        "grf_time_s": time.copy(),
        "grf_target_n": np.column_stack((np.zeros(frames), np.full(frames, grf_z))),
        "metadata_json": np.asarray(json.dumps({"schema": "cartesian_single_leg_1", "side": "right"})),
    }
    if cop_x is not None:
        reference["cop_target_m"] = np.full(frames, cop_x)
    return reference


def _tiny_shoe(directory: str) -> Shoe:
    """Write the two-column test shoe with its ankle 0.1 m above the column bottoms."""
    raw = _tiny_artifact()
    raw["visual_meshes"] = {
        "fullfoot_last": {
            "vertices_m": [[-0.02, -0.01, 0.02], [0.02, -0.01, 0.02], [0.02, 0.01, 0.02], [-0.02, 0.01, 0.02]],
            "triangles": [[0, 1, 2], [0, 2, 3]],
        }
    }
    raw["instron_fixtures"] = {
        "fullfoot_last": {
            "carrier_anchor_m": [[-0.01, 0.0, 0.02], [0.01, 0.0, 0.02]],
            "foam_free_top_m": [0.02, 0.02],
            "foam_bottom_m": [0.0, 0.0],
            "rest_length_m": [0.02, 0.02],
            "area_m2": [0.0001, 0.0001],
            "neighbors": [[1, -1, -1, -1], [0, -1, -1, -1]],
            "spacing_m": 0.01,
        }
    }
    path = Path(directory) / "shoe.json"
    path.write_text(json.dumps(raw))
    return Shoe(path, [0, 0, 0.1], 0.0)


def _chain() -> Chain:
    return chain_from_profile(_reference(), _PROFILE)


class TestHoganMechanics(unittest.TestCase):
    """Keep the pelvis-leg chain consistent with the existing leg model."""

    def test_free_fall_from_rest(self):
        """Accelerate every body at gravity with no rotation when unloaded at rest."""
        chain = _chain()
        q = np.array([0.1, 0.9, 1.4, 0.3, -0.5, 0.2])
        mass, bias = chain.dynamics(q, np.zeros(6))
        np.testing.assert_allclose(np.linalg.solve(mass, -bias), [0, -9.81, 0, 0, 0, 0], atol=1e-10)

    def test_point_jacobian_matches_finite_differences(self):
        """Differentiate a foot point position consistently with its analytic Jacobian."""
        chain = _chain()
        q = np.array([0.1, 0.9, 1.4, 0.3, -0.5, 0.2])
        local = np.array([0.08, -0.03])
        _, jacobian, _ = chain.point(q, 3, local)
        numeric = np.empty((2, 6))
        for i in range(6):
            step = np.zeros(6)
            step[i] = 1e-6
            numeric[:, i] = (chain.point(q + step, 3, local)[0] - chain.point(q - step, 3, local)[0]) / 2e-6
        np.testing.assert_allclose(jacobian, numeric, atol=1e-8)

    def test_leg_matches_cartesian_model_with_massless_pelvis(self):
        """Reproduce leg kinematics and projected dynamics of the Cartesian model."""
        leg = Body(
            [0.4, 0.4], [0.15, -0.05], _PROFILE["masses_kg"], _PROFILE["com_local_m"], _PROFILE["inertias_kg_m2"]
        )
        chain = Chain(
            [0.4, 0.4],
            [0.15, -0.05],
            [1e-12, *_PROFILE["masses_kg"]],
            [[0.0, 0.0], *_PROFILE["com_local_m"]],
            [1e-12, *_PROFILE["inertias_kg_m2"]],
        )
        state = np.array([[0.1, 0.9, -1.3, -0.6, 0.25]])
        velocity = np.array([[0.4, -0.2, 2.0, -3.0, 1.5]])
        q, v = from_leg_coordinates(state, velocity, np.array([1.45]), np.array([0.7]))
        np.testing.assert_allclose(chain.kinematics(q[0]), leg.kinematics(state[0]), atol=1e-12)
        # Leg velocities are a linear map of chain velocities: thigh rate = pelvis rate + hip rate.
        transform = np.zeros((5, 6))
        transform[[0, 1, 3, 4], [0, 1, 4, 5]] = 1.0
        transform[2, [2, 3]] = 1.0
        np.testing.assert_allclose(transform @ v[0], velocity[0], atol=1e-12)
        mass_leg, bias_leg = leg.dynamics(state[0], velocity[0])
        mass, bias = chain.dynamics(q[0], v[0])
        np.testing.assert_allclose(mass, transform.T @ mass_leg @ transform, atol=1e-9)
        np.testing.assert_allclose(bias, transform.T @ bias_leg, atol=1e-9)


class TestHoganPlan(unittest.TestCase):
    """Compute inverse-dynamics feedforward from measured force and COP."""

    def test_static_standing_needs_no_pelvis_residual(self):
        """Balance gravity without residuals when body-weight GRF acts under the COM."""
        # A forward trunk COM makes the joints carry load in this posture.
        chain = chain_from_profile(_reference(), _PROFILE, RestOfBody(com_local_m=(0.19, -0.05)))
        q, _ = from_leg_coordinates(_reference()["state"][:1], np.zeros((1, 5)), 0.5 * math.pi, 0.0)
        weight = chain.total_mass_kg * 9.81
        reference = _reference(grf_z=weight, cop_x=float(chain.com(q[0])[0]))
        plan = build_plan(reference, chain, dt_s=1e-3)
        np.testing.assert_allclose(plan.feedforward[:, :3], 0.0, atol=1e-6)
        self.assertGreater(np.max(np.abs(plan.feedforward[:, 3:])), 1.0)

    def test_grf_com_hip_removes_pelvis_force_residual(self):
        """Move a jittering marker hip so the chain COM follows the measured GRF."""
        chain = chain_from_profile(_reference(), _PROFILE, RestOfBody(com_local_m=(0.19, -0.05)))
        q, _ = from_leg_coordinates(_reference()["state"][:1], np.zeros((1, 5)), 0.5 * math.pi, 0.0)
        reference = _reference(
            duration_s=0.2, frames=41, grf_z=chain.total_mass_kg * 9.81, cop_x=float(chain.com(q[0])[0])
        )
        reference["state"][:, 1] += 0.01 * np.sin(2.0 * math.pi * 8.0 * reference["time_s"])
        markers = build_plan(reference, chain, dt_s=1e-3, hip_source="markers", cutoff_hz=None)
        com = build_plan(reference, chain, dt_s=1e-3, hip_source="grf_com", cutoff_hz=None)
        self.assertGreater(np.max(np.abs(markers.feedforward[:, 1])), 100.0)
        np.testing.assert_allclose(com.feedforward[:, :2], 0.0, atol=1e-6)
        np.testing.assert_allclose(np.diff(com.q[:, 1], 2), 0.0, atol=1e-9)

    def test_reach_pins_ankle_and_keeps_foot_angle(self):
        """Re-solve hip and knee so the ankle lands on a target without turning the foot."""
        chain = chain_from_profile(_reference(), _PROFILE)
        q, _ = from_leg_coordinates(
            np.array([[0.0, 0.9, -1.3, -0.6, 0.25]]), np.zeros((1, 5)), np.array([1.45]), np.array([0.0])
        )
        target = np.array([[0.05, 0.2]])
        solved = chain.reach(q, target)
        np.testing.assert_allclose(chain.kinematics(solved[0])[2], target[0], atol=1e-12)
        self.assertAlmostEqual(chain.angle(solved[0], 3), chain.angle(q[0], 3), places=12)
        self.assertLess(solved[0, 4], 0.0)

    def test_measured_contact_requires_cop(self):
        """Reject measured inverse dynamics when the reference has no COP."""
        with self.assertRaises(ValueError):
            build_plan(_reference(grf_z=100.0), _chain(), dt_s=1e-3)


class TestHoganImpedance(unittest.TestCase):
    """Schedule and persist impedance gains."""

    def test_gains_interpolate_and_hold(self):
        """Interpolate gains between knots and hold them outside the knot span."""
        impedance = Impedance([0.0, 1.0], np.array([np.full(6, 100.0), np.full(6, 300.0)]), np.ones((2, 6)))
        np.testing.assert_allclose(impedance.gains(0.25)[0], 150.0)
        np.testing.assert_allclose(impedance.gains(-1.0)[0], 100.0)
        np.testing.assert_allclose(impedance.gains(2.0)[0], 300.0)
        np.testing.assert_allclose(impedance.scaled(np.full(6, 2.0), np.ones(6)).gains(1.0)[0], 600.0)

    def test_save_and_load_round_trip(self):
        """Reload a saved schedule without pickle."""
        impedance = Impedance.critically_damped(np.full(6, 400.0), np.full(6, 4.0), 0.3, knot_count=3)
        np.testing.assert_allclose(impedance.damping, 80.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "impedance.npz"
            impedance.save(path, {"note": "test"})
            loaded = Impedance.load(path)
        np.testing.assert_array_equal(loaded.knot_s, impedance.knot_s)
        np.testing.assert_array_equal(loaded.stiffness, impedance.stiffness)


class TestHoganAdaptation(unittest.TestCase):
    """Map shoe descriptors to joint gain multipliers."""

    def test_series_adaptation_stiffens_the_leg_on_a_softer_shoe(self):
        """Raise joint stiffness on a softer shoe and leave pelvis gains unchanged."""
        reference = ShoeDescriptor(200000.0, 0.6, 1500.0, 0.2)
        softer = ShoeDescriptor(150000.0, 0.6, 1500.0, 0.2)
        adaptation = ShoeAdaptation("series", reference, leg_stiffness_n_m=20000.0)
        stiffness, damping = adaptation.factors(reference)
        np.testing.assert_allclose(stiffness, 1.0)
        stiffness, damping = adaptation.factors(softer)
        expected = 1.0 / (20000.0 * (1 / 20000.0 + 1 / 200000.0 - 1 / 150000.0))
        np.testing.assert_allclose(stiffness[3:], expected)
        np.testing.assert_allclose(stiffness[:3], 1.0)
        np.testing.assert_allclose(damping, np.sqrt(stiffness))
        self.assertGreater(expected, 1.0)

    def test_fixed_and_untrained_learned_adaptation_keep_gains(self):
        """Keep unit multipliers for fixed gains and zero learned sensitivities."""
        reference = ShoeDescriptor(200000.0, 0.6, 1500.0, 0.2)
        softer = ShoeDescriptor(100000.0, 0.8, 1500.0, 0.2)
        for adaptation in (
            ShoeAdaptation("fixed", reference),
            ShoeAdaptation("learned", reference, weights=np.zeros((2, 3))),
        ):
            np.testing.assert_allclose(adaptation.factors(softer)[0], 1.0)

    def test_leg_stiffness_uses_hip_ankle_shortening(self):
        """Divide peak vertical force by hip-to-ankle shortening during contact."""
        chain = _chain()
        knee = -np.linspace(0.0, 0.5, 11)
        state = np.column_stack((np.zeros(11), np.full(11, 0.9), np.full(11, -0.5 * math.pi), knee, -knee))
        q, v = from_leg_coordinates(state, np.zeros_like(state), 0.5 * math.pi, 0.0)
        grf = np.column_stack((np.zeros(11), np.full(11, 1000.0)))
        plan = Plan(np.linspace(0, 0.1, 11), q, v, np.zeros_like(q), np.zeros_like(q), grf, 0.0, "measured")
        shortening = 0.8 - math.sqrt(0.32 + 0.32 * math.cos(0.5))
        self.assertAlmostEqual(leg_stiffness(plan, chain), 1000.0 / shortening, places=6)

    def test_characterize_tiny_shoe(self):
        """Report positive secant stiffness and bounded energy return for a compression cycle."""
        with tempfile.TemporaryDirectory() as directory:
            shoe = _tiny_shoe(directory)
            shoe.foundation.reset()
            force, _ = shoe.apply([0.0, 0.095], [0.0, 0.0], 0.0, 0.0, 1e-4)
            descriptor, curve = characterize_shoe(shoe, peak_force_n=0.5 * force[1], speed_m_s=0.01, dt_s=1e-4)
        self.assertGreater(descriptor.stiffness_n_m, 0.0)
        self.assertGreater(descriptor.energy_return, 0.0)
        self.assertLessEqual(descriptor.energy_return, 1.0 + 1e-6)
        self.assertGreater(np.max(curve[:, 1]), 0.0)


class TestHoganRollout(unittest.TestCase):
    """Track a reference through shoe contact without prescribing the state."""

    def test_replayed_shoe_feedforward_holds_static_stance(self):
        """Hold a static compressed stance when feedforward replays the same shoe."""
        with tempfile.TemporaryDirectory() as directory:
            shoe = _tiny_shoe(directory)
            chain = _chain()
            plan = build_plan(
                _reference(), chain, dt_s=1e-4, contact="shoe_replay", shoe=shoe, contact_threshold_n=1e-3
            )
            mass, _ = chain.dynamics(plan.q[0], plan.v[0])
            impedance = Impedance.critically_damped([20000, 20000, 500, 500, 300, 300], np.diag(mass), plan.duration_s)
            trace, summary = simulate(plan, chain, impedance, shoe, config=Config(contact_threshold_n=1e-3))
        self.assertEqual(summary["status"], "completed", summary["failure"])
        np.testing.assert_allclose(trace["state"], plan.q[: len(trace["state"])], atol=1e-8)
        self.assertGreater(summary["peak_grf_n"][1], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
