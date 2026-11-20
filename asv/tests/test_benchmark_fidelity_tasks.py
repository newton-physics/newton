# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check task evidence reductions without advancing a solver."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton

sys.path.insert(0, str(Path(__file__).parents[1]))
from collision_fidelity.catalog import load_cases
from collision_fidelity.protocol import Timeline
from collision_fidelity.tasks import criterion_results, sample_task


class TestFidelityTasks(unittest.TestCase):
    """Distinguish measurements, negative witnesses, and missing acceptance tests."""

    def setUp(self):
        """Declare a required hand-supported hold with a prepared start."""
        self.protocol = {
            "family": "Robot manipulation",
            "hold_start": 1.0,
            "hold_end": 61.0,
            "initial_hold_prepared": True,
            "category_applicability": {"hold": True, "grasp": True, "place_release": True},
        }

    def test_finite_completion_is_not_task_pass(self):
        """Pass numerical completion alone while keeping unmeasured tasks incomplete."""
        result = criterion_results([], self.protocol, execution={"status": "completed", "completed": True})
        self.assertEqual(result["numerical"]["status"], "pass")
        self.assertEqual(result["grasp"]["status"], "not_applicable")
        self.assertEqual(result["hold"]["status"], "incomplete")
        self.assertEqual(result["overall"]["status"], "incomplete")

    def test_required_hold_separation_proves_failure(self):
        """Use complete finite all-hand separation as a sampled negative hold witness."""
        row = {
            "time_s": 60.0,
            "finite": True,
            "hands_complete": True,
            "object_complete": True,
            "hand_count": 2,
            "hand_separation_lower_bound_m": [0.1, 0.2],
        }
        result = criterion_results([row], self.protocol)
        self.assertEqual(result["hold"]["status"], "fail")
        self.assertEqual(result["hold"]["evidence"]["first_proven_separation_s"], 60.0)
        self.assertEqual(result["overall"]["status"], "fail")

    def test_overlap_or_missing_hand_is_not_contact(self):
        """Reject incomplete hand coverage and bounds overlap as retention evidence."""
        for gaps, count, complete in [([0.1], 2, True), ([0, 0], 2, True), ([0.1, 0.2], 2, False)]:
            row = {
                "time_s": 30.0,
                "finite": True,
                "hands_complete": complete,
                "object_complete": True,
                "hand_count": count,
                "hand_separation_lower_bound_m": gaps,
            }
            self.assertEqual(criterion_results([row], self.protocol)["hold"]["status"], "incomplete")

    def test_failure_outside_hold_is_not_hold_failure(self):
        """Keep release separation and nonfinite rows out of hold witnesses."""
        rows = [
            {
                "time_s": 62,
                "finite": True,
                "hands_complete": True,
                "object_complete": True,
                "hand_count": 1,
                "hand_separation_lower_bound_m": [1],
            },
            {
                "time_s": 30,
                "finite": False,
                "hands_complete": True,
                "object_complete": True,
                "hand_count": 1,
                "hand_separation_lower_bound_m": [1],
            },
        ]
        self.assertEqual(criterion_results(rows, self.protocol)["hold"]["status"], "incomplete")

    def test_nonfinite_overrides_completed_flag(self):
        """Fail numerical validity even when the requested timeline was completed."""
        result = criterion_results(
            [],
            self.protocol,
            execution={"status": "completed", "completed": True},
            measurements={"finite": {"all_samples_finite": False, "first_nonfinite_time_s": 25}},
        )
        self.assertEqual(result["numerical"]["status"], "fail")
        self.assertEqual(result["overall"]["status"], "fail")

    def test_unsupported_is_excluded(self):
        """Exclude unsupported execution from overall task scoring."""
        result = criterion_results([], self.protocol, execution={"status": "unsupported", "completed": False})
        self.assertEqual(result["overall"]["status"], "unsupported")
        self.assertFalse(result["overall"]["applicable"])

    def test_passive_case_has_no_manipulation_tasks(self):
        """Leave passive cards unscored for grasp, hold, transport, and release."""
        result = criterion_results(
            [],
            {"family": "Soft contact", "category_applicability": {}},
            execution={"status": "completed", "completed": True},
        )
        for name in ("grasp", "hold", "transport", "place_release", "friction_separation"):
            self.assertEqual(result[name]["status"], "not_applicable")

    def test_freefall_does_not_require_static_support_penetration(self):
        """Exclude fixed-support penetration for the two support-free controls."""
        cases = load_cases()
        for case_id in ("s3_cloth_10substeps", "s3_pads_10substeps"):
            with self.subTest(case=case_id):
                case = cases[case_id]
                self.assertFalse(case["applicability"]["penetration"])
                result = criterion_results([], Timeline(case).protocol)
                self.assertEqual(result["penetration"]["status"], "not_applicable")
                self.assertEqual(result["dynamic"]["status"], "incomplete")

    def test_bounds_use_actual_radius_and_margin(self):
        """Bound every particle shell and original rigid box with actual model radii."""
        model = SimpleNamespace(
            particle_radius=np.array([0.1, 0.3]),
            particle_mass=np.ones(2),
            shape_body=np.array([0]),
            shape_type=np.array([7]),
            shape_scale=np.array([[0.1, 0.1, 0.1]]),
            shape_margin=np.array([0.05]),
            shape_transform=np.array([[0, 0, 0, 0, 0, 0, 1]]),
        )
        state = SimpleNamespace(
            particle_q=np.array([[0, 0, 0], [1, 0, 0]]),
            particle_qd=np.zeros((2, 3)),
            body_q=np.array([[2, 0, 0, 0, 0, 0, 1]]),
            body_qd=np.zeros((1, 6)),
        )
        scene = SimpleNamespace(
            model=model, state=state, protocol=self.protocol, hands=[SimpleNamespace(shape_ids=[0], label="hand")]
        )
        row = sample_task(scene, 5.0)
        self.assertTrue(row["object_complete"])
        self.assertTrue(row["hands_complete"])
        self.assertAlmostEqual(row["whole_object_bounds_m"][1][0], 1.3)
        self.assertAlmostEqual(row["hand_separation_lower_bound_m"][0], 0.55)
        self.assertEqual(criterion_results([row], self.protocol)["hold"]["status"], "fail")

    def test_public_model_geometry_and_state_are_read_only(self):
        """Read actual Newton collision geometry without moving particles or rigid bodies."""
        builder = newton.ModelBuilder()
        builder.add_particle((0, 0, 0), (0, 0, 0), 1.0, radius=0.1)
        body = builder.add_body(xform=wp.transform((2, 0, 0), wp.quat_identity()), is_kinematic=True)
        shape = builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2, cfg=newton.ModelBuilder.ShapeConfig(margin=0.05))
        model = builder.finalize(device="cpu")
        current = model.state()
        before = current.particle_q.numpy().copy(), current.body_q.numpy().copy()
        scene = SimpleNamespace(model=model, state=current, hands=[SimpleNamespace(shape_ids=[shape])])
        row = sample_task(scene, 4.0, protocol=self.protocol)
        self.assertTrue(row["hands_complete"] and row["object_complete"])
        self.assertAlmostEqual(row["hand_separation_lower_bound_m"][0], 1.65, places=6)
        np.testing.assert_array_equal(current.particle_q.numpy(), before[0])
        np.testing.assert_array_equal(current.body_q.numpy(), before[1])


if __name__ == "__main__":
    unittest.main()
