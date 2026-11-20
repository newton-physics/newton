# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check cable recipe selection and substeps without advancing physics."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parents[1]))

from collision_fidelity.cable_scenes import build_scene
from collision_fidelity.catalog import load_cases


class TestCollisionFidelityCableInputs(unittest.TestCase):
    """Keep analytic cables distinct from the robot-hand replacement recipes."""

    def test_base_factory_rejects_removed_jaw_recipes(self):
        """Reject C8 IDs at selection while retaining the two C9 catalog cases."""
        for case_id in ("c8_bundle_squeeze", "c8_pile_pickup"):
            with self.subTest(case_id=case_id):
                with self.assertRaisesRegex(ValueError, "Unknown cable scene"):
                    build_scene(case_id, device="cpu")
        cases = load_cases()
        for case_id in ("c9_bundle_squeeze_withdraw", "c9_pile_pickup"):
            self.assertEqual(cases[case_id]["scene_id"], case_id)
            self.assertEqual(cases[case_id]["support"]["status"], "supported")

    def test_invalid_substeps_are_rejected(self):
        """Reject noninteger inputs and values above the ten-substep limit."""
        for substeps in (2.5, 2.0, "2", True, np.float64(2.5), 12):
            with self.subTest(substeps=substeps):
                with self.assertRaisesRegex(ValueError, "even integer"):
                    build_scene("c6_crossed_cables", device="cpu", substeps=substeps)

    def test_numpy_integer_builds_the_actual_cpu_model(self):
        """Accept integer scalar inputs and preserve the declared cable timestep."""
        scene = build_scene("c6_crossed_cables", device="cpu", substeps=np.int64(2))
        self.assertEqual(scene.substeps, 2)
        self.assertAlmostEqual(scene.dt, 1.0 / 120.0)
        self.assertEqual(len(scene.cable_bodies), 56)
        self.assertEqual(len(scene.fixture_ids), 4)
        self.assertEqual(scene.model.particle_count, 0)
        self.assertEqual(scene.time, 0.0)


if __name__ == "__main__":
    unittest.main()
