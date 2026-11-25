# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check authored soft-scene boundary commands on CPU."""

import sys
import unittest
from pathlib import Path

import numpy as np

import newton

sys.path.insert(0, str(Path(__file__).parents[1]))

from collision_fidelity.soft_scenes import SoftScene


class TestSoftCommands(unittest.TestCase):
    def test_unknown_keyword_is_rejected(self):
        """Do not silently ignore an unknown scene setting."""
        with self.assertRaisesRegex(TypeError, "unexpected_setting"):
            SoftScene("s3_cloth", device="cpu", unexpected_setting=2)

    def test_release_begins_at_declared_substep_start(self):
        """Keep the top face prescribed through 2.8 s, then release it."""
        scene = SoftScene("s2_lower", device="cpu", iterations=1)
        ids = scene.driven_indices
        scene.step_substep(2.8 - scene.dt)
        flags = scene.model.particle_flags.numpy()[ids]
        self.assertTrue(np.all(flags & int(newton.ParticleFlags.ACTIVE) == 0))
        np.testing.assert_allclose(
            scene.state.particle_q.numpy()[ids], scene.initial_positions[ids] + [0, 0, -0.4], atol=1e-7
        )
        scene.step_substep(2.8)
        self.assertTrue(np.all(scene.model.particle_flags.numpy()[ids] & int(newton.ParticleFlags.ACTIVE)))
        self.assertLess(float(scene.state.particle_qd.numpy()[ids, 2].min()), 0.0)


if __name__ == "__main__":
    unittest.main()
