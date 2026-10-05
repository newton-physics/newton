# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise full-buffer partial resets with two assets and alternating states."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.selection import DeformableSurfaceView
from newton.tests._selection_deformable_test_utils import _add_test_cloth


def _reset_cloths(cloths, states, indices_world, mask_cloths, positions_default, velocities_default):
    cloths.set_mask_from_indices(mask_cloths, indices_world)
    for state in states:
        cloths.set_particle_positions(state, positions_default, mask=mask_cloths)
        cloths.set_particle_velocities(state, velocities_default, mask=mask_cloths)


class TestDeformableResets(unittest.TestCase):
    """Reset one asset in selected worlds without disturbing other state."""

    def test_partial_resets_update_both_active_states(self):
        """Preserve matching rows and untouched assets across changing reset sets."""
        for device in wp.get_devices():
            for captured in (False, True) if device.is_cuda else (False,):
                with self.subTest(device=device, captured=captured):
                    prototype = newton.ModelBuilder()
                    _add_test_cloth(prototype, label="cloth_left")
                    _add_test_cloth(prototype, label="cloth_right")
                    builder = newton.ModelBuilder()
                    builder.replicate(prototype, 4)
                    model = builder.finalize(device=device)
                    cloths = DeformableSurfaceView(model, "cloth_left")
                    self.assertEqual(cloths.worlds, list(range(model.world_count)))
                    self.assertEqual(cloths.ranges("particle"), [(0, 4), (8, 12), (16, 20), (24, 28)])
                    states = [model.state(), model.state()]
                    positions_np = cloths.get_particle_positions(model).numpy().copy()
                    velocities_np = cloths.get_particle_velocities(model).numpy().copy()
                    positions_np[:, :, 2] += np.array([10, 20, 30, 40])[:, None]
                    velocities_np[:, :, 0] = np.array([1, 2, 3, 4])[:, None]
                    positions_default = wp.array(positions_np, dtype=wp.vec3, device=device)
                    velocities_default = wp.array(velocities_np, dtype=wp.vec3, device=device)
                    indices_world = wp.array([2, 0], dtype=wp.int32, device=device)
                    mask_cloths = wp.zeros(cloths.count, dtype=wp.bool, device=device)

                    reset_args = (cloths, states, indices_world, mask_cloths, positions_default, velocities_default)
                    _reset_cloths(*reset_args)
                    if captured:
                        with wp.ScopedCapture(device) as capture:
                            _reset_cloths(*reset_args)

                    for step, (indices, selected) in enumerate((([2, 0], [0, 2]), ([1, -1], [1]), ([3, 2], [2, 3]))):
                        # Fresh values expose stale mask bits that a second identical reset would hide.
                        before = []
                        for i, state in enumerate(states):
                            state.particle_q.fill_(100 + 10 * step + i)
                            state.particle_qd.fill_(-100 - 10 * step - i)
                            before.append((state.particle_q.numpy().copy(), state.particle_qd.numpy().copy()))
                        indices_world.assign(np.array(indices, dtype=np.int32))
                        if captured:
                            wp.capture_launch(capture.graph)
                        else:
                            _reset_cloths(*reset_args)
                        for state, (positions, velocities) in zip(states, before, strict=True):
                            # Each world holds four left-cloth particles, then four right-cloth particles.
                            positions.reshape(4, 8, 3)[selected, :4] = positions_np[selected]
                            velocities.reshape(4, 8, 3)[selected, :4] = velocities_np[selected]
                            np.testing.assert_array_equal(state.particle_q.numpy(), positions)
                            np.testing.assert_array_equal(state.particle_qd.numpy(), velocities)
                        np.testing.assert_array_equal(positions_default.numpy(), positions_np)
                        np.testing.assert_array_equal(velocities_default.numpy(), velocities_np)
                        states.reverse()


if __name__ == "__main__":
    unittest.main(verbosity=2)
