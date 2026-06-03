# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for newton.controllers — framework and ControllerPID."""

import unittest

import numpy as np
import warp as wp

from newton.controllers import ControlGroup, ControllerPID


class TestControllerPID(unittest.TestCase):
    def test_proportional_only(self):
        """kp * (setpoint - measurement) with no integral or derivative."""
        device = wp.get_device()
        indices = wp.array([0, 1, 2], dtype=wp.uint32, device=device)
        identity = wp.array([0, 1, 2], dtype=wp.uint32, device=device)
        output = wp.zeros(3, dtype=wp.float32, device=device)

        pid = ControllerPID(
            indices=indices,
            measurement=wp.zeros(3, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(3, dtype=wp.float32, device=device),
            setpoint=wp.array([1.0, 2.0, -1.0], dtype=wp.float32, device=device),
            setpoint_rate=wp.zeros(3, dtype=wp.float32, device=device),
            kp=(wp.array([2.0, 2.0, 2.0], dtype=wp.float32, device=device), identity),
            ki=(wp.array([0.0, 0.0, 0.0], dtype=wp.float32, device=device), identity),
            kd=(wp.array([0.0, 0.0, 0.0], dtype=wp.float32, device=device), identity),
            integral_max=(wp.array([np.inf, np.inf, np.inf], dtype=wp.float32, device=device), identity),
            output=output,
        )
        group = ControlGroup([pid])

        s0 = group.state()
        s1 = group.state()
        group.step(s0, s1, dt=0.01)

        np.testing.assert_allclose(output.numpy(), [2.0, 4.0, -2.0], atol=1e-6)

    def test_integral_accumulates(self):
        """With ki>0, repeated steps with constant error grow the integral linearly."""
        device = wp.get_device()
        indices = wp.array([0], dtype=wp.uint32, device=device)
        identity = wp.array([0], dtype=wp.uint32, device=device)
        output = wp.zeros(1, dtype=wp.float32, device=device)

        pid = ControllerPID(
            indices=indices,
            measurement=wp.zeros(1, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(1, dtype=wp.float32, device=device),
            setpoint=wp.array([1.0], dtype=wp.float32, device=device),
            setpoint_rate=wp.zeros(1, dtype=wp.float32, device=device),
            kp=(wp.array([0.0], dtype=wp.float32, device=device), identity),
            ki=(wp.array([0.5], dtype=wp.float32, device=device), identity),
            kd=(wp.array([0.0], dtype=wp.float32, device=device), identity),
            integral_max=(wp.array([np.inf], dtype=wp.float32, device=device), identity),
            output=output,
        )
        group = ControlGroup([pid])

        s0 = group.state()
        s1 = group.state()
        dt = 0.1
        running_integral = 0.0
        for step_i in range(5):
            running_integral += dt
            group.step(s0, s1, dt=dt)
            s0, s1 = s1, s0
            self.assertAlmostEqual(
                float(output.numpy()[0]),
                0.5 * running_integral,
                places=5,
                msg=f"step {step_i}: integral should be {running_integral:.3f}",
            )

    def test_anti_windup_clamps_integral(self):
        """integral_max bounds the accumulator symmetrically."""
        device = wp.get_device()
        indices = wp.array([0], dtype=wp.uint32, device=device)
        identity = wp.array([0], dtype=wp.uint32, device=device)
        output = wp.zeros(1, dtype=wp.float32, device=device)

        pid = ControllerPID(
            indices=indices,
            measurement=wp.zeros(1, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(1, dtype=wp.float32, device=device),
            setpoint=wp.array([1.0], dtype=wp.float32, device=device),
            setpoint_rate=wp.zeros(1, dtype=wp.float32, device=device),
            kp=(wp.array([0.0], dtype=wp.float32, device=device), identity),
            ki=(wp.array([1.0], dtype=wp.float32, device=device), identity),
            kd=(wp.array([0.0], dtype=wp.float32, device=device), identity),
            integral_max=(wp.array([0.3], dtype=wp.float32, device=device), identity),
            output=output,
        )
        group = ControlGroup([pid])

        s0 = group.state()
        s1 = group.state()
        # Without clamping the integral would reach 2.0 after 20 steps.
        for _ in range(21):
            group.step(s0, s1, dt=0.1)
            s0, s1 = s1, s0

        self.assertAlmostEqual(float(output.numpy()[0]), 0.3, places=5)
        self.assertAlmostEqual(float(s0.controller_states[0].integral.numpy()[0]), 0.3, places=5)

    def test_reset_zeros_integral(self):
        """Default reset_state is zero; group.reset clears the integral."""
        device = wp.get_device()
        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
        identity = wp.array([0, 1], dtype=wp.uint32, device=device)

        pid = ControllerPID(
            indices=indices,
            measurement=wp.zeros(2, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            setpoint=wp.array([1.0, 1.0], dtype=wp.float32, device=device),
            setpoint_rate=wp.zeros(2, dtype=wp.float32, device=device),
            kp=(wp.array([0.0, 0.0], dtype=wp.float32, device=device), identity),
            ki=(wp.array([1.0, 1.0], dtype=wp.float32, device=device), identity),
            kd=(wp.array([0.0, 0.0], dtype=wp.float32, device=device), identity),
            integral_max=(wp.array([np.inf, np.inf], dtype=wp.float32, device=device), identity),
            output=wp.zeros(2, dtype=wp.float32, device=device),
        )
        group = ControlGroup([pid])

        s0 = group.state()
        s1 = group.state()
        for _ in range(5):
            group.step(s0, s1, dt=0.1)
            s0, s1 = s1, s0
        self.assertTrue(np.all(s0.controller_states[0].integral.numpy() > 0.0))

        group.reset(s0, mask=wp.array([True, True], dtype=wp.bool, device=device))
        np.testing.assert_allclose(s0.controller_states[0].integral.numpy(), [0.0, 0.0], atol=1e-7)

    def test_reset_to_nonzero_target(self):
        """Mutating reset_state changes what reset writes; mask selects entries."""
        device = wp.get_device()
        indices = wp.array([0, 1, 2], dtype=wp.uint32, device=device)
        identity = wp.array([0, 1, 2], dtype=wp.uint32, device=device)

        pid = ControllerPID(
            indices=indices,
            measurement=wp.zeros(3, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(3, dtype=wp.float32, device=device),
            setpoint=wp.array([1.0, 1.0, 1.0], dtype=wp.float32, device=device),
            setpoint_rate=wp.zeros(3, dtype=wp.float32, device=device),
            kp=(wp.array([0.0, 0.0, 0.0], dtype=wp.float32, device=device), identity),
            ki=(wp.array([1.0, 1.0, 1.0], dtype=wp.float32, device=device), identity),
            kd=(wp.array([0.0, 0.0, 0.0], dtype=wp.float32, device=device), identity),
            integral_max=(wp.array([np.inf, np.inf, np.inf], dtype=wp.float32, device=device), identity),
            output=wp.zeros(3, dtype=wp.float32, device=device),
        )
        group = ControlGroup([pid])

        s0 = group.state()
        s1 = group.state()
        for _ in range(3):
            group.step(s0, s1, dt=0.1)
            s0, s1 = s1, s0

        pid.reset_state.integral.assign(wp.array([0.7, 0.8, 0.9], dtype=wp.float32, device=device))

        # Reset only slots 0 and 2 — slot 1 keeps its accumulated value.
        before = s0.controller_states[0].integral.numpy().copy()
        group.reset(s0, mask=wp.array([True, False, True], dtype=wp.bool, device=device))
        after = s0.controller_states[0].integral.numpy()

        self.assertAlmostEqual(float(after[0]), 0.7, places=5)
        self.assertAlmostEqual(float(after[1]), float(before[1]), places=5)
        self.assertAlmostEqual(float(after[2]), 0.9, places=5)


if __name__ == "__main__":
    unittest.main()
