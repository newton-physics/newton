# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for newton.controllers — framework and ControllerPID."""

import unittest

import numpy as np
import warp as wp

from newton.controllers import ControlGroup, ControllerPID


def _f(values, device):
    return wp.array(values, dtype=wp.float32, device=device)


def _u(values, device):
    return wp.array(values, dtype=wp.uint32, device=device)


def _build_pid(
    *,
    device,
    n=4,
    kp=1.0,
    ki=0.0,
    kd=0.0,
    integral_max=float("inf"),
):
    """Construct a PID + ControlGroup wired to local-layout arrays. Returns
    (group, pid, in_buffers, output_array)."""
    indices = _u(np.arange(n, dtype=np.uint32), device)
    identity = _u(np.arange(n, dtype=np.uint32), device)
    measurement = wp.zeros(n, dtype=wp.float32, device=device)
    measurement_rate = wp.zeros(n, dtype=wp.float32, device=device)
    setpoint = wp.zeros(n, dtype=wp.float32, device=device)
    setpoint_rate = wp.zeros(n, dtype=wp.float32, device=device)
    output = wp.zeros(n, dtype=wp.float32, device=device)
    pid = ControllerPID(
        indices=indices,
        measurement=measurement,
        measurement_rate=measurement_rate,
        setpoint=setpoint,
        setpoint_rate=setpoint_rate,
        kp=(_f([kp] * n, device), identity),
        ki=(_f([ki] * n, device), identity),
        kd=(_f([kd] * n, device), identity),
        integral_max=(_f([integral_max] * n, device), identity),
        output=output,
    )
    group = ControlGroup([pid])
    buffers = {
        "indices": indices,
        "measurement": measurement,
        "measurement_rate": measurement_rate,
        "setpoint": setpoint,
        "setpoint_rate": setpoint_rate,
    }
    return group, pid, buffers, output


class TestControllerPID(unittest.TestCase):
    def test_proportional_only(self):
        """kp * (setpoint - measurement) with no integral or derivative."""
        device = wp.get_device()
        group, _pid, bufs, output = _build_pid(device=device, n=3, kp=2.0)

        bufs["setpoint"].assign(_f([1.0, 2.0, -1.0], device))
        # measurement stays at 0; rates stay at 0.

        s0, s1 = group.state(), group.state()
        group.step(s0, s1, dt=0.01)

        np.testing.assert_allclose(output.numpy(), [2.0, 4.0, -2.0], atol=1e-6)

    def test_integral_accumulates(self):
        """With ki>0, repeated steps with constant error grow the integral
        linearly and add ki*integral to the output."""
        device = wp.get_device()
        ki = 0.5
        dt = 0.1
        group, _pid, bufs, output = _build_pid(device=device, n=1, kp=0.0, ki=ki, kd=0.0)
        bufs["setpoint"].assign(_f([1.0], device))

        s0, s1 = group.state(), group.state()

        running_integral = 0.0
        for step_i in range(5):
            running_integral += 1.0 * dt
            group.step(s0, s1, dt=dt)
            s0, s1 = s1, s0
            expected = ki * running_integral
            self.assertAlmostEqual(
                float(output.numpy()[0]),
                expected,
                places=5,
                msg=f"step {step_i}: integral should be {running_integral:.3f}",
            )

    def test_anti_windup_clamps_integral(self):
        """integral_max bounds the accumulator symmetrically."""
        device = wp.get_device()
        ki = 1.0
        dt = 0.1
        integral_max = 0.3
        group, _pid, bufs, output = _build_pid(
            device=device,
            n=1,
            kp=0.0,
            ki=ki,
            kd=0.0,
            integral_max=integral_max,
        )
        bufs["setpoint"].assign(_f([1.0], device))

        s0, s1 = group.state(), group.state()
        # Run many steps; without clamping the integral would reach 2.0.
        for _ in range(20):
            group.step(s0, s1, dt=dt)
            s0, s1 = s1, s0

        # Output should be ki * integral_max = 0.3 (not 2.0).
        self.assertAlmostEqual(float(output.numpy()[0]), ki * integral_max, places=5)
        self.assertAlmostEqual(float(s0.controller_states[0].integral.numpy()[0]), integral_max, places=5)

    def test_reset_zeros_integral(self):
        """Default reset_state is zero; group.reset clears the integral."""
        device = wp.get_device()
        group, _pid, bufs, _ = _build_pid(device=device, n=2, kp=0.0, ki=1.0, kd=0.0)
        bufs["setpoint"].assign(_f([1.0, 1.0], device))

        s0, s1 = group.state(), group.state()
        for _ in range(5):
            group.step(s0, s1, dt=0.1)
            s0, s1 = s1, s0
        # Integral should be ~0.5 in both DOFs.
        before = s0.controller_states[0].integral.numpy()
        self.assertTrue(np.all(before > 0.0))

        # Reset all DOFs.
        mask = wp.array([True, True], dtype=wp.bool, device=device)
        group.reset(s0, mask=mask)
        after = s0.controller_states[0].integral.numpy()
        np.testing.assert_allclose(after, [0.0, 0.0], atol=1e-7)

    def test_reset_to_nonzero_target(self):
        """Mutating reset_state changes what reset writes; mask selects entries."""
        device = wp.get_device()
        group, pid, bufs, _ = _build_pid(device=device, n=3, kp=0.0, ki=1.0, kd=0.0)
        bufs["setpoint"].assign(_f([1.0, 1.0, 1.0], device))

        # Build up integrals.
        s0, s1 = group.state(), group.state()
        for _ in range(3):
            group.step(s0, s1, dt=0.1)
            s0, s1 = s1, s0

        # Pre-populate the reset target with a non-zero pattern.
        pid.reset_state.integral.assign(_f([0.7, 0.8, 0.9], device))

        # Reset only DOFs 0 and 2 — DOF 1 should keep its accumulated value.
        before = s0.controller_states[0].integral.numpy().copy()
        mask = wp.array([True, False, True], dtype=wp.bool, device=device)
        group.reset(s0, mask=mask)
        after = s0.controller_states[0].integral.numpy()

        self.assertAlmostEqual(float(after[0]), 0.7, places=5)
        self.assertAlmostEqual(float(after[1]), float(before[1]), places=5)
        self.assertAlmostEqual(float(after[2]), 0.9, places=5)


if __name__ == "__main__":
    unittest.main()
