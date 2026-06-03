# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for newton.controllers — framework and ControllerPID."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.controllers import ControlGroup, ControllerDifferentialIK, ControllerPID


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


class TestControllerDifferentialIK(unittest.TestCase):
    def test_target_equals_current_gives_zero_velocity(self):
        """With target == current site pose, the DLS solve produces q_dot ≈ 0."""
        device = wp.get_device()

        # 2-link planar arm rotating about z. Each link is 1 unit long.
        # Joint 0: world → link0 at origin. Joint 1: link0 → link1 at link0's local (1,0,0).
        # Site is attached at link1's local (1,0,0) — the "tip" of link1.
        # At q=[0,0]: link1's frame at (1,0,0); site world position = (2,0,0).
        builder = newton.ModelBuilder()
        link0 = builder.add_link()
        link1 = builder.add_link()
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link0,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        j1 = builder.add_joint_revolute(
            parent=link0,
            child=link1,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([j0, j1], label="arm")

        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
        output_qd = wp.zeros(2, dtype=wp.float32, device=device)
        output_q = wp.zeros(2, dtype=wp.float32, device=device)
        diffik = ControllerDifferentialIK(
            model_builder=builder,
            indices=indices,
            end_effector_link=link1,
            site_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()),
            measurement=wp.zeros(2, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            target_pos=wp.array([wp.vec3(2.0, 0.0, 0.0)], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([0.05], dtype=wp.float32, device=device),
            output_qd=output_qd,
            output_q=output_q,
        )
        group = ControlGroup([diffik])

        s0 = group.state()
        s1 = group.state()
        group.step(s0, s1, dt=0.01)

        np.testing.assert_allclose(output_qd.numpy(), [0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(output_q.numpy(), [0.0, 0.0], atol=1e-5)

    def test_pulls_first_joint_toward_offset_target(self):
        """Target offset in +y from site at q=[0,0] drives positive q_dot on joint 0."""
        device = wp.get_device()

        builder = newton.ModelBuilder()
        link0 = builder.add_link()
        link1 = builder.add_link()
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link0,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        j1 = builder.add_joint_revolute(
            parent=link0,
            child=link1,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([j0, j1], label="arm")

        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
        output_qd = wp.zeros(2, dtype=wp.float32, device=device)
        output_q = wp.zeros(2, dtype=wp.float32, device=device)
        diffik = ControllerDifferentialIK(
            model_builder=builder,
            indices=indices,
            end_effector_link=link1,
            site_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()),
            measurement=wp.zeros(2, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            # Site at q=[0,0] is at (2, 0, 0); shift target +0.1 in y from there.
            target_pos=wp.array([wp.vec3(2.0, 0.1, 0.0)], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([0.05], dtype=wp.float32, device=device),
            output_qd=output_qd,
            output_q=output_q,
        )
        group = ControlGroup([diffik])

        s0 = group.state()
        s1 = group.state()
        group.step(s0, s1, dt=0.01)

        # Positive q0 rotation moves the site in +y direction at q=[0,0].
        self.assertGreater(float(output_qd.numpy()[0]), 0.0)

    def test_output_q_equals_current_q_plus_qdot_dt(self):
        """output_q is integrated as q_current + q_dot * dt."""
        device = wp.get_device()

        builder = newton.ModelBuilder()
        link0 = builder.add_link()
        link1 = builder.add_link()
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link0,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        j1 = builder.add_joint_revolute(
            parent=link0,
            child=link1,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([j0, j1], label="arm")

        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
        # Start with q = (0.2, -0.3) so q_current is non-zero in the integration check.
        joint_q = wp.array([0.2, -0.3], dtype=wp.float32, device=device)
        output_qd = wp.zeros(2, dtype=wp.float32, device=device)
        output_q = wp.zeros(2, dtype=wp.float32, device=device)
        diffik = ControllerDifferentialIK(
            model_builder=builder,
            indices=indices,
            end_effector_link=link1,
            site_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()),
            measurement=joint_q,
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            target_pos=wp.array([wp.vec3(1.5, 0.2, 0.0)], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([0.05], dtype=wp.float32, device=device),
            output_qd=output_qd,
            output_q=output_q,
        )
        group = ControlGroup([diffik])

        s0 = group.state()
        s1 = group.state()
        dt = 0.02
        group.step(s0, s1, dt=dt)

        expected_q = joint_q.numpy() + output_qd.numpy() * dt
        np.testing.assert_allclose(output_q.numpy(), expected_q, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
