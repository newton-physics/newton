# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for newton.controllers — framework and ControlLawPID."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.controllers import ControlLawDifferentialIK, ControlLawPID, Controller


# Tape-aware scalar sum (wp.utils.array_sum calls a native C reducer, so it
# isn't recorded by wp.Tape; we need a real Warp kernel for the grad tests).
@wp.kernel
def _sum_kernel(values: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(out, 0, values[i])


class TestControlLawPID(unittest.TestCase):
    def test_proportional_only(self):
        """kp * (setpoint - measurement) with no integral or derivative."""
        device = wp.get_device()
        indices = wp.array([0, 1, 2], dtype=wp.uint32, device=device)
        identity = wp.array([0, 1, 2], dtype=wp.uint32, device=device)
        output = wp.zeros(3, dtype=wp.float32, device=device)

        pid = ControlLawPID(
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
        group = Controller([pid])

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

        pid = ControlLawPID(
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
        group = Controller([pid])

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

        pid = ControlLawPID(
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
        group = Controller([pid])

        s0 = group.state()
        s1 = group.state()
        # Without clamping the integral would reach 2.0 after 20 steps.
        for _ in range(21):
            group.step(s0, s1, dt=0.1)
            s0, s1 = s1, s0

        self.assertAlmostEqual(float(output.numpy()[0]), 0.3, places=5)
        self.assertAlmostEqual(float(s0.control_law_states[0].integral.numpy()[0]), 0.3, places=5)

    def test_reset_zeros_integral(self):
        """Default reset_state is zero; group.reset clears the integral."""
        device = wp.get_device()
        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
        identity = wp.array([0, 1], dtype=wp.uint32, device=device)

        pid = ControlLawPID(
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
        group = Controller([pid])

        s0 = group.state()
        s1 = group.state()
        for _ in range(5):
            group.step(s0, s1, dt=0.1)
            s0, s1 = s1, s0
        self.assertTrue(np.all(s0.control_law_states[0].integral.numpy() > 0.0))

        group.reset(s0, mask=wp.array([True, True], dtype=wp.bool, device=device))
        np.testing.assert_allclose(s0.control_law_states[0].integral.numpy(), [0.0, 0.0], atol=1e-7)

    def test_reset_to_nonzero_target(self):
        """Mutating reset_state changes what reset writes; mask selects entries."""
        device = wp.get_device()
        indices = wp.array([0, 1, 2], dtype=wp.uint32, device=device)
        identity = wp.array([0, 1, 2], dtype=wp.uint32, device=device)

        pid = ControlLawPID(
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
        group = Controller([pid])

        s0 = group.state()
        s1 = group.state()
        for _ in range(3):
            group.step(s0, s1, dt=0.1)
            s0, s1 = s1, s0

        pid.reset_state.integral.assign(wp.array([0.7, 0.8, 0.9], dtype=wp.float32, device=device))

        # Reset only slots 0 and 2 — slot 1 keeps its accumulated value.
        before = s0.control_law_states[0].integral.numpy().copy()
        group.reset(s0, mask=wp.array([True, False, True], dtype=wp.bool, device=device))
        after = s0.control_law_states[0].integral.numpy()

        self.assertAlmostEqual(float(after[0]), 0.7, places=5)
        self.assertAlmostEqual(float(after[1]), float(before[1]), places=5)
        self.assertAlmostEqual(float(after[2]), 0.9, places=5)

    def test_gradient_flows_with_requires_grad(self):
        """With Controller(..., requires_grad=True), gradients from a loss on
        the output array flow back to a requires_grad=True setpoint."""
        device = wp.get_device()
        identity = wp.array([0, 1, 2], dtype=wp.uint32, device=device)
        output = wp.zeros(3, dtype=wp.float32, device=device, requires_grad=True)
        # Setpoint carries the gradient we want to recover.
        setpoint = wp.array([1.0, 2.0, -1.0], dtype=wp.float32, device=device, requires_grad=True)

        pid = ControlLawPID(
            indices=wp.array([0, 1, 2], dtype=wp.uint32, device=device),
            measurement=wp.zeros(3, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(3, dtype=wp.float32, device=device),
            setpoint=setpoint,
            setpoint_rate=wp.zeros(3, dtype=wp.float32, device=device),
            kp=(wp.array([2.0, 2.0, 2.0], dtype=wp.float32, device=device), identity),
            ki=(wp.array([0.0, 0.0, 0.0], dtype=wp.float32, device=device), identity),
            kd=(wp.array([0.0, 0.0, 0.0], dtype=wp.float32, device=device), identity),
            integral_max=(wp.array([np.inf, np.inf, np.inf], dtype=wp.float32, device=device), identity),
            output=output,
        )
        group = Controller([pid], requires_grad=True)

        s0 = group.state()
        s1 = group.state()
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            group.step(s0, s1, dt=0.01)
            wp.launch(_sum_kernel, dim=len(output), inputs=[output, loss])
        tape.backward(loss=loss)

        # Pure proportional with kp = 2: output[i] = 2 * (setpoint[i] - 0).
        # d(sum(output))/d(setpoint[i]) = 2 for every i.
        np.testing.assert_allclose(setpoint.grad.numpy(), [2.0, 2.0, 2.0], atol=1e-5)


class TestControlLawDifferentialIK(unittest.TestCase):
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
        # Site at the tip of link1 — 1 unit further along x in link1's local frame.
        builder.add_site(link1, label="tip", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
        output_qd = wp.zeros(2, dtype=wp.float32, device=device)
        output_q = wp.zeros(2, dtype=wp.float32, device=device)
        diffik = ControlLawDifferentialIK(
            model_builder=builder,
            indices=indices,
            site="tip",
            measurement=wp.zeros(2, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            target_pos=wp.array([wp.vec3(2.0, 0.0, 0.0)], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([0.05], dtype=wp.float32, device=device),
            gain=wp.array([1.0], dtype=wp.float32, device=device),
            output_qd=output_qd,
            output_q=output_q,
        )
        group = Controller([diffik])

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
        # Site at the tip of link1 — 1 unit further along x in link1's local frame.
        builder.add_site(link1, label="tip", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
        output_qd = wp.zeros(2, dtype=wp.float32, device=device)
        output_q = wp.zeros(2, dtype=wp.float32, device=device)
        diffik = ControlLawDifferentialIK(
            model_builder=builder,
            indices=indices,
            site="tip",
            measurement=wp.zeros(2, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            # Site at q=[0,0] is at (2, 0, 0); shift target +0.1 in y from there.
            target_pos=wp.array([wp.vec3(2.0, 0.1, 0.0)], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([0.05], dtype=wp.float32, device=device),
            gain=wp.array([1.0], dtype=wp.float32, device=device),
            output_qd=output_qd,
            output_q=output_q,
        )
        group = Controller([diffik])

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
        # Site at the tip of link1 — 1 unit further along x in link1's local frame.
        builder.add_site(link1, label="tip", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
        # Start with q = (0.2, -0.3) so q_current is non-zero in the integration check.
        joint_q = wp.array([0.2, -0.3], dtype=wp.float32, device=device)
        output_qd = wp.zeros(2, dtype=wp.float32, device=device)
        output_q = wp.zeros(2, dtype=wp.float32, device=device)
        diffik = ControlLawDifferentialIK(
            model_builder=builder,
            indices=indices,
            site="tip",
            measurement=joint_q,
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            target_pos=wp.array([wp.vec3(1.5, 0.2, 0.0)], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([0.05], dtype=wp.float32, device=device),
            gain=wp.array([1.0], dtype=wp.float32, device=device),
            output_qd=output_qd,
            output_q=output_q,
        )
        group = Controller([diffik])

        s0 = group.state()
        s1 = group.state()
        dt = 0.02
        group.step(s0, s1, dt=dt)

        expected_q = joint_q.numpy() + output_qd.numpy() * dt
        np.testing.assert_allclose(output_q.numpy(), expected_q, atol=1e-5)

    def test_one_dof_matches_analytical_dls(self):
        """End-to-end check against a hand-derived DLS solution.

        Robot: single revolute joint about world +z, one link of length 1.
        body_com defaults to (0, 0, 0). Tool site at body-frame (1, 0, 0)
        (the tip of the link). Initial config q = 0.

        At q = 0:
            link0 body frame in world         = identity
            site world position               = (1, 0, 0)
            site world orientation            = identity
            COM world position                = (0, 0, 0)
            offset = site_world - com_world   = (1, 0, 0)

        Newton's spatial Jacobian J for link0 is COM-frame (v_com_world,
        omega_world). With axis = (0, 0, 1) and the COM at the origin,
        a unit q_dot rotates the body about +z without translating the COM,
        so:
            J_COM = [0, 0, 0,                   (linear: v_com = 0)
                     0, 0, 1]^T                 (angular: omega = +z)

        The kernel converts each linear row to "at site" via
        v_site = v_com + cross(omega_col, offset). For our single column:
            cross((0,0,1), (1,0,0)) = (0, 1, 0)
        so
            J_site_linear  = (0, 0, 0) + (0, 1, 0) = (0, 1, 0)
            J_site_angular = (0, 0, 1)             (unchanged)
            J_site = [0, 1, 0, 0, 0, 1]^T          (6 x 1)

        Task-space error, with target_pos = (1, ERR_Y, 0) and
        target_quat = identity:
            pos_err = (0, ERR_Y, 0)
            q_err   = target_quat * conj(site_quat) = identity → rot_err = 0
            e       = [0, ERR_Y, 0, 0, 0, 0]^T

        The kernel solves the left-damped 6x6 system A y = e with
        A = J_site J_site^T + lambda^2 * I_6, then q_dot = GAIN * J_site^T y.
        For a 1-DOF system the equivalent right-damped form is a 1x1 solve
        and is easier to write down:
            q_dot = GAIN * J_site^T e / (J_site^T J_site + lambda^2)
                  = GAIN * (0*0 + 1*ERR_Y + 0 + 0 + 0 + 1*0) / (0+1+0+0+0+1 + lambda^2)
                  = GAIN * ERR_Y / (2 + lambda^2)
        Both DLS formulations give the same q_dot, so this is what the
        kernel must produce.
        """
        device = wp.get_device()

        ERR_Y = 0.1
        LAMBDA = 0.5
        # Non-unit gain so the test exercises the gain multiplier (a gain of
        # 1.0 would be indistinguishable from "no gain at all" in the
        # output, which doesn't tell us the kernel is honoring it).
        GAIN = 2.0

        builder = newton.ModelBuilder()
        link0 = builder.add_link()
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link0,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([j0], label="arm")
        builder.add_site(link0, label="tool", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

        indices = wp.array([0], dtype=wp.uint32, device=device)
        output_qd = wp.zeros(1, dtype=wp.float32, device=device)
        output_q = wp.zeros(1, dtype=wp.float32, device=device)
        diffik = ControlLawDifferentialIK(
            model_builder=builder,
            indices=indices,
            site="tool",
            measurement=wp.zeros(1, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(1, dtype=wp.float32, device=device),
            target_pos=wp.array([wp.vec3(1.0, ERR_Y, 0.0)], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA], dtype=wp.float32, device=device),
            gain=wp.array([GAIN], dtype=wp.float32, device=device),
            output_qd=output_qd,
            output_q=output_q,
        )
        group = Controller([diffik])

        s0 = group.state()
        s1 = group.state()
        group.step(s0, s1, dt=0.01)

        expected_qd = GAIN * ERR_Y / (2.0 + LAMBDA**2)
        self.assertAlmostEqual(float(output_qd.numpy()[0]), expected_qd, places=5)

    def test_runs_inside_wp_tape_without_crashing(self):
        """With Controller(..., requires_grad=True), the DiffIK controller
        can be wrapped in a wp.Tape and stepped without error.

        The DLS solve kernel is marked enable_backward=False because Warp
        1.14.0's tile_cholesky backward returns zero gradients (verified
        with a standalone test — forward is correct but the registered
        adjoint doesn't actually propagate). The forward pass still produces
        the correct analytical q_dot (verified separately by
        test_one_dof_matches_analytical_dls); this test only checks that
        the chain is tape-safe (no NaN, no crash) and asserts the
        documented zero-gradient-through-solve behaviour. When upstream
        Warp fixes tile_cholesky backward, the assertion below will start
        failing — at that point we can promote this test to assert the
        analytical gradient GAIN / (2 + LAMBDA**2).
        """
        device = wp.get_device()
        LAMBDA = 0.5
        GAIN = 1.5

        builder = newton.ModelBuilder()
        link0 = builder.add_link()
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link0,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([j0], label="arm")
        builder.add_site(link0, label="tool", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

        indices = wp.array([0], dtype=wp.uint32, device=device)
        output_qd = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        output_q = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        target_pos = wp.array([wp.vec3(1.0, 0.1, 0.0)], dtype=wp.vec3, device=device, requires_grad=True)
        diffik = ControlLawDifferentialIK(
            model_builder=builder,
            indices=indices,
            site="tool",
            measurement=wp.zeros(1, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(1, dtype=wp.float32, device=device),
            target_pos=target_pos,
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA], dtype=wp.float32, device=device),
            gain=wp.array([GAIN], dtype=wp.float32, device=device),
            output_qd=output_qd,
            output_q=output_q,
        )
        group = Controller([diffik], requires_grad=True)

        s0 = group.state()
        s1 = group.state()
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            group.step(s0, s1, dt=0.01)
            wp.launch(_sum_kernel, dim=len(output_qd), inputs=[output_qd, loss])
        tape.backward(loss=loss)

        # Forward matches the analytical DLS answer.
        expected_qd = GAIN * 0.1 / (2.0 + LAMBDA**2)
        self.assertAlmostEqual(float(output_qd.numpy()[0]), expected_qd, places=5)
        # Solve kernel is enable_backward=False (Warp 1.14.0 tile_cholesky
        # backward is non-functional), so target_pos.grad is blocked at the
        # solve and remains zero. Documented current behaviour.
        target_grad = target_pos.grad.numpy()
        np.testing.assert_allclose(target_grad[0], [0.0, 0.0, 0.0], atol=1e-7)

    def test_parallel_robots(self):
        """R=4 identical arms, each with a different target_pos.y. Verify the
        DLS solution is correctly applied per-robot.

        Same 1-DOF arm template as test_one_dof_matches_analytical_dls; the
        DiffIK is asked to internally replicate it 4 times via indices of
        length 4. With distinct target_pos.y per robot, the analytical per-robot
        q_dot is GAIN * target_pos[r].y / (2 + LAMBDA**2). This catches
        replication / indexing bugs that wouldn't show up at R=1.
        """
        device = wp.get_device()
        LAMBDA = 0.5
        GAIN = 1.0
        R = 4

        builder = newton.ModelBuilder()
        link0 = builder.add_link()
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link0,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([j0], label="arm")
        builder.add_site(link0, label="tool", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

        # One DOF per robot, R robots, so len(indices) = R = R * dofs_per_robot.
        indices = wp.array(np.arange(R, dtype=np.uint32), device=device)

        # Distinct y-offset target per robot so each robot's analytical q_dot is unique.
        target_ys = [0.10, 0.20, 0.05, -0.15]
        target_pos = wp.array([wp.vec3(1.0, y, 0.0) for y in target_ys], dtype=wp.vec3, device=device)

        output_qd = wp.zeros(R, dtype=wp.float32, device=device)
        output_q = wp.zeros(R, dtype=wp.float32, device=device)
        diffik = ControlLawDifferentialIK(
            model_builder=builder,
            indices=indices,
            site="tool",
            measurement=wp.zeros(R, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(R, dtype=wp.float32, device=device),
            target_pos=target_pos,
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * R, dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA] * R, dtype=wp.float32, device=device),
            gain=wp.array([GAIN] * R, dtype=wp.float32, device=device),
            output_qd=output_qd,
            output_q=output_q,
        )
        controller = Controller([diffik])

        s0, s1 = controller.state(), controller.state()
        controller.step(s0, s1, dt=0.01)

        expected_qd = np.array([GAIN * y / (2.0 + LAMBDA**2) for y in target_ys], dtype=np.float32)
        np.testing.assert_allclose(output_qd.numpy(), expected_qd, atol=1e-5)

    def test_robot_is_subset_of_scene(self):
        """The DiffIK's model is a strict subset of the full simulation scene.

        The "sim" contains an arm AND a separate pendulum articulation that
        the DiffIK does not know about. The DiffIK is constructed from an
        arm-only ModelBuilder, with `indices` selecting only the arm's DOF
        from the sim-sized flat joint_q / output buffers. After step(),
        the arm slot of output_qd carries the analytical q_dot and the
        pendulum slot is untouched (still 0).
        """
        device = wp.get_device()
        LAMBDA = 0.5
        GAIN = 1.5
        ERR_Y = 0.1

        # --- "Sim" scene: arm + an unrelated pendulum. Not finalized; just
        # used here to demonstrate the pattern and to figure out the right
        # buffer sizes / DOF indices for the DiffIK to talk to.
        sim_builder = newton.ModelBuilder()
        # Arm (joint 0 → DOF 0).
        sim_arm_link = sim_builder.add_link()
        sim_arm_joint = sim_builder.add_joint_revolute(
            parent=-1,
            child=sim_arm_link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        sim_builder.add_articulation([sim_arm_joint], label="arm")
        # Pendulum off to the side (joint 1 → DOF 1). Different axis + offset
        # so it's clearly a separate body.
        sim_pendulum_link = sim_builder.add_link()
        sim_pendulum_joint = sim_builder.add_joint_revolute(
            parent=-1,
            child=sim_pendulum_link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(5.0, 0.0, 0.0)),
            child_xform=wp.transform_identity(),
        )
        sim_builder.add_articulation([sim_pendulum_joint], label="pendulum")
        sim_n_dofs = sim_builder.joint_dof_count
        self.assertEqual(sim_n_dofs, 2)

        # --- Arm-only template that the DiffIK actually operates on. Disjoint
        # from sim_builder; the DiffIK builds its own internal Model from this.
        arm_builder = newton.ModelBuilder()
        arm_link = arm_builder.add_link()
        arm_joint = arm_builder.add_joint_revolute(
            parent=-1,
            child=arm_link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        arm_builder.add_articulation([arm_joint], label="arm")
        arm_builder.add_site(arm_link, label="tool", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

        # Sim-sized buffers; the DiffIK picks only the arm's DOF (index 0).
        arm_indices = wp.array([0], dtype=wp.uint32, device=device)
        measurement = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)
        measurement_rate = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)
        output_qd = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)
        output_q = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)

        diffik = ControlLawDifferentialIK(
            model_builder=arm_builder,
            indices=arm_indices,
            site="tool",
            measurement=measurement,
            measurement_rate=measurement_rate,
            target_pos=wp.array([wp.vec3(1.0, ERR_Y, 0.0)], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA], dtype=wp.float32, device=device),
            gain=wp.array([GAIN], dtype=wp.float32, device=device),
            output_qd=output_qd,
            output_q=output_q,
        )
        controller = Controller([diffik])

        s0, s1 = controller.state(), controller.state()
        dt = 0.01
        controller.step(s0, s1, dt=dt)

        # Arm DOF (index 0): analytical answer. Pendulum DOF (index 1): untouched.
        expected_qd_arm = GAIN * ERR_Y / (2.0 + LAMBDA**2)
        np.testing.assert_allclose(output_qd.numpy(), [expected_qd_arm, 0.0], atol=1e-5)
        # output_q = q_current + q_dot * dt; q_current was zeros.
        np.testing.assert_allclose(output_q.numpy(), [expected_qd_arm * dt, 0.0], atol=1e-5)

    def test_two_variants_in_template(self):
        """Template has K=2 articulations with different kinematics
        (effective link lengths). Verifies that the DLS solve produces the
        correct per-variant q_dot from differing body Jacobians, even though
        the K=2 articulations share the same body-local site offset.

        Variant 0 ("short"): joint at origin, identity child_xform. Body frame
                             at world (0, 0, 0). Site at body-local (1, 0, 0).
                             Site world at q=0: (1, 0, 0).
        Variant 1 ("long"):  joint at origin, child_xform translates the body
                             frame by +1 in x (via child_xform p=(-1, 0, 0)
                             — see Newton's joint anchor convention). Body
                             frame at world (1, 0, 0). Same body-local site
                             (1, 0, 0). Site world at q=0: (2, 0, 0).
                             Effectively twice the reach.

        Both articulations share the site label "tool", but the DiffIK's
        shape_label lookup returns the FIRST match (variant 0's site). The
        stored site_xform is therefore (1, 0, 0) in body-local frame — which
        happens to equal variant 1's site offset too, so the lookup result
        applies cleanly to both. The per-variant world-frame difference
        comes from each body's own body_q computed by eval_fk inside the
        controller.

        Analytical q_dot at q=0 with pos_err = (0, ERR_Y, 0), target_quat=identity:

        Variant 0 (existing 1-DOF analytical case):
            J_site = [0, 1, 0, 0, 0, 1]^T
            q_dot = ERR_Y / (2 + λ²)

        Variant 1 (derived):
            Body COM in world = (1, 0, 0). v_com = omega × r_com = (0, 1, 0).
            J_COM_linear = (0, 1, 0); J_COM_angular = (0, 0, 1).
            offset = site_world - com_world = (2,0,0) - (1,0,0) = (1, 0, 0).
            J_site_linear = J_COM_linear + cross(omega, offset)
                          = (0, 1, 0) + (0, 1, 0)
                          = (0, 2, 0).
            J_site = [0, 2, 0, 0, 0, 1]^T.
            q_dot = J^T e / (J^T J + λ²)
                  = (0*0 + 2*ERR_Y + 0 + 0 + 0 + 1*0) / (0+4+0+0+0+1 + λ²)
                  = 2 * ERR_Y / (5 + λ²).
        """
        device = wp.get_device()
        LAMBDA = 0.5
        GAIN = 1.0
        ERR_Y = 0.1

        builder = newton.ModelBuilder()

        # Variant 0: "short" — body frame at world origin at q=0.
        short_link = builder.add_link()
        short_joint = builder.add_joint_revolute(
            parent=-1,
            child=short_link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([short_joint], label="short_arm")
        builder.add_site(short_link, label="tool", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

        # Variant 1: "long" — body frame at world (1, 0, 0) at q=0 via
        # child_xform=(-1, 0, 0). Same body-local site offset (1, 0, 0) means
        # the site lands at world (2, 0, 0) — effectively twice the reach.
        long_link = builder.add_link()
        long_joint = builder.add_joint_revolute(
            parent=-1,
            child=long_link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(p=wp.vec3(-1.0, 0.0, 0.0)),
        )
        builder.add_articulation([long_joint], label="long_arm")
        builder.add_site(long_link, label="tool", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

        # K=2 articulations, R=1 (no replication) → num_robots=2; dofs_per_robot=1.
        # Robot 0 corresponds to variant 0 (short); robot 1 to variant 1 (long).
        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
        target_pos = wp.array(
            [wp.vec3(1.0, ERR_Y, 0.0), wp.vec3(2.0, ERR_Y, 0.0)],
            dtype=wp.vec3,
            device=device,
        )
        output_qd = wp.zeros(2, dtype=wp.float32, device=device)
        output_q = wp.zeros(2, dtype=wp.float32, device=device)
        diffik = ControlLawDifferentialIK(
            model_builder=builder,
            indices=indices,
            site="tool",
            measurement=wp.zeros(2, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            target_pos=target_pos,
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * 2, dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA, LAMBDA], dtype=wp.float32, device=device),
            gain=wp.array([GAIN, GAIN], dtype=wp.float32, device=device),
            output_qd=output_qd,
            output_q=output_q,
        )
        controller = Controller([diffik])
        s0, s1 = controller.state(), controller.state()
        controller.step(s0, s1, dt=0.01)

        expected_qd_short = GAIN * ERR_Y / (2.0 + LAMBDA**2)
        expected_qd_long = GAIN * 2.0 * ERR_Y / (5.0 + LAMBDA**2)
        np.testing.assert_allclose(
            output_qd.numpy(),
            [expected_qd_short, expected_qd_long],
            atol=1e-5,
        )

    def test_two_variants_with_different_site_offsets(self):
        """Template has K=2 articulations with identical kinematics but
        different body-local site xforms — both labeled ``"tool"``.

        This exercises the per-variant site lookup directly: the controller
        must find the site on each articulation and use that variant's own
        body-local offset, not just the first match. Same kinematics on both
        articulations isolates the site_xform difference as the only factor
        driving different q_dots — so if the lookup ever regressed back to
        "first match wins," both variants would compute the same q_dot and
        this test would fail.

        Variant 0: site at body-local (1, 0, 0) → site world (1, 0, 0).
        Variant 1: site at body-local (2, 0, 0) → site world (2, 0, 0).

        Analytical q_dot at q=0 with body COM at world origin
        (J_COM_linear = 0; only the cross-product correction contributes):

        Variant 0:
            offset = (1, 0, 0); J_site_linear = cross((0,0,1), (1,0,0)) = (0, 1, 0).
            J_site = [0, 1, 0, 0, 0, 1]^T.
            q_dot = ERR_Y / (2 + λ²).

        Variant 1:
            offset = (2, 0, 0); J_site_linear = cross((0,0,1), (2,0,0)) = (0, 2, 0).
            J_site = [0, 2, 0, 0, 0, 1]^T.
            q_dot = 2 * ERR_Y / (5 + λ²).
        """
        device = wp.get_device()
        LAMBDA = 0.5
        GAIN = 1.0
        ERR_Y = 0.1

        builder = newton.ModelBuilder()

        # Variant 0 — identical kinematics to variant 1, but a smaller site offset.
        v0_link = builder.add_link()
        v0_joint = builder.add_joint_revolute(
            parent=-1,
            child=v0_link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([v0_joint], label="arm_v0")
        builder.add_site(v0_link, label="tool", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

        # Variant 1 — same kinematics, but its "tool" site sits at twice the
        # body-local offset. With identical body_q, the only thing differing
        # between the two variants' DLS solves is the site xform looked up
        # by the controller.
        v1_link = builder.add_link()
        v1_joint = builder.add_joint_revolute(
            parent=-1,
            child=v1_link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([v1_joint], label="arm_v1")
        builder.add_site(v1_link, label="tool", xform=wp.transform(p=wp.vec3(2.0, 0.0, 0.0), q=wp.quat_identity()))

        # K=2, R=1 ⇒ num_robots=2. Targets matched to each variant's site at q=0.
        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
        target_pos = wp.array(
            [wp.vec3(1.0, ERR_Y, 0.0), wp.vec3(2.0, ERR_Y, 0.0)],
            dtype=wp.vec3,
            device=device,
        )
        output_qd = wp.zeros(2, dtype=wp.float32, device=device)
        output_q = wp.zeros(2, dtype=wp.float32, device=device)
        diffik = ControlLawDifferentialIK(
            model_builder=builder,
            indices=indices,
            site="tool",
            measurement=wp.zeros(2, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            target_pos=target_pos,
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * 2, dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA, LAMBDA], dtype=wp.float32, device=device),
            gain=wp.array([GAIN, GAIN], dtype=wp.float32, device=device),
            output_qd=output_qd,
            output_q=output_q,
        )
        controller = Controller([diffik])
        s0, s1 = controller.state(), controller.state()
        controller.step(s0, s1, dt=0.01)

        expected_qd_v0 = GAIN * ERR_Y / (2.0 + LAMBDA**2)
        expected_qd_v1 = GAIN * 2.0 * ERR_Y / (5.0 + LAMBDA**2)
        np.testing.assert_allclose(
            output_qd.numpy(),
            [expected_qd_v0, expected_qd_v1],
            atol=1e-5,
        )

    def test_parallel_robots_subset_of_scene(self):
        """Combined: R parallel arms living inside a sim scene that also
        contains R pendulums the DiffIK doesn't know about.

        Sim DOF layout (all arms added before all pendulums):
            indices 0..R-1  → arm DOFs (DiffIK-controlled)
            indices R..2R-1 → pendulum DOFs (untouched)

        Exercises both replication (R copies of the arm template) and the
        "subset of scene" pattern (DiffIK only writes the first R of 2R DOFs).
        """
        device = wp.get_device()
        LAMBDA = 0.5
        GAIN = 1.0
        R = 4

        # --- Sim scene: R arms followed by R pendulums.
        sim_builder = newton.ModelBuilder()
        for _ in range(R):
            link = sim_builder.add_link()
            joint = sim_builder.add_joint_revolute(
                parent=-1,
                child=link,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=wp.transform_identity(),
                child_xform=wp.transform_identity(),
            )
            sim_builder.add_articulation([joint], label="arm")
        for _ in range(R):
            link = sim_builder.add_link()
            joint = sim_builder.add_joint_revolute(
                parent=-1,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(p=wp.vec3(5.0, 0.0, 0.0)),
                child_xform=wp.transform_identity(),
            )
            sim_builder.add_articulation([joint], label="pendulum")
        sim_n_dofs = sim_builder.joint_dof_count
        self.assertEqual(sim_n_dofs, 2 * R)

        # --- Arm-only template (K=1). The DiffIK replicates internally to R.
        arm_builder = newton.ModelBuilder()
        arm_link = arm_builder.add_link()
        arm_joint = arm_builder.add_joint_revolute(
            parent=-1,
            child=arm_link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        arm_builder.add_articulation([arm_joint], label="arm")
        arm_builder.add_site(arm_link, label="tool", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

        # Indices pick only the arm DOFs (first R of 2R total).
        arm_indices = wp.array(np.arange(R, dtype=np.uint32), device=device)
        target_ys = [0.10, 0.20, 0.05, -0.15]
        target_pos = wp.array([wp.vec3(1.0, y, 0.0) for y in target_ys], dtype=wp.vec3, device=device)

        measurement = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)
        measurement_rate = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)
        output_qd = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)
        output_q = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)

        diffik = ControlLawDifferentialIK(
            model_builder=arm_builder,
            indices=arm_indices,
            site="tool",
            measurement=measurement,
            measurement_rate=measurement_rate,
            target_pos=target_pos,
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * R, dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA] * R, dtype=wp.float32, device=device),
            gain=wp.array([GAIN] * R, dtype=wp.float32, device=device),
            output_qd=output_qd,
            output_q=output_q,
        )
        controller = Controller([diffik])

        s0, s1 = controller.state(), controller.state()
        controller.step(s0, s1, dt=0.01)

        expected_arm_qd = np.array([GAIN * y / (2.0 + LAMBDA**2) for y in target_ys], dtype=np.float32)
        out = output_qd.numpy()
        # Arm slots match the analytical answer.
        np.testing.assert_allclose(out[:R], expected_arm_qd, atol=1e-5)
        # Pendulum slots untouched.
        np.testing.assert_allclose(out[R:], np.zeros(R, dtype=np.float32), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
