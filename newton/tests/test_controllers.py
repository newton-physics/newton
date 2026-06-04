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


if __name__ == "__main__":
    unittest.main()
