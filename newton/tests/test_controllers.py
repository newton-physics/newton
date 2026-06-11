# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for newton.controllers — framework, ControlLawPID, ControlLawDifferentialIK."""

import unittest
from types import SimpleNamespace

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


def _idx(values, device):
    return wp.array(values, dtype=wp.uint32, device=device)


def _iota(n, device):
    return wp.array(np.arange(n, dtype=np.uint32), device=device)


# ----------------------------------------------------------------------------
# ControlLawPID
# ----------------------------------------------------------------------------


class TestControlLawPID(unittest.TestCase):
    def test_proportional_only(self):
        """kp * (setpoint - measurement) with no integral or derivative."""
        device = wp.get_device()
        indices = _iota(3, device)
        output_arr = wp.zeros(3, dtype=wp.float32, device=device)

        input = SimpleNamespace(
            measurement=wp.zeros(3, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(3, dtype=wp.float32, device=device),
            setpoint=wp.array([1.0, 2.0, -1.0], dtype=wp.float32, device=device),
            setpoint_rate=wp.zeros(3, dtype=wp.float32, device=device),
            kp=wp.array([2.0, 2.0, 2.0], dtype=wp.float32, device=device),
            ki=wp.zeros(3, dtype=wp.float32, device=device),
            kd=wp.zeros(3, dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf, np.inf, np.inf], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output=output_arr)

        pid = ControlLawPID(
            label="pid",
            measurement=("measurement", indices),
            measurement_rate=("measurement_rate", indices),
            setpoint=("setpoint", indices),
            setpoint_rate=("setpoint_rate", indices),
            kp=("kp", indices),
            ki=("ki", indices),
            kd=("kd", indices),
            integral_max=("integral_max", indices),
            output=("output", indices),
        )
        group = Controller([pid])

        s0, s1 = group.state(), group.state()
        group.step(input, output, s0, s1, dt=0.01)

        np.testing.assert_allclose(output_arr.numpy(), [2.0, 4.0, -2.0], atol=1e-6)

    def test_integral_accumulates(self):
        """With ki>0, repeated steps with constant error grow the integral linearly."""
        device = wp.get_device()
        indices = _iota(1, device)
        output_arr = wp.zeros(1, dtype=wp.float32, device=device)

        input = SimpleNamespace(
            measurement=wp.zeros(1, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(1, dtype=wp.float32, device=device),
            setpoint=wp.array([1.0], dtype=wp.float32, device=device),
            setpoint_rate=wp.zeros(1, dtype=wp.float32, device=device),
            kp=wp.zeros(1, dtype=wp.float32, device=device),
            ki=wp.array([0.5], dtype=wp.float32, device=device),
            kd=wp.zeros(1, dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output=output_arr)

        pid = ControlLawPID(
            label="pid",
            measurement=("measurement", indices),
            measurement_rate=("measurement_rate", indices),
            setpoint=("setpoint", indices),
            setpoint_rate=("setpoint_rate", indices),
            kp=("kp", indices),
            ki=("ki", indices),
            kd=("kd", indices),
            integral_max=("integral_max", indices),
            output=("output", indices),
        )
        group = Controller([pid])

        s0, s1 = group.state(), group.state()
        dt = 0.1
        running_integral = 0.0
        for step_i in range(5):
            running_integral += dt
            group.step(input, output, s0, s1, dt=dt)
            s0, s1 = s1, s0
            self.assertAlmostEqual(
                float(output_arr.numpy()[0]),
                0.5 * running_integral,
                places=5,
                msg=f"step {step_i}: integral should be {running_integral:.3f}",
            )

    def test_anti_windup_clamps_integral(self):
        """integral_max bounds the accumulator symmetrically."""
        device = wp.get_device()
        indices = _iota(1, device)
        output_arr = wp.zeros(1, dtype=wp.float32, device=device)

        input = SimpleNamespace(
            measurement=wp.zeros(1, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(1, dtype=wp.float32, device=device),
            setpoint=wp.array([1.0], dtype=wp.float32, device=device),
            setpoint_rate=wp.zeros(1, dtype=wp.float32, device=device),
            kp=wp.zeros(1, dtype=wp.float32, device=device),
            ki=wp.array([1.0], dtype=wp.float32, device=device),
            kd=wp.zeros(1, dtype=wp.float32, device=device),
            integral_max=wp.array([0.3], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output=output_arr)

        pid = ControlLawPID(
            label="pid",
            measurement=("measurement", indices),
            measurement_rate=("measurement_rate", indices),
            setpoint=("setpoint", indices),
            setpoint_rate=("setpoint_rate", indices),
            kp=("kp", indices),
            ki=("ki", indices),
            kd=("kd", indices),
            integral_max=("integral_max", indices),
            output=("output", indices),
        )
        group = Controller([pid])

        s0, s1 = group.state(), group.state()
        # Without clamping the integral would reach 2.0 after 20 steps.
        for _ in range(21):
            group.step(input, output, s0, s1, dt=0.1)
            s0, s1 = s1, s0

        self.assertAlmostEqual(float(output_arr.numpy()[0]), 0.3, places=5)
        self.assertAlmostEqual(float(s0.control_law_states["pid"].integral.numpy()[0]), 0.3, places=5)

    def test_gradient_flows_with_requires_grad(self):
        """With Controller(..., requires_grad=True), gradients from a loss on
        the output array flow back to a requires_grad=True setpoint."""
        device = wp.get_device()
        indices = _iota(3, device)
        output_arr = wp.zeros(3, dtype=wp.float32, device=device, requires_grad=True)
        # Setpoint carries the gradient we want to recover.
        setpoint = wp.array([1.0, 2.0, -1.0], dtype=wp.float32, device=device, requires_grad=True)

        input = SimpleNamespace(
            measurement=wp.zeros(3, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(3, dtype=wp.float32, device=device),
            setpoint=setpoint,
            setpoint_rate=wp.zeros(3, dtype=wp.float32, device=device),
            kp=wp.array([2.0, 2.0, 2.0], dtype=wp.float32, device=device),
            ki=wp.zeros(3, dtype=wp.float32, device=device),
            kd=wp.zeros(3, dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf, np.inf, np.inf], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output=output_arr)

        pid = ControlLawPID(
            label="pid",
            measurement=("measurement", indices),
            measurement_rate=("measurement_rate", indices),
            setpoint=("setpoint", indices),
            setpoint_rate=("setpoint_rate", indices),
            kp=("kp", indices),
            ki=("ki", indices),
            kd=("kd", indices),
            integral_max=("integral_max", indices),
            output=("output", indices),
        )
        group = Controller([pid], requires_grad=True)

        s0, s1 = group.state(), group.state()
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            group.step(input, output, s0, s1, dt=0.01)
            wp.launch(_sum_kernel, dim=len(output_arr), inputs=[output_arr, loss])
        tape.backward(loss=loss)

        # Pure proportional with kp = 2: output[i] = 2 * (setpoint[i] - 0).
        # d(sum(output))/d(setpoint[i]) = 2 for every i.
        np.testing.assert_allclose(setpoint.grad.numpy(), [2.0, 2.0, 2.0], atol=1e-5)

    def test_per_dof_port_indices_diverge_from_output(self):
        """Per-DOF input ports can use a different layout than the output, as
        long as port_indices has the right length.

        Controller writes to ``output[output_idx[i]]`` — here ``output_idx = [5, 7]``.
        The kp source array has 3 entries; we point kp's port_indices at
        [1, 2] so slot 0 reads kp[1], slot 1 reads kp[2].
        """
        device = wp.get_device()
        output_idx = _idx([5, 7], device)
        # Output has slots 5 and 7 populated; others must remain 0.
        output_arr = wp.zeros(10, dtype=wp.float32, device=device)
        kp_arr = wp.array([99.0, 3.0, 4.0], dtype=wp.float32, device=device)
        kp_idx = _idx([1, 2], device)

        # Setpoint laid out at indices 5 and 7 (same layout as output).
        setpoint_np = np.zeros(10, dtype=np.float32)
        setpoint_np[5] = 1.0
        setpoint_np[7] = 1.0
        setpoint = wp.array(setpoint_np, dtype=wp.float32, device=device)

        input = SimpleNamespace(
            measurement=wp.zeros(10, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(10, dtype=wp.float32, device=device),
            setpoint=setpoint,
            setpoint_rate=wp.zeros(10, dtype=wp.float32, device=device),
            kp=kp_arr,
            ki=wp.zeros(2, dtype=wp.float32, device=device),
            kd=wp.zeros(2, dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf, np.inf], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output=output_arr)

        # Inputs whose array layout matches the output's get output_idx;
        # kp uses its own custom port_indices.
        ki_idx = _idx([0, 1], device)
        kd_idx = _idx([0, 1], device)
        imax_idx = _idx([0, 1], device)
        pid = ControlLawPID(
            label="pid",
            measurement=("measurement", output_idx),
            measurement_rate=("measurement_rate", output_idx),
            setpoint=("setpoint", output_idx),
            setpoint_rate=("setpoint_rate", output_idx),
            kp=("kp", kp_idx),
            ki=("ki", ki_idx),
            kd=("kd", kd_idx),
            integral_max=("integral_max", imax_idx),
            output=("output", output_idx),
        )
        group = Controller([pid])
        s0, s1 = group.state(), group.state()
        group.step(input, output, s0, s1, dt=0.01)

        # output[5] = kp[1] * 1 = 3.0; output[7] = kp[2] * 1 = 4.0.
        result = output_arr.numpy()
        self.assertAlmostEqual(float(result[5]), 3.0, places=5)
        self.assertAlmostEqual(float(result[7]), 4.0, places=5)
        for i in (0, 1, 2, 3, 4, 6, 8, 9):
            self.assertEqual(float(result[i]), 0.0)


# ----------------------------------------------------------------------------
# ControlLawDifferentialIK
# ----------------------------------------------------------------------------


def _build_planar_arm_one_link():
    """Single revolute joint, one link, site at body-local (1, 0, 0)."""
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
    return builder


def _build_planar_arm_two_link():
    """Two revolute joints, two unit-length links, site at link1's tip."""
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
    builder.add_site(link1, label="tip", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))
    return builder


class TestControlLawDifferentialIK(unittest.TestCase):
    def test_target_equals_current_gives_zero_velocity(self):
        """With target == current site pose, the DLS solve produces q_dot ~ 0."""
        device = wp.get_device()
        builder = _build_planar_arm_two_link()

        dof_idx = _iota(2, device)
        robot_idx = _iota(1, device)
        output_qd = wp.zeros(2, dtype=wp.float32, device=device)
        output_q = wp.zeros(2, dtype=wp.float32, device=device)
        input = SimpleNamespace(
            measurement=wp.zeros(2, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            target_pos=wp.array([wp.vec3(2.0, 0.0, 0.0)], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([0.05], dtype=wp.float32, device=device),
            gain=wp.array([1.0], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output_qd=output_qd, output_q=output_q)
        diffik = ControlLawDifferentialIK(
            label="ik",
            model_builder=builder,
            site="tip",
            measurement=("measurement", dof_idx),
            measurement_rate=("measurement_rate", dof_idx),
            target_pos=("target_pos", robot_idx),
            target_quat=("target_quat", robot_idx),
            damping=("damping", robot_idx),
            gain=("gain", robot_idx),
            output_qd=("output_qd", dof_idx),
            output_q=("output_q", dof_idx),
        )
        group = Controller([diffik])

        s0, s1 = group.state(), group.state()
        group.step(input, output, s0, s1, dt=0.01)

        np.testing.assert_allclose(output_qd.numpy(), [0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(output_q.numpy(), [0.0, 0.0], atol=1e-5)

    def test_pulls_first_joint_toward_offset_target(self):
        """Target offset in +y from site at q=[0,0] drives positive q_dot on joint 0."""
        device = wp.get_device()
        builder = _build_planar_arm_two_link()

        dof_idx = _iota(2, device)
        robot_idx = _iota(1, device)
        output_qd = wp.zeros(2, dtype=wp.float32, device=device)
        output_q = wp.zeros(2, dtype=wp.float32, device=device)
        input = SimpleNamespace(
            measurement=wp.zeros(2, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            target_pos=wp.array([wp.vec3(2.0, 0.1, 0.0)], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([0.05], dtype=wp.float32, device=device),
            gain=wp.array([1.0], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output_qd=output_qd, output_q=output_q)
        diffik = ControlLawDifferentialIK(
            label="ik",
            model_builder=builder,
            site="tip",
            measurement=("measurement", dof_idx),
            measurement_rate=("measurement_rate", dof_idx),
            target_pos=("target_pos", robot_idx),
            target_quat=("target_quat", robot_idx),
            damping=("damping", robot_idx),
            gain=("gain", robot_idx),
            output_qd=("output_qd", dof_idx),
            output_q=("output_q", dof_idx),
        )
        group = Controller([diffik])

        s0, s1 = group.state(), group.state()
        group.step(input, output, s0, s1, dt=0.01)

        # Positive q0 rotation moves the site in +y direction at q=[0,0].
        self.assertGreater(float(output_qd.numpy()[0]), 0.0)

    def test_output_q_equals_current_q_plus_qdot_dt(self):
        """output_q is integrated as q_current + q_dot * dt."""
        device = wp.get_device()
        builder = _build_planar_arm_two_link()

        dof_idx = _iota(2, device)
        robot_idx = _iota(1, device)
        joint_q = wp.array([0.2, -0.3], dtype=wp.float32, device=device)
        output_qd = wp.zeros(2, dtype=wp.float32, device=device)
        output_q = wp.zeros(2, dtype=wp.float32, device=device)
        input = SimpleNamespace(
            measurement=joint_q,
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            target_pos=wp.array([wp.vec3(1.5, 0.2, 0.0)], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([0.05], dtype=wp.float32, device=device),
            gain=wp.array([1.0], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output_qd=output_qd, output_q=output_q)
        diffik = ControlLawDifferentialIK(
            label="ik",
            model_builder=builder,
            site="tip",
            measurement=("measurement", dof_idx),
            measurement_rate=("measurement_rate", dof_idx),
            target_pos=("target_pos", robot_idx),
            target_quat=("target_quat", robot_idx),
            damping=("damping", robot_idx),
            gain=("gain", robot_idx),
            output_qd=("output_qd", dof_idx),
            output_q=("output_q", dof_idx),
        )
        group = Controller([diffik])

        s0, s1 = group.state(), group.state()
        dt = 0.02
        group.step(input, output, s0, s1, dt=dt)

        expected_q = joint_q.numpy() + output_qd.numpy() * dt
        np.testing.assert_allclose(output_q.numpy(), expected_q, atol=1e-5)

    def test_one_dof_matches_analytical_dls(self):
        """End-to-end check against a hand-derived DLS solution.

        Closed-form 1-DOF DLS:
            q_dot = GAIN * J_site^T e / (J_site^T J_site + lambda^2)
                  = GAIN * ERR_Y / (2 + lambda^2)
        """
        device = wp.get_device()
        ERR_Y = 0.1
        LAMBDA = 0.5
        GAIN = 2.0

        builder = _build_planar_arm_one_link()
        dof_idx = _iota(1, device)
        robot_idx = _iota(1, device)
        output_qd = wp.zeros(1, dtype=wp.float32, device=device)
        output_q = wp.zeros(1, dtype=wp.float32, device=device)
        input = SimpleNamespace(
            measurement=wp.zeros(1, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(1, dtype=wp.float32, device=device),
            target_pos=wp.array([wp.vec3(1.0, ERR_Y, 0.0)], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA], dtype=wp.float32, device=device),
            gain=wp.array([GAIN], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output_qd=output_qd, output_q=output_q)
        diffik = ControlLawDifferentialIK(
            label="ik",
            model_builder=builder,
            site="tool",
            measurement=("measurement", dof_idx),
            measurement_rate=("measurement_rate", dof_idx),
            target_pos=("target_pos", robot_idx),
            target_quat=("target_quat", robot_idx),
            damping=("damping", robot_idx),
            gain=("gain", robot_idx),
            output_qd=("output_qd", dof_idx),
            output_q=("output_q", dof_idx),
        )
        group = Controller([diffik])

        s0, s1 = group.state(), group.state()
        group.step(input, output, s0, s1, dt=0.01)

        expected_qd = GAIN * ERR_Y / (2.0 + LAMBDA**2)
        self.assertAlmostEqual(float(output_qd.numpy()[0]), expected_qd, places=5)

    def test_runs_inside_wp_tape_without_crashing(self):
        """Controller(..., requires_grad=True) can be wrapped in a wp.Tape.
        DLS solve kernel is enable_backward=False so target_pos.grad stays 0."""
        device = wp.get_device()
        LAMBDA = 0.5
        GAIN = 1.5

        builder = _build_planar_arm_one_link()
        dof_idx = _iota(1, device)
        robot_idx = _iota(1, device)
        output_qd = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        output_q = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        target_pos = wp.array([wp.vec3(1.0, 0.1, 0.0)], dtype=wp.vec3, device=device, requires_grad=True)
        input = SimpleNamespace(
            measurement=wp.zeros(1, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(1, dtype=wp.float32, device=device),
            target_pos=target_pos,
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA], dtype=wp.float32, device=device),
            gain=wp.array([GAIN], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output_qd=output_qd, output_q=output_q)
        diffik = ControlLawDifferentialIK(
            label="ik",
            model_builder=builder,
            site="tool",
            measurement=("measurement", dof_idx),
            measurement_rate=("measurement_rate", dof_idx),
            target_pos=("target_pos", robot_idx),
            target_quat=("target_quat", robot_idx),
            damping=("damping", robot_idx),
            gain=("gain", robot_idx),
            output_qd=("output_qd", dof_idx),
            output_q=("output_q", dof_idx),
        )
        group = Controller([diffik], requires_grad=True)

        s0, s1 = group.state(), group.state()
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            group.step(input, output, s0, s1, dt=0.01)
            wp.launch(_sum_kernel, dim=len(output_qd), inputs=[output_qd, loss])
        tape.backward(loss=loss)

        expected_qd = GAIN * 0.1 / (2.0 + LAMBDA**2)
        self.assertAlmostEqual(float(output_qd.numpy()[0]), expected_qd, places=5)
        np.testing.assert_allclose(target_pos.grad.numpy()[0], [0.0, 0.0, 0.0], atol=1e-7)

    def test_parallel_robots(self):
        """N=4 identical arms, each with a different target_pos.y.

        The user builds the 4-articulation template via
        ModelBuilder.replicate before passing to the controller (the
        controller does no replication of its own).
        """
        device = wp.get_device()
        LAMBDA = 0.5
        GAIN = 1.0
        N = 4

        template = _build_planar_arm_one_link()
        builder = newton.ModelBuilder()
        builder.replicate(template, world_count=N)

        dof_idx = _iota(N, device)
        robot_idx = _iota(N, device)
        target_ys = [0.10, 0.20, 0.05, -0.15]

        output_qd = wp.zeros(N, dtype=wp.float32, device=device)
        output_q = wp.zeros(N, dtype=wp.float32, device=device)
        input = SimpleNamespace(
            measurement=wp.zeros(N, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(N, dtype=wp.float32, device=device),
            target_pos=wp.array([wp.vec3(1.0, y, 0.0) for y in target_ys], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * N, dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA] * N, dtype=wp.float32, device=device),
            gain=wp.array([GAIN] * N, dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output_qd=output_qd, output_q=output_q)
        diffik = ControlLawDifferentialIK(
            label="ik",
            model_builder=builder,
            site="tool",
            measurement=("measurement", dof_idx),
            measurement_rate=("measurement_rate", dof_idx),
            target_pos=("target_pos", robot_idx),
            target_quat=("target_quat", robot_idx),
            damping=("damping", robot_idx),
            gain=("gain", robot_idx),
            output_qd=("output_qd", dof_idx),
            output_q=("output_q", dof_idx),
        )
        controller = Controller([diffik])

        s0, s1 = controller.state(), controller.state()
        controller.step(input, output, s0, s1, dt=0.01)

        expected_qd = np.array([GAIN * y / (2.0 + LAMBDA**2) for y in target_ys], dtype=np.float32)
        np.testing.assert_allclose(output_qd.numpy(), expected_qd, atol=1e-5)

    def test_robot_is_subset_of_scene(self):
        """The DiffIK's model contains a single arm; the sim is a wider
        scene whose joint layout includes one pendulum after the arm. The
        controller writes only to DOF index 0; the pendulum slot stays 0.
        """
        device = wp.get_device()
        LAMBDA = 0.5
        GAIN = 1.5
        ERR_Y = 0.1
        sim_n_dofs = 2  # arm dof 0 + pendulum dof 1

        builder = _build_planar_arm_one_link()
        dof_idx = _idx([0], device)
        robot_idx = _iota(1, device)
        output_qd = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)
        output_q = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)
        input = SimpleNamespace(
            measurement=wp.zeros(sim_n_dofs, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(sim_n_dofs, dtype=wp.float32, device=device),
            target_pos=wp.array([wp.vec3(1.0, ERR_Y, 0.0)], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA], dtype=wp.float32, device=device),
            gain=wp.array([GAIN], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output_qd=output_qd, output_q=output_q)
        diffik = ControlLawDifferentialIK(
            label="ik",
            model_builder=builder,
            site="tool",
            measurement=("measurement", dof_idx),
            measurement_rate=("measurement_rate", dof_idx),
            target_pos=("target_pos", robot_idx),
            target_quat=("target_quat", robot_idx),
            damping=("damping", robot_idx),
            gain=("gain", robot_idx),
            output_qd=("output_qd", dof_idx),
            output_q=("output_q", dof_idx),
        )
        controller = Controller([diffik])

        s0, s1 = controller.state(), controller.state()
        dt = 0.01
        controller.step(input, output, s0, s1, dt=dt)

        expected_qd_arm = GAIN * ERR_Y / (2.0 + LAMBDA**2)
        np.testing.assert_allclose(output_qd.numpy(), [expected_qd_arm, 0.0], atol=1e-5)
        np.testing.assert_allclose(output_q.numpy(), [expected_qd_arm * dt, 0.0], atol=1e-5)

    def test_two_articulations_different_kinematics(self):
        """Builder has N=2 articulations with different kinematics
        (effective link lengths via different child_xform).

        Articulation 0: joint at origin, identity child_xform -> site
                        world (1, 0, 0). q_dot = ERR_Y / (2 + lambda^2).
        Articulation 1: child_xform p=(-1,0,0) shifts the body to world
                        (1, 0, 0) -> site world (2, 0, 0). Twice the reach.
                        q_dot = 2 * ERR_Y / (5 + lambda^2).
        """
        device = wp.get_device()
        LAMBDA = 0.5
        GAIN = 1.0
        ERR_Y = 0.1

        builder = newton.ModelBuilder()

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

        dof_idx = _iota(2, device)
        robot_idx = _iota(2, device)
        output_qd = wp.zeros(2, dtype=wp.float32, device=device)
        output_q = wp.zeros(2, dtype=wp.float32, device=device)
        input = SimpleNamespace(
            measurement=wp.zeros(2, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            target_pos=wp.array(
                [wp.vec3(1.0, ERR_Y, 0.0), wp.vec3(2.0, ERR_Y, 0.0)],
                dtype=wp.vec3,
                device=device,
            ),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * 2, dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA, LAMBDA], dtype=wp.float32, device=device),
            gain=wp.array([GAIN, GAIN], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output_qd=output_qd, output_q=output_q)
        diffik = ControlLawDifferentialIK(
            label="ik",
            model_builder=builder,
            site="tool",
            measurement=("measurement", dof_idx),
            measurement_rate=("measurement_rate", dof_idx),
            target_pos=("target_pos", robot_idx),
            target_quat=("target_quat", robot_idx),
            damping=("damping", robot_idx),
            gain=("gain", robot_idx),
            output_qd=("output_qd", dof_idx),
            output_q=("output_q", dof_idx),
        )
        controller = Controller([diffik])
        s0, s1 = controller.state(), controller.state()
        controller.step(input, output, s0, s1, dt=0.01)

        expected_qd_short = GAIN * ERR_Y / (2.0 + LAMBDA**2)
        expected_qd_long = GAIN * 2.0 * ERR_Y / (5.0 + LAMBDA**2)
        np.testing.assert_allclose(
            output_qd.numpy(),
            [expected_qd_short, expected_qd_long],
            atol=1e-5,
        )

    def test_two_articulations_different_site_offsets(self):
        """N=2 articulations with identical kinematics but different
        body-local site xforms — same label ``"tool"`` on each.
        """
        device = wp.get_device()
        LAMBDA = 0.5
        GAIN = 1.0
        ERR_Y = 0.1

        builder = newton.ModelBuilder()

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

        dof_idx = _iota(2, device)
        robot_idx = _iota(2, device)
        output_qd = wp.zeros(2, dtype=wp.float32, device=device)
        output_q = wp.zeros(2, dtype=wp.float32, device=device)
        input = SimpleNamespace(
            measurement=wp.zeros(2, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            target_pos=wp.array(
                [wp.vec3(1.0, ERR_Y, 0.0), wp.vec3(2.0, ERR_Y, 0.0)],
                dtype=wp.vec3,
                device=device,
            ),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * 2, dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA, LAMBDA], dtype=wp.float32, device=device),
            gain=wp.array([GAIN, GAIN], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output_qd=output_qd, output_q=output_q)
        diffik = ControlLawDifferentialIK(
            label="ik",
            model_builder=builder,
            site="tool",
            measurement=("measurement", dof_idx),
            measurement_rate=("measurement_rate", dof_idx),
            target_pos=("target_pos", robot_idx),
            target_quat=("target_quat", robot_idx),
            damping=("damping", robot_idx),
            gain=("gain", robot_idx),
            output_qd=("output_qd", dof_idx),
            output_q=("output_q", dof_idx),
        )
        controller = Controller([diffik])
        s0, s1 = controller.state(), controller.state()
        controller.step(input, output, s0, s1, dt=0.01)

        expected_qd_v0 = GAIN * ERR_Y / (2.0 + LAMBDA**2)
        expected_qd_v1 = GAIN * 2.0 * ERR_Y / (5.0 + LAMBDA**2)
        np.testing.assert_allclose(
            output_qd.numpy(),
            [expected_qd_v0, expected_qd_v1],
            atol=1e-5,
        )

    def test_parallel_robots_subset_of_scene(self):
        """N parallel arms inside a wider sim scene whose joint layout
        also includes N pendulum DOFs after the arm DOFs.
        """
        device = wp.get_device()
        LAMBDA = 0.5
        GAIN = 1.0
        N = 4
        sim_n_dofs = 2 * N

        template = _build_planar_arm_one_link()
        builder = newton.ModelBuilder()
        builder.replicate(template, world_count=N)

        # Arm DOFs are the first N of 2N total.
        dof_idx = _idx(list(range(N)), device)
        robot_idx = _iota(N, device)
        target_ys = [0.10, 0.20, 0.05, -0.15]

        output_qd = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)
        output_q = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)
        input = SimpleNamespace(
            measurement=wp.zeros(sim_n_dofs, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(sim_n_dofs, dtype=wp.float32, device=device),
            target_pos=wp.array([wp.vec3(1.0, y, 0.0) for y in target_ys], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * N, dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA] * N, dtype=wp.float32, device=device),
            gain=wp.array([GAIN] * N, dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output_qd=output_qd, output_q=output_q)
        diffik = ControlLawDifferentialIK(
            label="ik",
            model_builder=builder,
            site="tool",
            measurement=("measurement", dof_idx),
            measurement_rate=("measurement_rate", dof_idx),
            target_pos=("target_pos", robot_idx),
            target_quat=("target_quat", robot_idx),
            damping=("damping", robot_idx),
            gain=("gain", robot_idx),
            output_qd=("output_qd", dof_idx),
            output_q=("output_q", dof_idx),
        )
        controller = Controller([diffik])

        s0, s1 = controller.state(), controller.state()
        controller.step(input, output, s0, s1, dt=0.01)

        expected_arm_qd = np.array([GAIN * y / (2.0 + LAMBDA**2) for y in target_ys], dtype=np.float32)
        out = output_qd.numpy()
        np.testing.assert_allclose(out[:N], expected_arm_qd, atol=1e-5)
        np.testing.assert_allclose(out[N:], np.zeros(N, dtype=np.float32), atol=1e-7)

    def test_per_robot_tuple_form_with_custom_indices(self):
        """Per-robot ports can share a source array across robots via
        explicit robot_indices.

        4-articulation arm builder; ``target_pos`` is a length-2 source
        array; robot_indices = [0, 0, 1, 1] maps robots 0-1 to target
        slot 0 and robots 2-3 to target slot 1.
        """
        device = wp.get_device()
        LAMBDA = 0.5
        GAIN = 1.0
        N = 4

        template = _build_planar_arm_one_link()
        builder = newton.ModelBuilder()
        builder.replicate(template, world_count=N)

        # Two distinct targets, shared across pairs of robots.
        target_pos_short = wp.array(
            [wp.vec3(1.0, 0.10, 0.0), wp.vec3(1.0, 0.20, 0.0)],
            dtype=wp.vec3,
            device=device,
        )
        target_pos_idx = _idx([0, 0, 1, 1], device)

        dof_idx = _iota(N, device)
        robot_idx = _iota(N, device)
        output_qd = wp.zeros(N, dtype=wp.float32, device=device)
        output_q = wp.zeros(N, dtype=wp.float32, device=device)
        input = SimpleNamespace(
            measurement=wp.zeros(N, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(N, dtype=wp.float32, device=device),
            target_pos=target_pos_short,
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * N, dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA] * N, dtype=wp.float32, device=device),
            gain=wp.array([GAIN] * N, dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output_qd=output_qd, output_q=output_q)
        diffik = ControlLawDifferentialIK(
            label="ik",
            model_builder=builder,
            site="tool",
            measurement=("measurement", dof_idx),
            measurement_rate=("measurement_rate", dof_idx),
            target_pos=("target_pos", target_pos_idx),
            target_quat=("target_quat", robot_idx),
            damping=("damping", robot_idx),
            gain=("gain", robot_idx),
            output_qd=("output_qd", dof_idx),
            output_q=("output_q", dof_idx),
        )
        controller = Controller([diffik])
        s0, s1 = controller.state(), controller.state()
        controller.step(input, output, s0, s1, dt=0.01)

        qd_0 = GAIN * 0.10 / (2.0 + LAMBDA**2)
        qd_1 = GAIN * 0.20 / (2.0 + LAMBDA**2)
        np.testing.assert_allclose(output_qd.numpy(), [qd_0, qd_0, qd_1, qd_1], atol=1e-5)


# ----------------------------------------------------------------------------
# Controller framework
# ----------------------------------------------------------------------------


class TestController(unittest.TestCase):
    def test_duplicate_labels_raise(self):
        """Two ControlLawPIDs sharing a label cannot compose into one Controller."""
        device = wp.get_device()
        indices = _iota(1, device)

        def make_pid(label):
            return ControlLawPID(
                label=label,
                measurement=("measurement", indices),
                measurement_rate=("measurement_rate", indices),
                setpoint=("setpoint", indices),
                setpoint_rate=("setpoint_rate", indices),
                kp=("kp", indices),
                ki=("ki", indices),
                kd=("kd", indices),
                integral_max=("integral_max", indices),
                output=("output", indices),
            )

        with self.assertRaises(ValueError):
            Controller([make_pid("dup"), make_pid("dup")])


if __name__ == "__main__":
    unittest.main()
