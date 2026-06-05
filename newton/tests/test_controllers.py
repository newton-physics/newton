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


class TestControlLawPID(unittest.TestCase):
    def test_proportional_only(self):
        """kp * (setpoint - measurement) with no integral or derivative."""
        device = wp.get_device()
        indices = wp.array([0, 1, 2], dtype=wp.uint32, device=device)
        output_arr = wp.zeros(3, dtype=wp.float32, device=device)

        input = SimpleNamespace(
            measurement=wp.zeros(3, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(3, dtype=wp.float32, device=device),
            setpoint=wp.array([1.0, 2.0, -1.0], dtype=wp.float32, device=device),
            setpoint_rate=wp.zeros(3, dtype=wp.float32, device=device),
            kp=wp.array([2.0, 2.0, 2.0], dtype=wp.float32, device=device),
            ki=wp.array([0.0, 0.0, 0.0], dtype=wp.float32, device=device),
            kd=wp.array([0.0, 0.0, 0.0], dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf, np.inf, np.inf], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output=output_arr)

        pid = ControlLawPID(
            indices=indices,
            measurement="measurement",
            measurement_rate="measurement_rate",
            setpoint="setpoint",
            setpoint_rate="setpoint_rate",
            kp="kp",
            ki="ki",
            kd="kd",
            integral_max="integral_max",
            output="output",
        )
        group = Controller([pid])

        s0 = group.state()
        s1 = group.state()
        group.step(input, output, s0, s1, dt=0.01)

        np.testing.assert_allclose(output_arr.numpy(), [2.0, 4.0, -2.0], atol=1e-6)

    def test_integral_accumulates(self):
        """With ki>0, repeated steps with constant error grow the integral linearly."""
        device = wp.get_device()
        indices = wp.array([0], dtype=wp.uint32, device=device)
        output_arr = wp.zeros(1, dtype=wp.float32, device=device)

        input = SimpleNamespace(
            measurement=wp.zeros(1, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(1, dtype=wp.float32, device=device),
            setpoint=wp.array([1.0], dtype=wp.float32, device=device),
            setpoint_rate=wp.zeros(1, dtype=wp.float32, device=device),
            kp=wp.array([0.0], dtype=wp.float32, device=device),
            ki=wp.array([0.5], dtype=wp.float32, device=device),
            kd=wp.array([0.0], dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output=output_arr)

        pid = ControlLawPID(
            indices=indices,
            measurement="measurement",
            measurement_rate="measurement_rate",
            setpoint="setpoint",
            setpoint_rate="setpoint_rate",
            kp="kp",
            ki="ki",
            kd="kd",
            integral_max="integral_max",
            output="output",
        )
        group = Controller([pid])

        s0 = group.state()
        s1 = group.state()
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
        indices = wp.array([0], dtype=wp.uint32, device=device)
        output_arr = wp.zeros(1, dtype=wp.float32, device=device)

        input = SimpleNamespace(
            measurement=wp.zeros(1, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(1, dtype=wp.float32, device=device),
            setpoint=wp.array([1.0], dtype=wp.float32, device=device),
            setpoint_rate=wp.zeros(1, dtype=wp.float32, device=device),
            kp=wp.array([0.0], dtype=wp.float32, device=device),
            ki=wp.array([1.0], dtype=wp.float32, device=device),
            kd=wp.array([0.0], dtype=wp.float32, device=device),
            integral_max=wp.array([0.3], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output=output_arr)

        pid = ControlLawPID(
            indices=indices,
            measurement="measurement",
            measurement_rate="measurement_rate",
            setpoint="setpoint",
            setpoint_rate="setpoint_rate",
            kp="kp",
            ki="ki",
            kd="kd",
            integral_max="integral_max",
            output="output",
        )
        group = Controller([pid])

        s0 = group.state()
        s1 = group.state()
        # Without clamping the integral would reach 2.0 after 20 steps.
        for _ in range(21):
            group.step(input, output, s0, s1, dt=0.1)
            s0, s1 = s1, s0

        self.assertAlmostEqual(float(output_arr.numpy()[0]), 0.3, places=5)
        self.assertAlmostEqual(float(s0.control_law_states[0].integral.numpy()[0]), 0.3, places=5)

    def test_reset_zeros_integral(self):
        """Default reset_state is zero; group.reset clears the integral."""
        device = wp.get_device()
        indices = wp.array([0, 1], dtype=wp.uint32, device=device)

        input = SimpleNamespace(
            measurement=wp.zeros(2, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            setpoint=wp.array([1.0, 1.0], dtype=wp.float32, device=device),
            setpoint_rate=wp.zeros(2, dtype=wp.float32, device=device),
            kp=wp.array([0.0, 0.0], dtype=wp.float32, device=device),
            ki=wp.array([1.0, 1.0], dtype=wp.float32, device=device),
            kd=wp.array([0.0, 0.0], dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf, np.inf], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output=wp.zeros(2, dtype=wp.float32, device=device))

        pid = ControlLawPID(
            indices=indices,
            measurement="measurement",
            measurement_rate="measurement_rate",
            setpoint="setpoint",
            setpoint_rate="setpoint_rate",
            kp="kp",
            ki="ki",
            kd="kd",
            integral_max="integral_max",
            output="output",
        )
        group = Controller([pid])

        s0 = group.state()
        s1 = group.state()
        for _ in range(5):
            group.step(input, output, s0, s1, dt=0.1)
            s0, s1 = s1, s0
        self.assertTrue(np.all(s0.control_law_states[0].integral.numpy() > 0.0))

        group.reset(s0, mask=wp.array([True, True], dtype=wp.bool, device=device))
        np.testing.assert_allclose(s0.control_law_states[0].integral.numpy(), [0.0, 0.0], atol=1e-7)

    def test_reset_to_nonzero_target(self):
        """Mutating reset_state changes what reset writes; mask selects entries."""
        device = wp.get_device()
        indices = wp.array([0, 1, 2], dtype=wp.uint32, device=device)

        input = SimpleNamespace(
            measurement=wp.zeros(3, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(3, dtype=wp.float32, device=device),
            setpoint=wp.array([1.0, 1.0, 1.0], dtype=wp.float32, device=device),
            setpoint_rate=wp.zeros(3, dtype=wp.float32, device=device),
            kp=wp.array([0.0, 0.0, 0.0], dtype=wp.float32, device=device),
            ki=wp.array([1.0, 1.0, 1.0], dtype=wp.float32, device=device),
            kd=wp.array([0.0, 0.0, 0.0], dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf, np.inf, np.inf], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output=wp.zeros(3, dtype=wp.float32, device=device))

        pid = ControlLawPID(
            indices=indices,
            measurement="measurement",
            measurement_rate="measurement_rate",
            setpoint="setpoint",
            setpoint_rate="setpoint_rate",
            kp="kp",
            ki="ki",
            kd="kd",
            integral_max="integral_max",
            output="output",
        )
        group = Controller([pid])

        s0 = group.state()
        s1 = group.state()
        for _ in range(3):
            group.step(input, output, s0, s1, dt=0.1)
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
        output_arr = wp.zeros(3, dtype=wp.float32, device=device, requires_grad=True)
        # Setpoint carries the gradient we want to recover.
        setpoint = wp.array([1.0, 2.0, -1.0], dtype=wp.float32, device=device, requires_grad=True)

        input = SimpleNamespace(
            measurement=wp.zeros(3, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(3, dtype=wp.float32, device=device),
            setpoint=setpoint,
            setpoint_rate=wp.zeros(3, dtype=wp.float32, device=device),
            kp=wp.array([2.0, 2.0, 2.0], dtype=wp.float32, device=device),
            ki=wp.array([0.0, 0.0, 0.0], dtype=wp.float32, device=device),
            kd=wp.array([0.0, 0.0, 0.0], dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf, np.inf, np.inf], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output=output_arr)

        pid = ControlLawPID(
            indices=wp.array([0, 1, 2], dtype=wp.uint32, device=device),
            measurement="measurement",
            measurement_rate="measurement_rate",
            setpoint="setpoint",
            setpoint_rate="setpoint_rate",
            kp="kp",
            ki="ki",
            kd="kd",
            integral_max="integral_max",
            output="output",
        )
        group = Controller([pid], requires_grad=True)

        s0 = group.state()
        s1 = group.state()
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            group.step(input, output, s0, s1, dt=0.01)
            wp.launch(_sum_kernel, dim=len(output_arr), inputs=[output_arr, loss])
        tape.backward(loss=loss)

        # Pure proportional with kp = 2: output[i] = 2 * (setpoint[i] - 0).
        # d(sum(output))/d(setpoint[i]) = 2 for every i.
        np.testing.assert_allclose(setpoint.grad.numpy(), [2.0, 2.0, 2.0], atol=1e-5)

    def test_per_dof_tuple_form_with_custom_indices(self):
        """Per-DOF port supplied as (attr_name, custom_port_indices) reads from
        a layout that differs from the controller's own ``indices``.

        Controller writes to ``output[indices[i]]`` — here ``indices = [5, 7]``.
        The kp source array has 3 entries laid out as [unused, kp_for_dof_0,
        kp_for_dof_1]. Passing ``kp=("kp", port_idx)`` with
        ``port_idx = [1, 2]`` tells the kernel "read kp[1] for slot 0 and
        kp[2] for slot 1," independent of the controller-level indices.
        """
        device = wp.get_device()
        indices = wp.array([5, 7], dtype=wp.uint32, device=device)
        # Output has slots 5 and 7 populated; others must remain 0.
        output_arr = wp.zeros(10, dtype=wp.float32, device=device)
        kp_arr = wp.array([99.0, 3.0, 4.0], dtype=wp.float32, device=device)
        kp_port_idx = wp.array([1, 2], dtype=wp.uint32, device=device)

        # Measurement / setpoint / gains laid out at indices 5 and 7
        # match the controller layout — only kp uses a custom port layout.
        measurement = wp.zeros(10, dtype=wp.float32, device=device)
        setpoint = wp.zeros(10, dtype=wp.float32, device=device)
        setpoint_np = np.zeros(10, dtype=np.float32)
        setpoint_np[5] = 1.0
        setpoint_np[7] = 1.0
        setpoint = wp.array(setpoint_np, dtype=wp.float32, device=device)

        input = SimpleNamespace(
            measurement=measurement,
            measurement_rate=wp.zeros(10, dtype=wp.float32, device=device),
            setpoint=setpoint,
            setpoint_rate=wp.zeros(10, dtype=wp.float32, device=device),
            kp=kp_arr,
            ki=wp.array([0.0, 0.0], dtype=wp.float32, device=device),
            kd=wp.array([0.0, 0.0], dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf, np.inf], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output=output_arr)

        pid = ControlLawPID(
            indices=indices,
            measurement="measurement",
            measurement_rate="measurement_rate",
            setpoint="setpoint",
            setpoint_rate="setpoint_rate",
            # Tuple form: custom port_indices on the kp port only.
            kp=("kp", kp_port_idx),
            ki="ki",
            kd="kd",
            integral_max="integral_max",
            output="output",
        )
        group = Controller([pid])
        s0, s1 = group.state(), group.state()
        group.step(input, output, s0, s1, dt=0.01)

        # output[5] = kp[1] * (setpoint[5] - 0) = 3.0; output[7] = kp[2] * 1 = 4.0.
        result = output_arr.numpy()
        self.assertAlmostEqual(float(result[5]), 3.0, places=5)
        self.assertAlmostEqual(float(result[7]), 4.0, places=5)
        # Untouched slots stay zero.
        for i in (0, 1, 2, 3, 4, 6, 8, 9):
            self.assertEqual(float(result[i]), 0.0)


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
        builder.add_site(link1, label="tip", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
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
            model_builder=builder,
            indices=indices,
            site="tip",
            measurement="measurement",
            measurement_rate="measurement_rate",
            target_pos="target_pos",
            target_quat="target_quat",
            damping="damping",
            gain="gain",
            output_qd="output_qd",
            output_q="output_q",
        )
        group = Controller([diffik])

        s0 = group.state()
        s1 = group.state()
        group.step(input, output, s0, s1, dt=0.01)

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
        builder.add_site(link1, label="tip", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
        output_qd = wp.zeros(2, dtype=wp.float32, device=device)
        output_q = wp.zeros(2, dtype=wp.float32, device=device)
        input = SimpleNamespace(
            measurement=wp.zeros(2, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(2, dtype=wp.float32, device=device),
            # Site at q=[0,0] is at (2, 0, 0); shift target +0.1 in y from there.
            target_pos=wp.array([wp.vec3(2.0, 0.1, 0.0)], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
            damping=wp.array([0.05], dtype=wp.float32, device=device),
            gain=wp.array([1.0], dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output_qd=output_qd, output_q=output_q)
        diffik = ControlLawDifferentialIK(
            model_builder=builder,
            indices=indices,
            site="tip",
            measurement="measurement",
            measurement_rate="measurement_rate",
            target_pos="target_pos",
            target_quat="target_quat",
            damping="damping",
            gain="gain",
            output_qd="output_qd",
            output_q="output_q",
        )
        group = Controller([diffik])

        s0 = group.state()
        s1 = group.state()
        group.step(input, output, s0, s1, dt=0.01)

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
        builder.add_site(link1, label="tip", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
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
            model_builder=builder,
            indices=indices,
            site="tip",
            measurement="measurement",
            measurement_rate="measurement_rate",
            target_pos="target_pos",
            target_quat="target_quat",
            damping="damping",
            gain="gain",
            output_qd="output_qd",
            output_q="output_q",
        )
        group = Controller([diffik])

        s0 = group.state()
        s1 = group.state()
        dt = 0.02
        group.step(input, output, s0, s1, dt=dt)

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

        Closed-form 1-DOF DLS:
            q_dot = GAIN * J_site^T e / (J_site^T J_site + lambda^2)
                  = GAIN * ERR_Y / (2 + lambda^2)
        """
        device = wp.get_device()

        ERR_Y = 0.1
        LAMBDA = 0.5
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
            model_builder=builder,
            indices=indices,
            site="tool",
            measurement="measurement",
            measurement_rate="measurement_rate",
            target_pos="target_pos",
            target_quat="target_quat",
            damping="damping",
            gain="gain",
            output_qd="output_qd",
            output_q="output_q",
        )
        group = Controller([diffik])

        s0 = group.state()
        s1 = group.state()
        group.step(input, output, s0, s1, dt=0.01)

        expected_qd = GAIN * ERR_Y / (2.0 + LAMBDA**2)
        self.assertAlmostEqual(float(output_qd.numpy()[0]), expected_qd, places=5)

    def test_runs_inside_wp_tape_without_crashing(self):
        """Controller(..., requires_grad=True) can be wrapped in a wp.Tape and
        stepped without error.

        DLS solve kernel is marked enable_backward=False because Warp 1.14.0's
        tile_cholesky backward returns zero gradients. Forward is correct;
        gradient through target_pos is blocked at the solve (asserts zero).
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
            model_builder=builder,
            indices=indices,
            site="tool",
            measurement="measurement",
            measurement_rate="measurement_rate",
            target_pos="target_pos",
            target_quat="target_quat",
            damping="damping",
            gain="gain",
            output_qd="output_qd",
            output_q="output_q",
        )
        group = Controller([diffik], requires_grad=True)

        s0 = group.state()
        s1 = group.state()
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            group.step(input, output, s0, s1, dt=0.01)
            wp.launch(_sum_kernel, dim=len(output_qd), inputs=[output_qd, loss])
        tape.backward(loss=loss)

        expected_qd = GAIN * 0.1 / (2.0 + LAMBDA**2)
        self.assertAlmostEqual(float(output_qd.numpy()[0]), expected_qd, places=5)
        target_grad = target_pos.grad.numpy()
        np.testing.assert_allclose(target_grad[0], [0.0, 0.0, 0.0], atol=1e-7)

    def test_parallel_robots(self):
        """R=4 identical arms, each with a different target_pos.y. Verify the
        DLS solution is correctly applied per-robot."""
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

        indices = wp.array(np.arange(R, dtype=np.uint32), device=device)
        target_ys = [0.10, 0.20, 0.05, -0.15]

        output_qd = wp.zeros(R, dtype=wp.float32, device=device)
        output_q = wp.zeros(R, dtype=wp.float32, device=device)
        input = SimpleNamespace(
            measurement=wp.zeros(R, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(R, dtype=wp.float32, device=device),
            target_pos=wp.array([wp.vec3(1.0, y, 0.0) for y in target_ys], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * R, dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA] * R, dtype=wp.float32, device=device),
            gain=wp.array([GAIN] * R, dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output_qd=output_qd, output_q=output_q)
        diffik = ControlLawDifferentialIK(
            model_builder=builder,
            indices=indices,
            site="tool",
            measurement="measurement",
            measurement_rate="measurement_rate",
            target_pos="target_pos",
            target_quat="target_quat",
            damping="damping",
            gain="gain",
            output_qd="output_qd",
            output_q="output_q",
        )
        controller = Controller([diffik])

        s0, s1 = controller.state(), controller.state()
        controller.step(input, output, s0, s1, dt=0.01)

        expected_qd = np.array([GAIN * y / (2.0 + LAMBDA**2) for y in target_ys], dtype=np.float32)
        np.testing.assert_allclose(output_qd.numpy(), expected_qd, atol=1e-5)

    def test_robot_is_subset_of_scene(self):
        """The DiffIK's model is a strict subset of the full simulation scene.

        Sim has an arm AND a separate pendulum. The DiffIK is constructed
        from an arm-only ModelBuilder; `indices` selects only the arm's DOF
        from sim-sized flat buffers. The pendulum slot of output_qd stays 0.
        """
        device = wp.get_device()
        LAMBDA = 0.5
        GAIN = 1.5
        ERR_Y = 0.1

        sim_builder = newton.ModelBuilder()
        sim_arm_link = sim_builder.add_link()
        sim_arm_joint = sim_builder.add_joint_revolute(
            parent=-1,
            child=sim_arm_link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        sim_builder.add_articulation([sim_arm_joint], label="arm")
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

        arm_indices = wp.array([0], dtype=wp.uint32, device=device)
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
            model_builder=arm_builder,
            indices=arm_indices,
            site="tool",
            measurement="measurement",
            measurement_rate="measurement_rate",
            target_pos="target_pos",
            target_quat="target_quat",
            damping="damping",
            gain="gain",
            output_qd="output_qd",
            output_q="output_q",
        )
        controller = Controller([diffik])

        s0, s1 = controller.state(), controller.state()
        dt = 0.01
        controller.step(input, output, s0, s1, dt=dt)

        expected_qd_arm = GAIN * ERR_Y / (2.0 + LAMBDA**2)
        np.testing.assert_allclose(output_qd.numpy(), [expected_qd_arm, 0.0], atol=1e-5)
        np.testing.assert_allclose(output_q.numpy(), [expected_qd_arm * dt, 0.0], atol=1e-5)

    def test_two_variants_in_template(self):
        """K=2 articulations with different kinematics (effective link lengths).

        Variant 0: joint at origin, identity child_xform → site world (1,0,0).
        Variant 1: child_xform p=(-1,0,0) shifts the body to world (1,0,0) →
                   site world (2,0,0). Effectively twice the reach.

        Analytical (at q=0):
            Variant 0: q_dot = ERR_Y / (2 + λ²)
            Variant 1: q_dot = 2 * ERR_Y / (5 + λ²)
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

        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
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
            model_builder=builder,
            indices=indices,
            site="tool",
            measurement="measurement",
            measurement_rate="measurement_rate",
            target_pos="target_pos",
            target_quat="target_quat",
            damping="damping",
            gain="gain",
            output_qd="output_qd",
            output_q="output_q",
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

    def test_two_variants_with_different_site_offsets(self):
        """K=2 articulations with identical kinematics but different
        body-local site xforms — both labeled ``"tool"``.

        Variant 0: site at body-local (1, 0, 0) → q_dot = ERR_Y / (2 + λ²).
        Variant 1: site at body-local (2, 0, 0) → q_dot = 2 * ERR_Y / (5 + λ²).
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

        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
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
            model_builder=builder,
            indices=indices,
            site="tool",
            measurement="measurement",
            measurement_rate="measurement_rate",
            target_pos="target_pos",
            target_quat="target_quat",
            damping="damping",
            gain="gain",
            output_qd="output_qd",
            output_q="output_q",
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
        """R parallel arms inside a sim scene that also contains R pendulums.

        Sim DOF layout: indices 0..R-1 → arm DOFs (DiffIK-controlled);
                        indices R..2R-1 → pendulum DOFs (untouched).
        """
        device = wp.get_device()
        LAMBDA = 0.5
        GAIN = 1.0
        R = 4

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

        arm_indices = wp.array(np.arange(R, dtype=np.uint32), device=device)
        target_ys = [0.10, 0.20, 0.05, -0.15]

        output_qd = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)
        output_q = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)
        input = SimpleNamespace(
            measurement=wp.zeros(sim_n_dofs, dtype=wp.float32, device=device),
            measurement_rate=wp.zeros(sim_n_dofs, dtype=wp.float32, device=device),
            target_pos=wp.array([wp.vec3(1.0, y, 0.0) for y in target_ys], dtype=wp.vec3, device=device),
            target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * R, dtype=wp.quat, device=device),
            damping=wp.array([LAMBDA] * R, dtype=wp.float32, device=device),
            gain=wp.array([GAIN] * R, dtype=wp.float32, device=device),
        )
        output = SimpleNamespace(output_qd=output_qd, output_q=output_q)
        diffik = ControlLawDifferentialIK(
            model_builder=arm_builder,
            indices=arm_indices,
            site="tool",
            measurement="measurement",
            measurement_rate="measurement_rate",
            target_pos="target_pos",
            target_quat="target_quat",
            damping="damping",
            gain="gain",
            output_qd="output_qd",
            output_q="output_q",
        )
        controller = Controller([diffik])

        s0, s1 = controller.state(), controller.state()
        controller.step(input, output, s0, s1, dt=0.01)

        expected_arm_qd = np.array([GAIN * y / (2.0 + LAMBDA**2) for y in target_ys], dtype=np.float32)
        out = output_qd.numpy()
        np.testing.assert_allclose(out[:R], expected_arm_qd, atol=1e-5)
        np.testing.assert_allclose(out[R:], np.zeros(R, dtype=np.float32), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
