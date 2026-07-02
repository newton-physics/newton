# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for newton.controllers."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.controllers import ControllerDifferentialKinematics, ControllerPID


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
# ControllerPID
# ----------------------------------------------------------------------------


class TestControllerPID(unittest.TestCase):
    def test_proportional_only(self):
        device = wp.get_device()
        default_idx = _iota(3, device)
        output_arr = wp.zeros(3, dtype=wp.float32, device=device)

        pid = ControllerPID(
            kp=wp.array([2.0, 2.0, 2.0], dtype=wp.float32, device=device),
            kd=wp.zeros(3, dtype=wp.float32, device=device),
            ki=wp.zeros(3, dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf, np.inf, np.inf], dtype=wp.float32, device=device),
            default_dof_indices=default_idx,
            joint_measured_attr="joint_q",
            joint_target_attr="joint_target_q",
            output_attr="output",
            device=device,
        )

        input_struct = SimpleNamespace(
            joint_q=wp.zeros(3, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(3, dtype=wp.float32, device=device),
            joint_target_q=wp.array([1.0, 2.0, -1.0], dtype=wp.float32, device=device),
            joint_target_qd=wp.zeros(3, dtype=wp.float32, device=device),
        )
        output_struct = SimpleNamespace(output=output_arr)

        s0, s1 = pid.state(), pid.state()
        pid.compute(input_struct, output_struct, s0, s1, time_step=0.01)

        np.testing.assert_allclose(output_arr.numpy(), [2.0, 4.0, -2.0], atol=1e-6)

    def test_integral_accumulates(self):
        device = wp.get_device()
        default_idx = _iota(1, device)
        output_arr = wp.zeros(1, dtype=wp.float32, device=device)

        pid = ControllerPID(
            kp=wp.zeros(1, dtype=wp.float32, device=device),
            kd=wp.zeros(1, dtype=wp.float32, device=device),
            ki=wp.array([0.5], dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf], dtype=wp.float32, device=device),
            default_dof_indices=default_idx,
            output_attr="output",
            device=device,
        )

        input_struct = SimpleNamespace(
            joint_q=wp.zeros(1, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(1, dtype=wp.float32, device=device),
            joint_target_q=wp.array([1.0], dtype=wp.float32, device=device),
            joint_target_qd=wp.zeros(1, dtype=wp.float32, device=device),
        )
        output_struct = SimpleNamespace(output=output_arr)

        s0, s1 = pid.state(), pid.state()
        dt = 0.1
        running_integral = 0.0
        for step_i in range(5):
            running_integral += dt
            pid.compute(input_struct, output_struct, s0, s1, time_step=dt)
            s0, s1 = s1, s0
            self.assertAlmostEqual(
                float(output_arr.numpy()[0]),
                0.5 * running_integral,
                places=5,
                msg=f"step {step_i}: integral should be {running_integral:.3f}",
            )

    def test_anti_windup_clamps_integral(self):
        device = wp.get_device()
        default_idx = _iota(1, device)
        output_arr = wp.zeros(1, dtype=wp.float32, device=device)

        pid = ControllerPID(
            kp=wp.zeros(1, dtype=wp.float32, device=device),
            kd=wp.zeros(1, dtype=wp.float32, device=device),
            ki=wp.array([1.0], dtype=wp.float32, device=device),
            integral_max=wp.array([0.3], dtype=wp.float32, device=device),
            default_dof_indices=default_idx,
            output_attr="output",
            device=device,
        )
        input_struct = SimpleNamespace(
            joint_q=wp.zeros(1, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(1, dtype=wp.float32, device=device),
            joint_target_q=wp.array([1.0], dtype=wp.float32, device=device),
            joint_target_qd=wp.zeros(1, dtype=wp.float32, device=device),
        )
        output_struct = SimpleNamespace(output=output_arr)

        s0, s1 = pid.state(), pid.state()
        for _ in range(21):
            pid.compute(input_struct, output_struct, s0, s1, time_step=0.1)
            s0, s1 = s1, s0

        self.assertAlmostEqual(float(output_arr.numpy()[0]), 0.3, places=5)
        self.assertAlmostEqual(float(s0.integral.numpy()[0]), 0.3, places=5)

    def test_gradient_flows_with_requires_grad(self):
        """Gradient from a loss on the output flows back to a requires_grad
        joint_target_q. Live ports resolve via getattr at step time, so
        autograd sees the user-supplied requires_grad array."""
        device = wp.get_device()
        default_idx = _iota(3, device)
        output_arr = wp.zeros(3, dtype=wp.float32, device=device, requires_grad=True)
        target = wp.array([1.0, 2.0, -1.0], dtype=wp.float32, device=device, requires_grad=True)

        pid = ControllerPID(
            kp=wp.array([2.0, 2.0, 2.0], dtype=wp.float32, device=device),
            kd=wp.zeros(3, dtype=wp.float32, device=device),
            ki=wp.zeros(3, dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf, np.inf, np.inf], dtype=wp.float32, device=device),
            default_dof_indices=default_idx,
            output_attr="output",
            device=device,
            requires_grad=True,
        )
        input_struct = SimpleNamespace(
            joint_q=wp.zeros(3, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(3, dtype=wp.float32, device=device),
            joint_target_q=target,
            joint_target_qd=wp.zeros(3, dtype=wp.float32, device=device),
        )
        output_struct = SimpleNamespace(output=output_arr)
        s0, s1 = pid.state(), pid.state()

        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            pid.compute(input_struct, output_struct, s0, s1, time_step=0.01)
            wp.launch(_sum_kernel, dim=len(output_arr), inputs=[output_arr, loss])
        tape.backward(loss=loss)

        np.testing.assert_allclose(target.grad.numpy(), [2.0, 2.0, 2.0], atol=1e-5)

    def test_live_idx_diverges_from_default(self):
        """A per-port `_idx` override lets a live input port read from a
        differently-laid-out source array than the output writes to."""
        device = wp.get_device()
        default_idx = _idx([5, 7], device)  # output slots
        output_arr = wp.zeros(10, dtype=wp.float32, device=device)
        # joint_target is laid out across a length-5 source; the controller's
        # two outputs read joint_target[1] and joint_target[2].
        target = wp.array([99.0, 1.0, 1.0, 99.0, 99.0], dtype=wp.float32, device=device)
        target_idx = _idx([1, 2], device)

        pid = ControllerPID(
            kp=wp.array([2.0, 2.0], dtype=wp.float32, device=device),
            kd=wp.zeros(2, dtype=wp.float32, device=device),
            ki=wp.zeros(2, dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf, np.inf], dtype=wp.float32, device=device),
            default_dof_indices=default_idx,
            joint_target_attr="joint_target_q",
            joint_target_idx=target_idx,
            output_attr="output",
            device=device,
        )
        input_struct = SimpleNamespace(
            joint_q=wp.zeros(10, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(10, dtype=wp.float32, device=device),
            joint_target_q=target,
            joint_target_qd=wp.zeros(10, dtype=wp.float32, device=device),
        )
        output_struct = SimpleNamespace(output=output_arr)
        s0, s1 = pid.state(), pid.state()
        pid.compute(input_struct, output_struct, s0, s1, time_step=0.01)

        result = output_arr.numpy()
        self.assertAlmostEqual(float(result[5]), 2.0, places=5)
        self.assertAlmostEqual(float(result[7]), 2.0, places=5)
        for i in (0, 1, 2, 3, 4, 6, 8, 9):
            self.assertEqual(float(result[i]), 0.0)

    def test_live_gain_from_input_struct(self):
        """A gain passed as a string lives on the input struct; the
        controller reads it at step time. Equivalent semantics to the baked
        form for natural-order indexing."""
        device = wp.get_device()
        default_idx = _iota(2, device)
        output_arr = wp.zeros(2, dtype=wp.float32, device=device)

        pid = ControllerPID(
            kp="kp",
            kd=wp.zeros(2, dtype=wp.float32, device=device),
            ki=wp.zeros(2, dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf, np.inf], dtype=wp.float32, device=device),
            default_dof_indices=default_idx,
            output_attr="output",
            device=device,
        )
        input_struct = SimpleNamespace(
            joint_q=wp.zeros(2, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(2, dtype=wp.float32, device=device),
            joint_target_q=wp.array([1.0, 2.0], dtype=wp.float32, device=device),
            joint_target_qd=wp.zeros(2, dtype=wp.float32, device=device),
            kp=wp.array([3.0, 4.0], dtype=wp.float32, device=device),
        )
        output_struct = SimpleNamespace(output=output_arr)
        s0, s1 = pid.state(), pid.state()
        pid.compute(input_struct, output_struct, s0, s1, time_step=0.01)

        np.testing.assert_allclose(output_arr.numpy(), [3.0, 8.0], atol=1e-6)

    def test_baked_gain_copied_not_referenced(self):
        """Mutating the user's baked gain array after construction has no
        effect on the controller — baked arrays are stored by copy."""
        device = wp.get_device()
        default_idx = _iota(2, device)
        kp_user = wp.array([2.0, 2.0], dtype=wp.float32, device=device)

        pid = ControllerPID(
            kp=kp_user,
            kd=wp.zeros(2, dtype=wp.float32, device=device),
            ki=wp.zeros(2, dtype=wp.float32, device=device),
            integral_max=wp.array([np.inf, np.inf], dtype=wp.float32, device=device),
            default_dof_indices=default_idx,
            output_attr="output",
            device=device,
        )
        # Mutate the user's kp_user post-construction — controller should not see this.
        kp_user.assign(np.array([100.0, 100.0], dtype=np.float32))

        input_struct = SimpleNamespace(
            joint_q=wp.zeros(2, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(2, dtype=wp.float32, device=device),
            joint_target_q=wp.array([1.0, 1.0], dtype=wp.float32, device=device),
            joint_target_qd=wp.zeros(2, dtype=wp.float32, device=device),
        )
        output_arr = wp.zeros(2, dtype=wp.float32, device=device)
        output_struct = SimpleNamespace(output=output_arr)
        s0, s1 = pid.state(), pid.state()
        pid.compute(input_struct, output_struct, s0, s1, time_step=0.01)

        np.testing.assert_allclose(output_arr.numpy(), [2.0, 2.0], atol=1e-6)

    def test_input_struct_factory(self):
        """`input_struct()` allocates a fresh dataclass with one field per
        live read port, sized for the controller's view."""
        device = wp.get_device()
        default_idx = _iota(3, device)
        pid = ControllerPID(
            kp="my_kp",
            kd=wp.zeros(3, dtype=wp.float32, device=device),
            ki=wp.zeros(3, dtype=wp.float32, device=device),
            integral_max=wp.full(3, value=float(np.inf), dtype=wp.float32, device=device),
            default_dof_indices=default_idx,
            device=device,
        )
        s = pid.input_struct()
        # Live ports + live gain (kp).
        self.assertTrue(hasattr(s, "joint_q"))
        self.assertTrue(hasattr(s, "joint_qd"))
        self.assertTrue(hasattr(s, "joint_target_q"))
        self.assertTrue(hasattr(s, "joint_target_qd"))
        self.assertTrue(hasattr(s, "my_kp"))
        # Baked gains (kd, ki, integral_max) do NOT appear on the input struct.
        self.assertFalse(hasattr(s, "kd"))
        self.assertFalse(hasattr(s, "ki"))
        self.assertFalse(hasattr(s, "integral_max"))
        self.assertEqual(s.joint_q.shape, (3,))
        self.assertEqual(s.my_kp.shape, (3,))


# ----------------------------------------------------------------------------
# ControllerDifferentialKinematics
# ----------------------------------------------------------------------------


def _build_planar_arm_one_link():
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


def _make_diffik(builder, site, default_idx, device, **overrides):
    return ControllerDifferentialKinematics(
        model_builder=builder,
        controlled_site_label=site,
        default_dof_indices=default_idx,
        bandwidth=wp.array([1.0], dtype=wp.float32, device=device),
        device=device,
        **overrides,
    )


class TestControllerDifferentialKinematics(unittest.TestCase):
    def test_target_equals_current_gives_zero_velocity(self):
        device = wp.get_device()
        builder = _build_planar_arm_two_link()
        default_idx = _iota(2, device)
        out_q = wp.zeros(2, dtype=wp.float32, device=device)
        out_qd = wp.zeros(2, dtype=wp.float32, device=device)

        diffik = ControllerDifferentialKinematics(
            model_builder=builder,
            controlled_site_label="tip",
            default_dof_indices=default_idx,
            bandwidth=wp.array([1.0], dtype=wp.float32, device=device),
            device=device,
        )
        input_struct = SimpleNamespace(
            joint_q=wp.zeros(2, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(2, dtype=wp.float32, device=device),
            site_target_position=wp.array([wp.vec3(2.0, 0.0, 0.0)], dtype=wp.vec3, device=device),
            site_target_quaternion=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
        )
        output_struct = SimpleNamespace(joint_target_q=out_q, joint_target_qd=out_qd)
        diffik.compute(input_struct, output_struct, None, None, time_step=0.01)

        np.testing.assert_allclose(out_qd.numpy(), [0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(out_q.numpy(), [0.0, 0.0], atol=1e-5)

    def test_pulls_first_joint_toward_offset_target(self):
        device = wp.get_device()
        builder = _build_planar_arm_two_link()
        default_idx = _iota(2, device)
        out_q = wp.zeros(2, dtype=wp.float32, device=device)
        out_qd = wp.zeros(2, dtype=wp.float32, device=device)
        diffik = ControllerDifferentialKinematics(
            model_builder=builder,
            controlled_site_label="tip",
            default_dof_indices=default_idx,
            bandwidth=wp.array([1.0], dtype=wp.float32, device=device),
            device=device,
        )
        input_struct = SimpleNamespace(
            joint_q=wp.zeros(2, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(2, dtype=wp.float32, device=device),
            site_target_position=wp.array([wp.vec3(2.0, 0.1, 0.0)], dtype=wp.vec3, device=device),
            site_target_quaternion=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
        )
        output_struct = SimpleNamespace(joint_target_q=out_q, joint_target_qd=out_qd)
        diffik.compute(input_struct, output_struct, None, None, time_step=0.01)

        self.assertGreater(float(out_qd.numpy()[0]), 0.0)

    def test_output_q_equals_current_q_plus_qdot_dt(self):
        device = wp.get_device()
        builder = _build_planar_arm_two_link()
        default_idx = _iota(2, device)
        joint_q = wp.array([0.2, -0.3], dtype=wp.float32, device=device)
        out_q = wp.zeros(2, dtype=wp.float32, device=device)
        out_qd = wp.zeros(2, dtype=wp.float32, device=device)
        diffik = ControllerDifferentialKinematics(
            model_builder=builder,
            controlled_site_label="tip",
            default_dof_indices=default_idx,
            bandwidth=wp.array([1.0], dtype=wp.float32, device=device),
            device=device,
        )
        input_struct = SimpleNamespace(
            joint_q=joint_q,
            joint_qd=wp.zeros(2, dtype=wp.float32, device=device),
            site_target_position=wp.array([wp.vec3(1.5, 0.2, 0.0)], dtype=wp.vec3, device=device),
            site_target_quaternion=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
        )
        output_struct = SimpleNamespace(joint_target_q=out_q, joint_target_qd=out_qd)
        dt = 0.02
        diffik.compute(input_struct, output_struct, None, None, time_step=dt)

        expected_q = joint_q.numpy() + out_qd.numpy() * dt
        np.testing.assert_allclose(out_q.numpy(), expected_q, atol=1e-5)

    def test_one_dof_matches_analytical_dls(self):
        device = wp.get_device()
        ERR_Y = 0.1
        LAMBDA = 0.5
        BAND = 2.0
        builder = _build_planar_arm_one_link()
        default_idx = _iota(1, device)
        out_q = wp.zeros(1, dtype=wp.float32, device=device)
        out_qd = wp.zeros(1, dtype=wp.float32, device=device)
        diffik = ControllerDifferentialKinematics(
            model_builder=builder,
            controlled_site_label="tool",
            default_dof_indices=default_idx,
            solver_damping=wp.array([LAMBDA], dtype=wp.float32, device=device),
            bandwidth=wp.array([BAND], dtype=wp.float32, device=device),
            device=device,
        )
        input_struct = SimpleNamespace(
            joint_q=wp.zeros(1, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(1, dtype=wp.float32, device=device),
            site_target_position=wp.array([wp.vec3(1.0, ERR_Y, 0.0)], dtype=wp.vec3, device=device),
            site_target_quaternion=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
        )
        output_struct = SimpleNamespace(joint_target_q=out_q, joint_target_qd=out_qd)
        diffik.compute(input_struct, output_struct, None, None, time_step=0.01)

        expected_qd = BAND * ERR_Y / (2.0 + LAMBDA**2)
        self.assertAlmostEqual(float(out_qd.numpy()[0]), expected_qd, places=5)

    def test_one_dof_matches_analytical_transpose(self):
        device = wp.get_device()
        ERR_Y = 0.1
        BAND = 2.0
        builder = _build_planar_arm_one_link()
        default_idx = _iota(1, device)
        out_q = wp.zeros(1, dtype=wp.float32, device=device)
        out_qd = wp.zeros(1, dtype=wp.float32, device=device)
        diffik = ControllerDifferentialKinematics(
            model_builder=builder,
            controlled_site_label="tool",
            default_dof_indices=default_idx,
            bandwidth=wp.array([BAND], dtype=wp.float32, device=device),
            ik_method=ControllerDifferentialKinematics.IkMethod.TRANSPOSE,
            device=device,
        )
        input_struct = SimpleNamespace(
            joint_q=wp.zeros(1, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(1, dtype=wp.float32, device=device),
            site_target_position=wp.array([wp.vec3(1.0, ERR_Y, 0.0)], dtype=wp.vec3, device=device),
            site_target_quaternion=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
        )
        output_struct = SimpleNamespace(joint_target_q=out_q, joint_target_qd=out_qd)
        diffik.compute(input_struct, output_struct, None, None, time_step=0.01)

        expected_qd = BAND * ERR_Y
        self.assertAlmostEqual(float(out_qd.numpy()[0]), expected_qd, places=5)

    def test_default_solver_damping(self):
        device = wp.get_device()
        builder = _build_planar_arm_one_link()
        default_idx = _iota(1, device)
        diffik = ControllerDifferentialKinematics(
            model_builder=builder,
            controlled_site_label="tool",
            default_dof_indices=default_idx,
            bandwidth=wp.array([1.0], dtype=wp.float32, device=device),
            device=device,
        )
        self.assertAlmostEqual(
            float(diffik._damping_baked.numpy()[0]),
            ControllerDifferentialKinematics.DEFAULT_SOLVER_DAMPING,
        )

    def test_runs_inside_wp_tape_without_crashing(self):
        device = wp.get_device()
        LAMBDA = 0.5
        BAND = 1.5
        builder = _build_planar_arm_one_link()
        default_idx = _iota(1, device)
        out_q = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        out_qd = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        target_pos = wp.array([wp.vec3(1.0, 0.1, 0.0)], dtype=wp.vec3, device=device, requires_grad=True)

        diffik = ControllerDifferentialKinematics(
            model_builder=builder,
            controlled_site_label="tool",
            default_dof_indices=default_idx,
            solver_damping=wp.array([LAMBDA], dtype=wp.float32, device=device),
            bandwidth=wp.array([BAND], dtype=wp.float32, device=device),
            device=device,
            requires_grad=True,
        )
        input_struct = SimpleNamespace(
            joint_q=wp.zeros(1, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(1, dtype=wp.float32, device=device),
            site_target_position=target_pos,
            site_target_quaternion=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
        )
        output_struct = SimpleNamespace(joint_target_q=out_q, joint_target_qd=out_qd)

        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            diffik.compute(input_struct, output_struct, None, None, time_step=0.01)
            wp.launch(_sum_kernel, dim=len(out_qd), inputs=[out_qd, loss])
        tape.backward(loss=loss)

        expected_qd = BAND * 0.1 / (2.0 + LAMBDA**2)
        self.assertAlmostEqual(float(out_qd.numpy()[0]), expected_qd, places=5)
        np.testing.assert_allclose(target_pos.grad.numpy()[0], [0.0, 0.0, 0.0], atol=1e-7)

    def test_parallel_robots(self):
        device = wp.get_device()
        LAMBDA = 0.5
        BAND = 1.0
        N = 4
        template = _build_planar_arm_one_link()
        builder = newton.ModelBuilder()
        builder.replicate(template, world_count=N)
        default_idx = _iota(N, device)
        target_ys = [0.10, 0.20, 0.05, -0.15]
        out_q = wp.zeros(N, dtype=wp.float32, device=device)
        out_qd = wp.zeros(N, dtype=wp.float32, device=device)

        diffik = ControllerDifferentialKinematics(
            model_builder=builder,
            controlled_site_label="tool",
            default_dof_indices=default_idx,
            solver_damping=wp.array([LAMBDA] * N, dtype=wp.float32, device=device),
            bandwidth=wp.array([BAND] * N, dtype=wp.float32, device=device),
            device=device,
        )
        input_struct = SimpleNamespace(
            joint_q=wp.zeros(N, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(N, dtype=wp.float32, device=device),
            site_target_position=wp.array([wp.vec3(1.0, y, 0.0) for y in target_ys], dtype=wp.vec3, device=device),
            site_target_quaternion=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * N, dtype=wp.quat, device=device),
        )
        output_struct = SimpleNamespace(joint_target_q=out_q, joint_target_qd=out_qd)
        diffik.compute(input_struct, output_struct, None, None, time_step=0.01)

        expected_qd = np.array([BAND * y / (2.0 + LAMBDA**2) for y in target_ys], dtype=np.float32)
        np.testing.assert_allclose(out_qd.numpy(), expected_qd, atol=1e-5)

    def test_robot_is_subset_of_scene(self):
        """Live per-DOF idx routes writes to arbitrary sim slots."""
        device = wp.get_device()
        LAMBDA = 0.5
        BAND = 1.5
        ERR_Y = 0.1
        sim_n_dofs = 2  # arm dof 0 + pendulum dof 1
        builder = _build_planar_arm_one_link()
        default_idx = _idx([0], device)
        out_q = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)
        out_qd = wp.zeros(sim_n_dofs, dtype=wp.float32, device=device)
        diffik = ControllerDifferentialKinematics(
            model_builder=builder,
            controlled_site_label="tool",
            default_dof_indices=default_idx,
            solver_damping=wp.array([LAMBDA], dtype=wp.float32, device=device),
            bandwidth=wp.array([BAND], dtype=wp.float32, device=device),
            device=device,
        )
        input_struct = SimpleNamespace(
            joint_q=wp.zeros(sim_n_dofs, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(sim_n_dofs, dtype=wp.float32, device=device),
            site_target_position=wp.array([wp.vec3(1.0, ERR_Y, 0.0)], dtype=wp.vec3, device=device),
            site_target_quaternion=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
        )
        output_struct = SimpleNamespace(joint_target_q=out_q, joint_target_qd=out_qd)
        dt = 0.01
        diffik.compute(input_struct, output_struct, None, None, time_step=dt)

        expected_qd_arm = BAND * ERR_Y / (2.0 + LAMBDA**2)
        np.testing.assert_allclose(out_qd.numpy(), [expected_qd_arm, 0.0], atol=1e-5)
        np.testing.assert_allclose(out_q.numpy(), [expected_qd_arm * dt, 0.0], atol=1e-5)

    def test_two_articulations_different_site_offsets(self):
        device = wp.get_device()
        LAMBDA = 0.5
        BAND = 1.0
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

        default_idx = _iota(2, device)
        out_q = wp.zeros(2, dtype=wp.float32, device=device)
        out_qd = wp.zeros(2, dtype=wp.float32, device=device)
        diffik = ControllerDifferentialKinematics(
            model_builder=builder,
            controlled_site_label="tool",
            default_dof_indices=default_idx,
            solver_damping=wp.array([LAMBDA, LAMBDA], dtype=wp.float32, device=device),
            bandwidth=wp.array([BAND, BAND], dtype=wp.float32, device=device),
            device=device,
        )
        input_struct = SimpleNamespace(
            joint_q=wp.zeros(2, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(2, dtype=wp.float32, device=device),
            site_target_position=wp.array(
                [wp.vec3(1.0, ERR_Y, 0.0), wp.vec3(2.0, ERR_Y, 0.0)],
                dtype=wp.vec3,
                device=device,
            ),
            site_target_quaternion=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * 2, dtype=wp.quat, device=device),
        )
        output_struct = SimpleNamespace(joint_target_q=out_q, joint_target_qd=out_qd)
        diffik.compute(input_struct, output_struct, None, None, time_step=0.01)

        expected_qd_v0 = BAND * ERR_Y / (2.0 + LAMBDA**2)
        expected_qd_v1 = BAND * 2.0 * ERR_Y / (5.0 + LAMBDA**2)
        np.testing.assert_allclose(out_qd.numpy(), [expected_qd_v0, expected_qd_v1], atol=1e-5)

    def test_per_robot_idx_share_target_across_robots(self):
        """Per-robot live-port idx lets multiple robots share a single source slot."""
        device = wp.get_device()
        LAMBDA = 0.5
        BAND = 1.0
        N = 4
        template = _build_planar_arm_one_link()
        builder = newton.ModelBuilder()
        builder.replicate(template, world_count=N)

        target_pos_short = wp.array(
            [wp.vec3(1.0, 0.10, 0.0), wp.vec3(1.0, 0.20, 0.0)],
            dtype=wp.vec3,
            device=device,
        )
        target_pos_idx = _idx([0, 0, 1, 1], device)
        default_idx = _iota(N, device)
        out_q = wp.zeros(N, dtype=wp.float32, device=device)
        out_qd = wp.zeros(N, dtype=wp.float32, device=device)

        diffik = ControllerDifferentialKinematics(
            model_builder=builder,
            controlled_site_label="tool",
            default_dof_indices=default_idx,
            solver_damping=wp.array([LAMBDA] * N, dtype=wp.float32, device=device),
            bandwidth=wp.array([BAND] * N, dtype=wp.float32, device=device),
            target_pos_idx=target_pos_idx,
            device=device,
        )
        input_struct = SimpleNamespace(
            joint_q=wp.zeros(N, dtype=wp.float32, device=device),
            joint_qd=wp.zeros(N, dtype=wp.float32, device=device),
            site_target_position=target_pos_short,
            site_target_quaternion=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * N, dtype=wp.quat, device=device),
        )
        output_struct = SimpleNamespace(joint_target_q=out_q, joint_target_qd=out_qd)
        diffik.compute(input_struct, output_struct, None, None, time_step=0.01)

        qd_0 = BAND * 0.10 / (2.0 + LAMBDA**2)
        qd_1 = BAND * 0.20 / (2.0 + LAMBDA**2)
        np.testing.assert_allclose(out_qd.numpy(), [qd_0, qd_0, qd_1, qd_1], atol=1e-5)

    def test_struct_factories(self):
        device = wp.get_device()
        builder = _build_planar_arm_one_link()
        default_idx = _iota(1, device)
        diffik = ControllerDifferentialKinematics(
            model_builder=builder,
            controlled_site_label="tool",
            default_dof_indices=default_idx,
            bandwidth="my_band",  # live
            device=device,
        )
        i = diffik.input_struct()
        self.assertTrue(hasattr(i, "joint_q"))
        self.assertTrue(hasattr(i, "joint_qd"))
        self.assertTrue(hasattr(i, "site_target_position"))
        self.assertTrue(hasattr(i, "site_target_quaternion"))
        self.assertTrue(hasattr(i, "my_band"))
        self.assertFalse(hasattr(i, "solver_damping"))  # baked
        self.assertEqual(i.site_target_position.dtype, wp.vec3)
        self.assertEqual(i.site_target_quaternion.dtype, wp.quat)
        o = diffik.output_struct()
        self.assertTrue(hasattr(o, "joint_target_q"))
        self.assertTrue(hasattr(o, "joint_target_qd"))


if __name__ == "__main__":
    unittest.main()
