# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for ControllerJointImpedance and ControllerJointImpedanceModelFree."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.controllers import ControllerJointImpedance, ControllerJointImpedanceModelFree

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _idx(values, device):
    """Return a wp.array[int32] of model indices."""
    return wp.array(np.array(values, dtype=np.int32), dtype=wp.int32, device=device)


def _dofs_arr(dofs_list, device):
    """Return a wp.array[int32] from a list of per-robot DOF counts."""
    return wp.array(np.array(dofs_list, dtype=np.int32), device=device)


def _gains(total_controlled_dofs, value, device):
    """Return a compact (total_controlled_dofs,) float32 gain array filled with value."""
    return wp.full(total_controlled_dofs, value, dtype=wp.float32, device=device)


def _flat(data, device):
    """Return a flat float32 Warp array from any array-like."""
    return wp.array(np.array(data, dtype=np.float32).flatten(), dtype=wp.float32, device=device)


def _build_single_prismatic():
    """Build a one-robot, one-DOF prismatic-joint ModelBuilder."""
    builder = newton.ModelBuilder()
    link = builder.add_link()
    j = builder.add_joint_prismatic(
        parent=-1,
        child=link,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j], label="robot")
    return builder


def _build_two_robot_mixed():
    """Build a ModelBuilder with robot 0 (2 revolute DOFs) and robot 1 (1 prismatic DOF)."""
    builder = newton.ModelBuilder()
    # Robot 0: 2-DOF revolute chain
    l0a = builder.add_link()
    l0b = builder.add_link()
    j0a = builder.add_joint_revolute(
        parent=-1,
        child=l0a,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    j0b = builder.add_joint_revolute(
        parent=l0a,
        child=l0b,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j0a, j0b], label="robot0")
    # Robot 1: 1-DOF prismatic
    l1 = builder.add_link()
    j1 = builder.add_joint_prismatic(
        parent=-1,
        child=l1,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j1], label="robot1")
    return builder


def _build_ball_then_revolute():
    """Build a one-robot model whose base is an uncontrollable 3-DOF ball joint.

    The ball joint spans four coordinates but three DOFs, so every joint after
    it has a different coordinate index than DOF index.
    """
    builder = newton.ModelBuilder()
    base = builder.add_link()
    arm = builder.add_link()
    j_ball = builder.add_joint_ball(
        parent=-1,
        child=base,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    j_rev = builder.add_joint_revolute(
        parent=base,
        child=arm,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j_ball, j_rev], label="robot")
    return builder, j_ball, j_rev


def _build_floating_base_fleet():
    """Build robot 0 (free base + 2 revolute) and robot 1 (1 revolute).

    The free joint spans seven coordinates but six DOFs, so coordinate and DOF
    indices diverge and the two robots have different controlled-DOF counts.
    Returns the builder and the three revolute joints, i.e. every joint but
    the (uncontrollable) free base.
    """
    builder = newton.ModelBuilder()
    base = builder.add_link(mass=1.0)
    a1 = builder.add_link(mass=1.0)
    a2 = builder.add_link(mass=1.0)
    jf = builder.add_joint_free(child=base)
    j1 = builder.add_joint_revolute(parent=base, child=a1, axis=wp.vec3(0.0, 0.0, 1.0))
    j2 = builder.add_joint_revolute(parent=a1, child=a2, axis=wp.vec3(0.0, 0.0, 1.0))
    builder.add_articulation([jf, j1, j2], label="robot0")
    link = builder.add_link(mass=1.0)
    j3 = builder.add_joint_revolute(parent=-1, child=link, axis=wp.vec3(0.0, 0.0, 1.0))
    builder.add_articulation([j3], label="robot1")
    return builder, [j1, j2, j3]


def _make_mf(
    *,
    dofs_list,
    kp,
    kd,
    device,
    use_gravity=False,
    use_coriolis=False,
    use_inertia=False,
    has_qdd=False,
):
    """Construct a ControllerJointImpedanceModelFree with compact gains."""
    total_controlled_dofs = sum(dofs_list)
    return ControllerJointImpedanceModelFree(
        controlled_dofs_per_robot=_dofs_arr(dofs_list, device),
        stiffness=_gains(total_controlled_dofs, kp, device),
        damping=_gains(total_controlled_dofs, kd, device),
        use_gravity_compensation=use_gravity,
        use_coriolis_compensation=use_coriolis,
        use_inertia_decoupling=use_inertia,
        has_qdd_feedforward=has_qdd,
        device=device,
    )


def _run_mf(ctrl, *, q, qd, q_des, qd_des, device, **extras):
    """Run one step on a ModelFree controller and return the torque array."""
    ins = ctrl.input()
    ins.joint_q = _flat(q, device)
    ins.joint_qd = _flat(qd, device)
    ins.joint_q_des = _flat(q_des, device)
    ins.joint_qd_des = _flat(qd_des, device)
    for k, v in extras.items():
        setattr(ins, k, v)
    outs = ctrl.output()
    ctrl.step(inputs=ins, outputs=outs, dt=0.01)
    return outs.joint_f.numpy()


# ---------------------------------------------------------------------------
# ControllerJointImpedanceModelFree — homogeneous
# ---------------------------------------------------------------------------


class TestControllerJointImpedanceModelFree(unittest.TestCase):
    def test_zero_error_gives_zero_torque(self):
        """Verify that zero position and velocity error produces zero torque."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[3], kp=10.0, kd=1.0, device=device)
        tau = _run_mf(
            ctrl, q=[0.1, 0.2, 0.3], qd=[0.0, 0.0, 0.0], q_des=[0.1, 0.2, 0.3], qd_des=[0.0, 0.0, 0.0], device=device
        )
        np.testing.assert_allclose(tau, np.zeros(3, dtype=np.float32), atol=1e-5)

    def test_position_error_produces_stiffness_torque(self):
        """Verify τ = Kp * (q_des - q) when Kd=0."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[3], kp=5.0, kd=0.0, device=device)
        tau = _run_mf(
            ctrl, q=[0.0, 0.0, 0.0], qd=[0.0, 0.0, 0.0], q_des=[1.0, 0.0, 0.0], qd_des=[0.0, 0.0, 0.0], device=device
        )
        np.testing.assert_allclose(tau, [5.0, 0.0, 0.0], atol=1e-5)

    def test_scalar_stiffness_broadcasts_to_every_dof(self):
        """Verify a scalar stiffness applies the same Kp to every controlled DOF."""
        device = wp.get_device()
        ctrl = ControllerJointImpedanceModelFree(
            controlled_dofs_per_robot=_dofs_arr([2, 1], device),
            stiffness=5.0,
            damping=0.0,
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        tau = _run_mf(ctrl, q=[0.0] * 3, qd=[0.0] * 3, q_des=[1.0, 2.0, 3.0], qd_des=[0.0] * 3, device=device)
        np.testing.assert_allclose(tau, [5.0, 10.0, 15.0], atol=1e-5)

    def test_velocity_error_produces_damping_torque(self):
        """Verify τ = Kd * (qd_des - qd) when Kp=0."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[3], kp=0.0, kd=2.0, device=device)
        tau = _run_mf(
            ctrl, q=[0.0, 0.0, 0.0], qd=[0.0, 0.0, 0.0], q_des=[0.0, 0.0, 0.0], qd_des=[0.0, 1.0, 0.0], device=device
        )
        np.testing.assert_allclose(tau, [0.0, 2.0, 0.0], atol=1e-5)

    def test_multiple_robots_independent(self):
        """Verify that torques for each robot depend only on that robot's error."""
        device = wp.get_device()
        model_robot_count, num_dofs = 3, 2
        ctrl = _make_mf(dofs_list=[num_dofs] * model_robot_count, kp=1.0, kd=0.0, device=device)
        q_des = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        q = np.zeros((model_robot_count, num_dofs), dtype=np.float32)
        tau = _run_mf(ctrl, q=q, qd=q * 0, q_des=q_des, qd_des=q * 0, device=device)
        np.testing.assert_allclose(tau, q_des.flatten(), atol=1e-5)

    def test_per_dof_gains_apply_independently(self):
        """Verify a compact gain array applies a distinct Kp to each controlled DOF.

        With 1-D gains there is no per-robot padding, so entry i of the gain
        array must line up with entry i of every other compact port regardless
        of how the DOFs are distributed across robots.
        """
        device = wp.get_device()
        ctrl = ControllerJointImpedanceModelFree(
            controlled_dofs_per_robot=_dofs_arr([2, 1], device),
            stiffness=wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device),
            damping=_gains(3, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        tau = _run_mf(ctrl, q=[0.0] * 3, qd=[0.0] * 3, q_des=[1.0, 1.0, 1.0], qd_des=[0.0] * 3, device=device)
        np.testing.assert_allclose(tau, [1.0, 2.0, 3.0], atol=1e-5)

    def test_inertia_decoupling_scales_by_mass_matrix(self):
        """Verify τ = M @ (Kp * Δq) when use_inertia_decoupling=True."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=1.0, kd=0.0, device=device, use_inertia=True)
        M = wp.array(np.eye(2, dtype=np.float32).reshape(1, 2, 2) * 2.0, dtype=wp.float32, device=device)
        tau = _run_mf(
            ctrl, q=[0.0, 0.0], qd=[0.0, 0.0], q_des=[1.0, 1.0], qd_des=[0.0, 0.0], device=device, mass_matrix=M
        )
        np.testing.assert_allclose(tau, [2.0, 2.0], atol=1e-5)

    def test_gravity_compensation_adds_to_tau(self):
        """Verify gravity_force is added to τ when use_gravity_compensation=True."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=0.0, kd=0.0, device=device, use_gravity=True)
        grav = wp.array([3.0, 4.0], dtype=wp.float32, device=device)
        tau = _run_mf(
            ctrl, q=[0.0, 0.0], qd=[0.0, 0.0], q_des=[0.0, 0.0], qd_des=[0.0, 0.0], device=device, gravity_force=grav
        )
        np.testing.assert_allclose(tau, [3.0, 4.0], atol=1e-5)

    def test_coriolis_compensation_adds_to_tau(self):
        """Verify coriolis_force is added to τ when use_coriolis_compensation=True."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=0.0, kd=0.0, device=device, use_coriolis=True)
        cor = wp.array([1.0, -1.0], dtype=wp.float32, device=device)
        tau = _run_mf(
            ctrl, q=[0.0, 0.0], qd=[0.0, 0.0], q_des=[0.0, 0.0], qd_des=[0.0, 0.0], device=device, coriolis_force=cor
        )
        np.testing.assert_allclose(tau, [1.0, -1.0], atol=1e-5)

    def test_qdd_feedforward_adds_before_inertia(self):
        """Verify qdd feedforward is included inside M @ (PD + qdd) when use_inertia=True."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=0.0, kd=0.0, device=device, use_inertia=True, has_qdd=True)
        M = wp.array(np.eye(2, dtype=np.float32).reshape(1, 2, 2) * 3.0, dtype=wp.float32, device=device)
        qdd = wp.array([1.0, 0.0], dtype=wp.float32, device=device)
        tau = _run_mf(
            ctrl,
            q=[0.0, 0.0],
            qd=[0.0, 0.0],
            q_des=[0.0, 0.0],
            qd_des=[0.0, 0.0],
            device=device,
            mass_matrix=M,
            joint_qdd=qdd,
        )
        np.testing.assert_allclose(tau, [3.0, 0.0], atol=1e-5)

    def test_live_stiffness_port(self):
        """Verify stiffness supplied via inputs.stiffness each step is applied correctly."""
        device = wp.get_device()
        ctrl = ControllerJointImpedanceModelFree(
            controlled_dofs_per_robot=_dofs_arr([2], device),
            stiffness=None,
            damping=wp.zeros(2, dtype=wp.float32, device=device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        ins = ctrl.input()
        ins.joint_q = wp.zeros(2, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(2, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array([2.0, 0.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(2, dtype=wp.float32, device=device)
        ins.stiffness = wp.array([3.0, 3.0], dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)
        np.testing.assert_allclose(outs.joint_f.numpy(), [6.0, 0.0], atol=1e-5)

    def test_is_graphable(self):
        """Verify the controller reports is_graphable() == True."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=1.0, kd=0.0, device=device)
        self.assertTrue(ctrl.is_graphable())

    def test_inputs_has_required_fields(self):
        """Verify input() returns a namespace with all declared port fields present."""
        device = wp.get_device()
        ctrl = _make_mf(
            dofs_list=[3],
            kp=1.0,
            kd=0.0,
            device=device,
            use_gravity=True,
            use_coriolis=True,
            use_inertia=True,
            has_qdd=True,
        )
        ins = ctrl.input()
        for field in (
            "joint_q",
            "joint_qd",
            "joint_q_des",
            "joint_qd_des",
            "joint_qdd",
            "mass_matrix",
            "gravity_force",
            "coriolis_force",
        ):
            self.assertTrue(hasattr(ins, field), f"Missing field: {field}")

    def test_all_ports_are_compact(self):
        """Verify every allocated 1-D port has exactly one entry per controlled DOF."""
        device = wp.get_device()
        ctrl = _make_mf(
            dofs_list=[3, 1],
            kp=1.0,
            kd=0.0,
            device=device,
            use_gravity=True,
            use_coriolis=True,
            has_qdd=True,
        )
        ins, outs = ctrl.input(), ctrl.output()
        self.assertEqual(ctrl.total_controlled_dofs, 4)
        for field in ("joint_q", "joint_qd", "joint_q_des", "joint_qd_des", "joint_qdd", "gravity_force"):
            self.assertEqual(getattr(ins, field).shape, (4,), f"{field} is not compact")
        self.assertEqual(outs.joint_f.shape, (4,))

    def test_indexed_view_input_gathers(self):
        """Verify a port bound to an indexed view reads through to the underlying array.

        This is how a caller feeds a simulation-sized array to a compact port
        without the controller owning an index table.
        """
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=1.0, kd=0.0, device=device)
        sim_q_des = wp.array([0.0, 5.0, 0.0, 3.0], dtype=wp.float32, device=device)
        idx = _idx([1, 3], device)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q_des = sim_q_des[idx]
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)
        np.testing.assert_allclose(outs.joint_f.numpy(), [5.0, 3.0], atol=1e-5)

    def test_indexed_view_input_is_live(self):
        """Verify a view bound once reflects later writes to the underlying array."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=1.0, kd=0.0, device=device)
        sim_q_des = wp.zeros(4, dtype=wp.float32, device=device)
        idx = _idx([1, 3], device)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q_des = sim_q_des[idx]
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)
        np.testing.assert_allclose(outs.joint_f.numpy(), [0.0, 0.0], atol=1e-5)

        sim_q_des.assign(np.array([0.0, 7.0, 0.0, 9.0], dtype=np.float32))
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)
        np.testing.assert_allclose(outs.joint_f.numpy(), [7.0, 9.0], atol=1e-5)

    def test_indexed_view_output_scatters(self):
        """Verify an output bound to an indexed view scatters torques into the underlying array."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=1.0, kd=0.0, device=device)
        sim_f = wp.zeros(4, dtype=wp.float32, device=device)
        idx = _idx([1, 3], device)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q_des = wp.array([2.0, 6.0], dtype=wp.float32, device=device)
        outs.joint_f = sim_f[idx]
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)
        np.testing.assert_allclose(sim_f.numpy(), [0.0, 2.0, 0.0, 6.0], atol=1e-5)

    def test_simulation_sized_port_raises(self):
        """Verify binding a simulation-sized array to a compact port raises.

        Ports are exact-length so that a caller who assumed model layout gets an
        error rather than silently reading the first ``total_controlled_dofs`` entries.
        """
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=1.0, kd=0.0, device=device)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q_des = wp.zeros(10, dtype=wp.float32, device=device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=outs, dt=0.01)

    def test_short_port_raises(self):
        """Verify a port shorter than total_controlled_dofs raises instead of reading out of bounds."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=1.0, kd=0.0, device=device)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q = wp.zeros(1, dtype=wp.float32, device=device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=outs, dt=0.01)

    def test_2d_controlled_dofs_per_robot_raises(self):
        """Verify a 2-D controlled_dofs_per_robot raises instead of silently deriving incorrect model_robot_count."""
        device = wp.get_device()
        # A (2, 3) array has size 6, which would otherwise be read as 6 robots.
        dofs_2d = wp.full((2, 3), 1, dtype=wp.int32, device=device)
        with self.assertRaises(ValueError):
            ControllerJointImpedanceModelFree(
                controlled_dofs_per_robot=dofs_2d,
                stiffness=_gains(6, 1.0, device),
                damping=_gains(6, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
                device=device,
            )

    def test_wrong_device_input_raises(self):
        """Verify an input array on another device raises instead of being dereferenced."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("needs a second device to mismatch against")
        ctrl = _make_mf(dofs_list=[2], kp=1.0, kd=0.0, device=device)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q_des = wp.array([1.0, 1.0], dtype=wp.float32, device="cpu")
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=outs, dt=0.01)

    def test_wrong_shape_live_gain_raises(self):
        """Verify a live gain whose length differs from total_controlled_dofs raises."""
        device = wp.get_device()
        ctrl = ControllerJointImpedanceModelFree(
            controlled_dofs_per_robot=_dofs_arr([2, 2], device),
            stiffness=None,  # live: read from inputs.stiffness each step
            damping=_gains(4, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        ins, outs = ctrl.input(), ctrl.output()
        ins.stiffness = wp.full(1, 7.0, dtype=wp.float32, device=device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=outs, dt=0.01)

    def test_controlled_dofs_per_robot_is_copied(self):
        """Verify editing the caller's controlled_dofs_per_robot after construction changes nothing.

        The kernels use it as a loop bound, while the flat-DOF tables and every
        buffer are sized from a snapshot taken at construction. An entry grown
        afterwards would walk the multiply past the end of those buffers.
        """
        device = wp.get_device()
        dofs = _dofs_arr([2], device)
        ctrl = ControllerJointImpedanceModelFree(
            controlled_dofs_per_robot=dofs,
            stiffness=_gains(2, 1.0, device),
            damping=_gains(2, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        before = _run_mf(ctrl, q=[0.0, 0.0], qd=[0.0, 0.0], q_des=[1.0, 2.0], qd_des=[0.0, 0.0], device=device)

        dofs.assign(np.array([5], dtype=np.int32))  # more DOFs than any buffer holds
        after = _run_mf(ctrl, q=[0.0, 0.0], qd=[0.0, 0.0], q_des=[1.0, 2.0], qd_des=[0.0, 0.0], device=device)

        np.testing.assert_allclose(after, before, atol=1e-6)
        np.testing.assert_array_equal(ctrl._controlled_dofs_per_robot.numpy(), [2])

    def test_mass_matrix_bound_to_a_view(self):
        """Verify a mass matrix bound to a view of a larger set of blocks is gathered correctly.

        The caller holds blocks for three robots and controls the middle one, so
        the view selects a single block that the multiply must read in place of
        the first.
        """
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=1.0, kd=0.0, device=device, use_inertia=True)

        fleet = np.zeros((3, 2, 2), dtype=np.float32)
        fleet[0] = np.eye(2) * 99.0  # a block the controller must not read
        fleet[1] = np.array([[2.0, 0.0], [0.0, 4.0]], dtype=np.float32)
        fleet[2] = np.eye(2) * -99.0
        blocks = wp.array(fleet, dtype=wp.float32, device=device)

        tau = _run_mf(
            ctrl,
            q=[0.0, 0.0],
            qd=[0.0, 0.0],
            q_des=[1.0, 1.0],
            qd_des=[0.0, 0.0],
            device=device,
            mass_matrix=blocks[_idx([1], device)],
        )
        np.testing.assert_allclose(tau, [2.0, 4.0], atol=1e-5)

    def test_wrong_shape_mass_matrix_raises(self):
        """Verify a mass matrix whose shape differs from (controlled_robot_count, max_controlled_dofs, max_controlled_dofs) raises."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2, 2], kp=1.0, kd=0.0, device=device, use_inertia=True)
        ins, outs = ctrl.input(), ctrl.output()
        ins.mass_matrix = wp.zeros((1, 1, 1), dtype=wp.float32, device=device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=outs, dt=0.01)

    def test_wrong_shape_constructor_gain_raises(self):
        """Verify a construction-time gain that is not 1-D of length total_controlled_dofs raises."""
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerJointImpedanceModelFree(
                controlled_dofs_per_robot=_dofs_arr([2], device),
                stiffness=wp.full((1, 2), 1.0, dtype=wp.float32, device=device),  # 2-D
                damping=_gains(2, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
                device=device,
            )

    def test_wrong_device_constructor_array_raises(self):
        """Verify every wp.array constructor argument is rejected when it is on the wrong device."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("needs a second device to mismatch against")
        other = "cpu"

        def _kwargs():
            return {
                "controlled_dofs_per_robot": _dofs_arr([2], device),
                "stiffness": _gains(2, 1.0, device),
                "damping": _gains(2, 0.0, device),
                "use_gravity_compensation": False,
                "use_coriolis_compensation": False,
                "use_inertia_decoupling": False,
                "device": device,
            }

        wrong = {
            "controlled_dofs_per_robot": _dofs_arr([2], other),
            "stiffness": _gains(2, 1.0, other),
            "damping": _gains(2, 0.0, other),
        }
        for name, bad_array in wrong.items():
            with self.subTest(argument=name), self.assertRaises(ValueError):
                ControllerJointImpedanceModelFree(**{**_kwargs(), name: bad_array})

    def test_wrong_dtype_constructor_array_raises(self):
        """Verify a wp.array constructor argument with the wrong dtype raises TypeError."""
        device = wp.get_device()
        with self.assertRaises(TypeError):
            ControllerJointImpedanceModelFree(
                controlled_dofs_per_robot=wp.zeros(2, dtype=wp.float32, device=device),  # want int32
                stiffness=_gains(2, 1.0, device),
                damping=_gains(2, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
                device=device,
            )

    def test_disabled_port_written_raises(self):
        """Verify writing a port whose feature is disabled raises instead of being ignored."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=1.0, kd=0.0, device=device)  # no feedforward, no dynamics terms
        ins = ctrl.input()
        ins.joint_q = _flat([0.0, 0.0], device)
        ins.joint_qd = _flat([0.0, 0.0], device)
        ins.joint_q_des = _flat([1.0, 1.0], device)
        ins.joint_qd_des = _flat([0.0, 0.0], device)
        ins.joint_qdd = _flat([100.0, 100.0], device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=ctrl.output(), dt=0.01)

    def test_zero_dof_robot_raises(self):
        """Verify a robot with zero controlled DOFs raises rather than occupying an empty slot.

        Every buffer is sized to the robots actually controlled, so a zero-DOF
        entry would reserve a mass-matrix block nothing ever reads.
        """
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerJointImpedanceModelFree(
                controlled_dofs_per_robot=_dofs_arr([0, 2], device),
                stiffness=_gains(2, 1.0, device),
                damping=_gains(2, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
                device=device,
            )

    def test_all_robots_zero_dofs_raises(self):
        """Verify a controlled_dofs_per_robot summing to zero raises rather than building an empty controller."""
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerJointImpedanceModelFree(
                controlled_dofs_per_robot=_dofs_arr([0, 0], device),
                stiffness=None,
                damping=None,
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
                device=device,
            )


# ---------------------------------------------------------------------------
# ControllerJointImpedanceModelFree — heterogeneous
# ---------------------------------------------------------------------------


class TestControllerJointImpedanceModelFreeHeterogeneous(unittest.TestCase):
    def test_heterogeneous_pd_torques(self):
        """Verify PD torques are correct for each robot with different DOF counts."""
        device = wp.get_device()
        # Robot 0: 2 DOFs, Kp=5; Robot 1: 1 DOF, Kp=5
        # Errors: robot0=[1,0], robot1=[2]  →  tau: robot0=[5,0], robot1=[10]
        ctrl = _make_mf(dofs_list=[2, 1], kp=5.0, kd=0.0, device=device)
        tau = _run_mf(ctrl, q=[0.0] * 3, qd=[0.0] * 3, q_des=[1.0, 0.0, 2.0], qd_des=[0.0] * 3, device=device)
        np.testing.assert_allclose(tau, [5.0, 0.0, 10.0], atol=1e-5)

    def test_heterogeneous_independence(self):
        """Verify robot 0's torques are zero when only robot 1 has a position error."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2, 1], kp=1.0, kd=0.0, device=device)
        tau = _run_mf(ctrl, q=[0.0] * 3, qd=[0.0] * 3, q_des=[0.0, 0.0, 3.0], qd_des=[0.0] * 3, device=device)
        np.testing.assert_allclose(tau[:2], [0.0, 0.0], atol=1e-5)
        self.assertAlmostEqual(tau[2], 3.0, places=5)

    def test_heterogeneous_output_is_compact(self):
        """Verify the torque output has exactly sum(controlled_dofs_per_robot) entries, with no padding."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[3, 1], kp=1.0, kd=0.0, device=device)
        tau = _run_mf(ctrl, q=[0.0] * 4, qd=[0.0] * 4, q_des=[99.0] * 4, qd_des=[0.0] * 4, device=device)
        self.assertEqual(tau.shape, (4,))
        np.testing.assert_allclose(tau, [99.0, 99.0, 99.0, 99.0], atol=1e-5)

    def test_heterogeneous_inertia_decoupling(self):
        """Verify M @ acc is computed per-robot with heterogeneous DOF counts."""
        device = wp.get_device()
        # Robot 0: 2 DOFs, M=2*I; Robot 1: 1 DOF, M=[[3]]
        # Errors: robot0=[1,1], robot1=[1] →  acc=[1,1] and [1]
        # tau: robot0 = 2*I @ [1,1] = [2,2], robot1 = [[3]] @ [1] = [3]
        ctrl = _make_mf(dofs_list=[2, 1], kp=1.0, kd=0.0, device=device, use_inertia=True)
        M_np = np.zeros((2, 2, 2), dtype=np.float32)
        M_np[0] = np.eye(2) * 2.0
        M_np[1, 0, 0] = 3.0
        M = wp.array(M_np, dtype=wp.float32, device=device)
        tau = _run_mf(ctrl, q=[0.0] * 3, qd=[0.0] * 3, q_des=[1.0] * 3, qd_des=[0.0] * 3, device=device, mass_matrix=M)
        np.testing.assert_allclose(tau, [2.0, 2.0, 3.0], atol=1e-5)

    def test_heterogeneous_off_diagonal_mass_matrix(self):
        """Verify the mass-matrix multiply reads only its own robot's block.

        The flat launch maps each compact DOF back to a (robot, row) pair, so a
        nonzero off-diagonal in robot 0's block must not leak into robot 1's
        torque and vice versa.
        """
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2, 1], kp=1.0, kd=0.0, device=device, use_inertia=True)
        M_np = np.zeros((2, 2, 2), dtype=np.float32)
        M_np[0] = np.array([[1.0, 0.5], [0.5, 1.0]])
        M_np[1, 0, 0] = 4.0
        M = wp.array(M_np, dtype=wp.float32, device=device)
        tau = _run_mf(
            ctrl,
            q=[0.0] * 3,
            qd=[0.0] * 3,
            q_des=[1.0, 0.0, 1.0],
            qd_des=[0.0] * 3,
            device=device,
            mass_matrix=M,
        )
        # robot0: [[1,0.5],[0.5,1]] @ [1,0] = [1, 0.5];  robot1: [[4]] @ [1] = [4]
        np.testing.assert_allclose(tau, [1.0, 0.5, 4.0], atol=1e-5)


# ---------------------------------------------------------------------------
# ControllerJointImpedance (model-based)
# ---------------------------------------------------------------------------


class TestControllerJointImpedance(unittest.TestCase):
    def _make_ctrl(self, device, *, kp=10.0, kd=1.0, use_inertia=False):
        """Build a ControllerJointImpedance for a single prismatic robot."""
        model = _build_single_prismatic().finalize(device=device)
        return ControllerJointImpedance(
            model,
            stiffness=_gains(1, kp, device),
            damping=_gains(1, kd, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=use_inertia,
        )

    def _run(self, ctrl, *, q_sim, qd_sim, q_des, qd_des, device):
        """Run one step and return the torque array."""
        ins = ctrl.input()
        ins.joint_q = _flat(q_sim, device)
        ins.joint_qd = _flat(qd_sim, device)
        ins.joint_q_des = _flat(q_des, device)
        ins.joint_qd_des = _flat(qd_des, device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)
        return outs.joint_f.numpy()

    def test_zero_error_gives_zero_torque(self):
        """Verify zero position and velocity error produces zero torque."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device)
        tau = self._run(ctrl, q_sim=[0.5], qd_sim=[0.0], q_des=[0.5], qd_des=[0.0], device=device)
        np.testing.assert_allclose(tau, [0.0], atol=1e-4)

    def test_position_error_produces_stiffness_torque(self):
        """Verify τ = Kp * (q_des - q) for a simple prismatic robot."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device, kp=5.0, kd=0.0)
        tau = self._run(ctrl, q_sim=[0.0], qd_sim=[0.0], q_des=[1.0], qd_des=[0.0], device=device)
        np.testing.assert_allclose(tau, [5.0], atol=1e-4)

    def test_damping_term(self):
        """Verify τ = Kd * (qd_des - qd) when Kp=0."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device, kp=0.0, kd=3.0)
        tau = self._run(ctrl, q_sim=[0.0], qd_sim=[0.0], q_des=[0.0], qd_des=[2.0], device=device)
        np.testing.assert_allclose(tau, [6.0], atol=1e-4)

    def test_scalar_stiffness_broadcasts_without_knowing_total_controlled_dofs(self):
        """Verify a scalar stiffness applies to every controlled DOF, resolved after articulations/joints."""
        device = wp.get_device()
        model = _build_two_robot_mixed().finalize(device=device)  # robot0: 2 DOFs, robot1: 1 DOF
        ctrl = ControllerJointImpedance(
            model,
            stiffness=4.0,
            damping=0.0,
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        tau = self._run(
            ctrl, q_sim=[0.0, 0.0, 0.0], qd_sim=[0.0, 0.0, 0.0], q_des=[1.0, 0.0, 2.0], qd_des=[0.0] * 3, device=device
        )
        np.testing.assert_allclose(tau, [4.0, 0.0, 8.0], atol=1e-4)

    def test_is_graphable_true(self):
        """Verify is_graphable() returns True."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device)
        self.assertTrue(ctrl.is_graphable())

    def test_device_and_requires_grad_derived_from_model(self):
        """Verify device and requires_grad come from the model, with no constructor override.

        Neither is a constructor argument: both would otherwise be redundant
        sources of truth the caller must keep in sync with the model.
        """
        device = wp.get_device()
        model = _build_single_prismatic().finalize(device=device, requires_grad=True)
        with wp.ScopedDevice("cpu" if device.is_cuda else device):
            ctrl = ControllerJointImpedance(
                model,
                stiffness=_gains(1, 1.0, device),
                damping=_gains(1, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
            )
        self.assertEqual(ctrl.device, device)
        self.assertTrue(ctrl.requires_grad)

    def test_ball_joint_uncontrolled_allowed(self):
        """Verify a multi-DOF ball joint is allowed as long as it is not controlled."""
        device = wp.get_device()
        builder, _j_ball, j_rev = _build_ball_then_revolute()
        model = builder.finalize(device=device)
        # Only the revolute joint is addressed by ``joints``, so the ball
        # joint is read for FK/dynamics but never controlled.
        ctrl = ControllerJointImpedance(
            model,
            joints=[j_rev],
            stiffness=_gains(1, 5.0, device),
            damping=_gains(1, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        self.assertIsNotNone(ctrl)
        # Full-model state: identity ball-joint quaternion (x, y, z, w) followed
        # by the revolute joint's coordinate/DOF; q_des/qd_des are compact and
        # address only the controlled revolute joint.
        tau = self._run(
            ctrl,
            q_sim=[0.0, 0.0, 0.0, 1.0, 0.0],
            qd_sim=[0.0, 0.0, 0.0, 0.0],
            q_des=[1.0],
            qd_des=[0.0],
            device=device,
        )
        np.testing.assert_allclose(tau, [5.0], atol=1e-4)

    def test_ball_joint_controlled_raises(self):
        """Verify explicitly naming a multi-DOF ball joint in ``joints`` raises ValueError.

        The default selection leaves an unsupported joint like this one
        uncontrolled instead (see ``test_default_selection_skips_uncontrollable_joints``),
        so this test names it explicitly to still exercise the check.
        """
        device = wp.get_device()
        builder = newton.ModelBuilder()
        link = builder.add_link()
        j = builder.add_joint_ball(
            parent=-1,
            child=link,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([j], label="ball_robot")
        model = builder.finalize(device=device)
        with self.assertRaises(ValueError):
            ControllerJointImpedance(
                model,
                joints=[j],
                stiffness=_gains(1, 1.0, device),
                damping=_gains(1, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
            )

    def test_default_selection_skips_uncontrollable_joints(self):
        """Verify the default ``joints`` selection leaves non-1x1 joints uncontrolled rather than raising.

        A model mixing a ball joint with a controllable revolute joint should
        not need to be pruned by hand: omitting ``joints`` controls only the
        revolute joint, the same result as naming it explicitly.
        """
        device = wp.get_device()
        builder, _j_ball, j_rev = _build_ball_then_revolute()
        model = builder.finalize(device=device)
        ctrl = ControllerJointImpedance(
            model,
            stiffness=_gains(1, 5.0, device),
            damping=_gains(1, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        self.assertEqual(ctrl.total_controlled_dofs, 1)
        np.testing.assert_array_equal(ctrl.qd_start.numpy(), [model.joint_qd_start.numpy()[j_rev]])

    def test_fixed_joint_allowed(self):
        """Verify that fixed joints (zero DOF) are accepted alongside revolute/prismatic joints."""
        device = wp.get_device()
        builder = newton.ModelBuilder()
        base = builder.add_link()
        arm = builder.add_link()
        j_fixed = builder.add_joint_fixed(
            parent=-1,
            child=base,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        j_rev = builder.add_joint_revolute(
            parent=base,
            child=arm,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([j_fixed, j_rev], label="robot")
        model = builder.finalize(device=device)
        # Should not raise — fixed joint is zero-DOF and irrelevant to the PD term.
        ctrl = ControllerJointImpedance(
            model,
            joints=[j_rev],
            stiffness=_gains(1, 10.0, device),
            damping=_gains(1, 1.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
        )
        self.assertIsNotNone(ctrl)

    def test_out_of_range_index_raises(self):
        """Verify an index outside the model's joint range raises."""
        device = wp.get_device()
        model = _build_single_prismatic().finalize(device=device)
        with self.assertRaises(ValueError):
            ControllerJointImpedance(
                model,
                joints=[99],
                stiffness=_gains(1, 1.0, device),
                damping=_gains(1, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
            )

    def test_joint_outside_any_articulation_raises(self):
        """Verify controlling a joint that belongs to no articulation raises."""
        device = wp.get_device()
        builder = newton.ModelBuilder()
        loose = builder.add_link()
        # A world-root joint left out of every articulation, so it is never
        # reached by the per-robot FK and dynamics evaluations.
        builder.add_joint_revolute(parent=-1, child=loose, axis=wp.vec3(0.0, 0.0, 1.0))
        controlled = builder.add_link()
        builder.add_articulation(
            [builder.add_joint_revolute(parent=-1, child=controlled, axis=wp.vec3(0.0, 0.0, 1.0))], label="robot"
        )
        model = builder.finalize(device=device)
        with self.assertRaises(ValueError):
            ControllerJointImpedance(
                model,
                joints=[0],  # the loose joint
                stiffness=_gains(1, 1.0, device),
                damping=_gains(1, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
            )

    def test_single_axis_d6_is_controllable(self):
        """Verify a D6 joint with one axis is controllable, spanning one coordinate and one DOF."""
        device = wp.get_device()
        builder = newton.ModelBuilder()
        link = builder.add_link()
        joint = builder.add_joint_d6(
            parent=-1,
            child=link,
            angular_axes=[newton.ModelBuilder.JointDofConfig(axis=wp.vec3(0.0, 0.0, 1.0))],
        )
        builder.add_articulation([joint], label="robot")
        model = builder.finalize(device=device)
        controller = ControllerJointImpedance(
            model,
            stiffness=_gains(1, 5.0, device),
            damping=_gains(1, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        ins, outs = controller.input(), controller.output()
        ins.joint_q_des.assign(np.array([1.0], dtype=np.float32))
        controller.step(inputs=ins, outputs=outs, dt=0.01)
        np.testing.assert_allclose(outs.joint_f.numpy(), [5.0], atol=1e-5)

    def test_heterogeneous_model(self):
        """Verify model-based controller works with a heterogeneous two-robot fleet."""
        device = wp.get_device()
        model = _build_two_robot_mixed().finalize(device=device)  # robot0: 2 DOFs, robot1: 1 DOF
        ctrl = ControllerJointImpedance(
            model,
            stiffness=_gains(3, 4.0, device),
            damping=_gains(3, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        tau = self._run(
            ctrl, q_sim=[0.0, 0.0, 0.0], qd_sim=[0.0, 0.0, 0.0], q_des=[1.0, 0.0, 2.0], qd_des=[0.0] * 3, device=device
        )
        # robot0 DOF0: 4*1=4, robot0 DOF1: 4*0=0, robot1 DOF0: 4*2=8
        np.testing.assert_allclose(tau, [4.0, 0.0, 8.0], atol=1e-4)

    def test_floating_base_fleet_scatters_to_correct_dofs(self):
        """Verify torques land at model DOF indices on a fleet whose coordinate and DOF spaces differ.

        Robot 0 has a free base (7 coordinates, 6 DOFs), so every controlled
        joint has a different coordinate index than DOF index. Reading positions
        through the coordinate index while writing torques through the DOF index
        is the whole reason the two arrays are separate.
        """
        device = wp.get_device()
        builder, revolute_joints = _build_floating_base_fleet()
        model = builder.finalize(device=device)
        # Coordinate and DOF indices must genuinely differ, or this proves nothing.
        q_idx = model.joint_q_start.numpy()[revolute_joints]
        qd_idx = model.joint_qd_start.numpy()[revolute_joints]
        self.assertFalse(np.array_equal(q_idx, qd_idx))

        ctrl = ControllerJointImpedance(
            model,
            joints=revolute_joints,
            stiffness=_gains(3, 5.0, device),
            damping=_gains(3, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        self.assertEqual(ctrl.total_controlled_dofs, 3)
        np.testing.assert_array_equal(ctrl._controlled_dofs_per_robot.numpy(), [2, 1])

        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q_des = _flat([1.0, 2.0, 3.0], device)
        sim_f = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        outs.joint_f = sim_f[ctrl.qd_start]
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        expected = np.zeros(model.joint_dof_count, dtype=np.float32)
        expected[ctrl.qd_start.numpy()] = [5.0, 10.0, 15.0]
        np.testing.assert_allclose(sim_f.numpy(), expected, atol=1e-4)

    def test_inertia_decoupling_uses_the_controlled_robots_own_block(self):
        """Verify M(q) is read from the controlled robot's own mass-matrix block.

        Only the second robot is controlled, so its packed index (0) differs
        from its model index (1). The two robots' inertias differ by 100x, so
        reading the wrong robot's block — or the wrong rows of the right one —
        misses by a factor no tolerance hides.
        """
        device = wp.get_device()
        inertia = wp.mat33(np.diag([0.1, 0.1, 0.1]).astype(np.float32))
        builder = newton.ModelBuilder()
        # Robot 0, uncontrolled: a heavy 1 m pendulum, Izz + m*r^2 = 100.1.
        heavy = builder.add_link(mass=100.0, com=wp.vec3(1.0, 0.0, 0.0), inertia=inertia, lock_inertia=True)
        builder.add_articulation(
            [builder.add_joint_revolute(parent=-1, child=heavy, axis=wp.vec3(0.0, 0.0, 1.0))], label="uncontrolled"
        )
        # Robot 1, controlled: a light 1 m pendulum, Izz + m*r^2 = 1.1.
        light = builder.add_link(mass=1.0, com=wp.vec3(1.0, 0.0, 0.0), inertia=inertia, lock_inertia=True)
        builder.add_articulation(
            [builder.add_joint_revolute(parent=-1, child=light, axis=wp.vec3(0.0, 0.0, 1.0))], label="controlled"
        )
        model = builder.finalize(device=device)

        ctrl = ControllerJointImpedance(
            model,
            articulations=["controlled"],
            stiffness=_gains(1, 10.0, device),
            damping=_gains(1, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=True,
        )
        # The controlled robot is model robot 1 but packed robot 0.
        np.testing.assert_array_equal(ctrl._model_robot_index.numpy(), [1])

        tau = self._run(ctrl, q_sim=[0.0, 0.0], qd_sim=[0.0, 0.0], q_des=[0.5], qd_des=[0.0], device=device)
        expected = (0.1 + 1.0 * 1.0**2) * 10.0 * 0.5  # (Izz + m*r^2) * Kp * dq
        np.testing.assert_allclose(tau, [expected], atol=1e-4)

    def test_subset_of_articulations(self):
        """Verify a model articulation may be left uncontrolled entirely."""
        device = wp.get_device()
        model = _build_two_robot_mixed().finalize(device=device)
        ctrl = ControllerJointImpedance(
            model,
            articulations=["robot1"],
            stiffness=_gains(1, 2.0, device),
            damping=_gains(1, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        # Only the controlled robot is carried: the uncontrolled one occupies no slot.
        np.testing.assert_array_equal(ctrl._controlled_dofs_per_robot.numpy(), [1])
        self.assertEqual((ctrl.model_robot_count, ctrl.controlled_robot_count), (2, 1))
        tau = self._run(ctrl, q_sim=[0.0] * 3, qd_sim=[0.0] * 3, q_des=[3.0], qd_des=[0.0], device=device)
        np.testing.assert_allclose(tau, [6.0], atol=1e-4)

    def test_full_state_propagates_to_uncontrolled_joint(self):
        """Verify step() copies the whole model state, not just the controlled DOFs.

        The mass matrix, gravity, and Coriolis terms depend on the state of
        the whole articulation, so an uncontrolled joint's coordinates must
        reach the internal model state exactly as supplied — not stay at
        whatever the state was initialized to.
        """
        device = wp.get_device()
        builder, j_ball, j_arm = _build_ball_then_revolute()
        model = builder.finalize(device=device)

        q_start = model.joint_q_start.numpy()
        ctrl = ControllerJointImpedance(
            model,
            joints=[j_arm],
            stiffness=_gains(1, 0.0, device),
            damping=_gains(1, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        ins, outs = ctrl.input(), ctrl.output()
        # Non-identity ball-joint quaternion (x, y, z, w) at coordinate slot
        # q_start[j_ball] — a value the controller never reads through its
        # single-DOF index arrays, only through the full-state copy.
        q_full = np.zeros(model.joint_coord_count, dtype=np.float32)
        q_full[q_start[j_ball] : q_start[j_ball] + 4] = [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
        ins.joint_q = wp.array(q_full, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        np.testing.assert_allclose(ctrl._model_state.joint_q.numpy(), q_full)

    def test_wrong_length_model_state_port_raises(self):
        """Verify inputs.joint_q must be exactly the model's coordinate count."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q = wp.zeros(5, dtype=wp.float32, device=device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=outs, dt=0.01)


if __name__ == "__main__":
    unittest.main()
