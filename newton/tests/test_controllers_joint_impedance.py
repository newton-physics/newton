# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for ControllerJointImpedance and ControllerJointImpedanceModelFree."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.controllers import ControllerJointImpedance, ControllerJointImpedanceModelFree

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iota(n, device):
    """Return a wp.array[uint32] of [0, 1, …, n-1]."""
    return wp.array(np.arange(n, dtype=np.uint32), device=device)


def _dofs_arr(dofs_list, device):
    """Return a wp.array[int32] from a list of per-robot DOF counts."""
    return wp.array(np.array(dofs_list, dtype=np.int32), device=device)


def _gains(num_robots, max_dofs, value, device):
    """Return a (num_robots, max_dofs) float32 gain array filled with value."""
    return wp.full((num_robots, max_dofs), value, dtype=wp.float32, device=device)


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


def _make_mf(
    *,
    num_robots,
    dofs_list,
    kp,
    kd,
    device,
    use_gravity=False,
    use_coriolis=False,
    use_inertia=False,
    has_qdd=False,
):
    """Construct a ControllerJointImpedanceModelFree with identity indices."""
    max_dofs = max(dofs_list)
    total_dofs = sum(dofs_list)
    return ControllerJointImpedanceModelFree(
        num_robots=num_robots,
        dofs_per_robot=_dofs_arr(dofs_list, device),
        max_dofs=max_dofs,
        default_dof_indices=_iota(total_dofs, device),
        stiffness=_gains(num_robots, max_dofs, kp, device),
        damping=_gains(num_robots, max_dofs, kd, device),
        use_gravity_compensation=use_gravity,
        use_coriolis_compensation=use_coriolis,
        use_inertia_decoupling=use_inertia,
        has_qdd_feedforward=has_qdd,
        device=device,
    )


def _run_mf(ctrl, *, q, qd, q_des, qd_des, device, **extras):
    """Run one compute step on a ModelFree controller and return the torque array."""
    ins = ctrl.input()
    ins.joint_q = _flat(q, device)
    ins.joint_qd = _flat(qd, device)
    ins.joint_q_des = _flat(q_des, device)
    ins.joint_qd_des = _flat(qd_des, device)
    for k, v in extras.items():
        setattr(ins, k, v)
    outs = ctrl.output()
    ctrl.compute(ins, outs, None, None, 0.01)
    return outs.joint_f.numpy()


# ---------------------------------------------------------------------------
# ControllerJointImpedanceModelFree — homogeneous
# ---------------------------------------------------------------------------


class TestControllerJointImpedanceModelFree(unittest.TestCase):
    def test_zero_error_gives_zero_torque(self):
        """Verify that zero position and velocity error produces zero torque."""
        device = wp.get_device()
        ctrl = _make_mf(num_robots=1, dofs_list=[3], kp=10.0, kd=1.0, device=device)
        tau = _run_mf(
            ctrl, q=[0.1, 0.2, 0.3], qd=[0.0, 0.0, 0.0], q_des=[0.1, 0.2, 0.3], qd_des=[0.0, 0.0, 0.0], device=device
        )
        np.testing.assert_allclose(tau, np.zeros(3, dtype=np.float32), atol=1e-5)

    def test_position_error_produces_stiffness_torque(self):
        """Verify τ = Kp * (q_des - q) when Kd=0."""
        device = wp.get_device()
        ctrl = _make_mf(num_robots=1, dofs_list=[3], kp=5.0, kd=0.0, device=device)
        tau = _run_mf(
            ctrl, q=[0.0, 0.0, 0.0], qd=[0.0, 0.0, 0.0], q_des=[1.0, 0.0, 0.0], qd_des=[0.0, 0.0, 0.0], device=device
        )
        np.testing.assert_allclose(tau, [5.0, 0.0, 0.0], atol=1e-5)

    def test_velocity_error_produces_damping_torque(self):
        """Verify τ = Kd * (qd_des - qd) when Kp=0."""
        device = wp.get_device()
        ctrl = _make_mf(num_robots=1, dofs_list=[3], kp=0.0, kd=2.0, device=device)
        tau = _run_mf(
            ctrl, q=[0.0, 0.0, 0.0], qd=[0.0, 0.0, 0.0], q_des=[0.0, 0.0, 0.0], qd_des=[0.0, 1.0, 0.0], device=device
        )
        np.testing.assert_allclose(tau, [0.0, 2.0, 0.0], atol=1e-5)

    def test_multiple_robots_independent(self):
        """Verify that torques for each robot depend only on that robot's error."""
        device = wp.get_device()
        num_robots, num_dofs = 3, 2
        ctrl = _make_mf(num_robots=num_robots, dofs_list=[num_dofs] * num_robots, kp=1.0, kd=0.0, device=device)
        q_des = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        q = np.zeros((num_robots, num_dofs), dtype=np.float32)
        tau = _run_mf(ctrl, q=q, qd=q * 0, q_des=q_des, qd_des=q * 0, device=device)
        np.testing.assert_allclose(tau, q_des.flatten(), atol=1e-5)

    def test_inertia_decoupling_scales_by_mass_matrix(self):
        """Verify τ = M @ (Kp * Δq) when use_inertia_decoupling=True."""
        device = wp.get_device()
        ctrl = _make_mf(num_robots=1, dofs_list=[2], kp=1.0, kd=0.0, device=device, use_inertia=True)
        M = wp.array(np.eye(2, dtype=np.float32).reshape(1, 2, 2) * 2.0, dtype=wp.float32, device=device)
        tau = _run_mf(
            ctrl, q=[0.0, 0.0], qd=[0.0, 0.0], q_des=[1.0, 1.0], qd_des=[0.0, 0.0], device=device, mass_matrix=M
        )
        np.testing.assert_allclose(tau, [2.0, 2.0], atol=1e-5)

    def test_gravity_compensation_adds_to_tau(self):
        """Verify gravity_force is added to τ when use_gravity_compensation=True."""
        device = wp.get_device()
        ctrl = _make_mf(num_robots=1, dofs_list=[2], kp=0.0, kd=0.0, device=device, use_gravity=True)
        grav = wp.array([3.0, 4.0], dtype=wp.float32, device=device)
        tau = _run_mf(
            ctrl, q=[0.0, 0.0], qd=[0.0, 0.0], q_des=[0.0, 0.0], qd_des=[0.0, 0.0], device=device, gravity_force=grav
        )
        np.testing.assert_allclose(tau, [3.0, 4.0], atol=1e-5)

    def test_coriolis_compensation_adds_to_tau(self):
        """Verify coriolis_force is added to τ when use_coriolis_compensation=True."""
        device = wp.get_device()
        ctrl = _make_mf(num_robots=1, dofs_list=[2], kp=0.0, kd=0.0, device=device, use_coriolis=True)
        cor = wp.array([1.0, -1.0], dtype=wp.float32, device=device)
        tau = _run_mf(
            ctrl, q=[0.0, 0.0], qd=[0.0, 0.0], q_des=[0.0, 0.0], qd_des=[0.0, 0.0], device=device, coriolis_force=cor
        )
        np.testing.assert_allclose(tau, [1.0, -1.0], atol=1e-5)

    def test_qdd_feedforward_adds_before_inertia(self):
        """Verify qdd feedforward is included inside M @ (PD + qdd) when use_inertia=True."""
        device = wp.get_device()
        ctrl = _make_mf(num_robots=1, dofs_list=[2], kp=0.0, kd=0.0, device=device, use_inertia=True, has_qdd=True)
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
        """Verify stiffness supplied via a live input attribute is applied correctly."""
        device = wp.get_device()
        ctrl = ControllerJointImpedanceModelFree(
            num_robots=1,
            dofs_per_robot=_dofs_arr([2], device),
            max_dofs=2,
            default_dof_indices=_iota(2, device),
            stiffness="kp",
            damping=wp.zeros((1, 2), dtype=wp.float32, device=device),
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
        ins.kp = wp.array([[3.0, 3.0]], dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.compute(ins, outs, None, None, 0.01)
        np.testing.assert_allclose(outs.joint_f.numpy(), [6.0, 0.0], atol=1e-5)

    def test_is_not_stateful(self):
        """Verify the controller reports is_stateful() == False."""
        device = wp.get_device()
        ctrl = _make_mf(num_robots=1, dofs_list=[2], kp=1.0, kd=0.0, device=device)
        self.assertFalse(ctrl.is_stateful())

    def test_is_graphable(self):
        """Verify the controller reports is_graphable() == True."""
        device = wp.get_device()
        ctrl = _make_mf(num_robots=1, dofs_list=[2], kp=1.0, kd=0.0, device=device)
        self.assertTrue(ctrl.is_graphable())

    def test_input_struct_has_required_fields(self):
        """Verify input() returns a namespace with all declared port fields present."""
        device = wp.get_device()
        ctrl = _make_mf(
            num_robots=1,
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

    def test_output_struct_has_joint_f(self):
        """Verify output() returns a flat array of size sum(dofs_per_robot)."""
        device = wp.get_device()
        ctrl = _make_mf(num_robots=1, dofs_list=[2], kp=1.0, kd=0.0, device=device)
        outs = ctrl.output()
        self.assertTrue(hasattr(outs, "joint_f"))
        self.assertEqual(outs.joint_f.shape, (2,))

    def test_custom_output_attr_name(self):
        """Verify joint_f_attr renames the output field on the output struct."""
        device = wp.get_device()
        ctrl = ControllerJointImpedanceModelFree(
            num_robots=1,
            dofs_per_robot=_dofs_arr([2], device),
            max_dofs=2,
            default_dof_indices=_iota(2, device),
            stiffness=wp.ones((1, 2), dtype=wp.float32, device=device),
            damping=wp.zeros((1, 2), dtype=wp.float32, device=device),
            joint_f_attr="tau_cmd",
            device=device,
        )
        outs = ctrl.output()
        self.assertTrue(hasattr(outs, "tau_cmd"))
        self.assertFalse(hasattr(outs, "joint_f"))

    def test_partial_sim_indices(self):
        """Verify gather/scatter correctly selects a controller-DOF subset from a larger sim array."""
        device = wp.get_device()
        indices = wp.array([1, 3], dtype=wp.uint32, device=device)
        ctrl = ControllerJointImpedanceModelFree(
            num_robots=1,
            dofs_per_robot=_dofs_arr([2], device),
            max_dofs=2,
            default_dof_indices=indices,
            stiffness=wp.ones((1, 2), dtype=wp.float32, device=device),
            damping=wp.zeros((1, 2), dtype=wp.float32, device=device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        ins = ctrl.input()
        ins.joint_q = wp.zeros(4, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(4, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array([0.0, 5.0, 0.0, 3.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(4, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.compute(ins, outs, None, None, 0.01)
        result = outs.joint_f.numpy()
        self.assertAlmostEqual(result[0], 0.0, places=5)
        self.assertAlmostEqual(result[1], 5.0, places=5)
        self.assertAlmostEqual(result[2], 0.0, places=5)
        self.assertAlmostEqual(result[3], 3.0, places=5)


# ---------------------------------------------------------------------------
# ControllerJointImpedanceModelFree — heterogeneous
# ---------------------------------------------------------------------------


class TestControllerJointImpedanceModelFreeHeterogeneous(unittest.TestCase):
    def test_heterogeneous_pd_torques(self):
        """Verify PD torques are correct for each robot with different DOF counts."""
        device = wp.get_device()
        # Robot 0: 2 DOFs, Kp=5; Robot 1: 1 DOF, Kp=5
        # Errors: robot0=[1,0], robot1=[2]  →  tau: robot0=[5,0], robot1=[10]
        dofs_list = [2, 1]
        max_dofs = 2
        ctrl = ControllerJointImpedanceModelFree(
            num_robots=2,
            dofs_per_robot=_dofs_arr(dofs_list, device),
            max_dofs=max_dofs,
            default_dof_indices=_iota(3, device),  # 2 + 1 = 3 total DOFs
            stiffness=_gains(2, max_dofs, 5.0, device),
            damping=_gains(2, max_dofs, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        ins = ctrl.input()
        ins.joint_q = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array([1.0, 0.0, 2.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(3, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.compute(ins, outs, None, None, 0.01)
        tau = outs.joint_f.numpy()
        np.testing.assert_allclose(tau, [5.0, 0.0, 10.0], atol=1e-5)

    def test_heterogeneous_independence(self):
        """Verify robot 0's torques are zero when only robot 1 has a position error."""
        device = wp.get_device()
        dofs_list = [2, 1]
        max_dofs = 2
        ctrl = ControllerJointImpedanceModelFree(
            num_robots=2,
            dofs_per_robot=_dofs_arr(dofs_list, device),
            max_dofs=max_dofs,
            default_dof_indices=_iota(3, device),
            stiffness=_gains(2, max_dofs, 1.0, device),
            damping=_gains(2, max_dofs, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        ins = ctrl.input()
        ins.joint_q = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array([0.0, 0.0, 3.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(3, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.compute(ins, outs, None, None, 0.01)
        tau = outs.joint_f.numpy()
        # Only robot 1's slot (index 2) should be nonzero
        np.testing.assert_allclose(tau[:2], [0.0, 0.0], atol=1e-5)
        self.assertAlmostEqual(tau[2], 3.0, places=5)

    def test_heterogeneous_padding_not_scattered(self):
        """Verify that padded slots never write to the output array."""
        device = wp.get_device()
        # Robot 0 has 3 DOFs, robot 1 has 1 DOF (max_dofs=3, padded slots: robot1 dofs 1,2)
        dofs_list = [3, 1]
        max_dofs = 3
        ctrl = ControllerJointImpedanceModelFree(
            num_robots=2,
            dofs_per_robot=_dofs_arr(dofs_list, device),
            max_dofs=max_dofs,
            default_dof_indices=_iota(4, device),
            stiffness=_gains(2, max_dofs, 1.0, device),
            damping=_gains(2, max_dofs, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        ins = ctrl.input()
        # Large desired values everywhere; only 4 real outputs should be written
        ins.joint_q = wp.zeros(4, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(4, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.full(4, 99.0, dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(4, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.compute(ins, outs, None, None, 0.01)
        tau = outs.joint_f.numpy()
        # Exactly 4 real DOFs: all should equal 99
        self.assertEqual(tau.shape[0], 4)
        np.testing.assert_allclose(tau, [99.0, 99.0, 99.0, 99.0], atol=1e-5)

    def test_heterogeneous_inertia_decoupling(self):
        """Verify M @ acc is computed per-robot with heterogeneous DOF counts."""
        device = wp.get_device()
        # Robot 0: 2 DOFs, M=2*I; Robot 1: 1 DOF, M=[[3]]
        # Errors: robot0=[1,1], robot1=[1] →  acc=[1,1] and [1]
        # tau: robot0 = 2*I @ [1,1] = [2,2], robot1 = [[3]] @ [1] = [3]
        dofs_list = [2, 1]
        max_dofs = 2
        ctrl = ControllerJointImpedanceModelFree(
            num_robots=2,
            dofs_per_robot=_dofs_arr(dofs_list, device),
            max_dofs=max_dofs,
            default_dof_indices=_iota(3, device),
            stiffness=_gains(2, max_dofs, 1.0, device),
            damping=_gains(2, max_dofs, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=True,
            device=device,
        )
        # Mass matrices padded to (2, 2, 2); robot1's second row/col is unused
        M_np = np.zeros((2, 2, 2), dtype=np.float32)
        M_np[0] = np.eye(2) * 2.0
        M_np[1, 0, 0] = 3.0
        M = wp.array(M_np, dtype=wp.float32, device=device)
        ins = ctrl.input()
        ins.joint_q = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.ones(3, dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(3, dtype=wp.float32, device=device)
        ins.mass_matrix = M
        outs = ctrl.output()
        ctrl.compute(ins, outs, None, None, 0.01)
        tau = outs.joint_f.numpy()
        np.testing.assert_allclose(tau, [2.0, 2.0, 3.0], atol=1e-5)


# ---------------------------------------------------------------------------
# ControllerJointImpedance (model-based)
# ---------------------------------------------------------------------------


class TestControllerJointImpedance(unittest.TestCase):
    def _make_ctrl(self, device, *, kp=10.0, kd=1.0, use_inertia=False):
        """Build a ControllerJointImpedance for a single prismatic robot."""
        builder = _build_single_prismatic()
        return ControllerJointImpedance(
            model_builder=builder,
            default_dof_indices=_iota(1, device),
            stiffness=_gains(1, 1, kp, device),
            damping=_gains(1, 1, kd, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=use_inertia,
            device=device,
        )

    def _run(self, ctrl, *, q_sim, qd_sim, q_des_sim, qd_des_sim, device):
        """Run one compute step and return the torque array."""
        ins = ctrl.input()
        ins.joint_q = wp.array(np.array(q_sim, dtype=np.float32), dtype=wp.float32, device=device)
        ins.joint_qd = wp.array(np.array(qd_sim, dtype=np.float32), dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array(np.array(q_des_sim, dtype=np.float32), dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.array(np.array(qd_des_sim, dtype=np.float32), dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.compute(ins, outs, None, None, 0.01)
        return outs.joint_f.numpy()

    def test_zero_error_gives_zero_torque(self):
        """Verify zero position and velocity error produces zero torque."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device)
        tau = self._run(ctrl, q_sim=[0.5], qd_sim=[0.0], q_des_sim=[0.5], qd_des_sim=[0.0], device=device)
        np.testing.assert_allclose(tau, [0.0], atol=1e-4)

    def test_position_error_produces_stiffness_torque(self):
        """Verify τ = Kp * (q_des - q) for a simple prismatic robot."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device, kp=5.0, kd=0.0)
        tau = self._run(ctrl, q_sim=[0.0], qd_sim=[0.0], q_des_sim=[1.0], qd_des_sim=[0.0], device=device)
        np.testing.assert_allclose(tau, [5.0], atol=1e-4)

    def test_damping_term(self):
        """Verify τ = Kd * (qd_des - qd) when Kp=0."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device, kp=0.0, kd=3.0)
        tau = self._run(ctrl, q_sim=[0.0], qd_sim=[0.0], q_des_sim=[0.0], qd_des_sim=[2.0], device=device)
        np.testing.assert_allclose(tau, [6.0], atol=1e-4)

    def test_is_stateful_false(self):
        """Verify is_stateful() returns False."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device)
        self.assertFalse(ctrl.is_stateful())

    def test_is_graphable_true(self):
        """Verify is_graphable() returns True."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device)
        self.assertTrue(ctrl.is_graphable())

    def test_non_scalar_joint_type_raises(self):
        """Verify that a model containing a non-1-DOF joint type raises ValueError."""
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
        with self.assertRaises(ValueError):
            ControllerJointImpedance(
                model_builder=builder,
                default_dof_indices=_iota(3, device),  # ball joint has 3 DOFs
                stiffness=_gains(1, 3, 1.0, device),
                damping=_gains(1, 3, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                device=device,
            )

    def test_wrong_index_length_raises(self):
        """Verify that a mismatched default_dof_indices length raises ValueError."""
        device = wp.get_device()
        builder = _build_single_prismatic()
        with self.assertRaises(ValueError):
            ControllerJointImpedance(
                model_builder=builder,
                default_dof_indices=_iota(5, device),
                stiffness=_gains(1, 1, 1.0, device),
                damping=_gains(1, 1, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                device=device,
            )

    def test_input_output_struct_shapes(self):
        """Verify input/output struct arrays have the expected flat shapes."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device)
        ins = ctrl.input()
        outs = ctrl.output()
        self.assertEqual(ins.joint_q.shape, (1,))
        self.assertEqual(outs.joint_f.shape, (1,))

    def test_heterogeneous_model(self):
        """Verify model-based controller works with a heterogeneous two-robot fleet."""
        device = wp.get_device()
        builder = _build_two_robot_mixed()  # robot0: 2 DOFs, robot1: 1 DOF
        max_dofs = 2
        ctrl = ControllerJointImpedance(
            model_builder=builder,
            default_dof_indices=_iota(3, device),  # 2 + 1 total DOFs
            stiffness=_gains(2, max_dofs, 4.0, device),
            damping=_gains(2, max_dofs, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        ins = ctrl.input()
        ins.joint_q = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array([1.0, 0.0, 2.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(3, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.compute(ins, outs, None, None, 0.01)
        tau = outs.joint_f.numpy()
        # robot0 DOF0: 4*1=4, robot0 DOF1: 4*0=0, robot1 DOF0: 4*2=8
        np.testing.assert_allclose(tau, [4.0, 0.0, 8.0], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
