# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Newton actuators."""

import importlib.util
import math
import os
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.utils.import_usd import parse_usd
from newton.actuators import (
    ActuatorParsed,
    ClampingDCMotor,
    ClampingMaxForce,
    ClampingPositionBased,
    ControllerNetLSTM,
    ControllerNetMLP,
    ControllerPD,
    ControllerPID,
    Delay,
    parse_actuator_prim,
)
from newton.selection import ArticulationView

try:
    from pxr import Usd

    HAS_USD = True
except ImportError:
    HAS_USD = False

_HAS_TORCH = importlib.util.find_spec("torch") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_dof_values(model, array, dof_indices, values):
    """Write scalar values into specific DOF positions of a Warp array."""
    arr_np = array.numpy()
    for dof, val in zip(dof_indices, values, strict=True):
        arr_np[dof] = val
    wp.copy(array, wp.array(arr_np, dtype=float, device=model.device))


# ---------------------------------------------------------------------------
# 1. Controllers
# ---------------------------------------------------------------------------


class TestControllerPD(unittest.TestCase):
    """PD controller: f = constant + act + kp*(target_pos - q) + kd*(target_vel - v)."""

    def test_compute(self):
        """Construct controller directly and call compute() with all terms."""
        n = 2
        kp_vals = [100.0, 200.0]
        kd_vals = [10.0, 20.0]
        const_vals = [5.0, -3.0]
        q = [0.3, -0.5]
        qd = [1.0, -2.0]
        tgt_pos = [1.0, 0.5]
        tgt_vel = [0.0, 1.0]
        ff = [3.0, -1.0]

        def _f(vals):
            return wp.array(vals, dtype=wp.float32)

        indices = wp.array(list(range(n)), dtype=wp.uint32)
        ctrl = ControllerPD(kp=_f(kp_vals), kd=_f(kd_vals), constant_force=_f(const_vals))
        forces = wp.zeros(n, dtype=wp.float32)

        ctrl.compute(
            positions=_f(q),
            velocities=_f(qd),
            target_pos=_f(tgt_pos),
            target_vel=_f(tgt_vel),
            feedforward=_f(ff),
            pos_indices=indices,
            vel_indices=indices,
            target_pos_indices=indices,
            target_vel_indices=indices,
            forces=forces,
            state=None,
            dt=0.01,
        )

        result = forces.numpy()
        for i in range(n):
            expected = const_vals[i] + ff[i] + kp_vals[i] * (tgt_pos[i] - q[i]) + kd_vals[i] * (tgt_vel[i] - qd[i])
            self.assertAlmostEqual(result[i], expected, places=4, msg=f"DOF {i}")


class TestControllerPID(unittest.TestCase):
    """PID controller: f = const + act + kp*e + ki*integral + kd*de."""

    def test_compute(self):
        """Construct controller directly and call compute() over multiple steps."""
        kp, ki, kd, const = 50.0, 10.0, 5.0, 2.0
        dt = 0.01
        q, qd = [0.0], [0.0]
        tgt_pos, tgt_vel = [1.0], [0.0]
        pos_error = tgt_pos[0] - q[0]
        vel_error = tgt_vel[0] - qd[0]
        device = wp.get_device()

        def _f(vals):
            return wp.array(vals, dtype=wp.float32, device=device)

        indices = wp.array([0], dtype=wp.uint32, device=device)
        ctrl = ControllerPID(
            kp=_f([kp]),
            ki=_f([ki]),
            kd=_f([kd]),
            integral_max=_f([math.inf]),
            constant_force=_f([const]),
        )
        ctrl.finalize(device, 1)

        state_a = ctrl.state(1, device)
        state_b = ctrl.state(1, device)

        integral = 0.0
        for step_i in range(3):
            forces = wp.zeros(1, dtype=wp.float32, device=device)
            integral += pos_error * dt
            expected = const + kp * pos_error + ki * integral + kd * vel_error

            current, nxt = (state_a, state_b) if step_i % 2 == 0 else (state_b, state_a)
            ctrl.compute(
                positions=_f(q),
                velocities=_f(qd),
                target_pos=_f(tgt_pos),
                target_vel=_f(tgt_vel),
                feedforward=None,
                pos_indices=indices,
                vel_indices=indices,
                target_pos_indices=indices,
                target_vel_indices=indices,
                forces=forces,
                state=current,
                dt=dt,
                device=device,
            )
            ctrl.update_state(current, nxt)

            self.assertAlmostEqual(forces.numpy()[0], expected, places=4, msg=f"step {step_i}")


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestControllerNetMLP(unittest.TestCase):
    """ControllerNetMLP — construct controller, call compute() directly."""

    def setUp(self):
        import torch

        self.torch = torch
        self.device = wp.get_device()
        self._torch_dev = torch.device(f"cuda:{self.device.ordinal}" if self.device.is_cuda else "cpu")

    def _mlp(self, in_dim, hidden=32):
        return self.torch.nn.Sequential(
            self.torch.nn.Linear(in_dim, hidden),
            self.torch.nn.ELU(),
            self.torch.nn.Linear(hidden, 1),
        ).to(self._torch_dev)

    def test_compute(self):
        """Constant-bias network produces known output; history rolls after update_state."""
        net = self.torch.nn.Sequential(self.torch.nn.Linear(2, 1, bias=True)).to(self._torch_dev)
        with self.torch.no_grad():
            net[0].weight.fill_(0.0)
            net[0].bias.fill_(42.0)

        n = 1
        ctrl = ControllerNetMLP(network=net)
        ctrl.finalize(self.device, n)
        state_a = ctrl.state(n, self.device)
        state_b = ctrl.state(n, self.device)

        indices = wp.array([0], dtype=wp.uint32, device=self.device)
        positions = wp.zeros(n, dtype=wp.float32, device=self.device)
        velocities = wp.zeros(n, dtype=wp.float32, device=self.device)
        target_pos = wp.array([1.0], dtype=wp.float32, device=self.device)
        target_vel = wp.zeros(n, dtype=wp.float32, device=self.device)
        forces = wp.zeros(n, dtype=wp.float32, device=self.device)

        ctrl.compute(
            positions,
            velocities,
            target_pos,
            target_vel,
            None,
            indices,
            indices,
            indices,
            indices,
            forces,
            state_a,
            0.01,
            self.device,
        )
        self.assertAlmostEqual(forces.numpy()[0], 42.0, places=3)

        ctrl.update_state(state_a, state_b)
        self.assertAlmostEqual(
            state_b.pos_error_history[0, 0].item(),
            1.0,
            places=4,
            msg="history should contain pos error from current step",
        )

    def test_invalid_input_order_raises(self):
        with self.assertRaises(ValueError):
            ControllerNetMLP(network=self._mlp(2), input_order="invalid")


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestControllerNetLSTM(unittest.TestCase):
    """ControllerNetLSTM — construct controller, call compute() directly."""

    def setUp(self):
        import torch

        self.torch = torch
        self.device = wp.get_device()
        self._torch_dev = torch.device(f"cuda:{self.device.ordinal}" if self.device.is_cuda else "cpu")

    def _lstm(self, hidden=8, layers=1):
        import torch

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(2, hidden, layers, batch_first=True)
                self.dec = torch.nn.Linear(hidden, 1)

            def forward(self, x, hc):
                out, (h, c) = self.lstm(x, hc)
                return self.dec(out[:, -1, :]), (h, c)

        return Net().to(self._torch_dev)

    def test_compute(self):
        """compute() produces non-zero force; hidden state evolves after update_state."""
        n = 1
        hidden, layers = 8, 1
        ctrl = ControllerNetLSTM(network=self._lstm(hidden=hidden, layers=layers))
        ctrl.finalize(self.device, n)

        state_a = ctrl.state(n, self.device)
        state_b = ctrl.state(n, self.device)
        self.assertTrue(self.torch.all(state_a.hidden == 0.0).item())

        indices = wp.array([0], dtype=wp.uint32, device=self.device)
        positions = wp.zeros(n, dtype=wp.float32, device=self.device)
        velocities = wp.array([1.0], dtype=wp.float32, device=self.device)
        target_pos = wp.array([1.0], dtype=wp.float32, device=self.device)
        target_vel = wp.zeros(n, dtype=wp.float32, device=self.device)
        forces = wp.zeros(n, dtype=wp.float32, device=self.device)

        ctrl.compute(
            positions,
            velocities,
            target_pos,
            target_vel,
            None,
            indices,
            indices,
            indices,
            indices,
            forces,
            state_a,
            0.01,
            self.device,
        )
        ctrl.update_state(state_a, state_b)

        self.assertNotAlmostEqual(forces.numpy()[0], 0.0, places=5, msg="LSTM should produce non-zero force")
        self.assertFalse(self.torch.all(state_b.hidden == 0.0).item(), "hidden state should evolve")


# ---------------------------------------------------------------------------
# 2. Delay
# ---------------------------------------------------------------------------


class TestDelay(unittest.TestCase):
    """Delay unit tests — construct Delay directly, call get_delayed_targets/update_state."""

    def test_buffer_shape(self):
        """State buffers have correct shape (buf_depth, N)."""
        n, max_delay = 2, 5
        device = wp.get_device()
        delays = wp.array([max_delay] * n, dtype=wp.int32, device=device)
        delay = Delay(delays, max_delay)
        delay.finalize(device, n)

        ds = delay.state(n, device)
        self.assertEqual(ds.buffer_pos.shape, (max_delay, n))
        self.assertEqual(ds.buffer_vel.shape, (max_delay, n))
        self.assertEqual(ds.buffer_act.shape, (max_delay, n))
        self.assertEqual(ds.write_idx, max_delay - 1)
        np.testing.assert_array_equal(ds.num_pushes.numpy(), [0, 0])

    def test_latency_behavior(self):
        """Delay=N gives exactly N steps of delay; empty buffer falls back to current targets."""
        n, delay_val = 1, 2
        device = wp.get_device()
        delays = wp.array([delay_val], dtype=wp.int32, device=device)
        delay = Delay(delays, delay_val)
        delay.finalize(device, n)

        indices = wp.array([0], dtype=wp.uint32, device=device)
        state_a = delay.state(n, device)
        state_b = delay.state(n, device)

        read_history = []
        for step_i in range(delay_val + 3):
            target_val = float(step_i + 1) * 10.0
            tgt_pos = wp.array([target_val], dtype=wp.float32, device=device)
            tgt_vel = wp.zeros(1, dtype=wp.float32, device=device)

            current, nxt = (state_a, state_b) if step_i % 2 == 0 else (state_b, state_a)
            out_pos, _out_vel, _out_act = delay.get_delayed_targets(tgt_pos, tgt_vel, None, indices, indices, current)
            read_history.append(out_pos.numpy()[0])
            delay.update_state(tgt_pos, tgt_vel, None, indices, indices, current, nxt)

        self.assertAlmostEqual(read_history[0], 10.0, places=4, msg="step 0: empty buffer -> current target")
        self.assertAlmostEqual(read_history[1], 10.0, places=4, msg="step 1: 1 entry, lag clamped -> oldest (10)")
        self.assertAlmostEqual(read_history[2], 10.0, places=4, msg="step 2: full delay=2 -> reads step 0 (10)")
        self.assertAlmostEqual(read_history[3], 20.0, places=4, msg="step 3: full delay=2 -> reads step 1 (20)")
        self.assertAlmostEqual(read_history[4], 30.0, places=4, msg="step 4: full delay=2 -> reads step 2 (30)")


# ---------------------------------------------------------------------------
# 3. Clamping
# ---------------------------------------------------------------------------


class TestClampingMaxForce(unittest.TestCase):
    """ClampingMaxForce: output is clamped to +/-max_force."""

    def test_modify_forces(self):
        """Construct clamping directly and call modify_forces()."""
        max_f = 50.0
        n = 3
        clamp = ClampingMaxForce(max_force=wp.array([max_f] * n, dtype=wp.float32))

        src_vals = [100.0, -80.0, 30.0]
        src = wp.array(src_vals, dtype=wp.float32)
        dst = wp.zeros(n, dtype=wp.float32)
        indices = wp.array(list(range(n)), dtype=wp.uint32)

        clamp.modify_forces(src, dst, wp.zeros(n, dtype=wp.float32), wp.zeros(n, dtype=wp.float32), indices, indices)

        result = dst.numpy()
        for i, s in enumerate(src_vals):
            expected = max(min(s, max_f), -max_f)
            self.assertAlmostEqual(result[i], expected, places=5, msg=f"DOF {i}")


class TestClampingDCMotor(unittest.TestCase):
    """DC motor torque-speed curve: clamp = saturation * (1 - v/v_limit)."""

    def test_modify_forces(self):
        """Construct clamping directly and call modify_forces() at several velocity points."""
        sat, v_lim, max_f = 100.0, 10.0, 200.0
        clamp = ClampingDCMotor(
            saturation_effort=wp.array([sat], dtype=wp.float32),
            velocity_limit=wp.array([v_lim], dtype=wp.float32),
            max_force=wp.array([max_f], dtype=wp.float32),
        )
        indices = wp.array([0], dtype=wp.uint32)
        raw_force = 500.0

        for qd in [0.0, 5.0, 10.0, -5.0]:
            src = wp.array([raw_force], dtype=wp.float32)
            dst = wp.zeros(1, dtype=wp.float32)
            vel = wp.array([qd], dtype=wp.float32)

            clamp.modify_forces(src, dst, wp.zeros(1, dtype=wp.float32), vel, indices, indices)

            tau_max = max(min(sat * (1.0 - qd / v_lim), max_f), 0.0)
            tau_min = max(min(sat * (-1.0 - qd / v_lim), 0.0), -max_f)
            expected = max(min(raw_force, tau_max), tau_min)
            self.assertAlmostEqual(dst.numpy()[0], expected, places=3, msg=f"qd={qd}")


class TestClampingPositionBased(unittest.TestCase):
    """Position-based clamping with angle-dependent lookup table."""

    def test_modify_forces(self):
        """Construct clamping directly and verify interpolated angle-dependent limits."""
        angles = (-1.0, 0.0, 1.0)
        torques = (10.0, 30.0, 50.0)
        device = wp.get_device()
        clamp = ClampingPositionBased(lookup_angles=angles, lookup_torques=torques)
        clamp.finalize(device, 1)

        raw_force = 999.0
        indices = wp.array([0], dtype=wp.uint32, device=device)

        for pos, expected_limit in [(-1.0, 10.0), (0.0, 30.0), (1.0, 50.0), (-0.5, 20.0), (0.5, 40.0)]:
            src = wp.array([raw_force], dtype=wp.float32, device=device)
            dst = wp.zeros(1, dtype=wp.float32, device=device)
            positions = wp.array([pos], dtype=wp.float32, device=device)

            clamp.modify_forces(
                src, dst, positions, wp.zeros(1, dtype=wp.float32, device=device), indices, indices, device=device
            )

            self.assertAlmostEqual(dst.numpy()[0], expected_limit, places=2, msg=f"pos={pos}")


# ---------------------------------------------------------------------------
# 4. Actuator pipeline — full step() integration
# ---------------------------------------------------------------------------


class TestActuatorStep(unittest.TestCase):
    """Integration test: full Actuator.step() with delay + PD + DC-motor clamping."""

    def test_full_pipeline(self):
        """Two-joint template x 3 envs, per-DOF delays (2 / 3), PD + DC motor.

        At each of 5 steps we verify:
            raw   = kp*(delayed_target - q) + kd*(0 - qd)
            τ_max = clamp(sat*(1 - qd/v_lim),  0,  max_f)
            τ_min = clamp(sat*(-1 - qd/v_lim), -max_f, 0)
            force = clamp(raw, τ_min, τ_max)
        """
        kp, kd = 50.0, 5.0
        sat, v_lim = 80.0, 20.0
        delay_a, delay_b = 2, 3
        num_envs = 3
        dt = 0.01

        template = newton.ModelBuilder()
        link_a = template.add_link()
        joint_a = template.add_joint_revolute(parent=-1, child=link_a, axis=newton.Axis.Z)
        link_b = template.add_link()
        joint_b = template.add_joint_revolute(parent=link_a, child=link_b, axis=newton.Axis.Z)
        template.add_articulation([joint_a, joint_b])
        dof_a = template.joint_qd_start[joint_a]
        dof_b = template.joint_qd_start[joint_b]
        dc_args = {"saturation_effort": sat, "velocity_limit": v_lim, "max_force": 1e6}
        template.add_actuator(
            ControllerPD,
            index=dof_a,
            kp=kp,
            kd=kd,
            delay=delay_a,
            clamping=[(ClampingDCMotor, dc_args)],
        )
        template.add_actuator(
            ControllerPD,
            index=dof_b,
            kp=kp,
            kd=kd,
            delay=delay_b,
            clamping=[(ClampingDCMotor, dc_args)],
        )

        builder = newton.ModelBuilder()
        builder.replicate(template, num_envs)
        model = builder.finalize()

        self.assertEqual(len(model.actuators), 1, "all DOFs share controller+clamping type")
        actuator = model.actuators[0]
        n = actuator.num_actuators
        self.assertEqual(n, 2 * num_envs)

        delays_np = actuator.delay.delays.numpy()
        expected_delays = [delay_a, delay_b] * num_envs
        np.testing.assert_array_equal(delays_np, expected_delays)

        state = model.state()
        sa = actuator.state()
        sb = actuator.state()

        qd_val = 2.0
        dofs = actuator.indices.numpy().tolist()
        _write_dof_values(model, state.joint_qd, dofs, [qd_val] * n)

        target_schedule = [10.0, 20.0, 30.0, 40.0, 50.0]
        written_targets: list[float] = []

        def _dc_clamp(raw: float, vel: float) -> float:
            tau_max = max(min(sat * (1.0 - vel / v_lim), 1e6), 0.0)
            tau_min = max(min(sat * (-1.0 - vel / v_lim), 0.0), -1e6)
            return max(min(raw, tau_max), tau_min)

        def _delayed_target(step_i: int, dof_delay: int) -> float:
            pushes = step_i
            if pushes == 0:
                return target_schedule[step_i]
            lag = min(dof_delay - 1, pushes - 1)
            return written_targets[step_i - 1 - lag]

        for step_i in range(5):
            control = model.control()
            tgt = target_schedule[step_i]
            _write_dof_values(model, control.joint_target_pos, dofs, [tgt] * n)
            written_targets.append(tgt)

            current, nxt = (sa, sb) if step_i % 2 == 0 else (sb, sa)
            actuator.step(state, control, current, nxt, dt)

            forces = control.joint_f.numpy()
            for local_i in range(n):
                d = dofs[local_i]
                dof_delay = expected_delays[local_i]
                delayed_tgt = _delayed_target(step_i, dof_delay)
                raw = kp * (delayed_tgt - 0.0) + kd * (0.0 - qd_val)
                expected = _dc_clamp(raw, qd_val)
                self.assertAlmostEqual(
                    forces[d],
                    expected,
                    places=3,
                    msg=f"step={step_i} dof={local_i} delay={dof_delay} "
                    f"delayed_tgt={delayed_tgt} raw={raw} expected={expected}",
                )

        ds = nxt.delay_state
        np.testing.assert_array_equal(
            ds.num_pushes.numpy(),
            [min(5, actuator.delay.buf_depth)] * n,
            err_msg="num_pushes should be clamped to buf_depth",
        )


# ---------------------------------------------------------------------------
# 5. Builder — from USD, programmatic, and free-joint replication
# ---------------------------------------------------------------------------


class TestActuatorBuilder(unittest.TestCase):
    """ModelBuilder actuator construction — grouping, params, state, and index layouts."""

    @unittest.skipUnless(HAS_USD, "pxr not installed")
    def test_from_usd(self):
        """Load actuators from a USD stage and verify params after finalize.

        The asset has two actuators:
          Joint1Actuator: PD (kp=100, kd=10) + MaxForce(50)
          Joint2Actuator: PD (kp=200, kd=20) + Delay(5)
        Different clamping/delay splits them into separate groups.
        """
        test_dir = os.path.dirname(__file__)
        usd_path = os.path.join(test_dir, "assets", "actuator_test.usda")
        if not os.path.exists(usd_path):
            self.skipTest(f"Test USD file not found: {usd_path}")

        builder = newton.ModelBuilder()
        result = parse_usd(builder, usd_path)
        self.assertGreater(result["actuator_count"], 0)
        model = builder.finalize()

        self.assertEqual(len(model.actuators), 2)
        clamped = next(a for a in model.actuators if a.clamping)
        delayed = next(a for a in model.actuators if a.delay is not None)

        self.assertEqual(clamped.num_actuators, 1)
        self.assertAlmostEqual(clamped.controller.kp.numpy()[0], 100.0, places=3)
        self.assertAlmostEqual(clamped.controller.kd.numpy()[0], 10.0, places=3)
        self.assertIsInstance(clamped.clamping[0], ClampingMaxForce)
        self.assertAlmostEqual(clamped.clamping[0].max_force.numpy()[0], 50.0, places=3)

        self.assertEqual(delayed.num_actuators, 1)
        self.assertAlmostEqual(delayed.controller.kp.numpy()[0], 200.0, places=3)
        self.assertAlmostEqual(delayed.controller.kd.numpy()[0], 20.0, places=3)
        np.testing.assert_array_equal(delayed.delay.delays.numpy(), [5])
        self.assertEqual(delayed.delay.buf_depth, 5)

        stage = Usd.Stage.Open(usd_path)
        parsed = parse_actuator_prim(stage.GetPrimAtPath("/World/Robot/Joint1Actuator"))
        self.assertIsNotNone(parsed)
        self.assertIsInstance(parsed, ActuatorParsed)
        self.assertEqual(parsed.controller_class, ControllerPD)

    def test_programmatic(self):
        """Mixed controller types, clamping, and delays via add_actuator.

        3-joint chain: PD, PID with DC motor clamping, PD with delay=4.
        Verifies grouping (3 groups), per-DOF params, and state shapes.
        """
        builder = newton.ModelBuilder()
        links = [builder.add_link() for _ in range(3)]
        joints = []
        for i, link in enumerate(links):
            parent = -1 if i == 0 else links[i - 1]
            joints.append(builder.add_joint_revolute(parent=parent, child=link, axis=newton.Axis.Z))
        builder.add_articulation(joints)
        dofs = [builder.joint_qd_start[j] for j in joints]

        builder.add_actuator(ControllerPD, index=dofs[0], kp=50.0, kd=5.0, constant_force=1.0)
        builder.add_actuator(
            ControllerPID,
            index=dofs[1],
            kp=100.0,
            ki=10.0,
            kd=20.0,
            clamping=[(ClampingDCMotor, {"saturation_effort": 80.0, "velocity_limit": 15.0, "max_force": 200.0})],
        )
        builder.add_actuator(ControllerPD, index=dofs[2], kp=150.0, delay=4)

        model = builder.finalize()
        self.assertEqual(len(model.actuators), 3)

        pd_plain = next(a for a in model.actuators if isinstance(a.controller, ControllerPD) and a.delay is None)
        pid_act = next(a for a in model.actuators if isinstance(a.controller, ControllerPID))
        pd_delay = next(a for a in model.actuators if isinstance(a.controller, ControllerPD) and a.delay is not None)

        self.assertEqual(pd_plain.num_actuators, 1)
        np.testing.assert_array_almost_equal(pd_plain.controller.kp.numpy(), [50.0])
        np.testing.assert_array_almost_equal(pd_plain.controller.kd.numpy(), [5.0])
        np.testing.assert_array_almost_equal(pd_plain.controller.constant_force.numpy(), [1.0])
        self.assertIsNone(pd_plain.state())

        self.assertEqual(pid_act.num_actuators, 1)
        np.testing.assert_array_almost_equal(pid_act.controller.kp.numpy(), [100.0])
        np.testing.assert_array_almost_equal(pid_act.controller.ki.numpy(), [10.0])
        np.testing.assert_array_almost_equal(pid_act.controller.kd.numpy(), [20.0])
        self.assertIsInstance(pid_act.clamping[0], ClampingDCMotor)
        self.assertAlmostEqual(pid_act.clamping[0].saturation_effort.numpy()[0], 80.0, places=3)
        pid_state = pid_act.state()
        self.assertIsNotNone(pid_state.controller_state)
        self.assertEqual(pid_state.controller_state.integral.shape, (1,))
        np.testing.assert_array_equal(pid_state.controller_state.integral.numpy(), [0.0])

        self.assertEqual(pd_delay.num_actuators, 1)
        np.testing.assert_array_almost_equal(pd_delay.controller.kp.numpy(), [150.0])
        np.testing.assert_array_equal(pd_delay.delay.delays.numpy(), [4])
        self.assertEqual(pd_delay.delay.buf_depth, 4)
        ds = pd_delay.state().delay_state
        self.assertEqual(ds.buffer_pos.shape, (4, 1))
        np.testing.assert_array_equal(ds.num_pushes.numpy(), [0])

    def test_free_joint_with_replication(self):
        """Free-joint base + 2 revolute children x 3 envs.

        Verifies:
        - pos_indices != indices when joint_q layout differs from joint_qd
        - Correct per-DOF parameter replication across environments
        - State shapes scale with num_envs
        """
        num_envs = 3

        template = newton.ModelBuilder()
        base = template.add_link()
        j_free = template.add_joint_free(child=base)
        link1 = template.add_link()
        j1 = template.add_joint_revolute(parent=base, child=link1, axis=newton.Axis.Z)
        link2 = template.add_link()
        j2 = template.add_joint_revolute(parent=link1, child=link2, axis=newton.Axis.Y)
        template.add_articulation([j_free, j1, j2])

        dof1 = template.joint_qd_start[j1]
        dof2 = template.joint_qd_start[j2]

        template.add_actuator(
            ControllerPD, index=dof1, kp=100.0, kd=10.0, pos_index=template.joint_q_start[j1], delay=2
        )
        template.add_actuator(
            ControllerPD, index=dof2, kp=200.0, kd=20.0, pos_index=template.joint_q_start[j2], delay=3
        )

        builder = newton.ModelBuilder()
        builder.replicate(template, num_envs)
        model = builder.finalize()

        self.assertEqual(len(model.actuators), 1)
        act = model.actuators[0]
        n = 2 * num_envs
        self.assertEqual(act.num_actuators, n)

        pos_idx = act.pos_indices.numpy()
        vel_idx = act.indices.numpy()
        self.assertFalse(
            np.array_equal(pos_idx, vel_idx),
            "pos_indices should differ from indices for free-joint articulations",
        )

        np.testing.assert_array_almost_equal(act.controller.kp.numpy(), [100.0, 200.0] * num_envs)
        np.testing.assert_array_almost_equal(act.controller.kd.numpy(), [10.0, 20.0] * num_envs)

        np.testing.assert_array_equal(act.delay.delays.numpy(), [2, 3] * num_envs)
        self.assertEqual(act.delay.buf_depth, 3)

        act_state = act.state()
        self.assertEqual(act_state.delay_state.buffer_pos.shape, (3, n))
        np.testing.assert_array_equal(act_state.delay_state.num_pushes.numpy(), [0] * n)


# ---------------------------------------------------------------------------
# 6. Parameter access via ArticulationView
# ---------------------------------------------------------------------------


class TestActuatorSelectionAPI(unittest.TestCase):
    """Tests for actuator parameter access via ArticulationView."""

    def run_test_actuator_selection(self, use_mask: bool, use_multiple_artics_per_view: bool):
        mjcf = """<?xml version="1.0" ?>
<mujoco model="myart">
    <worldbody>
    <body name="root" pos="0 0 0">
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      </body>
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      </body>
      <body name="link3" pos="-0.0 -0.9 0">
        <joint name="joint3" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

        num_joints_per_articulation = 3
        num_articulations_per_world = 2
        num_worlds = 3
        num_actuators = num_joints_per_articulation * num_articulations_per_world * num_worlds

        single_articulation_builder = newton.ModelBuilder()
        single_articulation_builder.add_mjcf(mjcf)

        joint_names = [
            "myart/worldbody/root/link1/joint1",
            "myart/worldbody/root/link2/joint2",
            "myart/worldbody/root/link3/joint3",
        ]
        for i, jname in enumerate(joint_names):
            j_idx = single_articulation_builder.joint_label.index(jname)
            dof = single_articulation_builder.joint_qd_start[j_idx]
            single_articulation_builder.add_actuator(ControllerPD, index=dof, kp=100.0 * (i + 1))

        single_world_builder = newton.ModelBuilder()
        for _i in range(num_articulations_per_world):
            single_world_builder.add_builder(single_articulation_builder)

        single_world_builder.articulation_label[1] = "art1"
        if use_multiple_artics_per_view:
            single_world_builder.articulation_label[0] = "art1"
        else:
            single_world_builder.articulation_label[0] = "art0"

        builder = newton.ModelBuilder()
        for _i in range(num_worlds):
            builder.add_world(single_world_builder)

        model = builder.finalize()

        joints_to_include = ["joint3"]
        joint_view = ArticulationView(model, "art1", include_joints=joints_to_include)

        actuator = model.actuators[0]

        kp_values = joint_view.get_actuator_parameter(actuator, actuator.controller, "kp").numpy().copy()

        if use_multiple_artics_per_view:
            self.assertEqual(kp_values.shape, (num_worlds, 2))
            np.testing.assert_array_almost_equal(kp_values, [[300.0, 300.0]] * num_worlds)
        else:
            self.assertEqual(kp_values.shape, (num_worlds, 1))
            np.testing.assert_array_almost_equal(kp_values, [[300.0]] * num_worlds)

        val = 1000.0
        for world_idx in range(kp_values.shape[0]):
            for dof_idx in range(kp_values.shape[1]):
                kp_values[world_idx, dof_idx] = val
                val += 100.0

        mask = None
        if use_mask:
            mask = wp.array([False, True, False], dtype=bool, device=model.device)

        wp_kp = wp.array(kp_values, dtype=float, device=model.device)
        joint_view.set_actuator_parameter(actuator, actuator.controller, "kp", wp_kp, mask=mask)

        expected_kp = []
        if use_mask:
            if use_multiple_artics_per_view:
                expected_kp = [
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    1200.0,
                    100.0,
                    200.0,
                    1300.0,
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    300.0,
                ]
            else:
                expected_kp = [
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    1100.0,
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    300.0,
                ]
        else:
            if use_multiple_artics_per_view:
                expected_kp = [
                    100.0,
                    200.0,
                    1000.0,
                    100.0,
                    200.0,
                    1100.0,
                    100.0,
                    200.0,
                    1200.0,
                    100.0,
                    200.0,
                    1300.0,
                    100.0,
                    200.0,
                    1400.0,
                    100.0,
                    200.0,
                    1500.0,
                ]
            else:
                expected_kp = [
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    1000.0,
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    1100.0,
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    1200.0,
                ]

        measured_kp = actuator.controller.kp.numpy()
        for i in range(num_actuators):
            self.assertAlmostEqual(
                expected_kp[i],
                measured_kp[i],
                places=4,
                msg=f"Expected kp[{i}]={expected_kp[i]}, got {measured_kp[i]}",
            )

    def test_actuator_selection_one_per_view_no_mask(self):
        self.run_test_actuator_selection(use_mask=False, use_multiple_artics_per_view=False)

    def test_actuator_selection_two_per_view_no_mask(self):
        self.run_test_actuator_selection(use_mask=False, use_multiple_artics_per_view=True)

    def test_actuator_selection_one_per_view_with_mask(self):
        self.run_test_actuator_selection(use_mask=True, use_multiple_artics_per_view=False)

    def test_actuator_selection_two_per_view_with_mask(self):
        self.run_test_actuator_selection(use_mask=True, use_multiple_artics_per_view=True)


if __name__ == "__main__":
    unittest.main()
