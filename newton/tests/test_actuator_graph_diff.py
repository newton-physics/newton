# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Graphability and differentiability tests for all actuator combinations.

Tests verify:
  - Differentiability: gradients flow through actuator.step() via wp.Tape
  - Graphability: multi-step actuator.step() can be captured and replayed
    as a CUDA graph using ScopedCapture / capture_launch

Combinations tested:
  - PD (stateless)
  - PD + MaxForce clamping
  - PD + DCMotor clamping
  - PD + PositionBased clamping
  - PD + Delay
  - PD + Delay + MaxForce
  - PID (stateful controller)
  - PID + Delay

Not committed — stash locally for manual verification.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.actuators import (
    ClampingDCMotor,
    ClampingMaxForce,
    ClampingPositionBased,
    ControllerPD,
    ControllerPID,
)


@wp.kernel
def _sum_kernel(src: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(out, 0, src[i])


NUM_JOINTS = 2
NUM_GRAPH_STEPS = 4


def _build(controller_class, controller_kwargs, clamping=None, delay=None, device=None, requires_grad=False):
    builder = newton.ModelBuilder()
    links = [builder.add_link() for _ in range(NUM_JOINTS)]
    joints = []
    for i, link in enumerate(links):
        parent = -1 if i == 0 else links[i - 1]
        joints.append(builder.add_joint_revolute(parent=parent, child=link, axis=newton.Axis.Z))
    builder.add_articulation(joints)
    dofs = [builder.joint_qd_start[j] for j in joints]
    for dof in dofs:
        builder.add_actuator(controller_class, index=dof, clamping=clamping, delay=delay, **controller_kwargs)
    return builder.finalize(device=device, requires_grad=requires_grad), dofs


def _write_dofs(model, array, dofs, values):
    arr_np = array.numpy()
    for dof, val in zip(dofs, values, strict=True):
        arr_np[dof] = val
    wp.copy(array, wp.array(arr_np, dtype=float, device=model.device))


def _reset_act_states(actuator, act_states):
    """Reset actuator state contents in-place (preserve GPU pointers)."""
    for s in act_states:
        if s is None:
            continue
        if s.delay_state is not None:
            s.delay_state.reset()
        if s.controller_state is not None:
            s.controller_state.reset()


# ---------------------------------------------------------------------------
# Differentiability tests
# ---------------------------------------------------------------------------


class TestDifferentiability(unittest.TestCase):
    """Verify gradients flow through actuator.step() for every combo."""

    def _run_diff_test(self, controller_class, controller_kwargs, clamping=None, delay=None):
        model, dofs = _build(controller_class, controller_kwargs, clamping=clamping, delay=delay, requires_grad=True)
        state = model.state(requires_grad=True)
        control = model.control(requires_grad=True)
        _write_dofs(model, control.joint_target_pos, dofs, [1.0] * NUM_JOINTS)
        loss = wp.zeros(1, dtype=float, requires_grad=True, device=model.device)

        actuator = model.actuators[0]
        is_stateful = actuator.is_stateful()
        state_a = actuator.state() if is_stateful else None
        state_b = actuator.state() if is_stateful else None

        tape = wp.Tape()
        with tape:
            actuator.step(state, control, state_a, state_b, 0.01)
            wp.launch(_sum_kernel, dim=len(control.joint_f), inputs=[control.joint_f], outputs=[loss])

        tape.backward(loss)
        grad = control.joint_target_pos.grad
        self.assertIsNotNone(grad, "gradient array should exist")
        for dof in dofs:
            self.assertNotEqual(grad.numpy()[dof], 0.0, f"gradient at DOF {dof} should be non-zero")

    def test_pd(self):
        self._run_diff_test(ControllerPD, {"kp": 1.0, "kd": 0.5})

    def test_pd_max_force(self):
        self._run_diff_test(ControllerPD, {"kp": 1.0}, clamping=[(ClampingMaxForce, {"max_force": 50.0})])

    def test_pd_dc_motor(self):
        self._run_diff_test(
            ControllerPD,
            {"kp": 1.0},
            clamping=[(ClampingDCMotor, {"saturation_effort": 100.0, "velocity_limit": 10.0, "max_force": 200.0})],
        )

    def test_pd_position_based(self):
        self._run_diff_test(
            ControllerPD,
            {"kp": 1.0},
            clamping=[(ClampingPositionBased, {"lookup_angles": (-1.0, 0.0, 1.0), "lookup_torques": (10.0, 30.0, 50.0)})],
        )

    def test_pd_delay(self):
        self._run_diff_test(ControllerPD, {"kp": 1.0}, delay=2)

    def test_pd_delay_max_force(self):
        self._run_diff_test(
            ControllerPD, {"kp": 1.0}, delay=1, clamping=[(ClampingMaxForce, {"max_force": 50.0})]
        )

    def test_pid(self):
        self._run_diff_test(ControllerPID, {"kp": 1.0, "ki": 0.5, "kd": 0.1})

    def test_pid_delay(self):
        self._run_diff_test(ControllerPID, {"kp": 1.0, "ki": 0.5, "kd": 0.1}, delay=2)

    def test_pid_max_force(self):
        self._run_diff_test(
            ControllerPID, {"kp": 1.0, "ki": 1.0, "kd": 0.0}, clamping=[(ClampingMaxForce, {"max_force": 50.0})]
        )

    def test_pid_delay_dc_motor(self):
        self._run_diff_test(
            ControllerPID,
            {"kp": 1.0, "ki": 1.0, "kd": 0.5},
            delay=1,
            clamping=[(ClampingDCMotor, {"saturation_effort": 100.0, "velocity_limit": 10.0, "max_force": 200.0})],
        )


# ---------------------------------------------------------------------------
# Graphability tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "CUDA graph requires GPU")
class TestGraphability(unittest.TestCase):
    """Verify multi-step actuator.step() can be captured as CUDA graph."""

    def _run_graph_test(self, controller_class, controller_kwargs, clamping=None, delay=None):
        device = wp.get_preferred_device()
        model, dofs = _build(controller_class, controller_kwargs, clamping=clamping, delay=delay, device=device)
        actuator = model.actuators[0]
        is_stateful = actuator.is_stateful()

        # --- Eager reference ---
        state = model.state()
        control = model.control()
        _write_dofs(model, control.joint_target_pos, dofs, [10.0] * NUM_JOINTS)
        act_states = [actuator.state(), actuator.state()] if is_stateful else [None, None]

        for _ in range(NUM_GRAPH_STEPS):
            actuator.step(state, control, act_states[0], act_states[1], 0.01)
            if is_stateful:
                act_states[0], act_states[1] = act_states[1], act_states[0]

        eager_forces = control.joint_f.numpy().copy()

        # --- Graph: allocate all objects once, reuse for capture & replay ---
        state = model.state()
        control = model.control()
        _write_dofs(model, control.joint_target_pos, dofs, [10.0] * NUM_JOINTS)
        act_states = [actuator.state(), actuator.state()] if is_stateful else [None, None]

        # Warm-up (kernels must be compiled before capture)
        for _ in range(NUM_GRAPH_STEPS):
            actuator.step(state, control, act_states[0], act_states[1], 0.01)
            if is_stateful:
                act_states[0], act_states[1] = act_states[1], act_states[0]

        # Reset the SAME arrays in-place for clean capture
        control.joint_f.zero_()
        _write_dofs(model, control.joint_target_pos, dofs, [10.0] * NUM_JOINTS)
        if is_stateful:
            _reset_act_states(actuator, act_states)

        def simulate():
            for _ in range(NUM_GRAPH_STEPS):
                actuator.step(state, control, act_states[0], act_states[1], 0.01)
                if is_stateful:
                    act_states[0], act_states[1] = act_states[1], act_states[0]

        with wp.ScopedCapture(device=device) as capture:
            simulate()

        # Reset in-place again, then replay on the SAME GPU memory
        control.joint_f.zero_()
        _write_dofs(model, control.joint_target_pos, dofs, [10.0] * NUM_JOINTS)
        if is_stateful:
            _reset_act_states(actuator, act_states)

        wp.capture_launch(capture.graph)

        graph_forces = control.joint_f.numpy()

        for dof in dofs:
            self.assertNotEqual(graph_forces[dof], 0.0, f"Graph replay should produce non-zero force at DOF {dof}")

        np.testing.assert_allclose(
            [graph_forces[d] for d in dofs],
            [eager_forces[d] for d in dofs],
            rtol=1e-5,
            err_msg="Graph replay must match eager execution",
        )

    def test_pd(self):
        self._run_graph_test(ControllerPD, {"kp": 1.0, "kd": 0.5})

    def test_pd_max_force(self):
        self._run_graph_test(ControllerPD, {"kp": 100.0}, clamping=[(ClampingMaxForce, {"max_force": 50.0})])

    def test_pd_dc_motor(self):
        self._run_graph_test(
            ControllerPD,
            {"kp": 100.0},
            clamping=[(ClampingDCMotor, {"saturation_effort": 100.0, "velocity_limit": 10.0, "max_force": 200.0})],
        )

    def test_pd_position_based(self):
        self._run_graph_test(
            ControllerPD,
            {"kp": 100.0},
            clamping=[(ClampingPositionBased, {"lookup_angles": (-1.0, 0.0, 1.0), "lookup_torques": (10.0, 30.0, 50.0)})],
        )

    def test_pd_delay(self):
        self._run_graph_test(ControllerPD, {"kp": 1.0}, delay=2)

    def test_pd_delay_max_force(self):
        self._run_graph_test(
            ControllerPD, {"kp": 100.0}, delay=1, clamping=[(ClampingMaxForce, {"max_force": 50.0})]
        )

    def test_pid(self):
        self._run_graph_test(ControllerPID, {"kp": 1.0, "ki": 0.5, "kd": 0.1})

    def test_pid_delay(self):
        self._run_graph_test(ControllerPID, {"kp": 1.0, "ki": 0.5, "kd": 0.1}, delay=2)

    def test_pid_max_force(self):
        self._run_graph_test(
            ControllerPID, {"kp": 100.0, "ki": 1.0, "kd": 0.0}, clamping=[(ClampingMaxForce, {"max_force": 50.0})]
        )

    def test_pid_delay_dc_motor(self):
        self._run_graph_test(
            ControllerPID,
            {"kp": 100.0, "ki": 1.0, "kd": 5.0},
            delay=1,
            clamping=[(ClampingDCMotor, {"saturation_effort": 100.0, "velocity_limit": 10.0, "max_force": 200.0})],
        )


if __name__ == "__main__":
    unittest.main()
