# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Newton actuators — parsing, building, stepping, and parameter access.

Covers the full actuator lifecycle:
  - Argument resolution and validation for each component
  - USD prim parsing (mock prims, no real USD stage needed)
  - ModelBuilder accumulation, grouping, and replication
  - State shapes and initialization
  - Force computation: PD, PID, clamping, delay, DC motor, position-based
  - Dual output (applied vs pre-clamp forces)
  - Parameter access via ArticulationView
  - Full USD stage parsing (skip-guarded)
  - Neural-network controllers (torch-dependent, skip-guarded)
"""

import importlib.util
import os
import tempfile
import unittest
from typing import ClassVar

import numpy as np
import warp as wp

import newton
from newton._src.utils.import_usd import parse_usd
from newton.actuators import (
    Actuator,
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


def _build_chain(num_joints, controller_class, controller_kwargs, clamping=None, delay=None):
    """Build a revolute chain with one actuator per joint, return (model, dof_indices)."""
    builder = newton.ModelBuilder()
    links = [builder.add_link() for _ in range(num_joints)]
    joints = []
    for i, link in enumerate(links):
        parent = -1 if i == 0 else links[i - 1]
        joints.append(builder.add_joint_revolute(parent=parent, child=link, axis=newton.Axis.Z))
    builder.add_articulation(joints)
    dofs = [builder.joint_qd_start[j] for j in joints]
    for dof in dofs:
        builder.add_actuator(
            controller_class,
            index=dof,
            clamping=clamping,
            delay=delay,
            **controller_kwargs,
        )
    return builder.finalize(), dofs


def _build_simple_model(num_joints, device="cpu"):
    """Build a minimal revolute chain without actuators, return (model, dof_indices)."""
    builder = newton.ModelBuilder()
    links = [builder.add_link() for _ in range(num_joints)]
    joints = []
    for i, link in enumerate(links):
        parent = -1 if i == 0 else links[i - 1]
        joints.append(builder.add_joint_revolute(parent=parent, child=link, axis=newton.Axis.Z))
    builder.add_articulation(joints)
    return builder.finalize(device=device), [builder.joint_qd_start[j] for j in joints]


def _write_dof_values(model, array, dof_indices, values):
    """Write scalar values into specific DOF positions of a Warp array."""
    arr_np = array.numpy()
    for dof, val in zip(dof_indices, values, strict=True):
        arr_np[dof] = val
    wp.copy(array, wp.array(arr_np, dtype=float, device=model.device))


# ---------------------------------------------------------------------------
# Mock USD prims
# ---------------------------------------------------------------------------


class _MockTokenListOp:
    def __init__(self, items):
        self._items = list(items)

    def GetAddedOrExplicitItems(self):
        return self._items


class _MockAttr:
    def __init__(self, value):
        self._value = value

    def HasAuthoredValue(self):
        return self._value is not None

    def Get(self):
        return self._value


class _MockRel:
    def __init__(self, targets):
        self._targets = targets

    def GetTargets(self):
        return self._targets


class _MockPrim:
    """Minimal stand-in for a USD prim."""

    def __init__(self, type_name, attrs=None, rels=None, schemas=None):
        self._type = type_name
        self._attrs = {k: _MockAttr(v) for k, v in (attrs or {}).items()}
        self._rels = {k: _MockRel(v) for k, v in (rels or {}).items()}
        self._schemas = schemas or []

    def GetTypeName(self):
        return self._type

    def GetAttribute(self, name):
        return self._attrs.get(name)

    def GetRelationship(self, name):
        return self._rels.get(name)

    def GetAppliedSchemas(self):
        return self._schemas

    def GetMetadata(self, key):
        if key == "apiSchemas":
            return _MockTokenListOp(self._schemas) if self._schemas else None
        return None


# ---------------------------------------------------------------------------
# 1. Argument resolution — verify defaults and validation
# ---------------------------------------------------------------------------


class TestResolveArguments(unittest.TestCase):
    """Each component's resolve_arguments should fill defaults and reject bad input."""

    def test_pd_defaults(self):
        r = ControllerPD.resolve_arguments({"kp": 50.0})
        self.assertEqual(r["kp"], 50.0)
        self.assertEqual(r["kd"], 0.0)
        self.assertEqual(r["constant_force"], 0.0)

    def test_delay_requires_delay_arg(self):
        with self.assertRaises(ValueError):
            Delay.resolve_arguments({})

    def test_dc_motor_requires_velocity_limit(self):
        with self.assertRaises(ValueError):
            ClampingDCMotor.resolve_arguments({})

    def test_dc_motor_requires_saturation_effort(self):
        with self.assertRaises(ValueError):
            ClampingDCMotor.resolve_arguments({"velocity_limit": 10.0})

    def test_dc_motor_accepts_valid_args(self):
        r = ClampingDCMotor.resolve_arguments({"velocity_limit": 10.0, "saturation_effort": 100.0})
        self.assertEqual(r["velocity_limit"], 10.0)
        self.assertEqual(r["saturation_effort"], 100.0)

    def test_position_based_requires_lookup(self):
        with self.assertRaises(ValueError):
            ClampingPositionBased.resolve_arguments({})


# ---------------------------------------------------------------------------
# 2. USD prim parsing (mock prims)
# ---------------------------------------------------------------------------


class TestActuatorParser(unittest.TestCase):
    """parse_actuator_prim resolves schema names and extracts kwargs correctly."""

    TARGET_REL: ClassVar[dict[str, list[str]]] = {"newton:actuator:targets": ["/World/Robot/Joint1"]}

    def _prim(self, attrs=None, rels=None, schemas=None, type_name="NewtonActuator"):
        return _MockPrim(type_name, attrs=attrs, rels=rels or self.TARGET_REL, schemas=schemas)

    def test_pd_controller(self):
        r = parse_actuator_prim(
            self._prim(
                attrs={"newton:actuator:kp": 100.0, "newton:actuator:kd": 10.0},
                schemas=["NewtonControllerPDAPI"],
            )
        )
        self.assertEqual(r.controller_class, ControllerPD)
        self.assertEqual(r.controller_kwargs["kp"], 100.0)
        self.assertEqual(r.controller_kwargs["kd"], 10.0)
        self.assertEqual(r.component_specs, [])

    def test_pid_controller(self):
        r = parse_actuator_prim(
            self._prim(
                attrs={"newton:actuator:kp": 100.0, "newton:actuator:ki": 5.0},
                schemas=["NewtonControllerPIDAPI"],
            )
        )
        self.assertEqual(r.controller_class, ControllerPID)
        self.assertEqual(r.controller_kwargs["ki"], 5.0)

    def test_pd_with_delay(self):
        r = parse_actuator_prim(
            self._prim(
                attrs={"newton:actuator:kp": 50.0, "newton:actuator:delay": 5},
                schemas=["NewtonControllerPDAPI", "NewtonDelayAPI"],
            )
        )
        self.assertEqual(len(r.component_specs), 1)
        cls, kwargs = r.component_specs[0]
        self.assertIs(cls, Delay)
        self.assertEqual(kwargs["delay"], 5)

    def test_pd_with_dc_motor(self):
        r = parse_actuator_prim(
            self._prim(
                attrs={
                    "newton:actuator:kp": 100.0,
                    "newton:actuator:velocityLimit": 10.0,
                    "newton:actuator:saturationEffort": 150.0,
                    "newton:actuator:maxForce": 200.0,
                },
                schemas=["NewtonControllerPDAPI", "NewtonClampingDCMotorAPI"],
            )
        )
        self.assertEqual(r.controller_class, ControllerPD)
        self.assertEqual(len(r.component_specs), 1)
        self.assertIs(r.component_specs[0][0], ClampingDCMotor)

    def test_dc_motor_zero_velocity_limit_raises(self):
        with self.assertRaises(ValueError):
            parse_actuator_prim(
                self._prim(
                    attrs={"newton:actuator:kp": 100.0, "newton:actuator:velocityLimit": 0.0},
                    schemas=["NewtonControllerPDAPI", "NewtonClampingDCMotorAPI"],
                )
            )

    def test_non_actuator_type_returns_none(self):
        p = _MockPrim("Mesh")
        self.assertIsNone(parse_actuator_prim(p))

    def test_pd_with_position_based_clamping(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# angle  torque\n-1.0  10.0\n0.0  30.0\n1.0  50.0\n")
            table_path = f.name

        try:
            r = parse_actuator_prim(
                self._prim(
                    attrs={"newton:actuator:kp": 100.0, "newton:actuator:lookupTablePath": table_path},
                    schemas=["NewtonControllerPDAPI", "NewtonClampingPositionBasedAPI"],
                )
            )
            self.assertEqual(r.controller_class, ControllerPD)
            self.assertEqual(len(r.component_specs), 1)
            cls, kwargs = r.component_specs[0]
            self.assertIs(cls, ClampingPositionBased)
            self.assertEqual(kwargs["lookup_table_path"], table_path)
        finally:
            os.unlink(table_path)

    def test_position_based_missing_path_raises(self):
        r = parse_actuator_prim(
            self._prim(
                attrs={"newton:actuator:kp": 100.0},
                schemas=["NewtonControllerPDAPI", "NewtonClampingPositionBasedAPI"],
            )
        )
        cls, kwargs = r.component_specs[0]
        self.assertIs(cls, ClampingPositionBased)
        with self.assertRaises(ValueError):
            cls.resolve_arguments(kwargs)

    def test_missing_target_returns_none(self):
        p = _MockPrim("NewtonActuator", attrs={"newton:actuator:kp": 100.0})
        self.assertIsNone(parse_actuator_prim(p))


# ---------------------------------------------------------------------------
# 3. Builder — accumulation, grouping, replication
# ---------------------------------------------------------------------------


class TestActuatorBuilder(unittest.TestCase):
    """Tests for ModelBuilder.add_actuator — grouping, replication, and scalar params."""

    def test_accumulation_and_parameters(self):
        builder = newton.ModelBuilder()

        bodies = [builder.add_body() for _ in range(3)]
        joints = []
        for i, body in enumerate(bodies):
            parent = -1 if i == 0 else bodies[i - 1]
            joints.append(builder.add_joint_revolute(parent=parent, child=body, axis=newton.Axis.Z))
        builder.add_articulation(joints)

        dofs = [builder.joint_qd_start[j] for j in joints]

        builder.add_actuator(ControllerPD, index=dofs[0], kp=50.0, constant_force=1.0)
        builder.add_actuator(ControllerPD, index=dofs[1], kp=100.0, kd=10.0)
        builder.add_actuator(
            ControllerPD,
            index=dofs[2],
            clamping=[(ClampingMaxForce, {"max_force": 50.0})],
            kp=150.0,
        )

        model = builder.finalize()

        self.assertEqual(len(model.actuators), 2)

        plain_act = next(a for a in model.actuators if not a.clamping)
        clamped_act = next(a for a in model.actuators if a.clamping)

        self.assertEqual(plain_act.num_actuators, 2)
        self.assertEqual(clamped_act.num_actuators, 1)

        np.testing.assert_array_equal(plain_act.indices.numpy(), [dofs[0], dofs[1]])
        np.testing.assert_array_equal(clamped_act.indices.numpy(), [dofs[2]])
        np.testing.assert_array_almost_equal(plain_act.controller.kp.numpy(), [50.0, 100.0])
        np.testing.assert_array_almost_equal(plain_act.controller.kd.numpy(), [0.0, 10.0])
        np.testing.assert_array_almost_equal(plain_act.controller.constant_force.numpy(), [1.0, 0.0])
        self.assertAlmostEqual(clamped_act.clamping[0].max_force.numpy()[0], 50.0)

    def test_mixed_types_with_replication(self):
        template = newton.ModelBuilder()

        body0 = template.add_body()
        body1 = template.add_body()
        body2 = template.add_body()

        joint0 = template.add_joint_revolute(parent=-1, child=body0, axis=newton.Axis.Z)
        joint1 = template.add_joint_revolute(parent=body0, child=body1, axis=newton.Axis.Y)
        joint2 = template.add_joint_revolute(parent=body1, child=body2, axis=newton.Axis.X)
        template.add_articulation([joint0, joint1, joint2])

        dof0 = template.joint_qd_start[joint0]
        dof1 = template.joint_qd_start[joint1]
        dof2 = template.joint_qd_start[joint2]

        template.add_actuator(ControllerPD, index=dof0, kp=100.0, kd=10.0)
        template.add_actuator(ControllerPID, index=dof1, kp=200.0, ki=5.0, kd=20.0)
        template.add_actuator(ControllerPD, index=dof2, kp=300.0)

        num_worlds = 3
        builder = newton.ModelBuilder()
        builder.replicate(template, num_worlds)

        model = builder.finalize()

        self.assertEqual(model.world_count, num_worlds)
        self.assertEqual(len(model.actuators), 2)

        pd_act = next(a for a in model.actuators if isinstance(a.controller, ControllerPD))
        pid_act = next(a for a in model.actuators if isinstance(a.controller, ControllerPID))

        self.assertEqual(pd_act.num_actuators, 2 * num_worlds)
        self.assertEqual(pid_act.num_actuators, num_worlds)

        np.testing.assert_array_almost_equal(pd_act.controller.kp.numpy(), [100.0, 300.0] * num_worlds)
        np.testing.assert_array_almost_equal(pid_act.controller.ki.numpy(), [5.0] * num_worlds)

        pd_indices = pd_act.indices.numpy()
        dofs_per_world = model.joint_dof_count // num_worlds

        for w in range(1, num_worlds):
            self.assertEqual(pd_indices[w * 2] - pd_indices[(w - 1) * 2], dofs_per_world)
            self.assertEqual(pd_indices[w * 2 + 1] - pd_indices[(w - 1) * 2 + 1], dofs_per_world)

    def test_delay_grouping(self):
        builder = newton.ModelBuilder()

        bodies = [builder.add_body() for _ in range(6)]
        joints = []
        for i, body in enumerate(bodies):
            parent = -1 if i == 0 else bodies[i - 1]
            joints.append(builder.add_joint_revolute(parent=parent, child=body, axis=newton.Axis.Z))
        builder.add_articulation(joints)

        dofs = [builder.joint_qd_start[j] for j in joints]

        builder.add_actuator(ControllerPD, index=dofs[0], kp=100.0)
        builder.add_actuator(ControllerPD, index=dofs[1], kp=150.0)
        builder.add_actuator(ControllerPD, index=dofs[2], delay=3, kp=200.0)
        builder.add_actuator(ControllerPD, index=dofs[3], delay=3, kp=250.0)
        builder.add_actuator(ControllerPD, index=dofs[4], delay=7, kp=300.0)
        builder.add_actuator(ControllerPD, index=dofs[5], delay=7, kp=350.0)

        model = builder.finalize()

        self.assertEqual(len(model.actuators), 3)

        plain_act = next(a for a in model.actuators if a.delay is None)
        delay3 = next(a for a in model.actuators if a.delay is not None and a.delay.delay == 3)
        delay7 = next(a for a in model.actuators if a.delay is not None and a.delay.delay == 7)

        self.assertEqual(plain_act.num_actuators, 2)
        self.assertEqual(delay3.num_actuators, 2)
        self.assertEqual(delay7.num_actuators, 2)

        np.testing.assert_array_almost_equal(delay3.controller.kp.numpy(), [200.0, 250.0])


# ---------------------------------------------------------------------------
# 4. State shapes and initialization
# ---------------------------------------------------------------------------


class TestActuatorState(unittest.TestCase):
    """State objects must have correct shapes and initial values."""

    def test_delay_buffer_shape(self):
        delay_steps, n = 5, 2
        buf_depth = delay_steps + 1
        model, _dofs = _build_chain(n, ControllerPD, {"kp": 1.0}, delay=delay_steps)
        actuator = next(a for a in model.actuators if a.delay is not None)
        ds = actuator.state().delay_state
        self.assertEqual(ds.buffer_pos.shape, (buf_depth, n))
        self.assertEqual(ds.buffer_vel.shape, (buf_depth, n))
        self.assertEqual(ds.buffer_act.shape, (buf_depth, n))
        self.assertEqual(ds.write_idx, buf_depth - 1)
        np.testing.assert_array_equal(ds.num_pushes.numpy(), [0, 0])
        self.assertEqual(len(actuator.delay.delays), n)

    def test_pid_integral_initialised_to_zero(self):
        n = 3
        model, _dofs = _build_chain(n, ControllerPID, {"kp": 0.0, "ki": 1.0, "kd": 0.0})
        actuator = model.actuators[0]
        s = actuator.state()
        np.testing.assert_array_equal(s.controller_state.integral.numpy(), np.zeros(n))

    def test_stateless_actuator_returns_none(self):
        model, _dofs = _build_chain(1, ControllerPD, {"kp": 1.0})
        actuator = model.actuators[0]
        self.assertIsNone(actuator.state())
        self.assertFalse(actuator.is_stateful())


# ---------------------------------------------------------------------------
# 5. Force computation — step() behavior
# ---------------------------------------------------------------------------


class TestActuatorStep(unittest.TestCase):
    """Actuator.step() with real Model/State/Control objects."""

    # -- PD controller -------------------------------------------------------

    def test_pd_position_error(self):
        """force = kp * (target_pos - q) when kd=0."""
        model, dofs = _build_chain(3, ControllerPD, {"kp": 100.0})
        state = model.state()
        control = model.control()
        _write_dof_values(model, control.joint_target_pos, dofs, [1.0, 2.0, 3.0])

        model.actuators[0].step(state, control)

        forces = control.joint_f.numpy()
        np.testing.assert_allclose([forces[d] for d in dofs], [100.0, 200.0, 300.0], rtol=1e-5)

    def test_pd_velocity_error(self):
        """force = kd * (target_vel - qd) when kp=0."""
        model, dofs = _build_chain(2, ControllerPD, {"kp": 0.0, "kd": 10.0})
        state = model.state()
        control = model.control()
        _write_dof_values(model, control.joint_target_vel, dofs, [5.0, -3.0])

        model.actuators[0].step(state, control)

        forces = control.joint_f.numpy()
        np.testing.assert_allclose([forces[d] for d in dofs], [50.0, -30.0], rtol=1e-5)

    def test_pd_feedforward(self):
        """Feedforward joint_act is added to output force."""
        model, dofs = _build_chain(2, ControllerPD, {"kp": 0.0})
        state = model.state()
        control = model.control()
        _write_dof_values(model, control.joint_act, dofs, [7.0, -3.0])

        model.actuators[0].step(state, control)

        forces = control.joint_f.numpy()
        np.testing.assert_allclose([forces[d] for d in dofs], [7.0, -3.0], rtol=1e-5)

    def test_pd_constant_force(self):
        """constant_force offset is included in output."""
        model, dofs = _build_chain(1, ControllerPD, {"kp": 0.0, "constant_force": 42.0})
        state = model.state()
        control = model.control()

        model.actuators[0].step(state, control)

        self.assertAlmostEqual(control.joint_f.numpy()[dofs[0]], 42.0, places=5)

    # -- PD + ClampingMaxForce -----------------------------------------------

    def test_max_force_clamp(self):
        """Force is clamped to ±max_force."""
        model, dofs = _build_chain(
            1,
            ControllerPD,
            {"kp": 100.0},
            clamping=[(ClampingMaxForce, {"max_force": 50.0})],
        )
        state = model.state()
        control = model.control()
        _write_dof_values(model, control.joint_target_pos, dofs, [1.0])

        model.actuators[0].step(state, control)

        self.assertAlmostEqual(control.joint_f.numpy()[dofs[0]], 50.0, places=5)

    # -- Dual output (applied vs pre-clamp) ----------------------------------

    def test_dual_output_applied_and_computed(self):
        """control_computed_output_attr writes the pre-clamp force separately."""
        model, dofs = _build_chain(
            1,
            ControllerPD,
            {"kp": 1000.0},
            clamping=[(ClampingMaxForce, {"max_force": 50.0})],
        )
        actuator = model.actuators[0]
        actuator.control_computed_output_attr = "joint_f_computed"

        state = model.state()
        control = model.control()
        control.joint_f_computed = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
        _write_dof_values(model, control.joint_target_pos, dofs, [1.0])

        actuator.step(state, control)

        self.assertAlmostEqual(control.joint_f.numpy()[dofs[0]], 50.0, places=3)
        self.assertAlmostEqual(control.joint_f_computed.numpy()[dofs[0]], 1000.0, places=3)

    def test_dual_output_computed_untouched_when_disabled(self):
        """Pre-clamp array is not written when control_computed_output_attr is None."""
        model, dofs = _build_chain(
            1,
            ControllerPD,
            {"kp": 1000.0},
            clamping=[(ClampingMaxForce, {"max_force": 50.0})],
        )
        state = model.state()
        control = model.control()
        control.joint_f_computed = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
        _write_dof_values(model, control.joint_target_pos, dofs, [1.0])

        model.actuators[0].step(state, control)

        self.assertAlmostEqual(control.joint_f.numpy()[dofs[0]], 50.0, places=3)
        self.assertAlmostEqual(control.joint_f_computed.numpy()[dofs[0]], 0.0, places=3)

    # -- PID controller ------------------------------------------------------

    def test_pid_integral_accumulation(self):
        """Integral term accumulates over multiple steps."""
        model, dofs = _build_chain(1, ControllerPID, {"kp": 0.0, "ki": 10.0, "kd": 0.0})
        state = model.state()
        control = model.control()
        _write_dof_values(model, control.joint_target_pos, dofs, [1.0])

        actuator = model.actuators[0]
        state_a = actuator.state()
        state_b = actuator.state()
        dt = 0.01

        forces_over_time = []
        for step_i in range(3):
            control.joint_f.zero_()
            current, nxt = (state_a, state_b) if step_i % 2 == 0 else (state_b, state_a)
            actuator.step(state, control, current, nxt, dt)
            forces_over_time.append(control.joint_f.numpy()[dofs[0]])

        np.testing.assert_allclose(forces_over_time, [0.1, 0.2, 0.3], rtol=1e-4)

    # -- Delay ---------------------------------------------------------------

    def test_delay_behavior(self):
        """Write-then-read: forces produced from step 0, lag clamped to history."""
        delay_steps = 2
        model, dofs = _build_chain(1, ControllerPD, {"kp": 1.0}, delay=delay_steps)
        state = model.state()

        actuator = model.actuators[0]
        state_a = actuator.state()
        state_b = actuator.state()
        dt = 0.01

        force_history = []
        for step_i in range(delay_steps + 2):
            control = model.control()
            target_val = float(step_i + 1) * 10.0
            _write_dof_values(model, control.joint_target_pos, dofs, [target_val])

            current, nxt = (state_a, state_b) if step_i % 2 == 0 else (state_b, state_a)
            actuator.step(state, control, current, nxt, dt)
            force_history.append(control.joint_f.numpy()[dofs[0]])

        # Step 0: lag clamped to 0 → reads just-pushed data (target=10) → force=10
        self.assertAlmostEqual(force_history[0], 10.0, places=4)
        # Step 1: lag clamped to 1 → reads step 0 data (target=10) → force=10
        self.assertAlmostEqual(force_history[1], 10.0, places=4)
        # Step 2: full delay=2 → reads step 0 data (target=10)
        self.assertAlmostEqual(force_history[2], 10.0, places=4)
        # Step 3: full delay=2 → reads step 1 data (target=20)
        self.assertAlmostEqual(force_history[3], 20.0, places=4)

    # -- DC motor clamping (torque-speed curve) ------------------------------

    def _step_dc_motor(self, qd):
        """Build a DC-motor-clamped actuator, step once at given velocity, return force."""
        model, dofs = _build_chain(
            1,
            ControllerPD,
            {"kp": 1000.0},
            clamping=[
                (
                    ClampingDCMotor,
                    {
                        "saturation_effort": 100.0,
                        "velocity_limit": 10.0,
                        "max_force": 200.0,
                    },
                )
            ],
        )
        state = model.state()
        _write_dof_values(model, state.joint_qd, dofs, [qd])
        control = model.control()
        _write_dof_values(model, control.joint_target_pos, dofs, [1.0])

        model.actuators[0].step(state, control)
        return control.joint_f.numpy()[dofs[0]]

    def test_dc_motor_zero_velocity_full_torque(self):
        self.assertAlmostEqual(self._step_dc_motor(0.0), 100.0, places=3)

    def test_dc_motor_half_velocity_halves_limit(self):
        self.assertAlmostEqual(self._step_dc_motor(5.0), 50.0, places=3)

    def test_dc_motor_at_velocity_limit_zero_torque(self):
        self.assertAlmostEqual(self._step_dc_motor(10.0), 0.0, places=3)

    def test_dc_motor_opposing_velocity_increases_limit(self):
        self.assertAlmostEqual(self._step_dc_motor(-5.0), 150.0, places=3)

    # -- Position-based clamping (angle-dependent) ---------------------------

    def _step_position_based(self, q):
        """Build a position-based-clamped actuator with delay=1, step until active, return force."""
        model, dofs = _build_chain(
            1,
            ControllerPD,
            {"kp": 1000.0},
            delay=1,
            clamping=[
                (
                    ClampingPositionBased,
                    {
                        "lookup_angles": (-1.0, 0.0, 1.0),
                        "lookup_torques": (10.0, 30.0, 50.0),
                    },
                )
            ],
        )
        state = model.state()
        _write_dof_values(model, state.joint_q, dofs, [q])

        actuator = model.actuators[0]
        sa, sb = actuator.state(), actuator.state()
        force = 0.0
        for step_i in range(2):
            control = model.control()
            _write_dof_values(model, control.joint_target_pos, dofs, [q + 10.0])
            current, nxt = (sa, sb) if step_i % 2 == 0 else (sb, sa)
            actuator.step(state, control, current, nxt, dt=0.01)
            force = control.joint_f.numpy()[dofs[0]]
        return force

    def test_position_based_table_endpoints(self):
        self.assertAlmostEqual(self._step_position_based(-1.0), 10.0, places=2)
        self.assertAlmostEqual(self._step_position_based(1.0), 50.0, places=2)

    def test_position_based_midpoint_interpolation(self):
        self.assertAlmostEqual(self._step_position_based(0.0), 30.0, places=2)
        self.assertAlmostEqual(self._step_position_based(-0.5), 20.0, places=2)
        self.assertAlmostEqual(self._step_position_based(0.5), 40.0, places=2)

    def test_position_based_different_tables_split(self):
        """DOFs with different lookup tables are placed in separate actuator groups."""
        table_a = {"lookup_angles": (-1.0, 0.0, 1.0), "lookup_torques": (10.0, 30.0, 50.0)}
        table_b = {"lookup_angles": (-1.0, 1.0), "lookup_torques": (100.0, 200.0)}

        builder = newton.ModelBuilder()
        links = [builder.add_link() for _ in range(3)]
        joints = []
        for i, link in enumerate(links):
            parent = -1 if i == 0 else links[i - 1]
            joints.append(builder.add_joint_revolute(parent=parent, child=link, axis=newton.Axis.Z))
        builder.add_articulation(joints)
        dofs = [builder.joint_qd_start[j] for j in joints]

        builder.add_actuator(ControllerPD, index=dofs[0], kp=1000.0, clamping=[(ClampingPositionBased, table_a)])
        builder.add_actuator(ControllerPD, index=dofs[1], kp=1000.0, clamping=[(ClampingPositionBased, table_a)])
        builder.add_actuator(ControllerPD, index=dofs[2], kp=1000.0, clamping=[(ClampingPositionBased, table_b)])
        model = builder.finalize()

        self.assertEqual(len(model.actuators), 2, "Same table should group, different table should split")


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


# ---------------------------------------------------------------------------
# 7. USD stage parsing (real USD, skip-guarded)
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_USD, "pxr not installed")
class TestActuatorUSDParsing(unittest.TestCase):
    """Tests for parsing actuators from real USD files."""

    def test_usd_parsing(self):
        test_dir = os.path.dirname(__file__)
        usd_path = os.path.join(test_dir, "assets", "actuator_test.usda")

        if not os.path.exists(usd_path):
            self.skipTest(f"Test USD file not found: {usd_path}")

        builder = newton.ModelBuilder()
        result = parse_usd(builder, usd_path)
        self.assertGreater(result["actuator_count"], 0)
        model = builder.finalize()
        self.assertGreater(len(model.actuators), 0)

        stage = Usd.Stage.Open(usd_path)
        actuator_prim = stage.GetPrimAtPath("/World/Robot/Joint1Actuator")
        parsed = parse_actuator_prim(actuator_prim)

        self.assertIsNotNone(parsed)
        self.assertIsInstance(parsed, ActuatorParsed)
        self.assertEqual(parsed.controller_class, ControllerPD)
        self.assertEqual(parsed.controller_kwargs.get("kp"), 100.0)
        self.assertEqual(parsed.controller_kwargs.get("kd"), 10.0)


# ---------------------------------------------------------------------------
# 8. Neural-network controllers (torch, skip-guarded)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestControllerNetMLP(unittest.TestCase):
    """ControllerNetMLP — creation, state shapes, step with clamping."""

    def setUp(self):
        wp.init()
        import torch

        self.torch = torch
        self.dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _mlp(self, in_dim, hidden=32):
        return self.torch.nn.Sequential(
            self.torch.nn.Linear(in_dim, hidden),
            self.torch.nn.ELU(),
            self.torch.nn.Linear(hidden, 1),
        )

    def test_is_stateful_and_not_graphable(self):
        _model, dofs = _build_simple_model(1, device=self.dev)
        act = Actuator(
            wp.array(dofs, dtype=wp.uint32, device=self.dev),
            controller=ControllerNetMLP(network=self._mlp(2)),
        )
        self.assertTrue(act.is_stateful())
        self.assertFalse(act.is_graphable())

    def test_state_history_shape(self):
        n = 3
        _model, dofs = _build_simple_model(n, device=self.dev)
        act = Actuator(
            wp.array(dofs, dtype=wp.uint32, device=self.dev),
            controller=ControllerNetMLP(network=self._mlp(6), input_idx=[0, 1, 2]),
        )
        cs = act.state().controller_state
        self.assertEqual(cs.pos_error_history.shape, (n, n))
        self.assertEqual(cs.vel_history.shape, (n, n))

    def test_constant_bias_clamped(self):
        """Network with large constant output is clamped to max_force."""
        net = self.torch.nn.Sequential(self.torch.nn.Linear(2, 1, bias=True))
        with self.torch.no_grad():
            net[0].weight.fill_(0.0)
            net[0].bias.fill_(999.0)

        model, dofs = _build_simple_model(1, device=self.dev)
        act = Actuator(
            wp.array(dofs, dtype=wp.uint32, device=self.dev),
            controller=ControllerNetMLP(network=net),
            clamping=[ClampingMaxForce(max_force=wp.array([10.0], dtype=wp.float32, device=self.dev))],
        )
        sa, sb = act.state(), act.state()

        state = model.state()
        control = model.control()
        act.step(state, control, sa, sb)

        control = model.control()
        act.step(state, control, sa, sb)

        self.assertAlmostEqual(control.joint_f.numpy()[dofs[0]], 10.0, places=3)

    def test_invalid_input_order_raises(self):
        with self.assertRaises(ValueError):
            ControllerNetMLP(network=self._mlp(2), input_order="invalid")


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestControllerNetLSTM(unittest.TestCase):
    """ControllerNetLSTM — creation, state shapes, state evolution."""

    def setUp(self):
        wp.init()
        import torch

        self.torch = torch
        self.dev = "cuda:0" if torch.cuda.is_available() else "cpu"

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

        return Net()

    def test_is_stateful_and_not_graphable(self):
        _model, dofs = _build_simple_model(1, device=self.dev)
        act = Actuator(
            wp.array(dofs, dtype=wp.uint32, device=self.dev),
            controller=ControllerNetLSTM(network=self._lstm()),
        )
        self.assertTrue(act.is_stateful())
        self.assertFalse(act.is_graphable())

    def test_state_shape(self):
        hidden, layers, n = 16, 2, 3
        _model, dofs = _build_simple_model(n, device=self.dev)
        act = Actuator(
            wp.array(dofs, dtype=wp.uint32, device=self.dev),
            controller=ControllerNetLSTM(network=self._lstm(hidden=hidden, layers=layers)),
        )
        cs = act.state().controller_state
        self.assertEqual(cs.hidden.shape, (layers, n, hidden))
        self.assertEqual(cs.cell.shape, (layers, n, hidden))

    def test_state_evolves_after_step(self):
        model, dofs = _build_simple_model(1, device=self.dev)
        act = Actuator(
            wp.array(dofs, dtype=wp.uint32, device=self.dev),
            controller=ControllerNetLSTM(network=self._lstm()),
        )
        sa, sb = act.state(), act.state()
        self.assertTrue(self.torch.all(sa.controller_state.hidden == 0.0).item())

        state = model.state()
        _write_dof_values(model, state.joint_qd, dofs, [1.0])
        control = model.control()
        _write_dof_values(model, control.joint_target_pos, dofs, [1.0])

        act.step(state, control, sa, sb)

        self.assertFalse(self.torch.all(sb.controller_state.hidden == 0.0).item())


if __name__ == "__main__":
    unittest.main()
