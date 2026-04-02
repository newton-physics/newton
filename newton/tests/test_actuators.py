# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for actuator classes."""

from __future__ import annotations

import importlib.util
import unittest
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton.actuators import (
    Actuator,
    ActuatorDCMotor,
    ActuatorDelayedPD,
    ActuatorPD,
    ActuatorPID,
    ActuatorRemotizedPD,
)

_HAS_TORCH = importlib.util.find_spec("torch") is not None


@dataclass
class MockSimState:
    """Mock simulation state for testing."""

    joint_q: wp.array[float]
    joint_qd: wp.array[float]
    tendon_length: wp.array[float] | None = None
    tendon_vel: wp.array[float] | None = None


@dataclass
class MockSimControl:
    """Mock simulation control for testing."""

    joint_target_pos: wp.array[float]
    joint_target_vel: wp.array[float]
    joint_act: wp.array[float]
    joint_f: wp.array[float]
    tendon_target_length: wp.array[float] | None = None
    tendon_target_vel: wp.array[float] | None = None
    tendon_force: wp.array[float] | None = None


class TestActuatorPDUnit(unittest.TestCase):
    """Tests for ActuatorPD."""

    def setUp(self):
        wp.init()

    def test_pd_actuator_creation(self):
        """Test that ActuatorPD can be created with valid parameters."""
        indices = wp.array([0, 1, 2], dtype=wp.uint32)
        actuator = ActuatorPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0, 100.0], dtype=wp.float32),
            kd=wp.array([10.0, 10.0, 10.0], dtype=wp.float32),
            max_force=wp.array([50.0, 50.0, 50.0], dtype=wp.float32),
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertIsNone(actuator.state())
        self.assertFalse(actuator.is_stateful())
        self.assertTrue(actuator.is_graphable())

    def test_pd_actuator_step(self):
        """Test that ActuatorPD.step() computes correct forces."""
        num_dofs = 3
        indices = wp.array([0, 1, 2], dtype=wp.uint32)

        actuator = ActuatorPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0, 100.0], dtype=wp.float32),
            kd=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
            max_force=wp.array([1000.0, 1000.0, 1000.0], dtype=wp.float32),
        )

        sim_state = MockSimState(
            joint_q=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
            joint_qd=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0, 2.0, 3.0], dtype=wp.float32),
            joint_target_vel=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
            joint_act=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
            joint_f=wp.zeros(num_dofs, dtype=wp.float32),
        )

        actuator.step(sim_state, sim_control, None, None)
        forces = sim_control.joint_f.numpy()
        np.testing.assert_allclose(forces, [100.0, 200.0, 300.0], rtol=1e-5)

    def test_pd_actuator_resolve_arguments(self):
        """Test that resolve_arguments fills defaults correctly."""
        resolved = ActuatorPD.resolve_arguments({"kp": 50.0})
        self.assertEqual(resolved["kp"], 50.0)
        self.assertEqual(resolved["kd"], 0.0)
        self.assertEqual(resolved["constant_force"], 0.0)


class TestActuatorDelayedPDUnit(unittest.TestCase):
    """Tests for ActuatorDelayedPD."""

    def setUp(self):
        wp.init()

    def test_delayed_pd_creation(self):
        """Test that ActuatorDelayedPD can be created with valid parameters."""
        indices = wp.array([0, 1], dtype=wp.uint32)
        actuator = ActuatorDelayedPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0], dtype=wp.float32),
            kd=wp.array([10.0, 10.0], dtype=wp.float32),
            delay=5,
            max_force=wp.array([50.0, 50.0], dtype=wp.float32),
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertTrue(actuator.is_stateful())
        self.assertTrue(actuator.is_graphable())

    def test_delayed_pd_state(self):
        """Test that ActuatorDelayedPD.state() returns properly initialized state."""
        delay = 5
        num_dofs = 2
        indices = wp.array([0, 1], dtype=wp.uint32)

        actuator = ActuatorDelayedPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0], dtype=wp.float32),
            kd=wp.array([10.0, 10.0], dtype=wp.float32),
            delay=delay,
            max_force=wp.array([50.0, 50.0], dtype=wp.float32),
        )

        state = actuator.state()
        self.assertIsInstance(state, ActuatorDelayedPD.State)
        self.assertEqual(state.write_idx, delay - 1)
        self.assertFalse(state.is_filled)
        self.assertEqual(state.buffer_pos.shape, (delay, num_dofs))
        self.assertEqual(state.buffer_vel.shape, (delay, num_dofs))
        self.assertEqual(state.buffer_act.shape, (delay, num_dofs))

    def test_delayed_pd_resolve_arguments_requires_delay(self):
        """Test that resolve_arguments raises error if delay not provided."""
        with self.assertRaises(ValueError):
            ActuatorDelayedPD.resolve_arguments({"kp": 50.0})

    def test_delayed_pd_delay_behavior(self):
        """Test that ActuatorDelayedPD correctly delays targets by N steps."""
        delay = 3
        num_dofs = 1
        indices = wp.array([0], dtype=wp.uint32)

        actuator = ActuatorDelayedPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([1.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            delay=delay,
            max_force=wp.array([1000.0], dtype=wp.float32),
        )

        stateA = actuator.state()
        stateB = actuator.state()

        target_history = []
        force_history = []

        for step in range(delay + 3):
            target_value = float(step + 1) * 10.0
            target_history.append(target_value)

            sim_state = MockSimState(
                joint_q=wp.array([0.0], dtype=wp.float32),
                joint_qd=wp.array([0.0], dtype=wp.float32),
            )
            sim_control = MockSimControl(
                joint_target_pos=wp.array([target_value], dtype=wp.float32),
                joint_target_vel=wp.array([0.0], dtype=wp.float32),
                joint_act=wp.array([0.0], dtype=wp.float32),
                joint_f=wp.zeros(num_dofs, dtype=wp.float32),
            )

            if step % 2 == 0:
                current, next_state = stateA, stateB
            else:
                current, next_state = stateB, stateA

            actuator.step(sim_state, sim_control, current, next_state, dt=0.01)
            force_history.append(sim_control.joint_f.numpy()[0])

        for i in range(delay):
            self.assertEqual(force_history[i], 0.0, f"Step {i}: expected 0 force during fill phase")

        self.assertAlmostEqual(force_history[3], target_history[0], places=5)
        self.assertAlmostEqual(force_history[4], target_history[1], places=5)
        self.assertAlmostEqual(force_history[5], target_history[2], places=5)


class TestActuatorPIDUnit(unittest.TestCase):
    """Tests for ActuatorPID."""

    def setUp(self):
        wp.init()

    def test_pid_actuator_creation(self):
        """Test that ActuatorPID can be created with valid parameters."""
        indices = wp.array([0, 1], dtype=wp.uint32)
        actuator = ActuatorPID(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0], dtype=wp.float32),
            ki=wp.array([10.0, 10.0], dtype=wp.float32),
            kd=wp.array([5.0, 5.0], dtype=wp.float32),
            max_force=wp.array([50.0, 50.0], dtype=wp.float32),
            integral_max=wp.array([10.0, 10.0], dtype=wp.float32),
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertTrue(actuator.is_stateful())
        self.assertTrue(actuator.is_graphable())

    def test_pid_actuator_state(self):
        """Test that ActuatorPID.state() returns properly initialized state."""
        num_dofs = 2
        indices = wp.array([0, 1], dtype=wp.uint32)

        actuator = ActuatorPID(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0], dtype=wp.float32),
            ki=wp.array([10.0, 10.0], dtype=wp.float32),
            kd=wp.array([5.0, 5.0], dtype=wp.float32),
            max_force=wp.array([50.0, 50.0], dtype=wp.float32),
            integral_max=wp.array([10.0, 10.0], dtype=wp.float32),
        )

        state = actuator.state()
        self.assertIsInstance(state, ActuatorPID.State)
        self.assertEqual(state.integral.shape[0], num_dofs)
        np.testing.assert_array_equal(state.integral.numpy(), [0.0, 0.0])


class TestActuatorDCMotorUnit(unittest.TestCase):
    """Tests for ActuatorDCMotor."""

    def setUp(self):
        wp.init()

    def test_dc_motor_creation(self):
        """Test that ActuatorDCMotor can be created with valid parameters."""
        indices = wp.array([0, 1], dtype=wp.uint32)
        actuator = ActuatorDCMotor(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0], dtype=wp.float32),
            kd=wp.array([10.0, 10.0], dtype=wp.float32),
            max_force=wp.array([50.0, 50.0], dtype=wp.float32),
            saturation_effort=wp.array([80.0, 80.0], dtype=wp.float32),
            velocity_limit=wp.array([10.0, 10.0], dtype=wp.float32),
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertFalse(actuator.is_stateful())
        self.assertIsNone(actuator.state())
        self.assertTrue(actuator.is_graphable())

    def test_dc_motor_resolve_arguments_requires_velocity_limit(self):
        """Test that resolve_arguments raises error if velocity_limit not provided."""
        with self.assertRaises(ValueError):
            ActuatorDCMotor.resolve_arguments({"kp": 50.0})

    def test_dc_motor_resolve_arguments(self):
        """Test that resolve_arguments fills defaults correctly."""
        resolved = ActuatorDCMotor.resolve_arguments({"kp": 50.0, "velocity_limit": 10.0})
        self.assertEqual(resolved["kp"], 50.0)
        self.assertEqual(resolved["kd"], 0.0)
        self.assertEqual(resolved["velocity_limit"], 10.0)

    def test_dc_motor_zero_velocity_full_torque(self):
        """At zero velocity, DC motor can produce full torque up to saturation/max_force."""
        indices = wp.array([0], dtype=wp.uint32)
        actuator = ActuatorDCMotor(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            max_force=wp.array([200.0], dtype=wp.float32),
            saturation_effort=wp.array([150.0], dtype=wp.float32),
            velocity_limit=wp.array([10.0], dtype=wp.float32),
        )

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32),
            joint_qd=wp.array([0.0], dtype=wp.float32),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32),
            joint_target_vel=wp.array([0.0], dtype=wp.float32),
            joint_act=wp.array([0.0], dtype=wp.float32),
            joint_f=wp.zeros(1, dtype=wp.float32),
        )

        actuator.step(sim_state, sim_control, None, None)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, 100.0, places=3)

    def test_dc_motor_velocity_reduces_max_torque(self):
        """At high velocity, available torque in direction of motion is reduced."""
        indices = wp.array([0], dtype=wp.uint32)
        actuator = ActuatorDCMotor(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([1000.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            max_force=wp.array([200.0], dtype=wp.float32),
            saturation_effort=wp.array([100.0], dtype=wp.float32),
            velocity_limit=wp.array([10.0], dtype=wp.float32),
        )

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32),
            joint_qd=wp.array([5.0], dtype=wp.float32),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32),
            joint_target_vel=wp.array([0.0], dtype=wp.float32),
            joint_act=wp.array([0.0], dtype=wp.float32),
            joint_f=wp.zeros(1, dtype=wp.float32),
        )

        actuator.step(sim_state, sim_control, None, None)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, 50.0, places=3)

    def test_dc_motor_at_velocity_limit(self):
        """At v_max, no torque can be produced in direction of motion."""
        indices = wp.array([0], dtype=wp.uint32)
        actuator = ActuatorDCMotor(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([1000.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            max_force=wp.array([200.0], dtype=wp.float32),
            saturation_effort=wp.array([100.0], dtype=wp.float32),
            velocity_limit=wp.array([10.0], dtype=wp.float32),
        )

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32),
            joint_qd=wp.array([10.0], dtype=wp.float32),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32),
            joint_target_vel=wp.array([0.0], dtype=wp.float32),
            joint_act=wp.array([0.0], dtype=wp.float32),
            joint_f=wp.zeros(1, dtype=wp.float32),
        )

        actuator.step(sim_state, sim_control, None, None)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, 0.0, places=3)

    def test_dc_motor_negative_velocity_increases_positive_limit(self):
        """Negative velocity allows more torque in the positive direction."""
        indices = wp.array([0], dtype=wp.uint32)
        actuator = ActuatorDCMotor(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([1000.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            max_force=wp.array([200.0], dtype=wp.float32),
            saturation_effort=wp.array([100.0], dtype=wp.float32),
            velocity_limit=wp.array([10.0], dtype=wp.float32),
        )

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32),
            joint_qd=wp.array([-5.0], dtype=wp.float32),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32),
            joint_target_vel=wp.array([0.0], dtype=wp.float32),
            joint_act=wp.array([0.0], dtype=wp.float32),
            joint_f=wp.zeros(1, dtype=wp.float32),
        )

        actuator.step(sim_state, sim_control, None, None)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, 150.0, places=3)


class TestActuatorRemotizedPDUnit(unittest.TestCase):
    """Tests for ActuatorRemotizedPD."""

    def setUp(self):
        wp.init()

    def _make_lookup(self):
        """Create a simple lookup table: torque limit varies from 10 to 50 over angles -1 to 1."""
        angles = wp.array([-1.0, 0.0, 1.0], dtype=wp.float32)
        torques = wp.array([10.0, 30.0, 50.0], dtype=wp.float32)
        return angles, torques

    def test_remotized_pd_creation(self):
        """Test that ActuatorRemotizedPD can be created with valid parameters."""
        indices = wp.array([0, 1], dtype=wp.uint32)
        angles, torques = self._make_lookup()

        actuator = ActuatorRemotizedPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0], dtype=wp.float32),
            kd=wp.array([10.0, 10.0], dtype=wp.float32),
            delay=3,
            lookup_angles=angles,
            lookup_torques=torques,
        )
        self.assertIsInstance(actuator, ActuatorDelayedPD)
        self.assertTrue(actuator.is_stateful())
        self.assertTrue(actuator.is_graphable())
        self.assertEqual(actuator.lookup_size, 3)

    def test_remotized_pd_resolve_arguments_requires_delay_and_lookup(self):
        """Test that resolve_arguments raises errors for missing required args."""
        with self.assertRaises(ValueError):
            ActuatorRemotizedPD.resolve_arguments({"kp": 50.0})
        with self.assertRaises(ValueError):
            ActuatorRemotizedPD.resolve_arguments({"kp": 50.0, "delay": 3})

    def test_remotized_pd_angle_dependent_clipping(self):
        """Test that torque is clamped based on the lookup table at the current joint angle."""
        delay = 2
        indices = wp.array([0], dtype=wp.uint32)
        angles, torques = self._make_lookup()

        actuator = ActuatorRemotizedPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([1000.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            delay=delay,
            lookup_angles=angles,
            lookup_torques=torques,
        )

        stateA = actuator.state()
        stateB = actuator.state()

        for step in range(delay + 1):
            sim_state = MockSimState(
                joint_q=wp.array([0.0], dtype=wp.float32),
                joint_qd=wp.array([0.0], dtype=wp.float32),
            )
            sim_control = MockSimControl(
                joint_target_pos=wp.array([1.0], dtype=wp.float32),
                joint_target_vel=wp.array([0.0], dtype=wp.float32),
                joint_act=wp.array([0.0], dtype=wp.float32),
                joint_f=wp.zeros(1, dtype=wp.float32),
            )
            if step % 2 == 0:
                current, next_s = stateA, stateB
            else:
                current, next_s = stateB, stateA
            actuator.step(sim_state, sim_control, current, next_s, dt=0.01)

        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, 30.0, places=3)

    def test_remotized_pd_different_angles(self):
        """Test that lookup interpolation works at different joint angles."""
        delay = 2
        indices = wp.array([0], dtype=wp.uint32)
        angles, torques = self._make_lookup()

        actuator = ActuatorRemotizedPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([1000.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            delay=delay,
            lookup_angles=angles,
            lookup_torques=torques,
        )

        for test_angle, expected_limit in [(-1.0, 10.0), (-0.5, 20.0), (0.5, 40.0), (1.0, 50.0)]:
            stateA = actuator.state()
            stateB = actuator.state()

            for step in range(delay + 1):
                sim_state = MockSimState(
                    joint_q=wp.array([test_angle], dtype=wp.float32),
                    joint_qd=wp.array([0.0], dtype=wp.float32),
                )
                sim_control = MockSimControl(
                    joint_target_pos=wp.array([test_angle + 10.0], dtype=wp.float32),
                    joint_target_vel=wp.array([0.0], dtype=wp.float32),
                    joint_act=wp.array([0.0], dtype=wp.float32),
                    joint_f=wp.zeros(1, dtype=wp.float32),
                )
                if step % 2 == 0:
                    current, next_s = stateA, stateB
                else:
                    current, next_s = stateB, stateA
                actuator.step(sim_state, sim_control, current, next_s, dt=0.01)

            force = sim_control.joint_f.numpy()[0]
            self.assertAlmostEqual(
                force, expected_limit, places=2, msg=f"At angle={test_angle}, expected limit={expected_limit}"
            )

    def test_remotized_pd_no_force_during_fill(self):
        """Test that no force is applied while the delay buffer is filling."""
        delay = 3
        indices = wp.array([0], dtype=wp.uint32)
        angles, torques = self._make_lookup()

        actuator = ActuatorRemotizedPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            delay=delay,
            lookup_angles=angles,
            lookup_torques=torques,
        )

        stateA = actuator.state()
        stateB = actuator.state()

        for step in range(delay):
            sim_state = MockSimState(
                joint_q=wp.array([0.0], dtype=wp.float32),
                joint_qd=wp.array([0.0], dtype=wp.float32),
            )
            sim_control = MockSimControl(
                joint_target_pos=wp.array([1.0], dtype=wp.float32),
                joint_target_vel=wp.array([0.0], dtype=wp.float32),
                joint_act=wp.array([0.0], dtype=wp.float32),
                joint_f=wp.zeros(1, dtype=wp.float32),
            )
            if step % 2 == 0:
                current, next_s = stateA, stateB
            else:
                current, next_s = stateB, stateA
            actuator.step(sim_state, sim_control, current, next_s, dt=0.01)
            force = sim_control.joint_f.numpy()[0]
            self.assertEqual(force, 0.0, f"Step {step}: expected 0 force during fill phase")


class MockAttribute:
    """Mock USD attribute for testing."""

    def __init__(self, value=None, name=""):
        self._value = value
        self._name = name

    def HasAuthoredValue(self):
        return self._value is not None

    def Get(self):
        return self._value

    def GetName(self):
        return self._name


class MockRelationship:
    """Mock USD relationship for testing."""

    def __init__(self, targets=None):
        self._targets = targets or []

    def GetTargets(self):
        return self._targets


class MockPrim:
    """Mock USD prim for testing ActuatorParser."""

    def __init__(self, type_name="", attributes=None, relationships=None, schemas=None):
        self._type_name = type_name
        self._attributes = attributes or {}
        self._relationships = relationships or {}
        self._schemas = schemas or []

    def GetTypeName(self):
        return self._type_name

    def GetAttribute(self, name):
        return self._attributes.get(name)

    def GetAttributes(self):
        return [MockAttribute(attr._value, name) for name, attr in self._attributes.items()]

    def GetRelationship(self, name):
        return self._relationships.get(name)

    def GetAppliedSchemas(self):
        return self._schemas


class TestActuatorParserUnit(unittest.TestCase):
    """Tests for ActuatorParser and USD parsing utilities."""

    def test_parse_pd_actuator_prim(self):
        """Test parsing a PD actuator prim."""
        from newton.actuators import ActuatorPD, parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={
                "newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp"),
                "newton:actuator:kd": MockAttribute(10.0, "newton:actuator:kd"),
            },
            relationships={
                "newton:actuator:target": MockRelationship(["/World/Robot/Joint1"]),
            },
            schemas=["PDControllerAPI"],
        )

        result = parse_actuator_prim(prim)

        self.assertIsNotNone(result)
        self.assertEqual(result.actuator_class, ActuatorPD)
        self.assertEqual(result.target_paths, ["/World/Robot/Joint1"])
        self.assertEqual(result.kwargs.get("kp"), 100.0)
        self.assertEqual(result.kwargs.get("kd"), 10.0)

    def test_parse_delayed_pd_actuator_prim(self):
        """Test parsing a Delayed PD actuator prim."""
        from newton.actuators import ActuatorDelayedPD, parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={
                "newton:actuator:kp": MockAttribute(50.0, "newton:actuator:kp"),
                "newton:actuator:delay": MockAttribute(5, "newton:actuator:delay"),
            },
            relationships={
                "newton:actuator:target": MockRelationship(["/World/Robot/Joint1"]),
            },
            schemas=["PDControllerAPI", "DelayAPI"],
        )

        result = parse_actuator_prim(prim)

        self.assertIsNotNone(result)
        self.assertEqual(result.actuator_class, ActuatorDelayedPD)
        self.assertEqual(result.kwargs.get("kp"), 50.0)
        self.assertEqual(result.kwargs.get("delay"), 5)

    def test_parse_pid_actuator_prim(self):
        """Test parsing a PID actuator prim."""
        from newton.actuators import ActuatorPID, parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={
                "newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp"),
                "newton:actuator:ki": MockAttribute(5.0, "newton:actuator:ki"),
                "newton:actuator:kd": MockAttribute(10.0, "newton:actuator:kd"),
            },
            relationships={
                "newton:actuator:target": MockRelationship(["/World/Robot/Joint1"]),
            },
            schemas=["PIDControllerAPI"],
        )

        result = parse_actuator_prim(prim)

        self.assertIsNotNone(result)
        self.assertEqual(result.actuator_class, ActuatorPID)
        self.assertEqual(result.kwargs.get("kp"), 100.0)
        self.assertEqual(result.kwargs.get("ki"), 5.0)

    def test_parse_multi_target_actuator(self):
        """Test parsing an actuator with multiple targets."""
        from newton.actuators import parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={
                "newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp"),
                "newton:actuator:transmission": MockAttribute([0.5, 0.3, 0.2], "newton:actuator:transmission"),
            },
            relationships={
                "newton:actuator:target": MockRelationship(
                    ["/World/Robot/Joint1", "/World/Robot/Joint2", "/World/Robot/Joint3"]
                ),
            },
            schemas=["PDControllerAPI"],
        )

        result = parse_actuator_prim(prim)

        self.assertIsNotNone(result)
        self.assertEqual(len(result.target_paths), 3)
        self.assertEqual(result.transmission, [0.5, 0.3, 0.2])

    def test_parse_non_actuator_prim_returns_none(self):
        """Test that non-Actuator prims return None."""
        from newton.actuators import parse_actuator_prim

        prim = MockPrim(type_name="Mesh", attributes={}, relationships={}, schemas=[])
        result = parse_actuator_prim(prim)
        self.assertIsNone(result)

    def test_parse_actuator_without_targets_returns_none(self):
        """Test that actuator without targets returns None."""
        from newton.actuators import parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={"newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp")},
            relationships={},
            schemas=["PDControllerAPI"],
        )
        result = parse_actuator_prim(prim)
        self.assertIsNone(result)

    def test_parse_dc_motor_actuator_prim(self):
        """Test parsing a DC motor actuator prim with PD + saturation params."""
        from newton.actuators import ActuatorDCMotor, parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={
                "newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp"),
                "newton:actuator:kd": MockAttribute(10.0, "newton:actuator:kd"),
                "newton:actuator:maxForce": MockAttribute(200.0, "newton:actuator:maxForce"),
                "newton:actuator:saturationEffort": MockAttribute(150.0, "newton:actuator:saturationEffort"),
                "newton:actuator:velocityLimit": MockAttribute(10.0, "newton:actuator:velocityLimit"),
            },
            relationships={
                "newton:actuator:target": MockRelationship(["/World/Robot/Joint1"]),
            },
        )

        result = parse_actuator_prim(prim)

        self.assertIsNotNone(result)
        self.assertEqual(result.actuator_class, ActuatorDCMotor)
        self.assertEqual(result.kwargs.get("kp"), 100.0)
        self.assertEqual(result.kwargs.get("saturation_effort"), 150.0)
        self.assertEqual(result.kwargs.get("velocity_limit"), 10.0)

    def test_parse_dc_motor_velocity_limit_zero_raises(self):
        """Test that velocity_limit=0 raises ValueError during parsing."""
        from newton.actuators import parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={
                "newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp"),
                "newton:actuator:saturationEffort": MockAttribute(150.0, "newton:actuator:saturationEffort"),
                "newton:actuator:velocityLimit": MockAttribute(0.0, "newton:actuator:velocityLimit"),
            },
            relationships={
                "newton:actuator:target": MockRelationship(["/World/Robot/Joint1"]),
            },
        )

        with self.assertRaises(ValueError):
            parse_actuator_prim(prim)

    def test_parse_dc_motor_velocity_limit_negative_raises(self):
        """Test that negative velocity_limit raises ValueError during parsing."""
        from newton.actuators import parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={
                "newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp"),
                "newton:actuator:saturationEffort": MockAttribute(150.0, "newton:actuator:saturationEffort"),
                "newton:actuator:velocityLimit": MockAttribute(-5.0, "newton:actuator:velocityLimit"),
            },
            relationships={
                "newton:actuator:target": MockRelationship(["/World/Robot/Joint1"]),
            },
        )

        with self.assertRaises(ValueError):
            parse_actuator_prim(prim)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestActuatorNetMLPUnit(unittest.TestCase):
    """Tests for ActuatorNetMLP."""

    def setUp(self):
        wp.init()
        import torch

        self.torch = torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.wp_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _make_mlp(self, input_dim, hidden=32):
        """Create a simple MLP: input_dim -> hidden -> 1."""
        return self.torch.nn.Sequential(
            self.torch.nn.Linear(input_dim, hidden),
            self.torch.nn.ELU(),
            self.torch.nn.Linear(hidden, 1),
        )

    def test_mlp_creation(self):
        """Test that ActuatorNetMLP can be created with valid parameters."""
        from newton.actuators import ActuatorNetMLP

        indices = wp.array([0, 1], dtype=wp.uint32, device=self.wp_device)
        network = self._make_mlp(input_dim=2)

        actuator = ActuatorNetMLP(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([50.0, 50.0], dtype=wp.float32, device=self.wp_device),
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertTrue(actuator.is_stateful())
        self.assertFalse(actuator.is_graphable())
        self.assertEqual(actuator.history_length, 1)

    def test_mlp_step_runs(self):
        """Test that step() executes without errors and produces output."""
        from newton.actuators import ActuatorNetMLP

        num_dofs = 2
        indices = wp.array([0, 1], dtype=wp.uint32, device=self.wp_device)
        network = self._make_mlp(input_dim=2)

        actuator = ActuatorNetMLP(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([1000.0, 1000.0], dtype=wp.float32, device=self.wp_device),
        )

        stateA = actuator.state()
        stateB = actuator.state()

        sim_state = MockSimState(
            joint_q=wp.array([0.0, 0.0], dtype=wp.float32, device=self.wp_device),
            joint_qd=wp.array([0.0, 0.0], dtype=wp.float32, device=self.wp_device),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0, 2.0], dtype=wp.float32, device=self.wp_device),
            joint_target_vel=wp.array([0.0, 0.0], dtype=wp.float32, device=self.wp_device),
            joint_act=wp.array([0.0, 0.0], dtype=wp.float32, device=self.wp_device),
            joint_f=wp.zeros(num_dofs, dtype=wp.float32, device=self.wp_device),
        )

        actuator.step(sim_state, sim_control, stateA, stateB)
        forces = sim_control.joint_f.numpy()
        self.assertEqual(forces.shape, (2,))

    def test_mlp_clamping(self):
        """Test that output is clamped to max_force."""
        from newton.actuators import ActuatorNetMLP

        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)

        network = self.torch.nn.Sequential(self.torch.nn.Linear(2, 1, bias=True))
        with self.torch.no_grad():
            network[0].weight.fill_(0.0)
            network[0].bias.fill_(999.0)

        max_force_val = 10.0
        actuator = ActuatorNetMLP(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([max_force_val], dtype=wp.float32, device=self.wp_device),
            torque_scale=1.0,
        )

        stateA = actuator.state()
        stateB = actuator.state()

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_qd=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32, device=self.wp_device),
            joint_target_vel=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_act=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_f=wp.zeros(1, dtype=wp.float32, device=self.wp_device),
        )

        actuator.step(sim_state, sim_control, stateA, stateB)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, max_force_val, places=3)

    def test_mlp_invalid_input_order(self):
        """Test that an invalid input_order raises ValueError at construction."""
        from newton.actuators import ActuatorNetMLP

        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)
        network = self._make_mlp(input_dim=2)

        with self.assertRaises(ValueError):
            ActuatorNetMLP(
                input_indices=indices,
                output_indices=indices,
                network=network,
                max_force=wp.array([50.0], dtype=wp.float32, device=self.wp_device),
                input_order="invalid",
            )


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestActuatorNetLSTMUnit(unittest.TestCase):
    """Tests for ActuatorNetLSTM."""

    def setUp(self):
        wp.init()
        import torch

        self.torch = torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.wp_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _make_lstm(self, hidden_size=8, num_layers=1):
        import torch

        class _SimpleLSTMNet(torch.nn.Module):
            def __init__(self, input_size=2, hidden_size=8, output_size=1, num_layers=1):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.decoder = torch.nn.Linear(hidden_size, output_size)

            def forward(self, x, hc):
                lstm_out, (h_new, c_new) = self.lstm(x, hc)
                output = self.decoder(lstm_out[:, -1, :])
                return output, (h_new, c_new)

        return _SimpleLSTMNet(input_size=2, hidden_size=hidden_size, num_layers=num_layers)

    def test_lstm_creation(self):
        """Test that ActuatorNetLSTM can be created with valid parameters."""
        from newton.actuators import ActuatorNetLSTM

        indices = wp.array([0, 1], dtype=wp.uint32, device=self.wp_device)
        network = self._make_lstm()

        actuator = ActuatorNetLSTM(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([50.0, 50.0], dtype=wp.float32, device=self.wp_device),
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertTrue(actuator.is_stateful())
        self.assertFalse(actuator.is_graphable())

    def test_lstm_state(self):
        """Test that state() returns properly shaped hidden and cell tensors."""
        from newton.actuators import ActuatorNetLSTM

        hidden_size = 16
        num_layers = 2
        indices = wp.array([0, 1, 2], dtype=wp.uint32, device=self.wp_device)
        network = self._make_lstm(hidden_size=hidden_size, num_layers=num_layers)

        actuator = ActuatorNetLSTM(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([50.0, 50.0, 50.0], dtype=wp.float32, device=self.wp_device),
        )

        state = actuator.state()
        self.assertIsInstance(state, ActuatorNetLSTM.State)
        self.assertEqual(state.hidden.shape, (num_layers, 3, hidden_size))
        self.assertEqual(state.cell.shape, (num_layers, 3, hidden_size))

    def test_lstm_step_runs(self):
        """Test that step() executes without errors and produces output."""
        from newton.actuators import ActuatorNetLSTM

        num_dofs = 2
        indices = wp.array([0, 1], dtype=wp.uint32, device=self.wp_device)
        network = self._make_lstm()

        actuator = ActuatorNetLSTM(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([1000.0, 1000.0], dtype=wp.float32, device=self.wp_device),
        )

        stateA = actuator.state()
        stateB = actuator.state()

        sim_state = MockSimState(
            joint_q=wp.array([0.0, 0.0], dtype=wp.float32, device=self.wp_device),
            joint_qd=wp.array([1.0, -1.0], dtype=wp.float32, device=self.wp_device),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0, 2.0], dtype=wp.float32, device=self.wp_device),
            joint_target_vel=wp.array([0.0, 0.0], dtype=wp.float32, device=self.wp_device),
            joint_act=wp.array([0.0, 0.0], dtype=wp.float32, device=self.wp_device),
            joint_f=wp.zeros(num_dofs, dtype=wp.float32, device=self.wp_device),
        )

        actuator.step(sim_state, sim_control, stateA, stateB)
        forces = sim_control.joint_f.numpy()
        self.assertEqual(forces.shape, (2,))

    def test_lstm_clamping(self):
        """Test that output is clamped to max_force."""
        from newton.actuators import ActuatorNetLSTM

        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)

        network = self._make_lstm(hidden_size=4)
        with self.torch.no_grad():
            network.decoder.weight.fill_(0.0)
            network.decoder.bias.fill_(500.0)

        max_force_val = 10.0
        actuator = ActuatorNetLSTM(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([max_force_val], dtype=wp.float32, device=self.wp_device),
        )

        stateA = actuator.state()
        stateB = actuator.state()

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_qd=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32, device=self.wp_device),
            joint_target_vel=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_act=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_f=wp.zeros(1, dtype=wp.float32, device=self.wp_device),
        )

        actuator.step(sim_state, sim_control, stateA, stateB)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, max_force_val, places=3)


if __name__ == "__main__":
    unittest.main()
