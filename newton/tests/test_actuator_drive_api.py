# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the actuator drive API migration."""

import types
import typing
import unittest

import warp as wp

import newton
import newton.actuators as actuators


class TestActuatorDriveAPI(unittest.TestCase):
    """Verify the canonical actuator drive API."""

    def test_joint_space_response_public_method_type_hints(self):
        """Keep JointSpaceResponse's public methods fully annotated."""
        self.assertEqual(
            typing.get_type_hints(actuators.JointSpaceResponse.__init__),
            {"model": newton.Model, "return": type(None)},
        )
        self.assertEqual(
            typing.get_type_hints(actuators.JointSpaceResponse.refresh),
            {"state": newton.State, "return": type(None)},
        )

    def test_actuator_passes_declared_inputs_without_validation(self):
        """Let drives validate or provide fallbacks for declared inputs."""

        class _RecordingDrive(actuators.DrivePD):
            custom_inputs = ("custom_control_input", "missing_control_input")

            def compute(self, *args, custom_inputs=None, **kwargs):
                self.seen_custom_inputs = custom_inputs
                return super().compute(*args, custom_inputs=custom_inputs, **kwargs)

        indices = wp.array([0], dtype=wp.uint32)
        drive = _RecordingDrive(
            kp=wp.array([0.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
        )
        actuator = actuators.Actuator(indices=indices, drive=drive)

        state = types.SimpleNamespace(
            joint_q=wp.zeros(1, dtype=wp.float32),
            joint_qd=wp.zeros(1, dtype=wp.float32),
            missing_control_input=object(),
        )
        control_input = object()
        control = types.SimpleNamespace(
            joint_target_q=wp.zeros(1, dtype=wp.float32),
            joint_target_qd=wp.zeros(1, dtype=wp.float32),
            joint_act=wp.zeros(1, dtype=wp.float32),
            joint_f=wp.zeros(1, dtype=wp.float32),
            custom_control_input=control_input,
        )

        actuator.step(state, control, dt=0.01)

        self.assertIs(drive.seen_custom_inputs["custom_control_input"], control_input)
        self.assertIsNone(drive.seen_custom_inputs["missing_control_input"])

    def test_explicit_drive_without_custom_inputs_keyword_remains_compatible(self):
        """Keep drives using the previous compute signature working."""

        class _LegacyDrive(actuators.DrivePD):
            def compute(self, *args, device=None):
                return super().compute(*args, device=device)

        actuator = actuators.Actuator(
            indices=wp.array([0], dtype=wp.uint32),
            drive=_LegacyDrive(
                kp=wp.array([1.0], dtype=wp.float32),
                kd=wp.array([0.0], dtype=wp.float32),
            ),
        )
        state = types.SimpleNamespace(
            joint_q=wp.zeros(1, dtype=wp.float32),
            joint_qd=wp.zeros(1, dtype=wp.float32),
        )
        control = types.SimpleNamespace(
            joint_target_q=wp.ones(1, dtype=wp.float32),
            joint_target_qd=wp.zeros(1, dtype=wp.float32),
            joint_act=None,
            joint_f=wp.zeros(1, dtype=wp.float32),
        )

        actuator.step(state, control, dt=0.01)

        self.assertAlmostEqual(float(control.joint_f.numpy()[0]), 1.0)

    def test_actuator_registers_drive_inputs_only(self):
        """Ignore undeclared input conventions on delay and clamping components."""
        drive = actuators.DrivePD(
            kp=wp.array([0.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
        )
        drive.custom_inputs = ("drive_input",)
        delay = actuators.Delay(delay_steps=wp.array([0], dtype=wp.int32), max_delay=1)
        delay.custom_inputs = ("delay_input",)
        clamping = actuators.ClampingMaxEffort(max_effort=wp.array([1.0], dtype=wp.float32))
        clamping.custom_inputs = ("clamping_input",)
        actuator = actuators.Actuator(
            indices=wp.array([0], dtype=wp.uint32),
            drive=drive,
            delay=delay,
            clamping=[clamping],
        )
        builder = newton.ModelBuilder()

        actuator.register_custom_attributes(builder)
        actuator.register_custom_attributes(builder)

        self.assertIn("drive_input", builder.custom_attributes)
        self.assertNotIn("delay_input", builder.custom_attributes)
        self.assertNotIn("clamping_input", builder.custom_attributes)
        sim_control = actuator.sim_control()
        self.assertTrue(hasattr(sim_control, "drive_input"))
        self.assertFalse(hasattr(sim_control, "delay_input"))
        self.assertFalse(hasattr(sim_control, "clamping_input"))
        declaration = builder.custom_attributes["drive_input"]
        self.assertEqual(declaration.dtype, wp.float32)
        self.assertEqual(declaration.frequency, newton.Model.AttributeFrequency.JOINT_DOF)
        self.assertEqual(declaration.assignment, newton.Model.AttributeAssignment.CONTROL)

    def test_builder_registers_declared_inputs_on_control(self):
        """Register actuator inputs on Control during model finalization."""

        class _CustomDrive(actuators.DrivePD):
            custom_inputs = ("estimated_load",)

        builder = newton.ModelBuilder()
        link = builder.add_link()
        joint = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z)
        builder.add_articulation([joint])
        builder.add_actuator(
            drive_class=_CustomDrive,
            index=builder.joint_qd_start[joint],
            kp=1.0,
        )

        model = builder.finalize()

        self.assertTrue(hasattr(model.control(), "estimated_load"))
        self.assertFalse(hasattr(model.state(), "estimated_load"))


if __name__ == "__main__":
    unittest.main()
