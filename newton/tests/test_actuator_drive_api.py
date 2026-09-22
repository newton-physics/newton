# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the actuator drive API migration."""

import types
import typing
import unittest
import warnings

import warp as wp

import newton
import newton.actuators as actuators


class TestActuatorDriveAPI(unittest.TestCase):
    """Verify canonical actuator names and deprecated compatibility aliases."""

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

    def test_public_base_names_and_deprecated_aliases(self):
        """Expose Base-suffixed names and warn for deprecated aliases."""
        aliases = {
            "Clamping": "ClampingBase",
            "Controller": "DriveBase",
            "ControllerPD": "DrivePD",
            "ControllerPID": "DrivePID",
            "ControllerNeuralMLP": "DriveNeuralMLP",
            "ControllerNeuralLSTM": "DriveNeuralLSTM",
        }

        for old_name, new_name in aliases.items():
            self.assertIn(new_name, actuators.__all__)
            self.assertNotIn(old_name, actuators.__all__)
            with self.assertWarnsRegex(DeprecationWarning, rf"{old_name}.*{new_name}"):
                old_value = getattr(actuators, old_name)
            self.assertIs(old_value, getattr(actuators, new_name))

    def test_actuator_deprecated_controller_keyword_and_attribute(self):
        """Keep the former Actuator constructor keyword and attribute functional."""
        indices = wp.array([0], dtype=wp.uint32)
        kp = wp.array([1.0], dtype=wp.float32)
        kd = wp.array([0.1], dtype=wp.float32)
        drive = actuators.DrivePD(kp=kp, kd=kd)

        actuator = actuators.Actuator(indices=indices, drive=drive)
        self.assertIs(actuator.drive, drive)

        with self.assertWarnsRegex(DeprecationWarning, r"Actuator\.controller.*Actuator\.drive"):
            self.assertIs(actuator.controller, drive)

        replacement = actuators.DrivePD(kp=kp, kd=kd)
        with self.assertWarnsRegex(DeprecationWarning, r"Actuator\.controller.*Actuator\.drive"):
            actuator.controller = replacement
        self.assertIs(actuator.drive, replacement)

        with self.assertWarnsRegex(DeprecationWarning, r"controller=.*drive="):
            legacy = actuators.Actuator(indices=indices, controller=drive)
        self.assertIs(legacy.drive, drive)

        with self.assertRaisesRegex(TypeError, "only one"):
            actuators.Actuator(indices=indices, drive=drive, controller=drive)

    def test_actuator_state_deprecated_controller_state(self):
        """Keep the former actuator-state keyword and attribute functional."""
        drive_state = actuators.DriveBase.State()
        state = actuators.Actuator.State(drive_state=drive_state)
        self.assertIs(state.drive_state, drive_state)

        with self.assertWarnsRegex(DeprecationWarning, r"controller_state.*drive_state"):
            legacy = actuators.Actuator.State(controller_state=drive_state)
        self.assertIs(legacy.drive_state, drive_state)

        with self.assertWarnsRegex(DeprecationWarning, r"controller_state.*drive_state"):
            self.assertIs(state.controller_state, drive_state)

        with self.assertWarnsRegex(DeprecationWarning, r"controller_state.*drive_state"):
            state.controller_state = None
        self.assertIsNone(state.drive_state)

        with self.assertRaisesRegex(TypeError, "only one"):
            actuators.Actuator.State(drive_state=drive_state, controller_state=drive_state)

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

    def test_builder_deprecated_controller_class_keyword(self):
        """Keep the former builder keyword functional with a warning."""
        builder = newton.ModelBuilder()
        builder.add_actuator(drive_class=actuators.DrivePD, index=0, kp=1.0)

        legacy_builder = newton.ModelBuilder()
        with self.assertWarnsRegex(DeprecationWarning, r"controller_class.*drive_class"):
            legacy_builder.add_actuator(controller_class=actuators.DrivePD, index=0, kp=1.0)

        with self.assertRaisesRegex(TypeError, "only one"):
            builder.add_actuator(
                drive_class=actuators.DrivePD,
                controller_class=actuators.DrivePD,
                index=0,
                kp=1.0,
            )

    def test_parsed_actuator_deprecated_controller_fields(self):
        """Keep the former parsed-actuator constructor and fields functional."""
        parsed = actuators.ActuatorParsed(drive_class=actuators.DrivePD, drive_kwargs={"kp": 1.0})
        self.assertIs(parsed.drive_class, actuators.DrivePD)
        self.assertEqual(parsed.drive_kwargs, {"kp": 1.0})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = actuators.ActuatorParsed(
                controller_class=actuators.DrivePD,
                controller_kwargs={"kp": 1.0},
            )
        self.assertEqual(len(caught), 2)
        self.assertTrue(all(issubclass(item.category, DeprecationWarning) for item in caught))
        self.assertIs(legacy.drive_class, actuators.DrivePD)
        self.assertEqual(legacy.drive_kwargs, {"kp": 1.0})

        with self.assertWarnsRegex(DeprecationWarning, r"controller_class.*drive_class"):
            self.assertIs(parsed.controller_class, actuators.DrivePD)
        with self.assertWarnsRegex(DeprecationWarning, r"controller_kwargs.*drive_kwargs"):
            self.assertEqual(parsed.controller_kwargs, {"kp": 1.0})

        with self.assertRaisesRegex(TypeError, "only one"):
            actuators.ActuatorParsed(
                drive_class=actuators.DrivePD,
                controller_class=actuators.DrivePD,
            )

        with self.assertRaisesRegex(TypeError, "only one"):
            actuators.ActuatorParsed(
                drive_class=actuators.DrivePD,
                drive_kwargs={"kp": 1.0},
                controller_kwargs={"kp": 1.0},
            )

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with self.assertRaisesRegex(TypeError, "only one"):
                actuators.ActuatorParsed(
                    drive_kwargs={"kp": 1.0},
                    controller_class=actuators.DrivePD,
                    controller_kwargs={"kp": 1.0},
                )

    def test_component_kind_deprecated_controller_member(self):
        """Keep the former component-kind member functional with a warning."""
        with self.assertWarnsRegex(DeprecationWarning, r"CONTROLLER.*DRIVE"):
            legacy = actuators.ComponentKind.CONTROLLER
        self.assertIs(legacy, actuators.ComponentKind.DRIVE)

        with self.assertWarnsRegex(DeprecationWarning, r"CONTROLLER.*DRIVE"):
            legacy = actuators.ComponentKind["CONTROLLER"]
        self.assertIs(legacy, actuators.ComponentKind.DRIVE)

        with self.assertWarnsRegex(DeprecationWarning, r"CONTROLLER.*DRIVE"):
            legacy = actuators.ComponentKind("controller")
        self.assertIs(legacy, actuators.ComponentKind.DRIVE)


if __name__ == "__main__":
    unittest.main()
