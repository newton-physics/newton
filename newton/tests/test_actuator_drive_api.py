# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the actuator drive API migration."""

import types
import typing
import unittest

import numpy as np
import warp as wp

import newton
import newton.actuators as actuators
from newton.exceptions import NewtonDeprecationWarning


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

    def test_compute_deprecated_forces_argument(self):
        """Keep the former name of the compute output argument working, as a call and as an override."""
        indices = wp.array([0], dtype=wp.uint32)
        kp = wp.array([2.0], dtype=wp.float32)
        kd = wp.array([0.0], dtype=wp.float32)
        q, target = wp.zeros(1, dtype=wp.float32), wp.ones(1, dtype=wp.float32)
        args = (q, q, target, q, None, indices, indices, indices, indices)

        with self.assertWarnsRegex(NewtonDeprecationWarning, r"rename it to 'output'"):

            class _LegacyDrive(actuators.DrivePD):
                def compute(
                    self,
                    positions,
                    velocities,
                    target_pos,
                    target_vel,
                    feedforward,
                    pos_indices,
                    vel_indices,
                    target_pos_indices,
                    target_vel_indices,
                    forces,
                    state,
                    dt,
                    device=None,
                    **kwargs,
                ):
                    return super().compute(
                        positions,
                        velocities,
                        target_pos,
                        target_vel,
                        feedforward,
                        pos_indices,
                        vel_indices,
                        target_pos_indices,
                        target_vel_indices,
                        forces,
                        state,
                        dt,
                        device,
                        **kwargs,
                    )

        for drive_class in (actuators.DrivePD, _LegacyDrive):
            with self.subTest(drive=drive_class.__name__):
                drive = drive_class(kp=kp, kd=kd)
                output = wp.zeros(1, dtype=wp.float32)
                drive.compute(*args, output, None, 0.01)
                self.assertAlmostEqual(float(output.numpy()[0]), 2.0)

                output.zero_()
                drive.compute(*args, output=output, state=None, dt=0.01)
                self.assertAlmostEqual(float(output.numpy()[0]), 2.0)

                forces = wp.zeros(1, dtype=wp.float32)
                with self.assertWarnsRegex(NewtonDeprecationWarning, r"forces argument.*use output"):
                    drive.compute(*args, forces=forces, state=None, dt=0.01)
                self.assertAlmostEqual(float(forces.numpy()[0]), 2.0)

                with self.assertRaisesRegex(TypeError, "both 'output' and its deprecated name 'forces'"):
                    drive.compute(*args, output=output, forces=forces, state=None, dt=0.01)

    def test_actuator_registers_drive_inputs_only(self):
        """Ignore undeclared input conventions on delay and clamping components."""
        drive = actuators.DrivePD(
            kp=wp.array([0.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
        )
        drive.custom_inputs = ("drive_input",)
        delay = actuators.InputProcessorDelay(delay_steps=wp.array([0], dtype=wp.int32), max_delay=1)
        delay.custom_inputs = ("delay_input",)
        clamping = actuators.ClampingSymmetric(limit=wp.array([1.0], dtype=wp.float32))
        clamping.custom_inputs = ("clamping_input",)
        actuator = actuators.Actuator(
            indices=wp.array([0], dtype=wp.uint32),
            drive=drive,
            input_processors=[delay],
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

    def test_deprecated_delay_arguments_match_input_processor(self):
        """Deprecated delay arguments warn and delay targets exactly like the Delay input processor."""

        def build(**actuator_kwargs) -> newton.Model:
            builder = newton.ModelBuilder()
            link = builder.add_link()
            joint = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z)
            builder.add_articulation([joint])
            builder.add_actuator(actuators.DrivePD, index=0, kp=10.0, **actuator_kwargs)
            return builder.finalize()

        def run(actuator: actuators.Actuator, model: newton.Model) -> list[float]:
            state, control = model.state(), model.control()
            sa, sb = actuator.state(), actuator.state()
            efforts = []
            for target in (1.0, 2.0, 3.0, 4.0):
                control.joint_target_q.fill_(target)
                control.joint_f.zero_()
                actuator.step(state, control, sa, sb, dt=0.01)
                sa, sb = sb, sa
                efforts.append(float(control.joint_f.numpy()[0]))
            return efforts

        model = build(input_processors=[(actuators.InputProcessorDelay, {"delay_steps": 2})])
        expected = run(model.actuators[0], model)
        np.testing.assert_allclose(expected, [10.0, 10.0, 10.0, 20.0])

        with self.assertWarnsRegex(NewtonDeprecationWarning, "1.9"):
            legacy_model = build(delay_steps=2)
        np.testing.assert_allclose(run(legacy_model.actuators[0], legacy_model), expected)

        with self.assertWarnsRegex(NewtonDeprecationWarning, "1.9"):
            self.assertIs(newton.actuators.Delay, actuators.InputProcessorDelay)

        legacy_drive = actuators.DrivePD(kp=wp.array([10.0], dtype=wp.float32), kd=wp.array([0.0], dtype=wp.float32))
        legacy_delay = actuators.InputProcessorDelay(delay_steps=wp.array([2], dtype=wp.int32))
        with self.assertWarnsRegex(NewtonDeprecationWarning, "1.9"):
            legacy_actuator = actuators.Actuator(
                indices=wp.array([0], dtype=wp.uint32), drive=legacy_drive, delay=legacy_delay
            )
        np.testing.assert_allclose(run(legacy_actuator, model), expected)

    def test_deprecated_delay_aliases_warn_and_forward(self):
        """Every deprecated delay alias warns with the 1.9 deprecation and reaches the input processor."""
        from newton.actuators import ComponentKind  # noqa: PLC0415

        drive = actuators.DrivePD(kp=wp.array([10.0], dtype=wp.float32), kd=wp.array([0.0], dtype=wp.float32))
        delay = actuators.InputProcessorDelay(delay_steps=wp.array([2], dtype=wp.int32))
        actuator = actuators.Actuator(indices=wp.array([0], dtype=wp.uint32), drive=drive, input_processors=[delay])
        deprecated = self.assertWarnsRegex(NewtonDeprecationWarning, "1.9")

        with deprecated:
            self.assertIs(actuator.delay, delay)
        replacement = actuators.InputProcessorDelay(delay_steps=wp.array([1], dtype=wp.int32))
        with deprecated:
            actuator.delay = replacement
        self.assertEqual(actuator.input_processors, [replacement])

        state = actuator.state()
        with deprecated:
            self.assertIs(state.delay_state, state.input_processor_states[0])
        with deprecated:
            legacy_state = actuators.Actuator.State(delay_state=state.input_processor_states[0])
        self.assertIs(legacy_state.input_processor_states[0], state.input_processor_states[0])

        with deprecated:
            entry = newton.ModelBuilder.ActuatorEntry(
                actuators.DrivePD, (), (), {}, [0], [0], [{"kp": 1.0, "kd": 0.0}], [{"delay_steps": 2}], [[]]
            )
        self.assertEqual(entry.input_processor_classes, (actuators.InputProcessorDelay,))
        with deprecated:
            self.assertEqual(entry.delay_args, [{"delay_steps": 2}])
        with deprecated:
            entry.delay_args = [{"delay_steps": 3}]
        self.assertEqual(entry.input_processor_args, [[{"delay_steps": 3}]])

        for legacy in (lambda: ComponentKind.DELAY, lambda: ComponentKind["DELAY"], lambda: ComponentKind("delay")):
            with deprecated:
                self.assertIs(legacy(), ComponentKind.INPUT_PROCESSOR)

        targets = wp.array([1.0], dtype=wp.float32)
        indices = wp.array([0], dtype=wp.uint32)
        delay_state, next_state = replacement.state(1, actuator.device), replacement.state(1, actuator.device)
        with deprecated:
            replacement.update_state(targets, targets, None, indices, indices, delay_state, next_state)
        np.testing.assert_array_equal(next_state.num_pushes.numpy(), [1])


if __name__ == "__main__":
    unittest.main()
