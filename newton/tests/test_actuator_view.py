# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone actuator parameter view."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.actuators import Actuator, ActuatorView, ClampingMaxEffort, ControllerPD, Delay
from newton.selection import ArticulationView


class TestActuatorView(unittest.TestCase):
    def make_actuator(
        self,
        kp: list[float],
        *,
        delay_steps: list[int] | None = None,
        max_effort: list[float] | None = None,
    ) -> Actuator:
        count = len(kp)
        device = wp.get_device()
        return Actuator(
            indices=wp.array(range(count), dtype=wp.uint32, device=device),
            controller=ControllerPD(
                kp=wp.array(kp, dtype=wp.float32, device=device),
                kd=wp.zeros(count, dtype=wp.float32, device=device),
            ),
            delay=(
                Delay(wp.array(delay_steps, dtype=wp.int32, device=device), max(delay_steps))
                if delay_steps is not None
                else None
            ),
            clamping=(
                [ClampingMaxEffort(wp.array(max_effort, dtype=wp.float32, device=device))]
                if max_effort is not None
                else None
            ),
        )

    def mapping(self) -> wp.array2d[int]:
        return wp.array([[0, -1, 1], [2, -1, 3]], dtype=int, device=wp.get_device())

    def make_articulation_view(self) -> tuple[ArticulationView, Actuator]:
        template = newton.ModelBuilder()
        body = template.add_link()
        joint = template.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
        template.add_articulation([joint], label="robot")
        template.add_actuator(ControllerPD, index=template.joint_qd_start[joint], kp=100.0, delay_steps=2)
        builder = newton.ModelBuilder()
        builder.replicate(template, 2)
        model = builder.finalize()
        return ArticulationView(model, "robot"), model.actuators[0]

    def make_empty_articulation_view(self) -> tuple[ArticulationView, Actuator]:
        source, actuator = self.make_articulation_view()
        return ArticulationView(source.model, "robot", include_joints=[]), actuator

    def test_get_gathers_requested_actuator(self):
        first = self.make_actuator([10.0, 20.0, 30.0, 40.0])
        second = self.make_actuator([50.0, 60.0, 70.0, 80.0])
        view = ActuatorView(
            {
                first: self.mapping(),
                second: wp.array([[-1, 0, -1], [-1, 2, -1]], dtype=int, device=wp.get_device()),
            }
        )

        values = view.get_actuator_parameter(first, "controller", "kp")

        np.testing.assert_array_equal(values.numpy(), [[10.0, 0.0, 20.0], [30.0, 0.0, 40.0]])

    def test_set_scatters_mapped_values(self):
        actuator = self.make_actuator([10.0, 20.0, 30.0, 40.0])
        view = ActuatorView({actuator: self.mapping()})
        values = wp.array([[11.0, 999.0, 21.0], [31.0, 999.0, 41.0]], dtype=wp.float32, device=wp.get_device())

        view.set_actuator_parameter(actuator, "controller", "kp", values)

        np.testing.assert_array_equal(actuator.controller.kp.numpy(), [11.0, 21.0, 31.0, 41.0])

    def test_set_honors_world_mask(self):
        actuator = self.make_actuator([10.0, 20.0, 30.0, 40.0])
        view = ActuatorView({actuator: self.mapping()})
        values = wp.array([[11.0, 999.0, 21.0], [31.0, 999.0, 41.0]], dtype=wp.float32, device=wp.get_device())

        view.set_actuator_parameter(
            actuator,
            "controller",
            "kp",
            values,
            mask=wp.array([False, True], dtype=bool, device=wp.get_device()),
        )

        np.testing.assert_array_equal(actuator.controller.kp.numpy(), [10.0, 20.0, 31.0, 41.0])

    def test_component_and_parameter_names_are_strings(self):
        actuator = self.make_actuator(
            [10.0, 20.0, 30.0, 40.0],
            delay_steps=[1, 2, 1, 2],
            max_effort=[100.0, 200.0, 300.0, 400.0],
        )
        view = ActuatorView({actuator: self.mapping()})

        delays = view.get_actuator_parameter(actuator, "delay", "delay_steps")
        efforts = view.get_actuator_parameter(actuator, "clamping.0", "max_effort")

        np.testing.assert_array_equal(delays.numpy(), [[1, 0, 2], [1, 0, 2]])
        np.testing.assert_array_equal(efforts.numpy(), [[100.0, 0.0, 200.0], [300.0, 0.0, 400.0]])

    def test_get_actuator_dof_mapping_survives_source_dictionary_mutation(self):
        actuator = self.make_actuator([10.0, 20.0, 30.0, 40.0])
        mapping = self.mapping()
        mappings = {actuator: mapping}
        view = ActuatorView(mappings)
        mappings.clear()

        self.assertIs(view.get_actuator_dof_mapping(actuator), mapping)

    def test_from_articulation_view_returns_cached_view(self):
        source, actuator = self.make_articulation_view()

        first = ActuatorView.from_articulation_view(source, [actuator])

        self.assertIs(first, source.get_actuator_view([actuator]))
        values = first.get_actuator_parameter(actuator, "controller", "kp")
        np.testing.assert_array_equal(values.numpy(), [[100.0], [100.0]])

    def test_legacy_get_zero_dofs_returns_float(self):
        source, actuator = self.make_empty_articulation_view()

        values = source.get_actuator_parameter(actuator, actuator.delay, "delay_steps")

        self.assertEqual(values.shape, (2, 0))
        self.assertEqual(values.dtype, wp.float32)

    def test_legacy_set_zero_dofs_skips_parameter_lookup(self):
        source, actuator = self.make_empty_articulation_view()

        self.assertIsNone(source.set_actuator_parameter(actuator, object(), "missing", []))


if __name__ == "__main__":
    unittest.main()
