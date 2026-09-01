# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone actuator parameter view."""

import unittest
from unittest.mock import patch

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
        indices: list[int] | None = None,
        delay_steps: list[int] | None = None,
        max_effort: list[float] | None = None,
    ) -> Actuator:
        """Build an actuator with parameters stored in the requested DOFs."""
        count = len(kp)
        device = wp.get_device()
        return Actuator(
            indices=wp.array(range(count) if indices is None else indices, dtype=wp.uint32, device=device),
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

    def make_view(self, *actuators: Actuator) -> ActuatorView:
        """Build a view over two worlds with three velocity DOFs each."""
        dof_indices = wp.array([[0, 1, 2], [3, 4, 5]], dtype=int, device=wp.get_device())
        return ActuatorView(list(actuators), dof_indices)

    def make_articulation_view(self) -> tuple[ArticulationView, Actuator]:
        """Build an articulation view containing one actuator in two worlds."""
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
        """Build an articulation view that selects no velocity DOFs."""
        source, actuator = self.make_articulation_view()
        return ArticulationView(source.model, "robot", include_joints=[]), actuator

    def test_constructor_builds_actuator_mappings(self):
        """Derive per-actuator mappings from selected global velocity DOFs."""
        first = self.make_actuator([10.0, 20.0, 30.0, 40.0], indices=[0, 2, 3, 5])
        second = self.make_actuator([50.0, 60.0], indices=[1, 4])
        view = self.make_view(first, second)

        values = view.get_actuator_parameter(first, "controller", "kp")

        np.testing.assert_array_equal(values.numpy(), [[10.0, 0.0, 20.0], [30.0, 0.0, 40.0]])
        np.testing.assert_array_equal(view.get_actuator_dof_mapping(second).numpy(), [[-1, 0, -1], [-1, 1, -1]])

    def test_set_scatters_mapped_values(self):
        """Write only values whose selected DOFs belong to the actuator."""
        actuator = self.make_actuator([10.0, 20.0, 30.0, 40.0], indices=[0, 2, 3, 5])
        view = self.make_view(actuator)
        values = wp.array([[11.0, 999.0, 21.0], [31.0, 999.0, 41.0]], dtype=wp.float32, device=wp.get_device())

        view.set_actuator_parameter(actuator, "controller", "kp", values)

        np.testing.assert_array_equal(actuator.controller.kp.numpy(), [11.0, 21.0, 31.0, 41.0])

    def test_set_honors_world_mask(self):
        """Update only worlds selected by the Boolean mask."""
        actuator = self.make_actuator([10.0, 20.0, 30.0, 40.0], indices=[0, 2, 3, 5])
        view = self.make_view(actuator)
        values = wp.array([[11.0, 999.0, 21.0], [31.0, 999.0, 41.0]], dtype=wp.float32, device=wp.get_device())

        view.set_actuator_parameter(
            actuator,
            "controller",
            "kp",
            values,
            mask=wp.array([False, True], dtype=bool, device=wp.get_device()),
        )

        np.testing.assert_array_equal(actuator.controller.kp.numpy(), [10.0, 20.0, 31.0, 41.0])

    def test_component_accepts_strings_and_objects(self):
        """Resolve string paths while preserving component-object access."""
        actuator = self.make_actuator(
            [10.0, 20.0, 30.0, 40.0],
            indices=[0, 2, 3, 5],
            delay_steps=[1, 2, 1, 2],
            max_effort=[100.0, 200.0, 300.0, 400.0],
        )
        view = self.make_view(actuator)

        delays = view.get_actuator_parameter(actuator, actuator.delay, "delay_steps")
        efforts = view.get_actuator_parameter(actuator, "clamping.0", "max_effort")

        np.testing.assert_array_equal(delays.numpy(), [[1, 0, 2], [1, 0, 2]])
        np.testing.assert_array_equal(efforts.numpy(), [[100.0, 0.0, 200.0], [300.0, 0.0, 400.0]])

    def test_articulation_view_returns_cached_actuator_view(self):
        """Build and cache a standalone view through an articulation view."""
        source, actuator = self.make_articulation_view()

        first = source.get_actuator_view([actuator])

        self.assertIs(first, source.get_actuator_view([actuator]))
        values = first.get_actuator_parameter(actuator, "controller", "kp")
        np.testing.assert_array_equal(values.numpy(), [[100.0], [100.0]])

    def test_legacy_get_zero_dofs_returns_float(self):
        """Preserve the selection API's float result for an empty DOF view."""
        source, actuator = self.make_empty_articulation_view()

        values = source.get_actuator_parameter(actuator, actuator.delay, "delay_steps")

        self.assertEqual(values.shape, (2, 0))
        self.assertEqual(values.dtype, wp.float32)

    def test_legacy_set_zero_dofs_skips_parameter_lookup(self):
        """Preserve parameter-lookup skipping for an empty DOF view."""
        source, actuator = self.make_empty_articulation_view()

        self.assertIsNone(source.set_actuator_parameter(actuator, object(), "missing", []))

    def test_constructor_rejects_empty_actuators(self):
        """Reject a view that cannot expose any actuator."""
        dof_indices = wp.empty((2, 0), dtype=int, device=wp.get_device())

        with self.assertRaisesRegex(ValueError, "at least one actuator"):
            ActuatorView([], dof_indices)

    def test_constructor_rejects_duplicate_dofs_per_world(self):
        """Reject duplicate columns that would race when scattering parameters."""
        actuator = self.make_actuator([10.0, 20.0, 30.0, 40.0], indices=[0, 2, 3, 5])
        dof_indices = wp.array([[0, 0, 2], [3, 4, 5]], dtype=int, device=wp.get_device())

        with self.assertRaisesRegex(ValueError, "unique within each world"):
            ActuatorView([actuator], dof_indices)

    def test_set_rejects_invalid_warp_inputs_before_launch(self):
        """Reject incompatible values and masks before launching a kernel."""
        actuator = self.make_actuator([10.0, 20.0, 30.0, 40.0], indices=[0, 2, 3, 5])
        view = self.make_view(actuator)
        invalid_inputs = (
            (wp.ones((2, 2), dtype=float), None, "values shape"),
            (wp.ones((2, 3), dtype=float), wp.ones(2, dtype=int), "Boolean mask"),
        )
        if wp.is_cuda_available():
            other_device = "cpu" if wp.get_device().is_cuda else "cuda:0"
            invalid_inputs += (
                (wp.ones((2, 3), dtype=float, device=other_device), None, "values on device"),
                (wp.ones((2, 3), dtype=float), wp.ones(2, dtype=bool, device=other_device), "mask on device"),
            )

        for values, mask, message in invalid_inputs:
            with self.subTest(message=message), patch.object(wp, "launch") as launch:
                with self.assertRaisesRegex(ValueError, message):
                    view.set_actuator_parameter(actuator, "controller", "kp", values, mask)
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
