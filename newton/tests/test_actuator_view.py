# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise standalone actuator parameters and automatic joint attribute selections."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.actuators import (
    Actuator,
    ClampingMaxEffort,
    ClampingPositionBased,
    Delay,
    DrivePD,
)
from newton.selection import ActuatorView

AttributeFrequency = newton.Model.AttributeFrequency


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
            drive=DrivePD(
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
        return ActuatorView(list(actuators), dof_indices=dof_indices)

    def test_parameter_mapping_and_masked_writes(self):
        """Gather sparse actuator parameters and scatter full or masked updates."""
        first = self.make_actuator([10.0, 20.0, 30.0, 40.0], indices=[0, 2, 3, 5])
        second = self.make_actuator([50.0, 60.0], indices=[1, 4])
        with self.assertRaisesRegex(ValueError, "dof_indices.*multiple actuators"):
            ActuatorView([first, second])
        view = self.make_view(first, second)

        values = view.get_actuator_parameter("kp", first.drive)

        np.testing.assert_array_equal(values.numpy(), [[10.0, 0.0, 20.0], [30.0, 0.0, 40.0]])
        np.testing.assert_array_equal(view.get_actuator_dof_mapping(second).numpy(), [[-1, 0, -1], [-1, 1, -1]])
        values = wp.array([[11.0, 999.0, 21.0], [31.0, 999.0, 41.0]], dtype=wp.float32, device=wp.get_device())
        view.set_actuator_parameter("kp", first.drive, values)
        np.testing.assert_array_equal(first.drive.kp.numpy(), [11.0, 21.0, 31.0, 41.0])
        values.fill_(50.0)
        view.set_actuator_parameter(
            "kp",
            first.drive,
            values,
            mask=wp.array([False, True], dtype=bool, device=wp.get_device()),
        )

        np.testing.assert_array_equal(first.drive.kp.numpy(), [11.0, 21.0, 50.0, 50.0])
        np.testing.assert_array_equal(second.drive.kp.numpy(), [50.0, 60.0])

    def test_delay_and_clamping_parameters(self):
        """Select parameters of delay and clamping components."""
        actuator = self.make_actuator(
            [10.0, 20.0, 30.0, 40.0],
            indices=[0, 2, 3, 5],
            delay_steps=[1, 2, 1, 2],
            max_effort=[100.0, 200.0, 300.0, 400.0],
        )
        view = self.make_view(actuator)
        delays = view.get_actuator_parameter("delay_steps", actuator.delay)
        efforts = view.get_actuator_parameter("max_effort", actuator.clamping[0])
        np.testing.assert_array_equal(delays.numpy(), [[1, 0, 2], [1, 0, 2]])
        np.testing.assert_array_equal(efforts.numpy(), [[100.0, 0.0, 200.0], [300.0, 0.0, 400.0]])
        view.set_actuator_parameter("delay_steps", actuator.delay, [[2, 0, 1], [2, 0, 1]])
        np.testing.assert_array_equal(actuator.delay.delay_steps.numpy(), [2, 1, 2, 1])

    def test_parameter_access_follows_current_components(self):
        """Resolve component ownership after replacement and reject foreign components."""
        actuator = self.make_actuator([10.0, 20.0, 30.0, 40.0], indices=[0, 2, 3, 5])
        foreign = self.make_actuator([1.0, 2.0, 3.0, 4.0], indices=[0, 2, 3, 5])
        view = self.make_view(actuator)
        with self.assertRaisesRegex(ValueError, "component owned"):
            view.get_actuator_parameter("kp", foreign.drive)
        with self.assertRaisesRegex(ValueError, "component owned"):
            view.set_actuator_parameter("kp", foreign.drive, [[0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(foreign.drive.kp.numpy(), [1, 2, 3, 4])

        original_drive = actuator.drive
        actuator.drive = foreign.drive
        view.set_actuator_parameter("kp", actuator.drive, [[5, 0, 6], [7, 0, 8]])
        np.testing.assert_array_equal(foreign.drive.kp.numpy(), [5, 6, 7, 8])
        np.testing.assert_array_equal(original_drive.kp.numpy(), [10, 20, 30, 40])
        with self.assertRaisesRegex(ValueError, "component owned"):
            view.get_actuator_parameter("kp", original_drive)

    def test_shared_component_uses_first_owner(self):
        """Access a component shared by actuators with identical DOFs."""
        actuator = self.make_actuator([10.0, 20.0, 30.0, 40.0], indices=[0, 2, 3, 5], max_effort=[1, 2, 3, 4])
        clamp = actuator.clamping[0]
        other = self.make_actuator([1.0, 2.0, 3.0, 4.0], indices=[0, 2, 3, 5])
        other.clamping.append(clamp)
        view = self.make_view(actuator, other)
        np.testing.assert_array_equal(view.get_actuator_parameter("max_effort", clamp).numpy(), [[1, 0, 2], [3, 0, 4]])
        view.set_actuator_parameter("max_effort", clamp, [[5, 0, 6], [7, 0, 8]])
        np.testing.assert_array_equal(clamp.max_effort.numpy(), [5, 6, 7, 8])

    def test_empty_selection_preserves_parameter_contract(self):
        """Preserve dtype and selection axes while validating empty parameter writes."""
        actuator = self.make_actuator([10, 20], delay_steps=[1, 2])
        view = ActuatorView([actuator], dof_indices=wp.empty((2, 1, 0), dtype=int, device=wp.get_device()))

        values = view.get_actuator_parameter("delay_steps", actuator.delay)

        self.assertEqual(values.shape, (2, 1, 0))
        self.assertEqual(values.dtype, wp.int32)

        with self.assertRaisesRegex(ValueError, "per-entry parameter"):
            view.set_actuator_parameter("missing", actuator.delay, [])

        empty_actuator = self.make_actuator([], indices=[])
        empty_view = ActuatorView(
            [empty_actuator], dof_indices=wp.array([[0, -1, 3]], dtype=int, device=wp.get_device())
        )
        np.testing.assert_array_equal(empty_view.get_actuator_dof_mapping(empty_actuator).numpy(), [[-1, -1, -1]])
        np.testing.assert_array_equal(
            empty_view.get_actuator_parameter("kp", empty_actuator.drive).numpy(), [[0, 0, 0]]
        )
        empty_view.set_actuator_parameter("kp", empty_actuator.drive, [[1, 2, 3]])
        self.assertEqual(empty_actuator.drive.kp.shape, (0,))

    def test_constructor_and_write_validation(self):
        """Validate explicit selections, the single-actuator default and masked writes."""
        actuator = self.make_actuator([10.0, 20.0, 30.0, 40.0], indices=[5, 0, 3, 2])
        with self.assertRaisesRegex(ValueError, "at least one actuator"):
            ActuatorView([], dof_indices=wp.empty((2, 0), dtype=int, device=wp.get_device()))
        duplicate_dofs = wp.array([[0, 0, 2], [3, 4, 5]], dtype=int, device=wp.get_device())
        with self.assertRaisesRegex(ValueError, "globally unique"):
            ActuatorView([actuator], dof_indices=duplicate_dofs)
        for dof in (np.iinfo(np.int32).max + 1, np.iinfo(np.uint32).max):
            with self.subTest(dof=dof), self.assertRaisesRegex(ValueError, "within int32 range"):
                ActuatorView([actuator], dof_indices=wp.array([dof], dtype=wp.uint32, device=wp.get_device()))
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
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    view.set_actuator_parameter("kp", actuator.drive, values, mask=mask)
                np.testing.assert_array_equal(actuator.drive.kp.numpy(), [10, 20, 30, 40])

        unsupported = wp.zeros((6, 1), dtype=float, device=wp.get_device())
        actuator.drive.tensor = unsupported[:4]
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            view.get_actuator_parameter("tensor", actuator.drive)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            view.set_attribute("joint_qd", {"joint_qd": unsupported}, np.ones((2, 3, 1)))
        np.testing.assert_array_equal(unsupported.numpy(), np.zeros((6, 1)))

        state = {"joint_qd": wp.array(range(6), dtype=float, device=wp.get_device())}
        for dof_indices in (None, actuator.indices):
            with self.subTest(explicit=dof_indices is not None):
                view = ActuatorView([actuator], dof_indices=dof_indices)
                self.assertIs(view.dof_indices, actuator.indices)
                np.testing.assert_array_equal(
                    view.get_actuator_parameter("kp", actuator.drive).numpy(), [10, 20, 30, 40]
                )
                np.testing.assert_array_equal(view.get_attribute("joint_qd", state).numpy(), [5, 0, 3, 2])
        view.set_actuator_parameter("kp", actuator.drive, [11, 21, 31, 41], mask=[True, False, True, False])
        np.testing.assert_array_equal(actuator.drive.kp.numpy(), [11, 20, 31, 40])

    def test_ambiguous_frequency_mappings_are_rejected(self):
        """Reject conflicting frequency mappings while preserving parameter access."""
        device = wp.get_device()
        actuator = Actuator(
            indices=wp.array([8, 2, 5], dtype=wp.uint32, device=device),
            pos_indices=wp.array([1, 6, 0], dtype=wp.uint32, device=device),
            target_pos_indices=wp.array([7, 4, 3], dtype=wp.uint32, device=device),
            effort_indices=wp.array([2, 0, 1], dtype=wp.uint32, device=device),
            control_output_attr="effort",
            drive=DrivePD(wp.array([10, 20, 30], dtype=float, device=device), wp.zeros(3, device=device)),
        )
        view = ActuatorView([actuator], dof_indices=wp.array([[[5, -1]], [[8, 2]]], dtype=int, device=device))
        np.testing.assert_array_equal(
            view.get_actuator_parameter("kp", actuator.drive).numpy(), [[[30, 0]], [[10, 20]]]
        )
        state = SimpleNamespace(joint_q=wp.array([0.3, 0.1, 0, 0, 0, 0, 0.2], dtype=float, device=device))
        control = SimpleNamespace(
            effort=wp.array(np.arange(9), dtype=float, device=device),
            get_attribute_frequency=lambda name: {"effort": AttributeFrequency.JOINT_DOF}[name],
        )
        for container, name in ((state, "joint_q"), (control, "effort")):
            with self.subTest(name=name):
                before = getattr(container, name).numpy().copy()
                with self.assertRaisesRegex(AttributeError, "disagree"):
                    view.get_attribute(name, container)
                with self.assertRaisesRegex(AttributeError, "disagree"):
                    view.set_attribute(name, container, [[[11, 999]], [[12, 13]]])
                np.testing.assert_array_equal(getattr(container, name).numpy(), before)

    def test_frequency_mapping_unions_actuators(self):
        """Combine common DOF and coordinate mappings across actuators."""
        device = wp.get_device()
        first = self.make_actuator([1, 1], indices=[90, 2])
        first.pos_indices = wp.array([3, 4], dtype=wp.uint32, device=device)
        first.target_pos_indices = first.pos_indices
        second = self.make_actuator([1, 1], indices=[2, 51])
        second.pos_indices = wp.array([4, 7], dtype=wp.uint32, device=device)
        second.target_pos_indices = second.pos_indices
        selected = wp.array([[[51, -1]], [[90, 2]]], dtype=int, device=device)
        view = ActuatorView([first, second], dof_indices=selected)
        source = {
            "joint_q": wp.array(range(9), dtype=float, device=device),
            "joint_qd": wp.array(range(91), dtype=float, device=device),
            "joint_target_q": wp.zeros(9, device=device),
        }
        np.testing.assert_array_equal(view.get_attribute("joint_q", source).numpy(), [[[7, 0]], [[3, 4]]])
        np.testing.assert_array_equal(view.get_attribute("joint_qd", source).numpy(), [[[51, 0]], [[90, 2]]])
        view.set_attribute("joint_target_q", source, [[[11, 999]], [[12, 13]]])
        np.testing.assert_array_equal(source["joint_target_q"].numpy(), [0, 0, 0, 12, 13, 0, 0, 11, 0])

    def test_masks_cover_leading_selection_axes(self):
        """Apply first-axis and leading-two-axis masks without articulation metadata."""
        device = wp.get_device()
        actuator = self.make_actuator([1] * 6)
        source = {"joint_qd": wp.zeros(6, device=device)}
        selected = wp.array([[[0, 1], [2, 3]], [[4, 5], [-1, -1]]], dtype=int, device=device)
        view = ActuatorView([actuator], dof_indices=selected)
        mask_storage = wp.array([[True, True, False, True], [False, True, True, True]], dtype=bool, device=device)
        view.set_attribute("joint_qd", source, np.arange(8).reshape(2, 2, 2) + 10, mask=mask_storage[:, ::2])
        np.testing.assert_array_equal(source["joint_qd"].numpy(), [10, 11, 0, 0, 0, 0])
        view.set_attribute("joint_qd", source, np.full((2, 2, 2), 20), mask=[False, True])
        np.testing.assert_array_equal(source["joint_qd"].numpy(), [10, 11, 0, 0, 20, 20])
        one = ActuatorView([actuator], dof_indices=wp.array([0, 2, 5], dtype=int, device=device))
        one.set_attribute("joint_qd", source, [30, 30, 30], mask=[True, False, True])
        np.testing.assert_array_equal(source["joint_qd"].numpy(), [30, 11, 0, 0, 20, 30])
        two = ActuatorView([actuator], dof_indices=wp.array([[0, 1], [4, 5]], dtype=int, device=device))
        with self.assertRaisesRegex(ValueError, "Boolean mask shape"):
            two.set_attribute("joint_qd", source, [[1, 1], [1, 1]], mask=[[True, False], [False, True]])

    def test_graph_capture_replays_live_arrays(self):
        """Capture coordinate, DOF and joint reads and replay values with a two-axis mask."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires CUDA")
        actuator = self.make_actuator([1, 1, 1], indices=[0, 1, 3])
        actuator.pos_indices = wp.array([2, 4, 0], dtype=wp.uint32, device=device)
        actuator.target_pos_indices = actuator.pos_indices
        selection = wp.array([[[0], [1]], [[3], [-1]]], dtype=int, device=device)
        old = {
            "user": {"position": wp.array([1, 2, 3, 4, 5], dtype=float, device=device)},
            "get_attribute_frequency": lambda name: {"user:position": AttributeFrequency.JOINT_COORD}[name],
        }
        current = {
            "joint_qd_start": wp.array([0, 2, 4], dtype=int, device=device),
            "joint_enabled": wp.array([True, False], dtype=bool, device=device),
            "user": {
                "position": wp.array([10, 20, 30, 40, 50], dtype=float, device=device),
                "command": wp.array([3, 5, 7, 9], dtype=float, device=device),
            },
        }
        current["get_attribute_frequency"] = lambda name: {
            "user:position": AttributeFrequency.JOINT_COORD,
            "user:command": AttributeFrequency.JOINT_DOF,
        }[name]
        control = {
            "target_position": wp.zeros(5, device=device),
            "get_attribute_frequency": lambda name: {"target_position": AttributeFrequency.JOINT_COORD}[name],
        }
        values = wp.array([[[11], [12]], [[13], [99]]], dtype=float, device=device)
        mask = wp.array([[True, False], [False, True]], dtype=bool, device=device)
        # Compile the kernels on another view, then capture this view's first access.
        warm = ActuatorView([actuator], dof_indices=selection)
        warm.get_attribute("user.position", old)
        warm.set_attribute("target_position", control, values, mask=mask)
        control["target_position"].zero_()
        view = ActuatorView([actuator], dof_indices=selection)
        view.get_attribute("joint_enabled", current)
        with wp.ScopedCapture(device) as capture:
            observed = view.get_attribute("user:position", current)
            commands = view.get_attribute("user.command", current)
            enabled = view.get_attribute("joint_enabled", current)
            view.set_attribute("target_position", control, values, mask=mask)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(observed.numpy(), [[[30], [50]], [[10], [0]]])
        np.testing.assert_array_equal(commands.numpy(), [[[3], [5]], [[9], [0]]])
        np.testing.assert_array_equal(enabled.numpy(), [[[True], [True]], [[False], [False]]])
        np.testing.assert_array_equal(control["target_position"].numpy(), [0, 0, 11, 0, 0])
        current["user"]["position"].assign(wp.array([100, 200, 300, 400, 500], dtype=float, device=device))
        current["user"]["command"].fill_(11)
        current["joint_enabled"].assign([False, True])
        values.fill_(21)
        mask.assign(wp.array([[False, True], [True, False]], dtype=bool, device=device))
        control["target_position"].zero_()
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(observed.numpy(), [[[300], [500]], [[100], [0]]])
        np.testing.assert_array_equal(commands.numpy(), [[[11], [11]], [[11], [0]]])
        np.testing.assert_array_equal(enabled.numpy(), [[[False], [False]], [[True], [False]]])
        np.testing.assert_array_equal(control["target_position"].numpy(), [21, 0, 0, 0, 21])

    def test_parameters_exclude_private_and_shared_arrays(self):
        """Reject private, shared and non-array attributes as parameters."""
        device = wp.get_device()
        actuator = self.make_actuator([1, 1], indices=[4, 8])
        clamp = ClampingPositionBased(lookup_positions=(0.0, 1.0), lookup_efforts=(10.0, 20.0))
        clamp.finalize(device, 2)
        actuator.clamping.append(clamp)
        view = ActuatorView([actuator], dof_indices=wp.array([[8, 4, -1]], dtype=int, device=device))
        actuator.drive._private_gain = actuator.drive.kp
        for name in ("custom_inputs", "_private_gain"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "public per-entry|per-entry parameter"):
                view.get_actuator_parameter(name, actuator.drive)
        with self.assertRaisesRegex(ValueError, "shared"):
            view.get_actuator_parameter("lookup_positions", clamp)

    def test_warp_vector_dtype(self):
        """Read and write vector elements without adding selection axes."""
        actuator = self.make_actuator([10, 20], indices=[4, 8])
        actuator.drive.vector = wp.array([[1, 2, 3], [4, 5, 6]], dtype=wp.vec3, device=wp.get_device())
        view = ActuatorView([actuator], dof_indices=wp.array([8, 4], dtype=int, device=wp.get_device()))
        vector = view.get_actuator_parameter("vector", actuator.drive)
        self.assertEqual(vector.dtype, wp.vec3)
        np.testing.assert_array_equal(vector.numpy(), [[4, 5, 6], [1, 2, 3]])
        view.set_actuator_parameter("vector", actuator.drive, [[7, 8, 9], [10, 11, 12]], mask=[True, False])
        np.testing.assert_array_equal(actuator.drive.vector.numpy(), [[1, 2, 3], [7, 8, 9]])

    def test_repeated_actuator_dofs_choose_last_entry(self):
        """Use the last repeated entry across signed and unsigned DOF boundaries."""
        max_dof = np.iinfo(np.int32).max
        actuator = self.make_actuator(
            [10, 20, 30, 40, 50], indices=[max_dof, 1, max_dof, max_dof + 1, np.iinfo(np.uint32).max]
        )
        view = ActuatorView([actuator], dof_indices=wp.array([[1, max_dof, -1]], dtype=int, device=wp.get_device()))
        np.testing.assert_array_equal(view.get_actuator_dof_mapping(actuator).numpy(), [[1, 2, -1]])
        np.testing.assert_array_equal(view.get_actuator_parameter("kp", actuator.drive).numpy(), [[20, 30, 0]])
        view.set_actuator_parameter("kp", actuator.drive, [[25, 35, 99]])
        np.testing.assert_array_equal(actuator.drive.kp.numpy(), [10, 25, 35, 40, 50])
        repeated = self.make_actuator([10, 20, 30], indices=[5, 0, 5])
        default_view = ActuatorView([repeated])
        np.testing.assert_array_equal(default_view.get_actuator_parameter("kp", repeated.drive).numpy(), [20, 30])

    def test_mapping_handles_singleton_search_boundaries(self):
        """Find the sole actuator key in a strided selection and reject neighbors on both sides."""
        actuator = self.make_actuator([10], indices=[9])
        selection = wp.array([8, -1, 9, -1, 10, -1, -1, -1], dtype=int, device=wp.get_device())[::2]
        view = ActuatorView([actuator], dof_indices=selection)

        np.testing.assert_array_equal(view.get_actuator_dof_mapping(actuator).numpy(), [-1, 0, -1, -1])
        np.testing.assert_array_equal(view.get_actuator_parameter("kp", actuator.drive).numpy(), [0, 10, 0, 0])
        view.set_actuator_parameter("kp", actuator.drive, [100, 11, 100, 100])
        np.testing.assert_array_equal(actuator.drive.kp.numpy(), [11])

    def test_registered_joint_attributes_follow_dof_selection(self):
        """Pad unowned DOFs while preserving registered mappings and caller indices."""
        device = wp.get_device()
        actuator = self.make_actuator([1, 1], indices=[1, 3])
        actuator.pos_indices = wp.array([2, 0], dtype=wp.uint32, device=device)
        actuator.target_pos_indices = actuator.pos_indices
        other = self.make_actuator([1], indices=[0])
        # Differences outside the selected DOFs do not make their projection ambiguous.
        other.effort_indices = wp.array([7], dtype=wp.uint32, device=device)
        view = ActuatorView([actuator, other], dof_indices=wp.array([[3, 2, 1]], dtype=int, device=device))
        state = newton.State()
        state.joint_qd = wp.array([1, 2, 3, 4], dtype=float, device=device)
        control = newton.Control()
        control.joint_act = wp.array([10, 20, 99, 30], dtype=float, device=device)
        np.testing.assert_array_equal(view.get_attribute("joint_qd", state).numpy(), [[4, 0, 2]])
        view.set_attribute("joint_act", control, [[7, 999, 8]])
        np.testing.assert_array_equal(control.joint_act.numpy(), [10, 8, 99, 7])
        np.testing.assert_array_equal(view.dof_indices.numpy(), [[3, 2, 1]])
        source = SimpleNamespace(
            user=SimpleNamespace(
                gain=wp.array(np.arange(8).reshape(4, 2), dtype=wp.vec2, device=device, requires_grad=True),
                coordinate=wp.array([10, 20, 30], dtype=float, device=device),
            ),
            get_attribute_frequency=lambda name: {
                "user:gain": AttributeFrequency.JOINT_DOF,
                "user:coordinate": AttributeFrequency.JOINT_COORD,
            }[name],
        )
        np.testing.assert_array_equal(view.get_attribute("user.coordinate", source).numpy(), [[10, 0, 30]])
        with wp.Tape() as tape:
            selected = view.get_attribute("user.gain", source)
        np.testing.assert_array_equal(selected.numpy(), [[[6, 7], [0, 0], [2, 3]]])
        tape.backward(grads={selected: wp.ones_like(selected)})
        np.testing.assert_array_equal(source.user.gain.grad.numpy(), [[0, 0], [1, 1], [0, 0], [1, 1]])
        source.user.gain = wp.full(4, wp.vec2(5.0), dtype=wp.vec2, device=device)
        for name in ("user.gain", "user:gain"):
            with self.subTest(name=name):
                np.testing.assert_array_equal(view.get_attribute(name, source).numpy(), [[[5, 5], [0, 0], [5, 5]]])
        source.user.coordinate = source.user.coordinate[:2]
        with self.assertRaisesRegex(ValueError, "at least 3 elements"):
            view.get_attribute("user.coordinate", source)

    def test_joint_projection_requires_topology_and_supported_frequency(self):
        """Pad unowned DOFs in joint projection and reject duplicate joint writes."""
        device = wp.get_device()
        actuator = self.make_actuator([1, 1, 1], indices=[1, 2, 3])
        actuator.effort_indices = wp.array([3, 2, 1], dtype=wp.uint32, device=device)
        view = ActuatorView([actuator], dof_indices=wp.array([[3, 0, 1]], dtype=int, device=device))
        model = newton.Model(device)
        # A three-DOF joint, a fixed joint, and a scalar joint exercise unequal spans.
        model.joint_count = 3
        model.joint_dof_count = 4
        model.joint_qd_start = wp.array([0, 3, 3, 4], dtype=int, device=device)
        model.joint_enabled = wp.array([True, False, False], dtype=bool, device=device)
        np.testing.assert_array_equal(view.get_attribute("joint_enabled", model).numpy(), [[False, False, True]])
        view.set_attribute("joint_enabled", model, [[True, True, False]])
        np.testing.assert_array_equal(model.joint_enabled.numpy(), [False, False, True])
        repeated = ActuatorView([actuator], dof_indices=wp.array([1, 2], dtype=int, device=device))
        np.testing.assert_array_equal(repeated.get_attribute("joint_enabled", model).numpy(), [False, False])
        with self.assertRaisesRegex(ValueError, "duplicate output indices"):
            repeated.set_attribute("joint_enabled", model, [True, False])
        np.testing.assert_array_equal(model.joint_enabled.numpy(), [False, False, True])
        with self.assertRaises(AttributeError):
            view.get_attribute("joint_enabled", SimpleNamespace(joint_enabled=model.joint_enabled))
        model.body_mass = wp.ones(1, device=device)
        with self.assertRaises(AttributeError):
            view.get_attribute("body_mass", model)


if __name__ == "__main__":
    unittest.main()
