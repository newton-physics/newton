# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import contextlib
import gc
import re
import unittest
import weakref
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.utils.selection import FrequencyLayout
from newton.actuators import Actuator, DrivePD
from newton.selection import ArticulationView
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_cuda_test_devices, get_test_devices


def origin_velocity_from_body_qd(model, body_q, body_qd, body_idx):
    """Recover body-origin velocity from COM-referenced `body_qd`."""
    rot = wp.quat(
        float(body_q[body_idx, 3]),
        float(body_q[body_idx, 4]),
        float(body_q[body_idx, 5]),
        float(body_q[body_idx, 6]),
    )
    com_local = model.body_com.numpy()[body_idx]
    com_world = np.array(
        wp.quat_rotate(rot, wp.vec3(float(com_local[0]), float(com_local[1]), float(com_local[2]))),
        dtype=np.float32,
    )
    return body_qd[body_idx, :3] - np.cross(body_qd[body_idx, 3:6], com_world)


class TestSelectionCacheLifetime(unittest.TestCase):
    def test_attribute_sources_release_with_view(self):
        """Release abandoned attribute sources while another view remains usable."""
        for layout in ("dense", "indexed", "sparse"):
            with self.subTest(layout=layout):
                live_model, live_view = self.make_view(layout)
                live_model.joint_q.fill_(17.0)
                live_buffer = live_view.get_dof_positions(live_model)
                live_values = live_buffer.numpy().copy()

                model, view = self.make_view(layout)
                state, control = model.state(), model.control()
                view.get_dof_positions(model)
                view.get_dof_positions(state)
                view.get_dof_forces(control)
                refs = {
                    name: weakref.ref(obj)
                    for name, obj in (("view", view), ("model", model), ("state", state), ("control", control))
                }
                del model, view, state, control
                gc.collect()

                assert_np_equal(live_buffer.numpy(), live_values)
                self.assertIs(live_view.get_dof_positions(live_model), live_buffer)
                assert_np_equal(live_view.get_dof_positions(live_model).numpy(), live_values)
                for name, ref in refs.items():
                    with self.subTest(source=name):
                        self.assertIsNone(ref(), f"Abandoned {name} retained for {layout}")

    def test_actuator_sources_release_with_view(self):
        """Release abandoned actuator mappings independently of a second live view."""
        for layout in ("dense", "indexed", "sparse"):
            with self.subTest(layout=layout):
                live_model, live_view = self.make_view(layout)
                live_actuator = self.make_actuator(live_model, 17.0)
                live_buffer = live_view.get_actuator_parameter(live_actuator, live_actuator.drive, "kp")
                live_values = live_buffer.numpy().copy()

                model, view = self.make_view(layout)
                actuator = self.make_actuator(model, 3.0)
                values = view.get_actuator_parameter(actuator, actuator.drive, "kp").numpy()
                assert_np_equal(values, np.full(values.shape, 3.0))
                refs = {
                    name: weakref.ref(obj) for name, obj in (("view", view), ("model", model), ("actuator", actuator))
                }
                del model, view, actuator
                gc.collect()

                assert_np_equal(live_buffer.numpy(), live_values)
                assert_np_equal(
                    live_view.get_actuator_parameter(live_actuator, live_actuator.drive, "kp").numpy(), live_values
                )
                for name, ref in refs.items():
                    with self.subTest(source=name):
                        self.assertIsNone(ref(), f"Abandoned {name} retained for {layout}")

    def test_attribute_sources_and_slices_remain_distinct(self):
        """Keep model, state, control, root slices, and full selections independent."""
        for layout in ("dense", "indexed", "sparse"):
            with self.subTest(layout=layout):
                model, view = self.make_view(layout)
                states = [model.state(), model.state()]
                controls = [model.control(), model.control()]
                sources = [model, *states]
                for value, source in enumerate(sources, 1):
                    source.joint_q.fill_(float(value))
                for value, control in enumerate(controls, 4):
                    control.joint_f.fill_(float(value))
                buffers = [view.get_dof_positions(source) for source in sources]
                force_buffers = [view.get_dof_forces(control) for control in controls]
                for value, (source, buffer) in enumerate(zip(sources, buffers, strict=True), 1):
                    root = view.get_root_transforms(source)
                    self.assertEqual(root.shape, (view.world_count, 1))
                    assert_np_equal(root.numpy(), np.full((*root.shape, 7), value))
                    assert_np_equal(buffer.numpy(), np.full(buffer.shape, value))
                    self.assertIs(view.get_dof_positions(source), buffer)
                    assert_np_equal(view.get_dof_positions(source).numpy(), buffer.numpy())
                for value, (control, buffer) in enumerate(zip(controls, force_buffers, strict=True), 4):
                    assert_np_equal(buffer.numpy(), np.full(buffer.shape, value))
                    self.assertIs(view.get_dof_forces(control), buffer)
                    assert_np_equal(view.get_dof_forces(control).numpy(), buffer.numpy())

    def test_returned_array_owns_backing_allocation(self):
        """Keep zero-copy data and gradient allocations alive beyond their view."""
        for requires_grad in (False, True):
            with self.subTest(requires_grad=requires_grad):
                model, view = self.make_view("dense", requires_grad=requires_grad)
                model.joint_q.fill_(5.0)
                source_ref = weakref.ref(model.joint_q)
                model_ref, view_ref = weakref.ref(model), weakref.ref(view)
                buffer = view.get_dof_positions(model)
                if requires_grad:
                    model.joint_q.grad.fill_(7.0)
                    grad_ref = weakref.ref(model.joint_q.grad)
                    gradient = buffer.grad
                del model, view
                gc.collect()

                self.assertIsNone(view_ref())
                self.assertIsNone(model_ref())
                self.assertIsNotNone(source_ref())
                assert_np_equal(buffer.numpy(), np.full(buffer.shape, 5.0))
                del buffer
                gc.collect()
                if requires_grad:
                    self.assertIsNotNone(grad_ref())
                    assert_np_equal(gradient.numpy(), np.full(gradient.shape, 7.0))
                    del gradient
                    gc.collect()
                    self.assertIsNone(grad_ref())
                self.assertIsNone(source_ref())

    @staticmethod
    def make_view(layout, requires_grad=False, device="cpu"):
        """Build selections with regular, indexed, or irregular world layouts."""
        robot = newton.ModelBuilder()
        parent = robot.add_link(label="robot/root")
        joints = [robot.add_joint_free(child=parent, label="robot/root_joint")]
        for index in range(3):
            child = robot.add_link(label=f"robot/link_{index}")
            joints.append(robot.add_joint_revolute(parent=parent, child=child, label=f"robot/joint_{index}"))
            parent = child
        robot.add_articulation(joints, label="robot")
        other = newton.ModelBuilder()
        link = other.add_link(label="other/link")
        other.add_articulation([other.add_joint_free(child=link)], label="other")
        scene = newton.ModelBuilder()
        worlds = (robot, other, robot, other, other, robot) if layout == "sparse" else (robot, robot)
        for world in worlds:
            scene.add_world(world)
        model = scene.finalize(device=device, requires_grad=requires_grad)
        selected = ["root_joint", "joint_0", "joint_2"] if layout == "indexed" else None
        return model, ArticulationView(model, "robot", include_joints=selected, verbose=False)

    @staticmethod
    def make_actuator(model, value):
        """Build an actuator covering every model DOF on the model's device."""
        count = model.joint_dof_count
        device = model.device
        return Actuator(
            indices=wp.array(np.arange(count), dtype=wp.uint32, device=device),
            drive=DrivePD(kp=wp.full(count, value, device=device), kd=wp.zeros(count, device=device)),
        )


def test_view_holds_sources_and_actuators_weakly(test, device):
    """Release states, controls, and actuators that are dropped while their view stays alive."""
    for layout in ("dense", "sparse"):
        with test.subTest(layout=layout):
            model, view = TestSelectionCacheLifetime.make_view(layout, device=device)
            state, control = model.state(), model.control()
            actuator = TestSelectionCacheLifetime.make_actuator(model, 3.0)
            view.get_dof_positions(state)
            view.get_dof_forces(control)
            view.get_actuator_parameter(actuator, actuator.drive, "kp")
            refs = {
                name: weakref.ref(obj) for name, obj in (("state", state), ("control", control), ("actuator", actuator))
            }
            del state, control, actuator
            gc.collect()

            for name, ref in refs.items():
                test.assertIsNone(ref(), f"{name} retained by a live view ({layout})")
            model.joint_q.fill_(2.0)
            positions = view.get_dof_positions(model).numpy()
            assert_np_equal(positions, np.full(positions.shape, 2.0))


def test_returned_array_survives_allocation_reuse(test, device):
    """Keep returned arrays and their gradients valid after their allocations would otherwise be reused."""
    for layout, requires_grad in (("dense", False), ("dense", True), ("indexed", False), ("sparse", False)):
        with test.subTest(layout=layout, requires_grad=requires_grad):
            model, view = TestSelectionCacheLifetime.make_view(layout, requires_grad=requires_grad, device=device)
            model.joint_q.fill_(5.0)
            source_ref = weakref.ref(model.joint_q)
            model_ref, view_ref = weakref.ref(model), weakref.ref(view)
            buffer = view.get_dof_positions(model)
            if requires_grad:
                model.joint_q.grad.fill_(7.0)
                gradient = view.get_dof_positions(model).grad
            byte_count = model.joint_q.capacity
            del model, view
            gc.collect()

            test.assertIsNone(view_ref())
            test.assertIsNone(model_ref())
            # reuse any freed allocation of the same size, so a dangling view would read 123
            churn = [wp.full(byte_count // 4, 123.0, dtype=float, device=device) for _ in range(64)]
            assert_np_equal(buffer.numpy(), np.full(buffer.shape, 5.0))
            if requires_grad:
                assert_np_equal(gradient.numpy(), np.full(gradient.shape, 7.0))
                del gradient
            if layout == "dense":
                test.assertIsNotNone(source_ref(), "returned zero-copy array does not keep its source alive")
            del buffer, churn
            gc.collect()
            test.assertIsNone(source_ref())


def test_getters_follow_replaced_source_arrays(test, device):
    """Read and write the current source arrays after they are replaced."""
    for layout in ("dense", "indexed", "sparse"):
        with test.subTest(layout=layout):
            model, view = TestSelectionCacheLifetime.make_view(layout, device=device)
            state, control = model.state(), model.control()
            first = view.get_dof_positions(state).numpy()
            view.get_root_transforms(state)
            view.get_dof_forces(control)
            old_q, old_f = state.joint_q, control.joint_f
            old_q_values, old_f_values = old_q.numpy().copy(), old_f.numpy().copy()

            state.joint_q = wp.full(old_q.shape, 4.0, dtype=old_q.dtype, device=device)
            control.joint_f = wp.full(old_f.shape, 6.0, dtype=old_f.dtype, device=device)
            q = view.get_dof_positions(state)
            assert_np_equal(q.numpy(), np.full(first.shape, 4.0))
            root = view.get_root_transforms(state).numpy()
            assert_np_equal(root, np.full(root.shape, 4.0))
            forces = view.get_dof_forces(control).numpy()
            assert_np_equal(forces, np.full(forces.shape, 6.0))

            view.set_dof_positions(state, np.full(q.shape, 9.0, dtype=np.float32))
            view.set_dof_forces(control, np.full(forces.shape, 8.0, dtype=np.float32))
            assert_np_equal(view.get_dof_positions(state).numpy(), np.full(q.shape, 9.0))
            assert_np_equal(view.get_dof_forces(control).numpy(), np.full(forces.shape, 8.0))
            # unselected coordinates (indexed joints, unselected worlds) keep the replacement's value
            test.assertEqual(set(np.unique(state.joint_q.numpy())), {9.0} if layout == "dense" else {4.0, 9.0})
            assert_np_equal(old_q.numpy(), old_q_values)
            assert_np_equal(old_f.numpy(), old_f_values)


def test_getters_follow_replaced_gradient(test, device):
    """Bind the current gradient after a source array's gradient is replaced."""
    model, view = TestSelectionCacheLifetime.make_view("dense", requires_grad=True, device=device)
    old_grad = model.joint_q.grad
    test.assertEqual(view.get_dof_positions(model).grad.ptr, old_grad.ptr)

    model.joint_q.grad = wp.full_like(model.joint_q, 3.0)
    gradient = view.get_dof_positions(model).grad
    test.assertEqual(gradient.ptr, model.joint_q.grad.ptr)
    assert_np_equal(gradient.numpy(), np.full(gradient.shape, 3.0))


for _test in (
    test_view_holds_sources_and_actuators_weakly,
    test_returned_array_survives_allocation_reuse,
    test_getters_follow_replaced_source_arrays,
    test_getters_follow_replaced_gradient,
):
    add_function_test(TestSelectionCacheLifetime, _test.__name__, _test, devices=get_test_devices())


class TestSelection(unittest.TestCase):
    def test_compiled_regex_selectors(self):
        builder = newton.ModelBuilder()
        articulation_labels = [
            "/World/envs/env_0/Robot_A",
            "/World/envs/env_0/Robot_B",
            "/World/envs/env_0/Robot_C",
            "/World/envs/env_0/Prop",
        ]
        for label in articulation_labels:
            base = builder.add_link(label=f"{label}/base")
            left_foot = builder.add_link(label=f"{label}/LF_FOOT")
            right_foot = builder.add_link(label=f"{label}/RF_FOOT")
            fixed_mount = builder.add_joint_free(child=base, label=f"{label}/fixed_mount")
            left_hip = builder.add_joint_revolute(parent=base, child=left_foot, label=f"{label}/LF_HIP")
            right_hip = builder.add_joint_revolute(parent=base, child=right_foot, label=f"{label}/RF_HIP")
            builder.add_articulation([fixed_mount, left_hip, right_hip], label=label)
        model = builder.finalize(device="cpu")

        view = ArticulationView(
            model,
            pattern=re.compile(r"/World/envs/env_[0-9]+/Robot_(A|B|C)"),
            include_links=re.compile(r"(LF|RF)_FOOT"),
            exclude_joints=re.compile(r"fixed_.*"),
        )

        assert_np_equal(view.articulation_ids.numpy(), [[0, 1, 2]])
        self.assertEqual(view.link_names, ["LF_FOOT", "RF_FOOT"])
        self.assertEqual(view.joint_names, ["LF_HIP", "RF_HIP"])
        self.assertEqual(view.link_count, 2)
        self.assertEqual(view.joint_count, 2)

        with self.assertRaisesRegex(KeyError, "No articulations matching pattern"):
            ArticulationView(model, pattern=re.compile(r"/World/envs/env_[0-9]+/Robot_Z"))

    def test_articulation_selector_lists(self):
        builder = newton.ModelBuilder()
        for label in ["robot_a", "robot_b", "prop"]:
            body = builder.add_link(label=f"{label}/body")
            joint = builder.add_joint_free(child=body, label=f"{label}/joint")
            builder.add_articulation([joint], label=label)
        model = builder.finalize()

        pattern_view = ArticulationView(model, pattern=["robot_*", "prop"])
        assert_np_equal(pattern_view.articulation_ids.numpy(), [[0, 1, 2]])

        index_view = ArticulationView(model, pattern=[0, 2])
        assert_np_equal(index_view.articulation_ids.numpy(), [[0, 2]])

        with self.assertRaisesRegex(ValueError, "must be unique and in ascending order"):
            ArticulationView(model, pattern=[2, 0])
        with self.assertRaisesRegex(ValueError, "must be unique and in ascending order"):
            ArticulationView(model, pattern=[0, 0])
        with self.assertRaisesRegex(ValueError, r"must be in range \[0, 3\)"):
            ArticulationView(model, pattern=[3])

        # each articulation has a single joint and link
        with self.assertRaisesRegex(ValueError, r"must be in range \[0, 1\)"):
            ArticulationView(model, pattern="robot_a", include_joints=[1])
        with self.assertRaisesRegex(ValueError, r"must be in range \[0, 1\)"):
            ArticulationView(model, pattern="robot_a", include_links=[1])

    def test_no_match(self):
        builder = newton.ModelBuilder()
        builder.add_body()
        model = builder.finalize()
        self.assertRaises(KeyError, ArticulationView, model, pattern="no_match")

    def test_sparse_world_articulation_view(self):
        """A view compacts matching heterogeneous worlds and gathers/scatters by absolute row."""

        def make_world(label: str, link_count: int):
            world = newton.ModelBuilder()
            parent = world.add_link(label=f"{label}/root")
            joints = [world.add_joint_free(child=parent, label=f"{label}/root_joint")]
            for index in range(1, link_count):
                child = world.add_link(label=f"{label}/link_{index}")
                joints.append(
                    world.add_joint_revolute(
                        parent=parent,
                        child=child,
                        axis=wp.vec3(0.0, 0.0, 1.0),
                        label=f"{label}/joint_{index}",
                    )
                )
                parent = child
            world.add_articulation(joints, label=label)
            return world

        robot_a = make_world("robot_a", 2)
        robot_b = make_world("robot_b", 4)
        scene = newton.ModelBuilder()
        for world in (robot_a, robot_b, robot_a, robot_b, robot_b, robot_a):
            scene.add_world(world)
        model = scene.finalize(device="cpu")

        view = ArticulationView(model, "robot_a", verbose=False)
        self.assertEqual(view.count, 3)
        self.assertEqual(view.world_count, 3)
        self.assertEqual(view.count_per_world, 1)
        self.assertTrue(view.is_sparse)
        self.assertTrue(view.uses_explicit_model_indices)
        assert_np_equal(view.world_ids.numpy(), [0, 2, 5])
        assert_np_equal(view.articulation_ids.numpy(), [[0], [2], [5]])

        q = np.arange(model.joint_coord_count, dtype=np.float32)
        model.joint_q.assign(q)
        dof_layout = view.frequency_layouts[newton.Model.AttributeFrequency.JOINT_COORD]
        absolute_q_indices = dof_layout.get_model_indices().numpy()
        actual = view.get_dof_positions(model).numpy()
        expected = q[absolute_q_indices]
        assert_np_equal(actual, expected)

        # A compact world mask addresses view worlds, while writes land in the correct
        # non-contiguous model rows and leave both the second match and other robot types alone.
        replacement = np.full(actual.shape, -7.0, dtype=np.float32)
        view.set_dof_positions(model, replacement, mask=[True, False, False])
        updated_q = model.joint_q.numpy()
        assert_np_equal(updated_q[absolute_q_indices[0]], replacement[0])
        assert_np_equal(updated_q[absolute_q_indices[1]], expected[1])
        untouched = np.ones(model.joint_coord_count, dtype=bool)
        untouched[absolute_q_indices[0].reshape(-1)] = False
        assert_np_equal(updated_q[untouched], q[untouched])

        expected_model_mask = np.array([True, False, True, False, False, True])
        assert_np_equal(view.get_model_articulation_mask().numpy(), expected_model_mask)

        # Selecting unlike topology under one view remains illegal; heterogeneity is
        # represented by separate sparse views, one per articulation type.
        with self.assertRaisesRegex(ValueError, "Articulations are not identical"):
            ArticulationView(model, "robot_*", verbose=False)

        # Alternating heterogeneous worlds are sparse by world ID but retain a regular
        # entity-row stride. Keep this common case on the direct strided binding path.
        regular_scene = newton.ModelBuilder()
        for world in (robot_a, robot_b, robot_a, robot_b, robot_a):
            regular_scene.add_world(world)
        regular_model = regular_scene.finalize(device="cpu")
        regular_view = ArticulationView(regular_model, "robot_a", verbose=False)
        self.assertTrue(regular_view.is_sparse)
        self.assertFalse(regular_view.uses_explicit_model_indices)
        assert_np_equal(regular_view.world_ids.numpy(), [0, 2, 4])

    def test_unsorted_include_indices_rejected(self):
        builder = newton.ModelBuilder()
        root = builder.add_link(label="root")
        middle = builder.add_link(label="middle")
        tip = builder.add_link(label="tip")
        root_joint = builder.add_joint_free(child=root, label="root_joint")
        middle_joint = builder.add_joint_revolute(parent=root, child=middle, label="middle_joint")
        tip_joint = builder.add_joint_revolute(parent=middle, child=tip, label="tip_joint")
        builder.add_articulation([root_joint, middle_joint, tip_joint], label="robot")
        model = builder.finalize()

        with self.assertRaisesRegex(ValueError, r"include_joints.*ascending order"):
            ArticulationView(model, "robot", include_joints=[2, 0])
        with self.assertRaisesRegex(ValueError, r"include_links.*ascending order"):
            ArticulationView(model, "robot", include_links=[2, 0])

        joint_view = ArticulationView(model, "robot", include_joints=[0, 2])
        self.assertEqual(joint_view.joint_names, ["root_joint", "tip_joint"])
        link_view = ArticulationView(model, "robot", include_links=[0, 2])
        self.assertEqual(link_view.link_names, ["root", "tip"])

    def test_empty_selection(self):
        builder = newton.ModelBuilder()
        body = builder.add_link()
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint], label="my_articulation")
        model = builder.finalize()
        control = model.control()
        selection = ArticulationView(model, pattern="my_articulation", exclude_joint_types=[newton.JointType.FREE])
        self.assertEqual(selection.count, 1)
        self.assertEqual(selection.get_root_transforms(model).shape, (1, 1))
        self.assertEqual(selection.get_dof_positions(model).shape, (1, 1, 0))
        self.assertEqual(selection.get_dof_velocities(model).shape, (1, 1, 0))
        self.assertEqual(selection.get_dof_forces(control).shape, (1, 1, 0))

    def test_fixed_joint_only_articulation(self):
        """Regression test for issue #920: ArticulationView with only fixed joints."""
        builder = newton.ModelBuilder()
        parent = builder.add_link()
        child = builder.add_link()
        j0 = builder.add_joint_fixed(parent=-1, child=parent)
        j1 = builder.add_joint_fixed(parent=parent, child=child)
        builder.add_articulation([j0, j1], label="fixed_only")
        model = builder.finalize()
        state = model.state()
        control = model.control()
        view = ArticulationView(model, pattern="fixed_only")
        self.assertEqual(view.count, 1)
        self.assertEqual(view.joint_dof_count, 0)
        self.assertEqual(view.joint_coord_count, 0)
        self.assertEqual(view.get_root_transforms(model).shape, (1, 1))
        self.assertEqual(view.get_dof_positions(state).shape, (1, 1, 0))
        self.assertEqual(view.get_dof_velocities(state).shape, (1, 1, 0))
        self.assertEqual(view.get_dof_forces(control).shape, (1, 1, 0))

    def test_root_base_classification_uses_dof_count(self):
        """Classify zero-DOF roots as fixed while preserving floating roots."""
        cases = (
            ("fixed", True, False),
            ("locked_d6", True, False),
            ("free", False, True),
        )

        for root_kind, expected_fixed, expected_floating in cases:
            with self.subTest(root_kind=root_kind):
                builder = newton.ModelBuilder()
                root = builder.add_link(label="root")

                if root_kind == "fixed":
                    root_joint = builder.add_joint_fixed(parent=-1, child=root)
                elif root_kind == "locked_d6":
                    root_joint = builder.add_joint_d6(parent=-1, child=root)
                else:
                    root_joint = builder.add_joint_free(parent=-1, child=root)

                builder.add_articulation([root_joint], label=root_kind)
                model = builder.finalize(device="cpu")
                view = ArticulationView(model, root_kind)

                self.assertEqual(view.is_fixed_base, expected_fixed)
                self.assertEqual(view.is_floating_base, expected_floating)

    def test_labels_preserve_full_paths(self):
        """Two-finger gripper whose distal bodies, finger joints, and tip
        shapes each share a colliding leaf name. ``*_names`` attributes
        collapse to the leaf; ``*_labels`` attributes expose the
        full slash-delimited labels from the template articulation so
        callers can still distinguish entries and recover selection order.
        """
        builder = newton.ModelBuilder()
        palm = builder.add_link(label="palm")
        left = builder.add_link(label="palm/left/fingertip")
        right = builder.add_link(label="palm/right/fingertip")
        builder.add_shape_box(body=left, hx=0.01, hy=0.01, hz=0.02, label="palm/left/tip_collision")
        builder.add_shape_box(body=right, hx=0.01, hy=0.01, hz=0.02, label="palm/right/tip_collision")
        j_root = builder.add_joint_free(parent=-1, child=palm, label="root")
        j_left = builder.add_joint_revolute(
            parent=palm, child=left, axis=(0.0, 0.0, 1.0), label="palm/left/fingertip_joint"
        )
        j_right = builder.add_joint_revolute(
            parent=palm, child=right, axis=(0.0, 0.0, 1.0), label="palm/right/fingertip_joint"
        )
        builder.add_articulation([j_root, j_left, j_right], label="gripper")
        model = builder.finalize()

        view = ArticulationView(model, "gripper", include_links="fingertip")

        # Leaf collisions are visible on the *_names attributes...
        self.assertEqual(view.link_count, 2)
        self.assertEqual(view.link_names, ["fingertip", "fingertip"])
        self.assertEqual(view.shape_names, ["tip_collision", "tip_collision"])

        # ...and disambiguated on the *_labels attributes.
        self.assertEqual(
            view.link_labels,
            ["palm/left/fingertip", "palm/right/fingertip"],
        )
        self.assertEqual(
            view.shape_labels,
            ["palm/left/tip_collision", "palm/right/tip_collision"],
        )
        self.assertIn("palm/left/fingertip_joint", view.joint_labels)
        self.assertIn("palm/right/fingertip_joint", view.joint_labels)
        self.assertEqual(len(view.joint_labels), view.joint_count)
        self.assertEqual(view.body_labels, view.link_labels)

    def test_duplicate_joint_child_is_one_link(self):
        """BODY-frequency link axis uses unique physical bodies, not joint slots."""
        builder = newton.ModelBuilder()
        root = builder.add_link(label="root")
        tip = builder.add_link(label="tip")
        builder.add_shape_box(body=tip, hx=0.01, hy=0.01, hz=0.01, label="tip_shape")

        j_root = builder.add_joint_free(parent=-1, child=root, label="root_joint")
        j_tip = builder.add_joint_revolute(parent=root, child=tip, axis=wp.vec3(0.0, 0.0, 1.0), label="tip_joint")
        with self.assertWarnsRegex(UserWarning, "undefined semantics"):
            j_tip_duplicate = builder.add_joint_fixed(parent=root, child=tip, label="tip_duplicate_joint")
        builder.add_articulation([j_root, j_tip, j_tip_duplicate], label="robot")
        model = builder.finalize()

        view = ArticulationView(model, "robot")

        self.assertEqual(list(model.body_label), ["root", "tip"])
        self.assertEqual(view.link_count, 2)
        self.assertEqual(view.link_names, ["root", "tip"])
        self.assertEqual(view.link_labels, ["root", "tip"])
        self.assertEqual(view.shape_count, 1)
        self.assertEqual(view.shape_labels, ["tip_shape"])

        body_layout = view.frequency_layouts[newton.Model.AttributeFrequency.BODY]
        self.assertEqual(body_layout.value_count, len(model.body_label))
        self.assertEqual(view.get_link_transforms(model).shape, (1, 1, 2))
        self.assertEqual(view.get_link_velocities(model).shape, (1, 1, 2))

    def _test_selection_shapes(self, floating: bool):
        # load articulation
        ant = newton.ModelBuilder()
        ant.add_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            ignore_names=["floor", "ground"],
            floating=floating,
        )

        L = 9  # num links
        J = 9  # num joints
        S = 13  # num shapes

        if floating:
            D = 14  # num joint dofs
            C = 15  # num joint coords
        else:
            D = 8  # num joint dofs
            C = 8  # num joint coords

        # scene with just one ant
        single_ant_model = ant.finalize()

        single_ant_view = ArticulationView(single_ant_model, "ant")
        self.assertEqual(single_ant_view.count, 1)
        self.assertEqual(single_ant_view.world_count, 1)
        self.assertEqual(single_ant_view.count_per_world, 1)
        self.assertEqual(single_ant_view.get_root_transforms(single_ant_model).shape, (1, 1))
        if floating:
            self.assertEqual(single_ant_view.get_root_velocities(single_ant_model).shape, (1, 1))
        else:
            self.assertIsNone(single_ant_view.get_root_velocities(single_ant_model))
        self.assertEqual(single_ant_view.get_link_transforms(single_ant_model).shape, (1, 1, L))
        self.assertEqual(single_ant_view.get_link_velocities(single_ant_model).shape, (1, 1, L))
        self.assertEqual(single_ant_view.get_dof_positions(single_ant_model).shape, (1, 1, C))
        self.assertEqual(single_ant_view.get_dof_velocities(single_ant_model).shape, (1, 1, D))
        self.assertEqual(single_ant_view.get_attribute("body_mass", single_ant_model).shape, (1, 1, L))
        self.assertEqual(single_ant_view.get_attribute("joint_type", single_ant_model).shape, (1, 1, J))
        self.assertEqual(single_ant_view.get_attribute("joint_dof_dim", single_ant_model).shape, (1, 1, J, 2))
        self.assertEqual(single_ant_view.get_attribute("joint_limit_ke", single_ant_model).shape, (1, 1, D))
        self.assertEqual(single_ant_view.get_attribute("shape_margin", single_ant_model).shape, (1, 1, S))

        W = 10  # num worlds

        # scene with one ant per world
        single_ant_per_world_scene = newton.ModelBuilder()
        single_ant_per_world_scene.replicate(ant, world_count=W)
        single_ant_per_world_model = single_ant_per_world_scene.finalize()

        single_ant_per_world_view = ArticulationView(single_ant_per_world_model, "ant")
        self.assertEqual(single_ant_per_world_view.count, W)
        self.assertEqual(single_ant_per_world_view.world_count, W)
        self.assertEqual(single_ant_per_world_view.count_per_world, 1)
        self.assertEqual(single_ant_per_world_view.get_root_transforms(single_ant_per_world_model).shape, (W, 1))
        if floating:
            self.assertEqual(single_ant_per_world_view.get_root_velocities(single_ant_per_world_model).shape, (W, 1))
        else:
            self.assertIsNone(single_ant_per_world_view.get_root_velocities(single_ant_per_world_model))
        self.assertEqual(single_ant_per_world_view.get_link_transforms(single_ant_per_world_model).shape, (W, 1, L))
        self.assertEqual(single_ant_per_world_view.get_link_velocities(single_ant_per_world_model).shape, (W, 1, L))
        self.assertEqual(single_ant_per_world_view.get_dof_positions(single_ant_per_world_model).shape, (W, 1, C))
        self.assertEqual(single_ant_per_world_view.get_dof_velocities(single_ant_per_world_model).shape, (W, 1, D))
        self.assertEqual(
            single_ant_per_world_view.get_attribute("body_mass", single_ant_per_world_model).shape, (W, 1, L)
        )
        self.assertEqual(
            single_ant_per_world_view.get_attribute("joint_type", single_ant_per_world_model).shape, (W, 1, J)
        )
        self.assertEqual(
            single_ant_per_world_view.get_attribute("joint_dof_dim", single_ant_per_world_model).shape, (W, 1, J, 2)
        )
        self.assertEqual(
            single_ant_per_world_view.get_attribute("joint_limit_ke", single_ant_per_world_model).shape, (W, 1, D)
        )
        self.assertEqual(
            single_ant_per_world_view.get_attribute("shape_margin", single_ant_per_world_model).shape, (W, 1, S)
        )

        A = 3  # num articulations per world

        # scene with multiple ants per world
        multi_ant_world = newton.ModelBuilder()
        for i in range(A):
            multi_ant_world.add_builder(ant, xform=wp.transform((0.0, 0.0, 1.0 + i), wp.quat_identity()))
        multi_ant_per_world_scene = newton.ModelBuilder()
        multi_ant_per_world_scene.replicate(multi_ant_world, world_count=W)
        multi_ant_per_world_model = multi_ant_per_world_scene.finalize()

        multi_ant_per_world_view = ArticulationView(multi_ant_per_world_model, "ant")
        self.assertEqual(multi_ant_per_world_view.count, W * A)
        self.assertEqual(multi_ant_per_world_view.world_count, W)
        self.assertEqual(multi_ant_per_world_view.count_per_world, A)
        self.assertEqual(multi_ant_per_world_view.get_root_transforms(multi_ant_per_world_model).shape, (W, A))
        if floating:
            self.assertEqual(multi_ant_per_world_view.get_root_velocities(multi_ant_per_world_model).shape, (W, A))
        else:
            self.assertIsNone(multi_ant_per_world_view.get_root_velocities(multi_ant_per_world_model))
        self.assertEqual(multi_ant_per_world_view.get_link_transforms(multi_ant_per_world_model).shape, (W, A, L))
        self.assertEqual(multi_ant_per_world_view.get_link_velocities(multi_ant_per_world_model).shape, (W, A, L))
        self.assertEqual(multi_ant_per_world_view.get_dof_positions(multi_ant_per_world_model).shape, (W, A, C))
        self.assertEqual(multi_ant_per_world_view.get_dof_velocities(multi_ant_per_world_model).shape, (W, A, D))
        self.assertEqual(
            multi_ant_per_world_view.get_attribute("body_mass", multi_ant_per_world_model).shape, (W, A, L)
        )
        self.assertEqual(
            multi_ant_per_world_view.get_attribute("joint_type", multi_ant_per_world_model).shape, (W, A, J)
        )
        self.assertEqual(
            multi_ant_per_world_view.get_attribute("joint_dof_dim", multi_ant_per_world_model).shape, (W, A, J, 2)
        )
        self.assertEqual(
            multi_ant_per_world_view.get_attribute("joint_limit_ke", multi_ant_per_world_model).shape, (W, A, D)
        )
        self.assertEqual(
            multi_ant_per_world_view.get_attribute("shape_margin", multi_ant_per_world_model).shape, (W, A, S)
        )

    def test_selection_shapes_floating_base(self):
        self._test_selection_shapes(floating=True)

    def test_selection_shapes_fixed_base(self):
        self._test_selection_shapes(floating=False)

    def test_selection_shape_values_noncontiguous(self):
        """Test that shape attribute values are correct when shape selection is non-contiguous."""
        # Build a 3-link chain: base -> link1 -> link2
        # Each link has one shape with a distinct margin value
        robot = newton.ModelBuilder()

        margins = [0.001, 0.002, 0.003]

        base = robot.add_link(xform=wp.transform([0, 0, 0], wp.quat_identity()), mass=1.0, label="base")
        robot.add_shape_box(
            base,
            hx=0.1,
            hy=0.1,
            hz=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(margin=margins[0]),
            label="shape_base",
        )

        link1 = robot.add_link(xform=wp.transform([0, 0, 0.5], wp.quat_identity()), mass=0.5, label="link1")
        robot.add_shape_capsule(
            link1,
            radius=0.05,
            half_height=0.2,
            cfg=newton.ModelBuilder.ShapeConfig(margin=margins[1]),
            label="shape_link1",
        )

        link2 = robot.add_link(xform=wp.transform([0, 0, 1.0], wp.quat_identity()), mass=0.3, label="link2")
        robot.add_shape_sphere(
            link2,
            radius=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(margin=margins[2]),
            label="shape_link2",
        )

        j0 = robot.add_joint_free(child=base)
        j1 = robot.add_joint_revolute(parent=base, child=link1, axis=[0, 1, 0])
        j2 = robot.add_joint_revolute(parent=link1, child=link2, axis=[0, 1, 0])
        robot.add_articulation([j0, j1, j2], label="robot")

        W = 3
        scene = newton.ModelBuilder()
        # add a ground plane first so shape indices are offset
        scene.add_shape_plane()
        scene.replicate(robot, world_count=W)
        model = scene.finalize()

        # exclude the middle link to make shape indices non-contiguous: [0, 2]
        view = ArticulationView(model, "robot", exclude_links=["link1"])
        self.assertFalse(view.shapes_contiguous, "Expected non-contiguous shape selection")
        self.assertEqual(view.shape_count, 2)

        # read shape_margin through ArticulationView and check values
        vals = view.get_attribute("shape_margin", model)
        self.assertEqual(vals.shape, (W, 1, 2))
        vals_np = vals.numpy()

        expected = [margins[0], margins[2]]  # base and link2 (link1 excluded)
        for w in range(W):
            for s, expected_margin in enumerate(expected):
                self.assertAlmostEqual(
                    float(vals_np[w, 0, s]),
                    expected_margin,
                    places=6,
                    msg=f"world={w}, shape={s}",
                )

    def test_eval_fk_translated_joint_chain_uses_view_mask(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)

        def add_translated_chain(label: str, x_offset: float):
            base = builder.add_link()
            slider = builder.add_link()

            builder.body_com[base] = wp.vec3(0.2, 0.0, 0.0)
            builder.body_com[slider] = wp.vec3(0.35, 0.0, -0.1)

            j0 = builder.add_joint_revolute(
                parent=-1,
                child=base,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(x_offset, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            )
            j1 = builder.add_joint_prismatic(
                parent=base,
                child=slider,
                axis=newton.Axis.X,
                parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.4), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.2, 0.0, -0.15), wp.quat_identity()),
            )
            builder.add_articulation([j0, j1], label=label)
            return base, slider, j0, j1

        target_base, target_slider, target_j0, target_j1 = add_translated_chain("translated_target", 0.0)
        other_base, other_slider, other_j0, other_j1 = add_translated_chain("translated_other", 5.0)

        model = builder.finalize()
        view = ArticulationView(model, "translated_target")

        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()

        q = model.joint_q.numpy().copy()
        qd = model.joint_qd.numpy().copy()

        q[q_start[target_j0]] = 0.55
        q[q_start[target_j1]] = 0.8
        qd[qd_start[target_j0]] = 1.1
        qd[qd_start[target_j1]] = -0.35

        q[q_start[other_j0]] = -0.3
        q[q_start[other_j1]] = 0.25
        qd[qd_start[other_j0]] = -0.7
        qd[qd_start[other_j1]] = 0.45

        dt = 1.0e-4
        q_next = q.copy()
        q_next[q_start[target_j0]] += qd[qd_start[target_j0]] * dt
        q_next[q_start[target_j1]] += qd[qd_start[target_j1]] * dt
        q_next[q_start[other_j0]] += qd[qd_start[other_j0]] * dt
        q_next[q_start[other_j1]] += qd[qd_start[other_j1]] * dt

        state = model.state()
        state_next = model.state()

        sentinel_q = state.body_q.numpy().copy()
        sentinel_q[:, :3] = -99.0
        sentinel_q[:, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        sentinel_qd = np.full_like(state.body_qd.numpy(), -77.0)

        state.body_q.assign(sentinel_q)
        state.body_qd.assign(sentinel_qd)
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        view.eval_fk(state)

        state_next.body_q.assign(sentinel_q)
        state_next.body_qd.assign(sentinel_qd)
        state_next.joint_q.assign(q_next)
        state_next.joint_qd.assign(qd)
        view.eval_fk(state_next)

        body_q = state.body_q.numpy().reshape(-1, 7)
        body_q_next = state_next.body_q.numpy().reshape(-1, 7)
        body_qd = state.body_qd.numpy().reshape(-1, 6)

        origin_vel_fd = (body_q_next[target_slider, :3] - body_q[target_slider, :3]) / dt
        origin_vel_from_body_qd = origin_velocity_from_body_qd(model, body_q, body_qd, target_slider)

        assert_np_equal(origin_vel_fd, origin_vel_from_body_qd, tol=5.0e-3)
        self.assertFalse(np.array_equal(body_q[target_base], sentinel_q[target_base]))
        assert_np_equal(body_q[other_base], sentinel_q[other_base], tol=0.0)
        assert_np_equal(body_q[other_slider], sentinel_q[other_slider], tol=0.0)
        assert_np_equal(body_qd[other_base], sentinel_qd[other_base], tol=0.0)
        assert_np_equal(body_qd[other_slider], sentinel_qd[other_slider], tol=0.0)

    def test_selection_mask(self):
        # load articulation
        ant = newton.ModelBuilder()
        ant.add_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            ignore_names=["floor", "ground"],
        )

        world_count = 4
        num_per_world = 3
        num_artis = world_count * num_per_world

        # scene with multiple ants per world
        world = newton.ModelBuilder()
        for i in range(num_per_world):
            world.add_builder(ant, xform=wp.transform((0.0, 0.0, 1.0 + i), wp.quat_identity()))
        scene = newton.ModelBuilder()
        scene.replicate(world, world_count=world_count)
        model = scene.finalize()

        view = ArticulationView(model, "ant")

        # test default mask
        model_mask = view.get_model_articulation_mask()
        expected = np.full(num_artis, 1, dtype=bool)
        assert_np_equal(model_mask.numpy(), expected)

        # test per-world mask
        model_mask = view.get_model_articulation_mask(mask=[0, 1, 1, 0])
        expected = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
        assert_np_equal(model_mask.numpy(), expected)

        world_mask = wp.array([0, 1, 1, 0], dtype=wp.bool, device=view.device)
        model_mask = view.get_model_articulation_mask(mask=world_mask)
        assert_np_equal(model_mask.numpy(), expected)

        # test world-arti mask
        m = [
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
        ]
        model_mask = view.get_model_articulation_mask(mask=m)
        expected = np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
        assert_np_equal(model_mask.numpy(), expected)

        world_articulation_mask = wp.array(m, dtype=wp.bool, device=view.device)
        model_mask = view.get_model_articulation_mask(mask=world_articulation_mask)
        assert_np_equal(model_mask.numpy(), expected)

    def test_selection_mask_rejects_invalid_warp_arrays(self):
        builder = newton.ModelBuilder()
        body = builder.add_link()
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint], label="robot")
        model = builder.finalize()
        view = ArticulationView(model, "robot")

        invalid_masks = (
            wp.empty(0, dtype=wp.bool, device=view.device),
            wp.ones(2, dtype=wp.bool, device=view.device),
            wp.ones((1, 2), dtype=wp.bool, device=view.device),
            wp.ones((1, 1, 1), dtype=wp.bool, device=view.device),
            wp.ones(1, dtype=wp.int32, device=view.device),
        )
        for mask in invalid_masks:
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                with mock.patch.object(wp, "launch") as launch:
                    with self.assertRaisesRegex(ValueError, "Boolean mask"):
                        view.get_model_articulation_mask(mask)
                    launch.assert_not_called()

        if wp.is_cuda_available():
            other_device = "cpu" if view.device.is_cuda else "cuda:0"
            mask = wp.ones(1, dtype=wp.bool, device=other_device)
            with self.subTest(device=mask.device):
                with mock.patch.object(wp, "launch") as launch:
                    with self.assertRaisesRegex(ValueError, "device"):
                        view.get_model_articulation_mask(mask)
                    launch.assert_not_called()

    def run_test_joint_selection(self, use_mask: bool, use_multiple_artics_per_view: bool):
        """Test an ArticulationView that includes a subset of joints and that we
        can write attributes to the subset of joints with and without a mask. Test
        that we can write to model/state/control."""
        mjcf = """<?xml version="1.0" ?>
<mujoco model="myart">
    <worldbody>
    <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>

      <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      </body>

      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      </body>

      <!-- Third child link with prismatic joint along x -->
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
        num_joints = num_joints_per_articulation * num_articulations_per_world * num_worlds

        # Create a single articulation with 3 joints.
        single_articuation_builder = newton.ModelBuilder()
        single_articuation_builder.add_mjcf(mjcf, ignore_inertial_definitions=False)

        # Create a world with 2 articulations
        single_world_builder = newton.ModelBuilder()
        for _i in range(0, num_articulations_per_world):
            single_world_builder.add_builder(single_articuation_builder)

        # Customise the articulation keys in single_world_builder
        single_world_builder.articulation_label[1] = "art1"
        if use_multiple_artics_per_view:
            single_world_builder.articulation_label[0] = "art1"
        else:
            single_world_builder.articulation_label[0] = "art0"

        # Create 3 worlds with two articulations per world and 3 joints per articulation.
        builder = newton.ModelBuilder()
        for _i in range(0, num_worlds):
            builder.add_world(single_world_builder)

        # Create the model
        model = builder.finalize()
        state_0 = model.state()
        control = model.control()

        # Create a view of "art1/joint3"
        joints_to_include = ["joint3"]
        joint_view = ArticulationView(model, "art1", include_joints=joints_to_include)

        # Get the attributes associated with "joint3"
        joint_dof_positions = joint_view.get_dof_positions(model).numpy().copy()
        joint_limit_lower = joint_view.get_attribute("joint_limit_lower", model).numpy().copy()
        joint_target_pos = joint_view.get_attribute("joint_target_q", model).numpy().copy()

        # Modify the attributes associated with "joint3"
        val = 1.0
        for world_idx in range(joint_dof_positions.shape[0]):
            for arti_idx in range(joint_dof_positions.shape[1]):
                for joint_idx in range(joint_dof_positions.shape[2]):
                    joint_dof_positions[world_idx, arti_idx, joint_idx] = val
                    joint_limit_lower[world_idx, arti_idx, joint_idx] += val
                    joint_target_pos[world_idx, arti_idx, joint_idx] += 2.0 * val
                    val += 1.0

        mask = None
        if use_mask:
            if use_multiple_artics_per_view:
                mask = wp.array([[False, False], [False, True], [False, False]], dtype=bool, device=model.device)
            else:
                mask = wp.array([[False], [True], [False]], dtype=bool, device=model.device)

        expected_dof_positions = []
        expected_joint_limit_lower = []
        expected_joint_target_pos = []
        if use_mask:
            if use_multiple_artics_per_view:
                expected_dof_positions = [
                    0.0,  # world0/artic0
                    0.0,
                    0.0,
                    0.0,  # world0/artic1
                    0.0,
                    0.0,
                    0.0,  # world1/artic0
                    0.0,
                    0.0,
                    0.0,  # world1/artic1
                    0.0,
                    4.0,
                    0.0,  # world2/artic0
                    0.0,
                    0.0,
                    0.0,  # world2/artic1
                    0.0,
                    0.0,
                ]
                expected_joint_limit_lower = [
                    -50.5,  # world0/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world0/artic1
                    -50.5,
                    -50.5,
                    -50.5,  # world1/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world1/artic1
                    -50.5,
                    -46.5,
                    -50.5,  # world2/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world2/artic1
                    -50.5,
                    -50.5,
                ]
                expected_joint_target_pos = [
                    0.0,  # world0/artic0
                    0.0,
                    0.0,
                    0.0,  # world0/artic1
                    0.0,
                    0.0,
                    0.0,  # world1/artic0
                    0.0,
                    0.0,
                    0.0,  # world1/artic1
                    0.0,
                    8.0,
                    0.0,  # world2/artic0
                    0.0,
                    0.0,
                    0.0,  # world2/artic1
                    0.0,
                    0.0,
                ]
            else:
                expected_dof_positions = [
                    0.0,  # world0/artic0
                    0.0,
                    0.0,
                    0.0,  # world0/artic1
                    0.0,
                    0.0,
                    0.0,  # world1/artic0
                    0.0,
                    0.0,
                    0.0,  # world1/artic1
                    0.0,
                    2.0,
                    0.0,  # world2/artic0
                    0.0,
                    0.0,
                    0.0,  # world2/artic1
                    0.0,
                    0.0,
                ]
                expected_joint_limit_lower = [
                    -50.5,  # world0/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world0/artic1
                    -50.5,
                    -50.5,
                    -50.5,  # world1/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world1/artic1
                    -50.5,
                    -48.5,
                    -50.5,  # world2/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world2/artic1
                    -50.5,
                    -50.5,
                ]
                expected_joint_target_pos = [
                    0.0,  # world0/artic0
                    0.0,
                    0.0,
                    0.0,  # world0/artic1
                    0.0,
                    0.0,
                    0.0,  # world1/artic0
                    0.0,
                    0.0,
                    0.0,  # world1/artic1
                    0.0,
                    4.0,
                    0.0,  # world2/artic0
                    0.0,
                    0.0,
                    0.0,  # world2/artic1
                    0.0,
                    0.0,
                ]
        else:
            if use_multiple_artics_per_view:
                expected_dof_positions = [
                    0.0,  # world0/artic0
                    0.0,
                    1.0,
                    0.0,  # world0/artic1
                    0.0,
                    2.0,
                    0.0,  # world1/artic0
                    0.0,
                    3.0,
                    0.0,  # world1/artic1
                    0.0,
                    4.0,
                    0.0,  # world2/artic0
                    0.0,
                    5.0,
                    0.0,  # world2/artic1
                    0.0,
                    6.0,
                ]
                expected_joint_limit_lower = [
                    -50.5,  # world0/artic0
                    -50.5,
                    -49.5,
                    -50.5,  # world0/artic1
                    -50.5,
                    -48.5,
                    -50.5,  # world1/artic0
                    -50.5,
                    -47.5,
                    -50.5,  # world1/artic1
                    -50.5,
                    -46.5,
                    -50.5,  # world2/artic0
                    -50.5,
                    -45.5,
                    -50.5,  # world2/artic1
                    -50.5,
                    -44.5,
                ]
                expected_joint_target_pos = [
                    0.0,  # world0/artic0
                    0.0,
                    2.0,
                    0.0,  # world0/artic1
                    0.0,
                    4.0,
                    0.0,  # world1/artic0
                    0.0,
                    6.0,
                    0.0,  # world1/artic1
                    0.0,
                    8.0,
                    0.0,  # world2/artic0
                    0.0,
                    10.0,
                    0.0,  # world2/artic1
                    0.0,
                    12.0,
                ]
            else:
                expected_dof_positions = [
                    0.0,  # world0/artic0
                    0.0,
                    0.0,
                    0.0,  # world0/artic1
                    0.0,
                    1.0,
                    0.0,  # world1/artic0
                    0.0,
                    0.0,
                    0.0,  # world1/artic1
                    0.0,
                    2.0,
                    0.0,  # world2/artic0
                    0.0,
                    0.0,
                    0.0,  # world2/artic1
                    0.0,
                    3.0,
                ]
                expected_joint_limit_lower = [
                    -50.5,  # world0/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world0/artic1
                    -50.5,
                    -49.5,
                    -50.5,  # world1/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world1/artic1
                    -50.5,
                    -48.5,
                    -50.5,  # world2/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world2/artic1
                    -50.5,
                    -47.5,
                ]
                expected_joint_target_pos = [
                    0.0,  # world0/artic0
                    0.0,
                    0.0,
                    0.0,  # world0/artic1
                    0.0,
                    2.0,
                    0.0,  # world1/artic0
                    0.0,
                    0.0,
                    0.0,  # world1/artic1
                    0.0,
                    4.0,
                    0.0,  # world2/artic0
                    0.0,
                    0.0,
                    0.0,  # world2/artic1
                    0.0,
                    6.0,
                ]

        # Set the values associated with "joint3"
        wp_joint_dof_positions = wp.array(joint_dof_positions, dtype=float, device=model.device)
        wp_joint_limit_lowers = wp.array(joint_limit_lower, dtype=float, device=model.device)
        wp_joint_target_pos = wp.array(joint_target_pos, dtype=float, device=model.device)
        joint_view.set_dof_positions(state_0, wp_joint_dof_positions, mask)
        joint_view.set_dof_positions(model, wp_joint_dof_positions, mask)
        joint_view.set_attribute("joint_limit_lower", model, wp_joint_limit_lowers, mask)
        joint_view.set_attribute("joint_target_q", control, wp_joint_target_pos, mask)
        joint_view.set_attribute("joint_target_q", model, wp_joint_target_pos, mask)

        # Get the updated values from model, state, control.
        measured_state_joint_dof_positions = state_0.joint_q.numpy()
        measured_model_joint_dof_positions = model.joint_q.numpy()
        measured_model_joint_limit_lower = model.joint_limit_lower.numpy()
        measured_control_joint_target_pos = control.joint_target_q.numpy()
        measured_model_joint_target_pos = model.joint_target_q.numpy()

        # Test that the modified values were correctly set in model, state and control
        for i in range(0, num_joints):
            measured = measured_state_joint_dof_positions[i]
            expected = expected_dof_positions[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected state joint dof position value {i}: {expected}, Measured value: {measured}",
            )

            measured = measured_model_joint_dof_positions[i]
            expected = expected_dof_positions[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected model joint dof position value {i}: {expected}, Measured value: {measured}",
            )

            measured = measured_model_joint_limit_lower[i]
            expected = expected_joint_limit_lower[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected model joint limit lower value {i}: {expected}, Measured value: {measured}",
            )

            measured = measured_control_joint_target_pos[i]
            expected = expected_joint_target_pos[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected control joint target pos value {i}: {expected}, Measured value: {measured}",
            )

            measured = measured_model_joint_target_pos[i]
            expected = expected_joint_target_pos[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected model joint target pos value {i}: {expected}, Measured value: {measured}",
            )

    def run_test_link_selection(self, use_mask: bool, use_multiple_artics_per_view: bool):
        """Test an ArticulationView that excludes a subset of links and that we
        can write attributes to the subset of links with and without a mask"""
        mjcf = """<?xml version="1.0" ?>
<mujoco model="myart">
    <worldbody>
    <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">
       <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>

          <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

      <!-- Third child link with prismatic joint along x -->
      <body name="link3" pos="-0.0 -0.9 0">
        <joint name="joint3" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
        num_links_per_articulation = 4
        num_articulations_per_world = 2
        num_worlds = 3
        num_links = num_links_per_articulation * num_articulations_per_world * num_worlds

        # Create a single articulation
        single_articulation_builder = newton.ModelBuilder()
        single_articulation_builder.add_mjcf(mjcf, ignore_inertial_definitions=False)

        # Create a world with 2 articulations
        single_world_builder = newton.ModelBuilder()
        for _i in range(0, num_articulations_per_world):
            single_world_builder.add_builder(single_articulation_builder)

        # Customise the articulation keys in single_world_builder
        single_world_builder.articulation_label[0] = "art0"
        if use_multiple_artics_per_view:
            single_world_builder.articulation_label[1] = "art0"
        else:
            single_world_builder.articulation_label[1] = "art1"

        # Create 3 worlds with 2 articulations per world and 4 links per articulation.
        builder = newton.ModelBuilder()
        for _i in range(0, num_worlds):
            builder.add_world(single_world_builder)

        # Create the model
        model = builder.finalize()
        state_0 = model.state()

        # create a view of art0/"link1" and art0/"link2" by excluding "root" and "link3"
        links_to_exclude = ["root", "link3"]
        link_view = ArticulationView(model, "art0", exclude_links=links_to_exclude)

        # Get the attributes associated with "art0/link1" and "art0/link2"
        link_masses = link_view.get_attribute("body_mass", model).numpy().copy()
        link_vels = link_view.get_attribute("body_qd", model).numpy().copy()

        # Modify the attributes associated with "art0/link1" and "art0/link2"
        val = 1.0
        for world_idx in range(link_masses.shape[0]):
            for arti_idx in range(link_masses.shape[1]):
                for link_idx in range(link_masses.shape[2]):
                    link_masses[world_idx, arti_idx, link_idx] += val
                    link_vels[world_idx, arti_idx, link_idx] = [val, val, val, val, val, val]
                    val += 1.0

        mask = None
        if use_mask:
            if use_multiple_artics_per_view:
                mask = wp.array([[False, False], [False, True], [False, False]], dtype=bool, device=model.device)
            else:
                mask = wp.array([[False], [True], [False]], dtype=bool, device=model.device)

        wp_link_masses = wp.array(link_masses, dtype=float, device=model.device)
        wp_link_vels = wp.array(link_vels, dtype=float, device=model.device)
        link_view.set_attribute("body_mass", model, wp_link_masses, mask)
        link_view.set_attribute("body_qd", model, wp_link_vels, mask)
        link_view.set_attribute("body_qd", state_0, wp_link_vels, mask)

        expected_body_masses = []
        expected_body_vels = []
        if use_mask:
            if use_multiple_artics_per_view:
                expected_body_masses = [
                    1.0,  # world0/artic0
                    1.0,
                    1.0,
                    1.0,
                    1.0,  # world0/artic1
                    1.0,
                    1.0,
                    1.0,
                    1.0,  # world1/artic0
                    1.0,
                    1.0,
                    1.0,
                    1.0,  # world1/artic1
                    8.0,
                    9.0,
                    1.0,
                    1.0,  # world2/artic0
                    1.0,
                    1.0,
                    1.0,
                    1.0,  # world2/artic1
                    1.0,
                    1.0,
                    1.0,
                ]
                expected_body_vels = [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/root
                    [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],  # world1/artic1/link1
                    [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],  # world1/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link3
                ]
            else:
                expected_body_masses = [
                    1.0,  # world0/artic0
                    1.0,
                    1.0,
                    1.0,
                    1.0,  # world0/artic1
                    1.0,
                    1.0,
                    1.0,
                    1.0,  # world1/artic0
                    4.0,
                    5.0,
                    1.0,
                    1.0,  # world1/artic1
                    1.0,
                    1.0,
                    1.0,
                    1.0,  # world2/artic0
                    1.0,
                    1.0,
                    1.0,
                    1.0,  # world2/artic1
                    1.0,
                    1.0,
                    1.0,
                ]
                expected_body_vels = [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/root
                    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],  # world1/artic0/link1
                    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],  # world1/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link3
                ]
        else:
            if use_multiple_artics_per_view:
                expected_body_masses = [
                    1.0,  # world0/artic0
                    2.0,
                    3.0,
                    1.0,
                    1.0,  # world0/artic1
                    4.0,
                    5.0,
                    1.0,
                    1.0,  # world1/artic0
                    6.0,
                    7.0,
                    1.0,
                    1.0,  # world1/artic1
                    8.0,
                    9.0,
                    1.0,
                    1.0,  # world2/artic0
                    10.0,
                    11.0,
                    1.0,
                    1.0,  # world2/artic1
                    12.0,
                    13.0,
                    1.0,
                ]
                expected_body_vels = [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/root
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # world0/artic0/link1
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],  # world0/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/root
                    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],  # world0/artic1/link1
                    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],  # world0/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/root
                    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],  # world1/artic0/link1
                    [6.0, 6.0, 6.0, 6.0, 6.0, 6.0],  # world1/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/root
                    [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],  # world1/artic1/link1
                    [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],  # world1/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/root
                    [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],  # world2/artic0/link1
                    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],  # world2/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/root
                    [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],  # world2/artic1/link1
                    [12.0, 12.0, 12.0, 12.0, 12.0, 12.0],  # world2/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link3
                ]
            else:
                expected_body_masses = [
                    1.0,  # world0/artic0
                    2.0,
                    3.0,
                    1.0,
                    1.0,  # world0/artic1
                    1.0,
                    1.0,
                    1.0,
                    1.0,  # world1/artic0
                    4.0,
                    5.0,
                    1.0,
                    1.0,  # world1/artic1
                    1.0,
                    1.0,
                    1.0,
                    1.0,  # world2/artic0
                    6.0,
                    7.0,
                    1.0,
                    1.0,  # world2/artic1
                    1.0,
                    1.0,
                    1.0,
                ]
                expected_body_vels = [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/root
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # world0/artic0/link1
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],  # world0/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/root
                    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],  # world1/artic0/link1
                    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],  # world1/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/root
                    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],  # world2/artic0/link1
                    [6.0, 6.0, 6.0, 6.0, 6.0, 6.0],  # world2/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link3
                ]

        # Get the updated body masses
        measured_body_masses = model.body_mass.numpy()
        measured_model_body_vels = model.body_qd.numpy()
        measured_state_body_vels = state_0.body_qd.numpy()

        # Test that the modified values were correctly set in model
        for i in range(0, num_links):
            measured = measured_body_masses[i]
            expected = expected_body_masses[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected body mass value {i}: {expected}, Measured value: {measured}",
            )

            for j in range(0, 6):
                measured = measured_model_body_vels[i][j]
                expected = expected_body_vels[i][j]
                self.assertAlmostEqual(
                    expected,
                    measured,
                    places=4,
                    msg=f"Expected body velocity value {i}: {expected}, Measured value: {measured}",
                )

            for j in range(0, 6):
                measured = measured_state_body_vels[i][j]
                expected = expected_body_vels[i][j]
                self.assertAlmostEqual(
                    expected,
                    measured,
                    places=4,
                    msg=f"Expected body velocity value {i}: {expected}, Measured value: {measured}",
                )

    def test_joint_selection_one_per_view_no_mask(self):
        self.run_test_joint_selection(use_mask=False, use_multiple_artics_per_view=False)

    def test_joint_selection_two_per_view_no_mask(self):
        self.run_test_joint_selection(use_mask=False, use_multiple_artics_per_view=True)

    def test_joint_selection_one_per_view_with_mask(self):
        self.run_test_joint_selection(use_mask=True, use_multiple_artics_per_view=False)

    def test_joint_selection_two_per_view_with_mask(self):
        self.run_test_joint_selection(use_mask=True, use_multiple_artics_per_view=True)

    def test_link_selection_one_per_view_no_mask(self):
        self.run_test_link_selection(use_mask=False, use_multiple_artics_per_view=False)

    def test_link_selection_two_per_view_no_mask(self):
        self.run_test_link_selection(use_mask=False, use_multiple_artics_per_view=True)

    def test_link_selection_one_per_view_with_mask(self):
        self.run_test_link_selection(use_mask=True, use_multiple_artics_per_view=False)

    def test_link_selection_two_per_view_with_mask(self):
        self.run_test_link_selection(use_mask=True, use_multiple_artics_per_view=True)

    def test_get_attribute_extended_state(self):
        """Test that get_attribute works for extended state attributes."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.request_state_attributes("body_qdd", "body_parent_f", "mujoco:qfrc_actuator")

        link = builder.add_link()
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(
            -1,
            link,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()),
            axis=wp.vec3(0, 1, 0),
        )
        builder.add_articulation([joint], label="art")
        model = builder.finalize()
        state = model.state()

        view = ArticulationView(model, "art")

        # body_qdd and body_parent_f should be retrievable via get_attribute on state
        body_qdd = view.get_attribute("body_qdd", state)
        self.assertEqual(body_qdd.shape[2], 1)  # 1 link

        body_parent_f = view.get_attribute("body_parent_f", state)
        self.assertEqual(body_parent_f.shape[2], 1)  # 1 link

        qfrc_actuator = view.get_attribute("mujoco.qfrc_actuator", state)
        self.assertEqual(qfrc_actuator.shape[2], 1)  # 1 revolute DOF

    def test_loop_closing_joint_selection_is_opt_in(self):
        """ArticulationView excludes loop-closing joints unless requested."""
        builder = newton.ModelBuilder()
        root = builder.add_link(label="root")
        middle = builder.add_link(label="middle")
        tip = builder.add_link(label="tip")
        j_root = builder.add_joint_revolute(-1, root, label="root_joint")
        j_middle = builder.add_joint_revolute(root, middle, label="middle_joint")
        j_tip = builder.add_joint_revolute(middle, tip, label="tip_joint")
        builder.add_articulation([j_root, j_middle, j_tip], label="robot")
        builder.add_joint_ball(tip, root, label="loop_joint")

        model = builder.finalize()
        np.testing.assert_array_equal(model.articulation_start.numpy(), np.array([0, 4], dtype=np.int32))
        np.testing.assert_array_equal(model.articulation_end.numpy(), np.array([3], dtype=np.int32))

        view = ArticulationView(model, "robot")
        self.assertEqual(view.joint_names, ["root_joint", "middle_joint", "tip_joint"])

        view_with_loop = ArticulationView(model, "robot", include_loop_closing_joints=True)
        self.assertEqual(view_with_loop.joint_names, ["root_joint", "middle_joint", "tip_joint", "loop_joint"])


def _make_robot_world(label: str, link_count: int):
    """One floating-base chain articulation with ``link_count`` links (6 + link_count - 1 DOFs)."""
    world = newton.ModelBuilder()
    parent = world.add_link(label=f"{label}/root")
    joints = [world.add_joint_free(child=parent, label=f"{label}/root_joint")]
    for index in range(1, link_count):
        child = world.add_link(label=f"{label}/link_{index}")
        joints.append(
            world.add_joint_revolute(
                parent=parent,
                child=child,
                axis=wp.vec3(0.0, 0.0, 1.0),
                label=f"{label}/joint_{index}",
            )
        )
        parent = child
    world.add_articulation(joints, label=label)
    return world


def _make_sparse_robot_model(layout: str, device):
    """Worlds of a 7-DOF robot_a and a 9-DOF robot_b in a regular, irregular or dense order."""
    robot_a = _make_robot_world("robot_a", 2)
    robot_b = _make_robot_world("robot_b", 4)
    if layout == "regular":
        worlds = (robot_a, robot_b, robot_a, robot_b, robot_a)
    elif layout == "irregular":
        worlds = (robot_a, robot_b, robot_a, robot_b, robot_b, robot_a)
    else:
        worlds = (robot_a, robot_a, robot_a)
    scene = newton.ModelBuilder()
    for world in worlds:
        scene.add_world(world)
    return scene.finalize(device=device)


def _articulation_dofs(model, articulation_ids):
    """Absolute model DOF indices ``[world, dof]`` of single-articulation-per-world selections."""
    articulation_start = model.articulation_start.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    rows = []
    for articulation in articulation_ids:
        first_joint = articulation_start[articulation]
        end_joint = articulation_start[articulation + 1]
        rows.append(list(range(joint_qd_start[first_joint], joint_qd_start[end_joint])))
    return np.array(rows, dtype=np.int64)


def _reversed_row_actuator(model, device):
    """One actuator row per model DOF, listed in reverse global DOF order, with distinct gains.

    Returns ``(actuator, kp, row_of_dof)``. The actuator row differs from the DOF index, so a DOF/row
    mix-up is visible, and the rows of the last world come first, so the actuator does not own an
    equal consecutive block of rows per world in world order.
    """
    dof_count = model.joint_dof_count
    kp = np.arange(dof_count, dtype=np.float32) * 10.0 + 1.0
    actuator = Actuator(
        indices=wp.array(np.arange(dof_count)[::-1].copy(), dtype=wp.uint32, device=device),
        drive=DrivePD(kp=wp.array(kp, device=device), kd=wp.zeros(dof_count, device=device)),
    )
    row_of_dof = np.empty(dof_count, dtype=np.int64)
    row_of_dof[np.arange(dof_count)[::-1]] = np.arange(dof_count)
    return actuator, kp, row_of_dof


def _check_actuator_rows(test, device, cases):
    for layout, expected_sparse, expected_explicit in cases:
        with test.subTest(layout=layout):
            model = _make_sparse_robot_model(layout, device)
            view = ArticulationView(model, "robot_a", verbose=False)
            test.assertEqual(view.is_sparse, expected_sparse)
            test.assertEqual(view.uses_explicit_model_indices, expected_explicit)
            articulation_ids = view.articulation_ids.numpy().reshape(-1)
            selected_dofs = _articulation_dofs(model, articulation_ids)
            test.assertEqual(selected_dofs.shape, (3, 7))

            actuator, kp, row_of_dof = _reversed_row_actuator(model, device)
            actual = view.get_actuator_parameter(actuator, actuator.drive, "kp").numpy()
            assert_np_equal(actual, kp[row_of_dof[selected_dofs]])

            replacement = -(np.arange(actual.size, dtype=np.float32).reshape(actual.shape) + 1.0)
            view.set_actuator_parameter(
                actuator,
                actuator.drive,
                "kp",
                replacement,
                mask=wp.array([True, False, True], dtype=bool, device=device),
            )
            updated = actuator.drive.kp.numpy()
            expected = kp.copy()
            expected[row_of_dof[selected_dofs[0]]] = replacement[0]
            expected[row_of_dof[selected_dofs[2]]] = replacement[2]
            assert_np_equal(updated, expected)


def test_sparse_world_actuator_parameters_use_selected_rows(test, device):
    """Actuator gather/scatter through sparse views addresses the selected articulations' rows."""
    _check_actuator_rows(test, device, (("regular", True, False), ("irregular", True, True)))


def test_dense_view_reversed_actuator_order(test, device):
    """A dense view reads the right rows when the actuator lists them in reverse global DOF order.

    The view is not sparse, but the actuator does not own an equal, consecutive block of rows per
    world in world order, so replicating the first world's pattern would address the wrong rows.
    """
    _check_actuator_rows(test, device, (("dense", False, False),))


def test_sparse_world_partial_actuator_parameters(test, device):
    """DOFs without an actuator read as zero and are never written through a sparse view."""
    for layout in ("regular", "irregular"):
        with test.subTest(layout=layout):
            model = _make_sparse_robot_model(layout, device)
            view = ArticulationView(model, "robot_a", verbose=False)
            selected_dofs = _articulation_dofs(model, view.articulation_ids.numpy().reshape(-1))
            # Actuate only the revolute DOFs of every articulation in the model (skip free roots).
            joint_type = model.joint_type.numpy()
            joint_qd_start = model.joint_qd_start.numpy()
            actuated = [
                int(joint_qd_start[j]) for j in range(model.joint_count) if joint_type[j] == newton.JointType.REVOLUTE
            ]
            kp = np.arange(len(actuated), dtype=np.float32) + 100.0
            actuator = Actuator(
                indices=wp.array(actuated, dtype=wp.uint32, device=device),
                drive=DrivePD(kp=wp.array(kp, device=device), kd=wp.zeros(len(actuated), device=device)),
            )
            row_of_dof = {dof: row for row, dof in enumerate(actuated)}
            expected = np.array(
                [[kp[row_of_dof[d]] if d in row_of_dof else 0.0 for d in world] for world in selected_dofs],
                dtype=np.float32,
            )
            assert_np_equal(view.get_actuator_parameter(actuator, actuator.drive, "kp").numpy(), expected)

            view.set_actuator_parameter(actuator, actuator.drive, "kp", np.full(expected.shape, -1.0, np.float32))
            updated = actuator.drive.kp.numpy()
            written = {row_of_dof[d] for world in selected_dofs for d in world if d in row_of_dof}
            for row, value in enumerate(updated):
                test.assertEqual(value, -1.0 if row in written else kp[row])


def test_sparse_world_root_transforms_and_velocities(test, device):
    """Floating-base root access through sparse views gathers and scatters the selected roots."""
    for layout in ("regular", "irregular"):
        with test.subTest(layout=layout):
            model = _make_sparse_robot_model(layout, device)
            view = ArticulationView(model, "robot_a", verbose=False)
            test.assertTrue(view.is_floating_base)
            articulation_ids = view.articulation_ids.numpy().reshape(-1)
            articulation_start = model.articulation_start.numpy()
            root_joints = articulation_start[articulation_ids]
            q_start = model.joint_q_start.numpy()[root_joints]
            qd_start = model.joint_qd_start.numpy()[root_joints]

            state = model.state()
            q = np.arange(model.joint_coord_count, dtype=np.float32)
            qd = np.arange(model.joint_dof_count, dtype=np.float32) + 0.5
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)

            transforms = view.get_root_transforms(state).numpy().reshape(3, 7)
            velocities = view.get_root_velocities(state).numpy().reshape(3, 6)
            assert_np_equal(transforms, np.stack([q[s : s + 7] for s in q_start]))
            assert_np_equal(velocities, np.stack([qd[s : s + 6] for s in qd_start]))
            # Repeated access reuses the cached index map.
            assert_np_equal(view.get_root_transforms(state).numpy().reshape(3, 7), transforms)

            new_transforms = np.tile(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], np.float32), (3, 1, 1))
            new_velocities = np.full((3, 1, 6), -2.0, np.float32)
            mask = wp.array([False, True, False], dtype=bool, device=device)
            view.set_root_transforms(state, new_transforms, mask=mask)
            view.set_root_velocities(state, new_velocities, mask=mask)
            expected_q = q.copy()
            expected_q[q_start[1] : q_start[1] + 7] = new_transforms[1, 0]
            expected_qd = qd.copy()
            expected_qd[qd_start[1] : qd_start[1] + 6] = new_velocities[1, 0]
            assert_np_equal(state.joint_q.numpy(), expected_q)
            assert_np_equal(state.joint_qd.numpy(), expected_qd)


def test_actuator_mapping_built_once_and_cached(test, device):
    """The first actuator-parameter access builds the DOF mapping; later gets and sets reuse it."""
    for layout in ("regular", "irregular", "dense"):
        with test.subTest(layout=layout):
            model = _make_sparse_robot_model(layout, device)
            view = ArticulationView(model, "robot_a", verbose=False)
            actuator, kp, row_of_dof = _reversed_row_actuator(model, device)
            selected_dofs = _articulation_dofs(model, view.articulation_ids.numpy().reshape(-1))
            with mock.patch.object(
                ArticulationView,
                "_create_actuator_dof_mapping",
                autospec=True,
                side_effect=ArticulationView._create_actuator_dof_mapping,
            ) as create:
                cold = view.get_actuator_parameter(actuator, actuator.drive, "kp").numpy()
                test.assertEqual(create.call_count, 1)
                mapping = view._actuator_dof_mapping_cache[actuator]
                warm = view.get_actuator_parameter(actuator, actuator.drive, "kp").numpy()
                view.set_actuator_parameter(actuator, actuator.drive, "kp", cold)
                view.get_actuator_parameter(actuator, actuator.drive, "kd")
                test.assertEqual(create.call_count, 1)
                test.assertIs(view._actuator_dof_mapping_cache[actuator], mapping)
            assert_np_equal(cold, kp[row_of_dof[selected_dofs]])
            assert_np_equal(warm, cold)


def _world_major_actuator(model, device):
    """One actuator row per model DOF, in global DOF order (world-major), with distinct gains."""
    dof_count = model.joint_dof_count
    kp = np.arange(dof_count, dtype=np.float32) * 10.0 + 1.0
    actuator = Actuator(
        indices=wp.array(np.arange(dof_count), dtype=wp.uint32, device=device),
        drive=DrivePD(kp=wp.array(kp, device=device), kd=wp.zeros(dof_count, device=device)),
    )
    return actuator, kp, np.arange(dof_count, dtype=np.int64)


@contextlib.contextmanager
def _no_host_transfers():
    """Fail on device-to-host readbacks and on arrays created from host data."""
    array_init = wp.array.__init__

    def init_without_host_data(self, data=None, *args, **kwargs):
        if data is not None:
            raise AssertionError("array created from host data during capture")
        array_init(self, None, *args, **kwargs)

    with (
        mock.patch.object(wp.array, "numpy", side_effect=AssertionError("host readback during capture")),
        mock.patch.object(wp.array, "__init__", init_without_host_data),
    ):
        yield


def test_actuator_parameters_capture_cold(test, device):
    """A first actuator access captures and replays: the mapping build runs on the device only.

    Kernels are compiled with a separate view first, so the captured views start with an empty
    mapping cache. Replays use the values present at launch time, and a second capture reuses
    the cached mapping.
    """
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture of allocations requires the mempool")
    cases = (
        ("dense", _world_major_actuator),
        ("dense", _reversed_row_actuator),
        ("regular", _reversed_row_actuator),
        ("irregular", _reversed_row_actuator),
    )
    for layout, make_actuator in cases:
        with test.subTest(layout=layout, actuator=make_actuator.__name__):
            model = _make_sparse_robot_model(layout, device)
            warm_view = ArticulationView(model, "robot_a", verbose=False)
            warm_actuator = make_actuator(model, device)[0]
            warm_view.set_actuator_parameter(warm_actuator, warm_actuator.drive, "kp", np.zeros((3, 7), np.float32))

            view = ArticulationView(model, "robot_a", verbose=False)
            actuator, kp, row_of_dof = make_actuator(model, device)
            selected_dofs = _articulation_dofs(model, view.articulation_ids.numpy().reshape(-1))
            test.assertEqual(len(view._actuator_dof_mapping_cache), 0)

            gathered = wp.zeros((3, 7), dtype=float, device=device)
            values = wp.zeros((3, 7), dtype=float, device=device)
            mask = wp.array([False, True, False], dtype=bool, device=device)
            for _ in range(2):  # cold capture, then a capture that reuses the cached mapping
                with _no_host_transfers():
                    with wp.ScopedCapture(device) as capture:
                        view.set_actuator_parameter(actuator, actuator.drive, "kp", values, mask=mask)
                        wp.copy(gathered, view.get_actuator_parameter(actuator, actuator.drive, "kp"))
                test.assertEqual(len(view._actuator_dof_mapping_cache), 1)
                for replay in range(2):
                    actuator.drive.kp.assign(kp)
                    new_values = -(np.arange(21, dtype=np.float32).reshape(3, 7) + 1.0 + 100.0 * replay)
                    values.assign(new_values)
                    wp.capture_launch(capture.graph)
                    expected_kp = kp.copy()
                    expected_kp[row_of_dof[selected_dofs[1]]] = new_values[1]
                    assert_np_equal(actuator.drive.kp.numpy(), expected_kp)
                    assert_np_equal(gathered.numpy(), expected_kp[row_of_dof[selected_dofs]])


def test_actuator_capture_without_mempool_raises(test, device):
    """Without the memory pool, a cold access inside capture raises a clear error and caches nothing."""
    model = _make_sparse_robot_model("dense", device)
    view = ArticulationView(model, "robot_a", verbose=False)
    actuator, kp, _row_of_dof = _world_major_actuator(model, device)
    view.get_actuator_parameter(actuator, actuator.drive, "kd")  # compile kernels on another mapping
    view._actuator_dof_mapping_cache.clear()
    was_enabled = wp.is_mempool_enabled(device)
    wp.set_mempool_enabled(device, False)
    try:
        with test.assertRaisesRegex(RuntimeError, "memory pool"):
            with wp.ScopedCapture(device):
                view.get_actuator_parameter(actuator, actuator.drive, "kp")
    finally:
        wp.set_mempool_enabled(device, was_enabled)
    test.assertEqual(len(view._actuator_dof_mapping_cache), 0)
    assert_np_equal(view.get_actuator_parameter(actuator, actuator.drive, "kp").numpy(), kp.reshape(3, 7))


class _BorrowedActuatorView:
    """A model-free view that borrows ``ArticulationView``'s actuator-parameter methods.

    Downstream callers that build actuators without a :class:`~newton.Model` (for example IsaacLab's
    PhysX-family actuator adapters) provide only the placement attributes those methods read, and
    store ``device`` as the alias string they were given rather than a :class:`warp.Device`.
    """

    def __init__(self, world_count: int, dof_count: int, device: str):
        self.world_count = world_count
        self.count_per_world = 1
        self.device = device
        self.full_mask = wp.ones(world_count, dtype=wp.bool, device=device)
        self._actuator_dof_mapping_cache = {}
        self.frequency_layouts = {
            newton.Model.AttributeFrequency.JOINT_DOF: FrequencyLayout(
                offset=0,
                stride_between_worlds=dof_count,
                stride_within_worlds=dof_count,
                value_count=dof_count,
                indices=list(range(dof_count)),
                device=device,
            )
        }

    get_actuator_parameter = ArticulationView.get_actuator_parameter
    set_actuator_parameter = ArticulationView.set_actuator_parameter
    _get_actuator_dof_mapping = ArticulationView._get_actuator_dof_mapping
    _create_actuator_dof_mapping = ArticulationView._create_actuator_dof_mapping
    _resolve_world_mask = ArticulationView._resolve_world_mask


def test_borrowed_view_with_device_alias(test, device):
    """Borrowed actuator-parameter methods accept a view whose ``device`` is an alias string.

    Covers eager get/set and, on CUDA, cold and warmed capture with replay.
    """
    alias = wp.get_device(device).alias
    test.assertIsInstance(alias, str)
    view = _BorrowedActuatorView(world_count=2, dof_count=3, device=alias)
    kp = np.arange(1.0, 7.0, dtype=np.float32)
    actuator = Actuator(
        indices=wp.array(np.arange(6), dtype=wp.uint32, device=alias),
        drive=DrivePD(kp=wp.array(kp, device=alias), kd=wp.zeros(6, device=alias)),
    )

    if wp.get_device(alias).is_cuda:
        if not wp.is_mempool_enabled(alias):
            test.skipTest("CUDA graph capture of allocations requires the mempool")
        # Compile kernels on another view, then capture the target view's first access.
        _BorrowedActuatorView(world_count=2, dof_count=3, device=alias).get_actuator_parameter(
            actuator, actuator.drive, "kd"
        )
        cold = wp.zeros((2, 3), dtype=float, device=alias)
        with _no_host_transfers():
            with wp.ScopedCapture(alias) as capture:
                wp.copy(cold, view.get_actuator_parameter(actuator, actuator.drive, "kp"))
        wp.capture_launch(capture.graph)
        assert_np_equal(cold.numpy(), kp.reshape(2, 3))

    assert_np_equal(view.get_actuator_parameter(actuator, actuator.drive, "kp").numpy(), kp.reshape(2, 3))
    view.set_actuator_parameter(
        actuator,
        actuator.drive,
        "kp",
        np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], np.float32),
        mask=wp.array([False, True], dtype=bool, device=alias),
    )
    expected = np.array([1.0, 2.0, 3.0, -4.0, -5.0, -6.0], np.float32)
    assert_np_equal(actuator.drive.kp.numpy(), expected)

    if not wp.get_device(alias).is_cuda:
        return
    # The calls above cached the mapping, so the same accesses capture and replay.
    gathered = wp.zeros((2, 3), dtype=float, device=alias)
    values = wp.zeros((2, 3), dtype=float, device=alias)
    mask = wp.array([True, False], dtype=bool, device=alias)
    with _no_host_transfers():
        with wp.ScopedCapture(alias) as capture:
            view.set_actuator_parameter(actuator, actuator.drive, "kp", values, mask=mask)
            wp.copy(gathered, view.get_actuator_parameter(actuator, actuator.drive, "kp"))
    values.assign(np.array([[10.0, 20.0, 30.0], [0.0, 0.0, 0.0]], np.float32))
    wp.capture_launch(capture.graph)
    expected[:3] = (10.0, 20.0, 30.0)
    assert_np_equal(actuator.drive.kp.numpy(), expected)
    assert_np_equal(gathered.numpy(), expected.reshape(2, 3))


class TestSelectionActuatorMapping(unittest.TestCase):
    pass


_devices = get_test_devices()
_cuda_devices = get_cuda_test_devices()
for _name, _func, _func_devices in (
    (
        "test_sparse_world_actuator_parameters_use_selected_rows",
        test_sparse_world_actuator_parameters_use_selected_rows,
        _devices,
    ),
    ("test_dense_view_reversed_actuator_order", test_dense_view_reversed_actuator_order, _devices),
    ("test_sparse_world_partial_actuator_parameters", test_sparse_world_partial_actuator_parameters, _devices),
    ("test_sparse_world_root_transforms_and_velocities", test_sparse_world_root_transforms_and_velocities, _devices),
    ("test_actuator_mapping_built_once_and_cached", test_actuator_mapping_built_once_and_cached, _devices),
    ("test_borrowed_view_with_device_alias", test_borrowed_view_with_device_alias, _devices),
    ("test_actuator_parameters_capture_cold", test_actuator_parameters_capture_cold, _cuda_devices),
    ("test_actuator_capture_without_mempool_raises", test_actuator_capture_without_mempool_raises, _cuda_devices),
):
    add_function_test(TestSelectionActuatorMapping, _name, _func, devices=_func_devices)


class TestSelectionFixedTendons(unittest.TestCase):
    """Tests for fixed tendon support in ArticulationView."""

    TENDON_MJCF = """<?xml version="1.0" ?>
<mujoco model="two_prismatic_links">
  <compiler angle="degree"/>
  <option timestep="0.002" gravity="0 0 0"/>

  <worldbody>
    <body name="root" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom type="cylinder" size="0.05 0.025" rgba="1 0 0 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom type="cylinder" size="0.05 0.025" rgba="0 0 1 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>

  <tendon>
    <fixed name="coupling_tendon" stiffness="2.0" damping="1.0" springlength="0.0">
      <joint joint="joint1" coef="1"/>
      <joint joint="joint2" coef="1"/>
    </fixed>
  </tendon>
</mujoco>
"""

    def test_tendon_count(self):
        """Test that tendon count is correctly detected."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_mjcf(self.TENDON_MJCF)
        model = builder.finalize()

        view = ArticulationView(model, "two_prismatic_links")
        self.assertEqual(view.tendon_count, 1)

    def test_tendon_selection_shapes(self):
        """Test that tendon selection API returns correct shapes."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_mjcf(self.TENDON_MJCF)
        model = builder.finalize()

        view = ArticulationView(model, "two_prismatic_links")
        T = 1  # num tendons

        # Test generic attribute access
        stiffness = view.get_attribute("mujoco.tendon_stiffness", model)
        self.assertEqual(stiffness.shape, (1, 1, T))

        damping = view.get_attribute("mujoco.tendon_damping", model)
        self.assertEqual(damping.shape, (1, 1, T))

        tendon_range = view.get_attribute("mujoco.tendon_range", model)
        self.assertEqual(tendon_range.shape, (1, 1, T))  # vec2 trailing dim

        tendon_coef = view.get_attribute("mujoco.tendon_coef", model)
        self.assertEqual(tendon_coef.shape, (1, 1, 2))

    def test_tendon_generic_api(self):
        """Test that tendon attributes are accessible via generic get/set_attribute."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_mjcf(self.TENDON_MJCF)
        model = builder.finalize()

        view = ArticulationView(model, "two_prismatic_links")
        T = 1

        # Test getters via generic API
        stiffness = view.get_attribute("mujoco.tendon_stiffness", model)
        self.assertEqual(stiffness.shape, (1, 1, T))
        assert_np_equal(stiffness.numpy(), np.array([[[2.0]]]))

        damping = view.get_attribute("mujoco.tendon_damping", model)
        self.assertEqual(damping.shape, (1, 1, T))
        assert_np_equal(damping.numpy(), np.array([[[1.0]]]))

        springlength = view.get_attribute("mujoco.tendon_springlength", model)
        self.assertEqual(springlength.shape, (1, 1, T))

        tendon_range = view.get_attribute("mujoco.tendon_range", model)
        self.assertEqual(tendon_range.shape, (1, 1, T))

        # Test setters via generic API
        view.set_attribute("mujoco.tendon_damping", model, np.array([[[2.5]]]))
        damping = view.get_attribute("mujoco.tendon_damping", model)
        assert_np_equal(damping.numpy(), np.array([[[2.5]]]))

    def test_tendon_multi_world(self):
        """Test that tendon selection works with multiple worlds."""
        individual_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        individual_builder.add_mjcf(self.TENDON_MJCF)

        W = 4  # num worlds
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        scene.replicate(individual_builder, world_count=W)
        model = scene.finalize()

        view = ArticulationView(model, "two_prismatic_links")
        T = 1

        self.assertEqual(view.world_count, W)
        self.assertEqual(view.count_per_world, 1)
        self.assertEqual(view.tendon_count, T)

        stiffness = view.get_attribute("mujoco.tendon_stiffness", model)
        self.assertEqual(stiffness.shape, (W, 1, T))

        # Verify values are correct across all worlds
        expected = np.full((W, 1, T), 2.0)
        assert_np_equal(stiffness.numpy(), expected)

    def test_tendon_set_values(self):
        """Test that setting tendon values works correctly."""
        individual_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        individual_builder.add_mjcf(self.TENDON_MJCF)

        W = 2  # num worlds
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        scene.replicate(individual_builder, world_count=W)
        model = scene.finalize()

        view = ArticulationView(model, "two_prismatic_links")

        # Set new stiffness values via generic API
        new_stiffness = np.array([[[5.0]], [[10.0]]])
        view.set_attribute("mujoco.tendon_stiffness", model, new_stiffness)

        # Verify values were set
        stiffness = view.get_attribute("mujoco.tendon_stiffness", model)
        assert_np_equal(stiffness.numpy(), new_stiffness)

    def test_tendon_names(self):
        """Test that tendon names are correctly populated."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_mjcf(self.TENDON_MJCF)
        model = builder.finalize()

        view = ArticulationView(model, "two_prismatic_links")

        # Check tendon_names is populated
        self.assertEqual(len(view.tendon_names), 1)
        self.assertEqual(view.tendon_names[0], "coupling_tendon")

        # Check that we can look up index from name
        idx = view.tendon_names.index("coupling_tendon")
        self.assertEqual(idx, 0)

    def test_no_tendons_in_articulation(self):
        """Test that articulations without tendons have tendon_count=0."""
        # Use nv_ant.xml which has no tendons
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            ignore_names=["floor", "ground"],
        )
        model = builder.finalize()

        view = ArticulationView(model, "ant")
        self.assertEqual(view.tendon_count, 0)
        self.assertEqual(len(view.tendon_names), 0)

    def test_no_tendons_but_model_has_tendons(self):
        """Test accessing tendon attributes on articulation without tendons when model has tendons elsewhere."""
        # Create a model with one articulation that has tendons and one without
        with_tendons_mjcf = self.TENDON_MJCF

        no_tendons_mjcf = """<?xml version="1.0" ?>
<mujoco model="no_tendons_robot">
  <compiler angle="degree"/>
  <option timestep="0.002" gravity="0 0 0"/>

  <worldbody>
    <body name="simple_robot" pos="0 0 0">
      <joint name="simple_joint" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_mjcf(with_tendons_mjcf)
        builder.add_mjcf(no_tendons_mjcf)
        model = builder.finalize()

        # Select the articulation without tendons
        view = ArticulationView(model, "no_tendons_robot")
        self.assertEqual(view.tendon_count, 0)

        # Attempting to access tendon attributes should raise an error
        # This tests line 969: no tendons found in the selected articulations
        with self.assertRaises(AttributeError) as ctx:
            view.get_attribute("mujoco.tendon_stiffness", model)
        self.assertIn("no rows were found", str(ctx.exception))

    def test_multiple_articulations_per_world(self):
        """Test tendon selection with multiple articulations in a single world."""
        # Build a single articulation with tendons
        individual_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        individual_builder.add_mjcf(self.TENDON_MJCF)

        # Create a world with multiple copies of the articulation
        A = 2  # articulations per world
        multi_robot_world = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for i in range(A):
            multi_robot_world.add_builder(
                individual_builder, xform=wp.transform((i * 2.0, 0.0, 0.0), wp.quat_identity())
            )

        # Replicate to multiple worlds
        W = 2  # num worlds
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        scene.replicate(multi_robot_world, world_count=W)
        model = scene.finalize()

        # Select all articulations
        view = ArticulationView(model, "two_prismatic_links")

        # Should have W worlds, A articulations per world, 1 tendon per articulation
        self.assertEqual(view.world_count, W)
        self.assertEqual(view.count_per_world, A)
        self.assertEqual(view.tendon_count, 1)

        # Test that we can read tendon attributes
        stiffness = view.get_attribute("mujoco.tendon_stiffness", model)
        self.assertEqual(stiffness.shape, (W, A, 1))

        # All stiffness values should be 2.0 (from TENDON_MJCF)
        expected = np.full((W, A, 1), 2.0)
        assert_np_equal(stiffness.numpy(), expected)


class TestSelectionMuJoCoActuators(unittest.TestCase):
    """Tests for MuJoCo actuator custom frequencies in ArticulationView."""

    ACTUATOR_MJCF = """
<mujoco model="actuated">
  <worldbody>
    <body name="link">
      <joint name="hinge" type="hinge"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="drive" joint="hinge"/>
  </actuator>
</mujoco>
"""

    def test_actuator_frequency_uses_declared_articulation_owner(self):
        """Expose MuJoCo actuator controls through their declared owner metadata."""
        robot = newton.ModelBuilder()
        robot.add_mjcf(self.ACTUATOR_MJCF)
        scene = newton.ModelBuilder()
        scene.replicate(robot, world_count=2)
        model = scene.finalize()
        control = model.control()

        view = ArticulationView(model, "actuated")
        self.assertEqual(view.custom_frequency_counts["mujoco:actuator"], 1)
        self.assertEqual(view.custom_frequency_labels["mujoco:actuator"], ["drive"])
        assert_np_equal(model.custom_frequency_articulation["mujoco:actuator"].numpy(), [0, 1])

        values = np.array([[[1.0]], [[2.0]]], dtype=np.float32)
        view.set_attribute("mujoco.ctrl", control, values)
        assert_np_equal(view.get_attribute("mujoco.ctrl", control).numpy(), values)

    def test_actuator_owners_survive_merge_followed_by_import(self):
        """Preserve remapped actuator owners when later imports append rows."""
        robot = newton.ModelBuilder()
        robot.add_mjcf(self.ACTUATOR_MJCF)

        world = newton.ModelBuilder()
        world.add_builder(robot, label_prefix="a")
        world.add_builder(robot, label_prefix="b")
        world.add_mjcf(self.ACTUATOR_MJCF)

        scene = newton.ModelBuilder()
        scene.replicate(world, world_count=2)
        model = scene.finalize()

        assert_np_equal(model.custom_frequency_articulation["mujoco:actuator"].numpy(), np.arange(6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
