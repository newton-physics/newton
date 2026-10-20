# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import gc
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

from newton import Mesh, Model, ModelBuilder, Rod, eval_fk
from newton._src.sim.builder import _ARRAY_BACKED_ATTRIBUTE_DTYPES, _materialize_array_backed_list
from newton.actuators import DrivePD
from newton.tests.unittest_utils import add_function_test, get_test_devices
from newton.utils import compute_world_offsets


class TestModelBuilderReplicate(unittest.TestCase):
    @staticmethod
    def _make_source() -> ModelBuilder:
        builder = ModelBuilder()

        particles = [
            builder.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(), 1.0),
            builder.add_particle(wp.vec3(1.0, 0.0, 0.0), wp.vec3(), 1.0),
            builder.add_particle(wp.vec3(0.0, 1.0, 0.0), wp.vec3(), 1.0),
            builder.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(), 1.0),
        ]
        builder.add_spring(particles[0], particles[1], 100.0, 1.0, 0.0)
        builder.add_triangle(*particles[:3])
        builder.add_tetrahedron(*particles)
        builder.add_edge(*particles)

        root = builder.add_link(
            xform=wp.transform((1.0, 2.0, 3.0), wp.quat_rpy(0.1, 0.2, 0.3)),
            label="root",
        )
        child = builder.add_link(xform=wp.transform((1.0, 2.0, 4.0), wp.quat_identity()), label="child")
        root_joint = builder.add_joint_free(
            parent=-1,
            child=root,
            parent_xform=wp.transform((0.2, 0.3, 0.4), wp.quat_rpy(0.2, 0.1, 0.0)),
            label="free",
        )
        child_joint = builder.add_joint_revolute(parent=root, child=child, axis=(0.0, 1.0, 0.0), label="hinge")
        builder.add_articulation([root_joint, child_joint], label="robot")

        fixed = builder.add_link(xform=wp.transform((0.5, 0.0, 0.0), wp.quat_identity()), label="fixed")
        fixed_joint = builder.add_joint_fixed(
            parent=-1,
            child=fixed,
            parent_xform=wp.transform((0.1, 0.2, 0.3), wp.quat_rpy(0.3, 0.2, 0.1)),
            label="fixed_joint",
        )
        builder.add_articulation([fixed_joint], label="fixed_articulation")

        root_shape = builder.add_shape_box(body=root, hx=0.2, hy=0.3, hz=0.4, label="box")
        child_shape = builder.add_shape_sphere(body=child, radius=0.2, label="ball")
        builder.add_shape_sphere(
            body=-1,
            radius=0.1,
            xform=wp.transform((2.0, 0.0, 0.0), wp.quat_identity()),
            label="static",
        )
        builder.add_shape_box(body=fixed, hx=0.1, hy=0.1, hz=0.1, label="fixed_shape")
        builder.add_shape_collision_filter_pair(root_shape, child_shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            builder.add_constraint_mimic(child_joint, root_joint, coef0=0.25, coef1=-1.0, label="mimic")

        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="thing", namespace="test"))
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ref_body",
                dtype=wp.int32,
                frequency="test:thing",
                namespace="test",
                references="body",
                default=-1,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="world",
                dtype=wp.int32,
                frequency="test:thing",
                namespace="test",
                references="world",
                default=-1,
            )
        )
        builder.add_custom_values(**{"test:ref_body": child, "test:world": -1})
        builder.add_actuator(
            DrivePD,
            index=builder.joint_qd_start[child_joint],
            pos_index=builder.joint_q_start[child_joint],
            kp=100.0,
            kd=10.0,
        )
        builder.add_muscle([root, child], [wp.vec3(), wp.vec3(0.0, 0.0, 0.5)], 1.0, 1.0, 1.0, 1.0, 0.0)
        builder.body_color_groups = [np.asarray([root, child, fixed])]
        # Uneven group sizes exercise the balancing in the merge combine.
        builder.set_coloring([[0], [1, 2, 3]])

        builder._record_curve_deformable_object("cable", (root, child + 1), (root_joint, child_joint + 1))
        builder._record_surface_deformable_object("cloth", (0, 3), (0, 1), (0, 1))
        builder._record_volume_deformable_object("soft", (0, 4), (0, 1))
        return builder

    @staticmethod
    def _make_destination() -> ModelBuilder:
        builder = ModelBuilder()
        body = builder.add_body(label="global")
        builder.add_shape_sphere(body, radius=0.1, label="global_shape")
        builder.begin_world()
        existing = builder.add_body(label="existing")
        builder.add_shape_box(existing, hx=0.1, hy=0.1, hz=0.1, label="existing_shape")
        builder.end_world()
        return builder

    def assert_builder_merge_state_equal(self, expected: ModelBuilder, actual: ModelBuilder) -> None:
        manually_verified = (
            "_raw_array_access_active",
            "_array_backed_attributes",
            "_shape_collision_filter_pairs",
            "actuator_entries",
            "custom_attributes",
        )
        for name in sorted(set(vars(expected)) | set(vars(actual))):
            if name in manually_verified:
                continue
            expected_value = getattr(expected, name)
            actual_value = getattr(actual, name)
            with self.subTest(attribute=name):
                if name.endswith("_color_groups"):
                    # Group lengths may differ, so element-wise np.asarray would fail.
                    self.assertEqual(len(expected_value), len(actual_value))
                    for expected_group, actual_group in zip(expected_value, actual_value, strict=True):
                        np.testing.assert_array_equal(expected_group, actual_group)
                elif isinstance(expected_value, list) or wp.types.type_is_vector(type(expected_value)):
                    np.testing.assert_array_equal(np.asarray(expected_value), np.asarray(actual_value))
                elif (
                    isinstance(expected_value, (bool, float, int, str, tuple, set, frozenset, dict))
                    or expected_value is None
                ):
                    self.assertEqual(expected_value, actual_value)
                elif hasattr(expected_value, "__dict__"):
                    self.assertEqual(vars(expected_value), vars(actual_value))
                else:
                    self.fail(f"unhandled attribute type {type(expected_value).__name__}; add an explicit comparison")

        self.assertEqual(expected.body_shapes, actual.body_shapes)
        self.assertEqual(expected.joint_parents, actual.joint_parents)
        self.assertEqual(expected.joint_children, actual.joint_children)
        self.assertEqual(tuple(expected.shape_collision_filter_pairs), tuple(actual.shape_collision_filter_pairs))

        self.assertEqual(set(expected.custom_attributes), set(actual.custom_attributes))
        for name, expected_attr in expected.custom_attributes.items():
            self.assertEqual(expected_attr.values, actual.custom_attributes[name].values)

        self.assertEqual(set(expected.actuator_entries), set(actual.actuator_entries))
        for key, expected_entry in expected.actuator_entries.items():
            actual_entry = actual.actuator_entries[key]
            for field in ("indices", "pos_indices", "drive_args", "delay_args", "clamping_args"):
                self.assertEqual(getattr(expected_entry, field), getattr(actual_entry, field))

    def test_replicate_matches_add_world_loop(self):
        """Match repeated world composition with replicated builder arrays."""
        world_count = 4
        source = self._make_source()
        for spacing in ((0.0, 0.0, 0.0), (2.0, 3.0, 0.0)):
            with self.subTest(spacing=spacing):
                expected = self._make_destination()
                for offset in compute_world_offsets(world_count, spacing, expected.up_axis):
                    expected.add_world(source, wp.transform(offset, wp.quat_identity()))

                actual = self._make_destination()
                actual.replicate(source, world_count, spacing)

                self.assert_builder_merge_state_equal(expected, actual)

    def test_composition_transforms_particle_state(self):
        """Transform copied particle positions and velocities without changing the prefix."""
        source = ModelBuilder()
        source.add_particle((1.0, 2.0, 3.0), (2.0, -1.0, 0.5), 1.0)
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        offset = np.array([3.0, -2.0, 1.0])
        xforms = [
            wp.transform(),
            wp.transform(offset),
            wp.transform(offset, wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2)),
        ]
        for method in ("add_builder", "replicate"):
            with self.subTest(method=method):
                target = ModelBuilder()
                target.add_particle((7.0, 8.0, 9.0), (4.0, 5.0, 6.0), 1.0)
                if method == "replicate":
                    target.replicate(source, len(xforms), xforms=xforms)
                else:
                    for xform in xforms:
                        target.add_builder(source, xform)
                for i, (rot, shift) in enumerate(
                    ((np.eye(3), np.zeros(3)), (np.eye(3), offset), (rotation, offset)), 1
                ):
                    with self.subTest(copy=i, field="position"):
                        np.testing.assert_allclose(
                            target.particle_q[i], rot @ np.array(source.particle_q[0]) + shift, atol=1e-6, rtol=0
                        )
                    with self.subTest(copy=i, field="velocity"):
                        np.testing.assert_allclose(
                            target.particle_qd[i], rot @ np.array(source.particle_qd[0]), atol=1e-6, rtol=0
                        )
                np.testing.assert_array_equal(target.particle_q[0], [7.0, 8.0, 9.0])
                np.testing.assert_array_equal(target.particle_qd[0], [4.0, 5.0, 6.0])
                np.testing.assert_array_equal(source.particle_q, [[1.0, 2.0, 3.0]])
                np.testing.assert_array_equal(source.particle_qd, [[2.0, -1.0, 0.5]])

    def test_add_builder_preserves_mixed_particle_velocity_types(self):
        """Preserve each velocity representation and native operations when composing particles."""
        velocities = [(1.0, 2.0, 3.0), [4.0, 5.0, 6.0], wp.vec3(7.0, 8.0, 9.0), np.array([10.0, 11.0, 12.0])]
        rotation = np.diag([-1.0, -1.0, 1.0])
        for first in range(len(velocities)):
            source = ModelBuilder()
            for velocity in velocities[first:] + velocities[:first]:
                source.add_particle((1.0, 2.0, 3.0), velocity, 1.0)
            original = np.asarray(source.particle_qd).copy()
            for xform, rot in (
                (None, np.eye(3)),
                (wp.transform((2.0, 3.0, 4.0)), np.eye(3)),
                (wp.transform((2.0, 3.0, 4.0), wp.quat(0.0, 0.0, 1.0, 0.0)), rotation),
            ):
                with self.subTest(first=first, xform=xform):
                    target = ModelBuilder()
                    target.add_particle((0.0, 0.0, 0.0), wp.vec3(-1.0, -2.0, -3.0), 1.0)
                    target.add_builder(source, xform)
                    for before, after in zip(source.particle_qd, target.particle_qd[1:], strict=True):
                        self.assertIs(type(after), type(before))
                        np.testing.assert_array_equal(after, rot @ np.asarray(before))
                        if isinstance(before, wp.vec3):
                            self.assertEqual(after.x, (rot @ np.asarray(before))[0])
                            np.testing.assert_array_equal(after * 0.5, 0.5 * (rot @ np.asarray(before)))
                    self.assertIsInstance(target.particle_qd[0], wp.vec3)
                    np.testing.assert_array_equal(target.particle_qd[0], [-1.0, -2.0, -3.0])
                    np.testing.assert_array_equal(source.particle_qd, original)
                    model = target.finalize(device="cpu")
                    np.testing.assert_array_equal(model.particle_qd.numpy()[1:], original @ rot.T)

                    nested = ModelBuilder()
                    nested.add_builder(target, wp.transform(q=wp.quat(0.0, 0.0, 1.0, 0.0)))
                    for before, after in zip(target.particle_qd, nested.particle_qd, strict=True):
                        self.assertIs(type(after), type(before))
                        np.testing.assert_array_equal(after, rotation @ np.asarray(before))

    def test_composition_transforms_free_state_and_targets(self):
        """Transform free-root state and targets in the parent frame in both target layouts."""
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        anchor_rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        anchor_position = np.array([0.25, -0.5, 1.0])
        offset = np.array([3.0, -2.0, 1.0])
        xforms = [
            wp.transform(),
            wp.transform(offset),
            wp.transform(offset, wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2)),
        ]
        for coord_layout in (False, True):
            with mock.patch("newton.use_coord_layout_targets", coord_layout), warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The legacy DOF-shaped joint_target_q layout is deprecated.*",
                    category=DeprecationWarning,
                )
                source = ModelBuilder()
                root = source.add_link(
                    xform=wp.transform((1.0, 2.0, 3.0)), mass=1.0, inertia=wp.mat33(np.eye(3)), com=(0.2, 0.1, 0.0)
                )
                free = source.add_joint_free(
                    root,
                    parent_xform=wp.transform(anchor_position, wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2)),
                )
                child = source.add_link(xform=wp.transform((1.0, 2.0, 4.0)), mass=1.0, inertia=wp.mat33(np.eye(3)))
                hinge = source.add_joint_revolute(root, child, target_pos=0.3, target_vel=0.4)
                source.add_articulation([free, hinge])
                source.joint_qd[:] = [1.0, 2.0, 0.5, 0.5, -1.0, 0.25, 0.7]
                target_position = np.array([2.0, 1.0, -0.5])
                target_angles = wp.vec3(0.2, -0.3, 0.4)
                target_quat = wp.quat_from_euler(target_angles, 2, 1, 0)
                source.joint_target_q[:7] = [
                    *target_position,
                    *(target_quat if coord_layout else [*target_angles, 1.0]),
                ]
                source.joint_target_qd[:6] = [2.0, -0.5, 1.0, -0.5, 0.25, 1.0]
                original_model = source.finalize(device="cpu")
                original_state = original_model.state()
                eval_fk(original_model, original_model.joint_q, original_model.joint_qd, original_state)
                source.body_qd = [wp.spatial_vector(*row) for row in original_state.body_qd.numpy()]
                source_state = {
                    name: np.asarray(getattr(source, name)).copy()
                    for name in ("body_qd", "joint_q", "joint_qd", "joint_target_q", "joint_target_qd")
                }
                for method in ("add_builder", "replicate"):
                    with self.subTest(layout=coord_layout, method=method):
                        target = ModelBuilder()
                        target.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
                        prefix_target = np.asarray(target.joint_target_q).copy()
                        if method == "replicate":
                            target.replicate(source, len(xforms), xforms=xforms)
                        else:
                            for xform in xforms:
                                target.add_builder(source, xform)
                        model = target.finalize(device="cpu")
                        control = model.control()
                        state = model.state()
                        fk_state = model.state()
                        eval_fk(model, model.joint_q, model.joint_qd, fk_state)
                        body_velocities = state.body_qd.numpy()
                        joint_velocities = state.joint_qd.numpy()
                        target_velocities = control.joint_target_qd.numpy()
                        target_positions = control.joint_target_q.numpy()
                        for i, (rot, shift) in enumerate(
                            ((np.eye(3), np.zeros(3)), (np.eye(3), offset), (rotation, offset))
                        ):
                            body_start, q_start, qd_start = 1 + 2 * i, 7 + 8 * i, 6 + 7 * i
                            local_rotation = anchor_rotation.T @ rot @ anchor_rotation
                            with self.subTest(copy=i, field="body_qd"):
                                expected = (np.asarray(source.body_qd).reshape(-1, 3) @ rot.T).reshape(2, 6)
                                np.testing.assert_allclose(
                                    body_velocities[body_start : body_start + 2], expected, atol=2e-6, rtol=0
                                )
                            for field, velocities in (
                                ("joint_qd", joint_velocities),
                                ("joint_target_qd", target_velocities),
                            ):
                                with self.subTest(copy=i, field=field):
                                    expected = (
                                        np.asarray(getattr(source, field)[:6]).reshape(2, 3) @ local_rotation.T
                                    ).ravel()
                                    np.testing.assert_allclose(
                                        velocities[qd_start : qd_start + 6], expected, atol=2e-6, rtol=0
                                    )
                            target_start = q_start if coord_layout else qd_start
                            values = target_positions[target_start : target_start + (7 if coord_layout else 6)]
                            with self.subTest(copy=i, field="target_position"):
                                world_position = anchor_rotation @ values[:3] + anchor_position
                                np.testing.assert_allclose(
                                    world_position,
                                    rot @ (anchor_rotation @ target_position + anchor_position) + shift,
                                    atol=2e-6,
                                    rtol=0,
                                )
                            with self.subTest(copy=i, field="target_orientation"):
                                quat = (
                                    wp.quat(*values[3:])
                                    if coord_layout
                                    else wp.quat_from_euler(wp.vec3(*values[3:]), 2, 1, 0)
                                )
                                np.testing.assert_allclose(
                                    anchor_rotation @ np.asarray(wp.quat_to_matrix(quat)).reshape(3, 3),
                                    rot @ anchor_rotation @ np.asarray(wp.quat_to_matrix(target_quat)).reshape(3, 3),
                                    atol=2e-6,
                                    rtol=0,
                                )
                            self.assertAlmostEqual(target.joint_qd[qd_start + 6], 0.7)
                            self.assertAlmostEqual(target.joint_target_q[q_start + 7], 0.3)
                            self.assertAlmostEqual(target.joint_target_qd[qd_start + 6], 0.4)
                        np.testing.assert_allclose(fk_state.body_qd.numpy(), body_velocities, atol=2e-6, rtol=0)
                        np.testing.assert_array_equal(target.joint_target_q[:7], prefix_target)
                        for name, values in source_state.items():
                            np.testing.assert_array_equal(getattr(source, name), values)

    def test_composition_transforms_multiple_free_targets(self):
        """Keep distinct free-root targets and anchors independent across copied worlds."""
        source = ModelBuilder()
        anchors = [
            wp.transform((0.2, 0.3, 0.4), wp.quat(0.5, 0.5, 0.5, 0.5)),
            wp.transform((-0.4, 0.7, 0.1), wp.quat(0.0, 1.0, 0.0, 0.0)),
        ]
        anchor_rotations = [np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), np.diag([-1, 1, -1])]
        for i, anchor in enumerate(anchors):
            root = source.add_link()
            free = source.add_joint_free(root, parent_xform=anchor)
            source.add_articulation([free])
            source.joint_target_q[7 * i : 7 * i + 3] = [i + 1, i + 2, i + 3]
            source.joint_target_qd[6 * i : 6 * i + 6] = [i + 1, 2, 3, 4, 5, i + 6]
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        shift = np.array([2.0, -1.0, 3.0])
        xform = wp.transform(shift, wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2))
        for method in ("add_builder", "replicate"):
            with self.subTest(method=method):
                target = ModelBuilder()
                if method == "replicate":
                    target.replicate(source, 2, xforms=[wp.transform(), xform])
                else:
                    target.add_builder(source)
                    target.add_builder(source, xform)
                np.testing.assert_array_equal(target.joint_target_q[:14], source.joint_target_q)
                for i, (anchor, anchor_rot) in enumerate(zip(anchors, anchor_rotations, strict=True)):
                    original = np.asarray(source.joint_target_q[7 * i : 7 * i + 3])
                    expected = anchor_rot.T @ (rot @ (anchor_rot @ original + anchor.p) + shift - anchor.p)
                    np.testing.assert_allclose(
                        target.joint_target_q[14 + 7 * i : 17 + 7 * i], expected, atol=2e-6, rtol=0
                    )
                    original_twist = np.asarray(source.joint_target_qd[6 * i : 6 * i + 6]).reshape(2, 3)
                    expected_twist = original_twist @ (anchor_rot.T @ rot @ anchor_rot).T
                    np.testing.assert_allclose(
                        target.joint_target_qd[12 + 6 * i : 18 + 6 * i], expected_twist.ravel(), atol=2e-6, rtol=0
                    )

    def test_composition_preserves_body_velocity_operations(self):
        """Preserve native spatial-vector operations before finalizing rotated copies."""
        source = ModelBuilder()
        body = source.add_body(mass=1.0)
        source.body_qd[body] = wp.spatial_vector(1, 2, 3, 4, 5, 6)
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        xform = wp.transform((1, 2, 3), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2))
        for method in ("add_builder", "replicate"):
            with self.subTest(method=method):
                target = ModelBuilder()
                if method == "add_builder":
                    target.add_builder(source, xform)
                else:
                    target.replicate(source, 1, xforms=[xform])
                np.testing.assert_allclose(wp.spatial_top(target.body_qd[0]), rotation @ [1, 2, 3], atol=1e-6)
                np.testing.assert_allclose(wp.spatial_bottom(target.body_qd[0]), rotation @ [4, 5, 6], atol=1e-6)
                copied = ModelBuilder()
                copied.replicate(target, 2)
                for velocity in copied.body_qd:
                    np.testing.assert_allclose(wp.spatial_top(velocity), rotation @ [1, 2, 3], atol=1e-6)
                    np.testing.assert_allclose(wp.spatial_bottom(velocity), rotation @ [4, 5, 6], atol=1e-6)
        np.testing.assert_array_equal(source.body_qd[0], [1, 2, 3, 4, 5, 6])

    def test_composition_transforms_tetrahedral_reference(self):
        """Rotate tetrahedral references while retaining deformation and existing data."""
        rest = np.array([[0.1, 0.2, 0.3], [1.3, 0.2, 0.3], [0.4, 1.6, 0.3], [0.3, 0.5, 1.4]])
        rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        offset = np.array([1.0, -2.0, 3.0])
        xforms = [wp.transform(), wp.transform(offset), wp.transform(offset, wp.quat(0.5, 0.5, 0.5, 0.5))]
        source = ModelBuilder()
        for point in rest:
            source.add_particle(point, (0, 0, 0), 1.0)
        source.add_tetrahedron(0, 1, 2, 3)
        rest_inverse = np.asarray(source.tet_poses[0]).copy()
        for deformed in (False, True):
            points = rest.copy()
            if deformed:
                points[1] += [0.15, -0.07, 0.04]
            source.particle_q = points.tolist()
            original_gradient = (points[1:] - points[0]).T @ rest_inverse
            for method in ("add_builder", "replicate"):
                with self.subTest(deformed=deformed, method=method):
                    target = ModelBuilder()
                    for point in rest + 5:
                        target.add_particle(point, (0, 0, 0), 1.0)
                    target.add_tetrahedron(0, 1, 2, 3)
                    prefix_q = np.asarray(target.particle_q).copy()
                    prefix_pose = np.asarray(target.tet_poses).copy()
                    if method == "replicate":
                        target.replicate(source, len(xforms), xforms=xforms)
                    else:
                        for xform in xforms:
                            target.add_builder(source, xform)
                    for i, (rot, shift) in enumerate(
                        ((np.eye(3), np.zeros(3)), (np.eye(3), offset), (rotation, offset)), 1
                    ):
                        with self.subTest(copy=i):
                            # Reconstruct the reference from independently transformed rest vertices.
                            reference = rest @ rot.T + shift
                            expected_inverse = np.linalg.inv((reference[1:] - reference[0]).T)
                            actual_inverse = np.asarray(target.tet_poses[i])
                            np.testing.assert_allclose(actual_inverse, expected_inverse, atol=2e-6, rtol=0)
                            actual_points = np.asarray(target.particle_q[4 * i : 4 * (i + 1)])
                            np.testing.assert_allclose(actual_points, points @ rot.T + shift, atol=2e-6, rtol=0)
                            gradient = (actual_points[1:] - actual_points[0]).T @ actual_inverse
                            np.testing.assert_allclose(gradient, rot @ original_gradient @ rot.T, atol=2e-6, rtol=0)
                    np.testing.assert_array_equal(target.particle_q[:4], prefix_q)
                    np.testing.assert_array_equal(target.tet_poses[:1], prefix_pose)
                    np.testing.assert_array_equal(source.tet_poses[0], rest_inverse)
                    np.testing.assert_array_equal(source.particle_q, points)

    def test_copy_rotation_rotates_world_velocities(self):
        """Verify that yaw rotates world-frame position and twist and that a pure translation leaves velocity unchanged."""

        def make_source() -> ModelBuilder:
            builder = ModelBuilder()
            root = builder.add_link(xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()), label="root")
            builder.body_qd[root] = wp.spatial_vector(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
            free = builder.add_joint_free(child=root, label="free")
            qd_start = builder.joint_qd_start[free]
            builder.joint_qd[qd_start : qd_start + 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            child = builder.add_link(xform=wp.transform((2.0, 0.0, 0.0), wp.quat_identity()), label="child")
            hinge = builder.add_joint_revolute(parent=root, child=child, axis=(0.0, 0.0, 1.0), label="hinge")
            builder.joint_qd[builder.joint_qd_start[hinge]] = 0.5
            builder.body_qd[child] = wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
            return builder

        yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2.0)
        yaw_xform = wp.transform((0.0, 0.0, 0.0), yaw)
        for merge in ("add_world", "replicate"):
            with self.subTest(merge=merge):
                scene = ModelBuilder()
                if merge == "add_world":
                    scene.add_world(make_source(), yaw_xform)
                else:
                    scene.replicate(make_source(), 1, xforms=[yaw_xform])

                body_q = np.asarray(scene.body_q, dtype=np.float32)
                body_qd = np.asarray(scene.body_qd, dtype=np.float32)
                np.testing.assert_allclose(body_q[0, :3], [0.0, 1.0, 0.0], atol=1e-5)
                np.testing.assert_allclose(body_q[1, :3], [0.0, 2.0, 0.0], atol=1e-5)
                expected_yaw = np.broadcast_to(np.asarray(yaw, dtype=np.float32), (2, 4))
                np.testing.assert_allclose(body_q[:, 3:], expected_yaw, atol=1e-5)
                np.testing.assert_allclose(body_qd[0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0], atol=1e-5)
                np.testing.assert_allclose(body_qd[1], [0.0, 0.0, 1.0, -1.0, 0.0, 0.0], atol=1e-5)
                joint_qd = np.asarray(scene.joint_qd, dtype=np.float32)
                np.testing.assert_allclose(joint_qd[:6], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0], atol=1e-5)
                np.testing.assert_allclose(joint_qd[6], 0.5, atol=1e-6)
                free_q = np.asarray(scene.joint_q[:7], dtype=np.float32)
                np.testing.assert_allclose(free_q[:3], [0.0, 1.0, 0.0], atol=1e-5)

        source = make_source()
        translated = ModelBuilder()
        translated.add_world(source, wp.transform((5.0, -2.0, 3.0), wp.quat_identity()))
        np.testing.assert_array_equal(np.asarray(translated.body_qd), np.asarray(source.body_qd))
        np.testing.assert_array_equal(np.asarray(translated.joint_qd), np.asarray(source.joint_qd))
        np.testing.assert_allclose(np.asarray(translated.body_q[0])[:3], [6.0, -2.0, 3.0], atol=1e-6)

        replicated = ModelBuilder()
        replicated.replicate(source, 1, xforms=[wp.transform((5.0, -2.0, 3.0), wp.quat_identity())])
        np.testing.assert_array_equal(np.asarray(replicated.body_qd), np.asarray(source.body_qd))
        np.testing.assert_array_equal(np.asarray(replicated.joint_qd), np.asarray(source.joint_qd))

    def test_replicate_matches_add_world_loop_with_explicit_transforms(self):
        source = self._make_source()
        xforms = [
            wp.transform((1.0, 2.0, 3.0), wp.quat_rpy(0.1, 0.2, 0.3)),
            wp.transform((-2.0, 1.0, 0.5), wp.quat_rpy(-0.2, 0.4, 0.1)),
        ]

        expected = self._make_destination()
        for xform in xforms:
            expected.add_world(source, xform)

        actual = self._make_destination()
        actual.replicate(source, len(xforms), xforms=xforms)

        self.assert_builder_merge_state_equal(expected, actual)

    def test_replicate_remaps_joint_mimic_references(self):
        """Verify replication remaps dense mimic references per world."""
        source = ModelBuilder()
        body0 = source.add_link()
        body1 = source.add_link()
        reference = source.add_joint_revolute(-1, body0)
        follower = source.add_joint_revolute(body0, body1)
        source.add_articulation([reference, follower])
        source.set_joint_mimic(follower, reference, coeffs=(0.25, -2.0))

        builder = ModelBuilder()
        builder.replicate(source, 3)
        model = builder.finalize()

        np.testing.assert_array_equal(model.joint_mimic_joint.numpy(), [-1, 0, -1, 2, -1, 4])
        np.testing.assert_allclose(
            model.joint_mimic_coeffs.numpy(),
            [(0.0, 1.0), (0.25, -2.0)] * 3,
        )

    def test_replicate_rejects_mismatched_explicit_transforms(self):
        with self.assertRaisesRegex(ValueError, "xforms must contain 2 entries, got 1"):
            ModelBuilder().replicate(self._make_source(), 2, xforms=[wp.transform_identity()])

    def test_replicate_does_not_call_add_world(self):
        source = self._make_source()
        builder = self._make_destination()

        with mock.patch.object(builder, "add_world", side_effect=AssertionError("unexpected scalar merge")):
            builder.replicate(source, 2)

    def test_replicated_worlds_cache_contact_pairs_without_filters(self):
        """Reuse one contact-pair template when replicated worlds have no filters."""
        source = ModelBuilder()
        for _ in range(4):
            body = source.add_body()
            source.add_shape_sphere(body, radius=0.1)

        builder = ModelBuilder()
        builder.replicate(source, 4)
        self.assertEqual(len(builder._shape_collision_filter_pairs), 0)

        model = builder.finalize(device="cpu")
        self.assertEqual(model.shape_contact_pair_count, 4 * 6)
        data = model._shape_contact_pair_data
        with mock.patch.object(data, "_store_pairs", wraps=data._store_pairs) as store_pairs:
            self.assertEqual(len(model.shape_contact_pairs), 4 * 6)
        # Enumerate six pairs once; replay the other three worlds directly.
        self.assertEqual(sum(len(call.args[1]) for call in store_pairs.call_args_list), 6)

    def test_add_world_uses_public_composition(self):
        class TrackingBuilder(ModelBuilder):
            def __init__(self):
                super().__init__()
                self.calls = []

            def begin_world(self, *args, **kwargs):
                self.calls.append("begin_world")
                return super().begin_world(*args, **kwargs)

            def add_builder(self, *args, **kwargs):
                self.calls.append("add_builder")
                return super().add_builder(*args, **kwargs)

            def end_world(self):
                self.calls.append("end_world")
                return super().end_world()

        source = ModelBuilder()
        source.add_body()
        destination = TrackingBuilder()

        destination.add_world(source)

        self.assertEqual(destination.calls, ["begin_world", "add_builder", "end_world"])

    def test_identity_transforms_do_not_alias_source(self):
        source = ModelBuilder()
        source.add_body(xform=wp.transform((1.0, 2.0, 3.0), wp.quat_identity()))

        replicated = ModelBuilder()
        replicated.replicate(source, 2)

        self.assertIsNot(replicated.body_q[0], source.body_q[0])
        self.assertIsNot(replicated.body_q[0], replicated.body_q[1])
        replicated.body_q[0][0] = 9.0
        self.assertEqual(source.body_q[0][0], 1.0)
        self.assertEqual(replicated.body_q[1][0], 1.0)

        merged = ModelBuilder()
        merged.add_builder(source, xform=wp.transform_identity())
        self.assertIsNot(merged.body_q[0], source.body_q[0])

        merged_no_xform = ModelBuilder()
        merged_no_xform.add_builder(source)
        self.assertIsNot(merged_no_xform.body_q[0], source.body_q[0])

    def test_merge_ignores_unknown_builder_attributes(self):
        class AnnotatedBuilder(ModelBuilder):
            def __init__(self):
                super().__init__()
                self.annotations = []

        source = AnnotatedBuilder()
        source.add_body()
        source.annotations.append("note")
        source.ad_hoc_tags = ["tag"]

        destination = ModelBuilder()
        destination.add_builder(source)
        destination.replicate(source, 2)
        self.assertEqual(destination.body_count, 3)

    def test_merge_accepts_array_valued_fields(self):
        source = self._make_source()
        joint_q = list(source.joint_q)
        particle_qd = [tuple(qd) for qd in source.particle_qd]
        source.joint_q = np.asarray(source.joint_q)
        source.particle_qd = np.asarray(source.particle_qd)

        destination = ModelBuilder()
        destination.add_builder(source)

        self.assertEqual(destination.joint_q, joint_q)
        self.assertEqual([tuple(qd) for qd in destination.particle_qd], particle_qd)

    def test_merge_accepts_array_valued_transforms_under_xform(self):
        # A rotated xform routes static shapes, root joints, and body_q through
        # the transform-multiply paths, which must accept ndarray rows.
        xform = wp.transform((0.5, -0.3, 0.25), wp.quat_rpy(0.4, -0.2, 0.7))
        source = self._make_source()

        expected = ModelBuilder()
        expected.add_builder(source, xform=xform)

        source.shape_transform = np.asarray(source.shape_transform)
        source.joint_X_p = np.asarray(source.joint_X_p)
        source.body_q = np.asarray(source.body_q)

        actual = ModelBuilder()
        actual.add_builder(source, xform=xform)

        for attr in ("shape_transform", "joint_X_p", "body_q", "joint_q"):
            np.testing.assert_array_equal(
                np.asarray(getattr(actual, attr), dtype=np.float32),
                np.asarray(getattr(expected, attr), dtype=np.float32),
                err_msg=attr,
            )

    def test_replicate_combines_color_groups_across_worlds(self):
        source = ModelBuilder()
        for i in range(6):
            source.add_particle(wp.vec3(float(i), 0.0, 0.0), wp.vec3(), 1.0)
        # Plain lists must be tolerated alongside ndarray groups.
        source.particle_color_groups = [[0, 1], np.arange(2, 6)]

        expected = ModelBuilder()
        for _ in range(3):
            expected.add_world(source)

        replicated = ModelBuilder()
        replicated.replicate(source, 3)

        self.assertEqual(len(replicated.particle_color_groups), len(expected.particle_color_groups))
        for expected_group, actual_group in zip(
            expected.particle_color_groups, replicated.particle_color_groups, strict=True
        ):
            np.testing.assert_array_equal(actual_group, expected_group)

    def test_zero_spacing_replication_copies_joint_q_exactly(self):
        """Bypass offset construction while preserving exact joint coordinates."""
        source = ModelBuilder()
        body = source.add_link()
        source.add_joint_free(
            parent=-1, child=body, parent_xform=wp.transform((0.2, 0.3, 0.4), wp.quat_rpy(0.123, 0.456, 0.789))
        )

        replicated = ModelBuilder()
        with mock.patch(
            "newton._src.sim.builder.compute_world_offsets",
            side_effect=AssertionError("unexpected offset construction"),
        ):
            replicated.replicate(source, 2)

        expected = np.tile(np.asarray(source.joint_q, dtype=np.float32), 2)
        np.testing.assert_array_equal(np.asarray(replicated.joint_q, dtype=np.float32), expected)

    def test_validation_failure_does_not_mutate_destination(self):
        def make_builder(default: int, value: int) -> ModelBuilder:
            builder = ModelBuilder()
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="probe",
                    frequency=Model.AttributeFrequency.BODY,
                    dtype=wp.int32,
                    default=default,
                )
            )
            builder.add_body(custom_attributes={"probe": value})
            return builder

        for operation in ("replicate", "add_world"):
            with self.subTest(operation=operation):
                destination = make_builder(0, 1)
                source = make_builder(2, 3)
                state_before = {
                    name: tuple(value) for name, value in vars(destination).items() if isinstance(value, list)
                }

                with self.assertRaisesRegex(ValueError, "default mismatch"):
                    if operation == "replicate":
                        destination.replicate(source, 2)
                    else:
                        destination.add_world(source)

                self.assertEqual(destination.world_count, 0)
                self.assertEqual(destination.current_world, -1)
                for name, expected in state_before.items():
                    with self.subTest(attribute=name):
                        self.assertEqual(tuple(getattr(destination, name)), expected)

    def test_incompatible_custom_attribute_spec_does_not_merge(self):
        destination = ModelBuilder()
        destination.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="probe",
                frequency=Model.AttributeFrequency.BODY,
                dtype=wp.int32,
                default=0,
            )
        )
        destination.add_body(custom_attributes={"probe": 10})

        for with_values in (False, True):
            with self.subTest(with_values=with_values):
                source = ModelBuilder()
                source.add_custom_attribute(
                    ModelBuilder.CustomAttribute(
                        name="probe",
                        frequency=Model.AttributeFrequency.SHAPE,
                        dtype=wp.int32,
                        default=0,
                    )
                )
                if with_values:
                    body = source.add_body()
                    source.add_shape_sphere(body, radius=0.1, custom_attributes={"probe": 20})

                with self.assertRaisesRegex(ValueError, "incompatible spec"):
                    destination.add_builder(source)

                self.assertEqual(destination.body_count, 1)
                self.assertEqual(destination.shape_count, 0)
                self.assertEqual(destination.custom_attributes["probe"].values, {0: 10})

    def test_incompatible_custom_frequency_does_not_merge(self):
        def filter_a(prim, context):
            return True

        def filter_b(prim, context):
            return False

        destination = ModelBuilder()
        destination.add_custom_frequency(ModelBuilder.CustomFrequency(name="probe", usd_prim_filter=filter_a))
        source = ModelBuilder()
        source.add_custom_frequency(ModelBuilder.CustomFrequency(name="probe", usd_prim_filter=filter_b))

        with self.assertRaisesRegex(ValueError, "different callbacks"):
            destination.add_world(source)

        self.assertEqual(destination.world_count, 0)
        self.assertEqual(destination.current_world, -1)
        self.assertIs(destination.custom_frequencies["probe"].usd_prim_filter, filter_a)

    def test_custom_reference_values_preserve_scalar_integer_sentinels(self):
        def make_builder() -> ModelBuilder:
            builder = ModelBuilder()
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="ref_body",
                    frequency=Model.AttributeFrequency.BODY,
                    dtype=wp.int32,
                    references="body",
                    default=-1,
                )
            )
            return builder

        source = make_builder()
        source.add_body(custom_attributes={"ref_body": np.int32(1)})
        source.add_body(custom_attributes={"ref_body": np.int32(-1)})
        source.add_body(custom_attributes={"ref_body": wp.int32(-1)})

        destination = make_builder()
        destination.add_body()
        destination.add_builder(source)

        values = destination.custom_attributes["ref_body"].values
        self.assertEqual(int(values[1]), 2)
        self.assertEqual(int(values[2]), -1)
        self.assertEqual(int(values[3]), -1)

    def test_replicate_has_explicit_finalized_semantics(self):
        source = ModelBuilder()
        body = source.add_body(xform=wp.transform((1.0, 2.0, 3.0), wp.quat_identity()), label="body")
        source.add_shape_sphere(body, radius=0.25)

        scene = ModelBuilder()
        scene.replicate(source, 2, spacing=(2.0, 0.0, 0.0))

        self.assertEqual(scene.world_count, 2)
        self.assertEqual(scene.body_world, [0, 1])
        self.assertEqual(scene.shape_world, [0, 1])
        self.assertEqual(scene.shape_body, [0, 1])
        np.testing.assert_array_equal(
            np.asarray(scene.body_q),
            np.asarray(
                [
                    (0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
                    (2.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
                ],
                dtype=np.float32,
            ),
        )

        model = scene.finalize(device="cpu")
        np.testing.assert_array_equal(model.body_world.numpy(), np.asarray([0, 1], dtype=np.int32))
        np.testing.assert_array_equal(model.shape_body.numpy(), np.asarray([0, 1], dtype=np.int32))
        np.testing.assert_array_equal(
            model.body_q.numpy(),
            np.asarray(
                [
                    (0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
                    (2.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
                ],
                dtype=np.float32,
            ),
        )

    def test_replicate_preserves_muscles_and_body_coloring(self):
        source = ModelBuilder()
        body0 = source.add_body()
        body1 = source.add_body()
        source.add_muscle([body0, body1], [wp.vec3(), wp.vec3(0.0, 0.0, 1.0)], 1.0, 2.0, 3.0, 4.0, 5.0)
        source.body_color_groups = [np.asarray([body0, body1])]

        scene = ModelBuilder()
        scene.replicate(source, 2)

        self.assertEqual(scene.muscle_start, [0, 2])
        self.assertEqual(scene.muscle_bodies, [0, 1, 2, 3])
        self.assertEqual(scene.muscle_params, source.muscle_params * 2)
        self.assertEqual(scene.muscle_activations, source.muscle_activations * 2)
        np.testing.assert_array_equal(scene.body_color_groups[0], np.asarray([0, 1, 2, 3]))

    def test_replicate_materializes_array_backing_on_public_access(self):
        """Materialize an array-backed attribute on ordinary access."""
        source = ModelBuilder()
        source.add_particle(wp.vec3(1.0, 2.0, 3.0), wp.vec3(), 2.0)

        scene = ModelBuilder()
        scene.replicate(source, 2)
        other_scene = ModelBuilder()
        other_scene.replicate(source, 1)

        self.assertIn("particle_q", scene._array_backed_attributes)
        self.assertNotIn("particle_q", vars(scene))
        self.assertIsInstance(scene._array_backed_attributes["particle_q"][0], np.ndarray)
        self.assertIsNot(scene._array_backed_attributes, other_scene._array_backed_attributes)

        scene.replicate(source, 1)
        self.assertIn("particle_q", scene._array_backed_attributes)

        particle_q = scene.particle_q
        self.assertIsInstance(particle_q, list)
        self.assertNotIn("particle_q", scene._array_backed_attributes)
        self.assertIn("particle_q", other_scene._array_backed_attributes)
        np.testing.assert_array_equal(np.asarray(particle_q), np.tile(np.asarray(source.particle_q), (3, 1)))

        particle_q.append(wp.vec3(4.0, 5.0, 6.0))
        self.assertEqual(len(scene.particle_q), 4)

    def test_array_backed_access_preserves_descriptor_attribute_errors(self):
        """Preserve AttributeErrors raised while evaluating unrelated descriptors."""

        class DerivedBuilder(ModelBuilder):
            @property
            def my_property(self):
                return self._missing_internal_attribute

        with self.assertRaisesRegex(AttributeError, "_missing_internal_attribute"):
            _ = DerivedBuilder().my_property

    def test_ctypes_materialization_converts_wider_array_dtype(self):
        """Convert wider backing values before reconstructing Warp ctypes elements."""
        source = ModelBuilder()
        source.add_particle(wp.vec3(1.0, 2.0, 3.0), wp.vec3(), 1.0)

        scene = ModelBuilder()
        scene.add_particle([4.0, 5.0, 6.0], wp.vec3(), 1.0)
        scene.replicate(source, 1)

        self.assertEqual(scene._array_backed_attributes["particle_q"][0].dtype, np.dtype(np.float64))
        self.assertTrue(all(isinstance(position, wp.vec3) for position in scene.particle_q))
        np.testing.assert_array_equal(scene.particle_q, [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])

    def test_tuple_materialization_restores_elements(self):
        """Reconstruct tuple elements with one bulk list conversion."""
        values = np.arange(8, dtype=np.int32).reshape((2, 4))

        materialized = _materialize_array_backed_list(values, ("tuple", None))

        self.assertEqual(materialized, [tuple(range(4)), tuple(range(4, 8))])
        self.assertEqual(_materialize_array_backed_list(np.empty((2, 0)), ("tuple", None)), [(), ()])
        self.assertEqual(_materialize_array_backed_list(np.empty((0, 3)), ("tuple", None)), [])

    def test_list_materialization_restores_element_shape(self):
        """Reconstruct nested list elements with one bulk list conversion."""
        values = np.arange(24, dtype=np.int32).reshape((2, 3, 4))

        materialized = _materialize_array_backed_list(values, ("list", (3, 4)))

        self.assertEqual(materialized, values.tolist())
        self.assertEqual(_materialize_array_backed_list(np.empty((2, 0)), ("list", (0,))), [[], []])
        self.assertEqual(_materialize_array_backed_list(np.empty((0, 3)), ("list", (3,))), [])

    def test_materialization_restores_gc_after_failure(self):
        """Disable cyclic GC during materialization and restore its prior state after failure."""
        original_gc_enabled = gc.isenabled()

        class FailingElement:
            @classmethod
            def from_buffer_copy(cls, row):
                self.assertFalse(gc.isenabled())
                raise RuntimeError("expected failure")

        try:
            for initially_enabled in (False, True):
                with self.subTest(initially_enabled=initially_enabled):
                    if initially_enabled:
                        gc.enable()
                    else:
                        gc.disable()

                    with self.assertRaisesRegex(RuntimeError, "expected failure"):
                        _materialize_array_backed_list(np.ones((1, 1), dtype=np.float32), ("ctypes", FailingElement))

                    self.assertEqual(gc.isenabled(), initially_enabled)
        finally:
            if original_gc_enabled:
                gc.enable()
            else:
                gc.disable()

    def test_replicate_and_finalize_restore_gc_after_failure(self):
        """Disable cyclic GC during each operation and restore its prior state after failure."""
        source = ModelBuilder()
        original_gc_enabled = gc.isenabled()

        try:
            for operation, internal_method, args, kwargs in (
                ("replicate", "_merge_builder_copies", (source, 1), {}),
                ("finalize", "_finalize_impl", (), {"skip_all_validations": True}),
            ):
                for initially_enabled in (False, True):
                    with self.subTest(operation=operation, initially_enabled=initially_enabled):
                        if initially_enabled:
                            gc.enable()
                        else:
                            gc.disable()

                        builder = ModelBuilder()

                        def fail_while_gc_is_disabled(*args, **kwargs):
                            self.assertFalse(gc.isenabled())
                            raise RuntimeError("expected failure")

                        with mock.patch.object(builder, internal_method, side_effect=fail_while_gc_is_disabled):
                            with self.assertRaisesRegex(RuntimeError, "expected failure"):
                                getattr(builder, operation)(*args, **kwargs)

                        self.assertEqual(gc.isenabled(), initially_enabled)
        finally:
            if original_gc_enabled:
                gc.enable()
            else:
                gc.disable()

    def test_add_world_materializes_array_backing(self):
        """Materialize array-backed attributes before adding another world."""
        source = ModelBuilder()
        source.add_particle(wp.vec3(1.0, 2.0, 3.0), wp.vec3(), 2.0)

        scene = ModelBuilder()
        scene.replicate(source, 2)
        self.assertIn("particle_q", scene._array_backed_attributes)

        scene.add_world(source)

        self.assertIsInstance(scene.particle_q, list)
        self.assertNotIn("particle_q", scene._array_backed_attributes)
        self.assertEqual(scene.particle_world, [0, 1, 2])
        np.testing.assert_array_equal(np.asarray(scene.particle_q), np.tile(np.asarray(source.particle_q), (3, 1)))

    def test_array_backed_dtype_is_at_least_declared_dtype(self):
        """Do not truncate offsets when a source reference array uses a narrow dtype."""
        source = ModelBuilder()
        particles = [source.add_particle(wp.vec3(), wp.vec3(), 1.0) for _ in range(2)]
        source.add_spring(*particles, 1.0, 1.0, 0.0)
        source.spring_indices = np.asarray(source.spring_indices, dtype=np.int8)

        scene = ModelBuilder()
        for _ in range(128):
            scene.add_particle(wp.vec3(), wp.vec3(), 1.0)
        scene.replicate(source, 1)

        spring_indices = scene._array_backed_attributes["spring_indices"][0]
        self.assertEqual(spring_indices.dtype, np.dtype(np.int32))
        np.testing.assert_array_equal(spring_indices, [128, 129])

    def test_finalize_consumes_array_backing_without_materializing(self):
        """Finalize directly from array-backed attributes without materializing them."""
        source = ModelBuilder()
        source.add_particle(wp.vec3(1.0, 2.0, 3.0), wp.vec3(), 2.0)

        scene = ModelBuilder()
        scene.replicate(source, 2)
        particle_q = scene._array_backed_attributes["particle_q"][0]

        model = scene.finalize(device="cpu")

        self.assertIn("particle_q", scene._array_backed_attributes)
        self.assertIs(scene._array_backed_attributes["particle_q"][0], particle_q)
        np.testing.assert_array_equal(model.particle_q.numpy(), particle_q)

    def test_finalize_matches_materialized_finalize(self):
        """Match materialized finalization for array-backed attributes."""
        for world_count in (1, 3):
            with self.subTest(world_count=world_count):
                source = self._make_source()

                array_backed = self._make_destination()
                array_backed.replicate(source, world_count)
                array_backed_names = set(array_backed._array_backed_attributes)
                self.assertTrue(array_backed_names)
                self.assertTrue(
                    {
                        "body_inertia",
                        "joint_target_q",
                        "muscle_params",
                        "tri_poses",
                        "tet_poses",
                    }.issubset(array_backed_names)
                )

                materialized = self._make_destination()
                materialized.replicate(source, world_count)
                for name in _ARRAY_BACKED_ATTRIBUTE_DTYPES:
                    getattr(materialized, name)
                self.assertFalse(materialized._array_backed_attributes)

                expected = materialized.finalize(device="cpu")
                actual = array_backed.finalize(device="cpu")

                self.assertEqual(actual.world_count, expected.world_count)
                self.assertEqual(set(array_backed._array_backed_attributes), array_backed_names)
                for name in sorted(array_backed_names):
                    with self.subTest(attribute=name):
                        expected_array = getattr(expected, name)
                        actual_array = getattr(actual, name)
                        self.assertIsInstance(expected_array, wp.array)
                        self.assertIsInstance(actual_array, wp.array)
                        self.assertIs(expected_array.dtype, actual_array.dtype)
                        np.testing.assert_array_equal(expected_array.numpy(), actual_array.numpy())

    def test_array_backed_finalize_preserves_rod_target_coordinates(self):
        """Preserve replicated Rod target quaternions during finalization."""
        source = ModelBuilder()
        source.add_rod(
            rod=Rod([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.0, 2.0)], radius=0.1),
            body_frame_origin="com",
        )

        array_backed = ModelBuilder()
        array_backed.replicate(source, 2)
        self.assertIn("joint_target_q", array_backed._array_backed_attributes)

        actual = array_backed.finalize(device="cpu")

        self.assertEqual(actual.joint_target_q.shape, (actual.joint_coord_count,))
        np.testing.assert_allclose(actual.joint_target_q.numpy(), np.tile(source.joint_target_q, 2), atol=1.0e-6)
        targets = actual.joint_target_q.numpy().reshape(-1, 7)
        np.testing.assert_allclose(np.linalg.norm(targets[:, 3:], axis=1), 1.0, atol=1.0e-6)

    def test_array_backed_joint_validation_returns_early_when_all_joints_are_articulated(self):
        """Skip orphan-joint validation and keep topology array-backed when every joint is articulated."""
        scene = ModelBuilder()
        scene.replicate(self._make_source(), 2)
        array_backed_names = set(scene._array_backed_attributes)
        self.assertTrue({"body_q", "joint_articulation", "joint_child", "joint_parent"}.issubset(array_backed_names))

        with mock.patch("newton._src.sim.builder.np.isin") as isin:
            isin.side_effect = AssertionError("unexpected orphan-joint validation")
            scene._validate_joints()

        isin.assert_not_called()
        self.assertEqual(set(scene._array_backed_attributes), array_backed_names)

    def test_array_backed_joint_validation_still_rejects_orphan_joints(self):
        """Run general joint validation when an array-backed articulation entry is negative."""
        source = ModelBuilder()
        parent = source.add_link()
        child = source.add_link()
        source.add_joint_fixed(parent, child)
        scene = ModelBuilder()
        scene.replicate(source, 2)

        with self.assertRaisesRegex(ValueError, "not belonging to any articulation"):
            scene.finalize(device="cpu")

    def test_all_builder_lists_have_merge_metadata(self):
        """Cover every builder list with valid, coordinate-aware merge metadata."""
        list_attributes = {name for name, value in vars(ModelBuilder()).items() if isinstance(value, list)}
        specs = ModelBuilder._builder_merge_attribute_specs()
        self.assertEqual(set(specs), list_attributes)
        self.assertIs(specs["shape_body"], Model._CORE_ATTRIBUTE_SPECS["shape_body"])
        self.assertEqual(specs["joint_target_q"].frequency, Model.AttributeFrequency.JOINT_COORD)

        base_attributes, base_lists = ModelBuilder._base_builder_attributes()

        added = {"_missing_merge_metadata"}
        with (
            mock.patch.object(ModelBuilder, "_BASE_ATTRIBUTES", base_attributes | added),
            mock.patch.object(ModelBuilder, "_BASE_LIST_ATTRIBUTES", base_lists | added),
        ):
            with self.assertRaisesRegex(RuntimeError, "_missing_merge_metadata"):
                ModelBuilder._build_builder_merge_attribute_specs()

        removed = {"body_lock_inertia"}
        with (
            mock.patch.object(ModelBuilder, "_BASE_ATTRIBUTES", base_attributes - removed),
            mock.patch.object(ModelBuilder, "_BASE_LIST_ATTRIBUTES", base_lists - removed),
        ):
            with self.assertRaisesRegex(RuntimeError, "body_lock_inertia"):
                ModelBuilder._build_builder_merge_attribute_specs()

        with mock.patch.object(ModelBuilder, "_BASE_LIST_ATTRIBUTES", base_lists - removed):
            with self.assertRaisesRegex(RuntimeError, "body_lock_inertia"):
                ModelBuilder._build_builder_merge_attribute_specs()

    def test_array_backed_attributes_have_generic_merge_metadata(self):
        """Require generic merge metadata and NumPy-compatible dtypes for array-backed attributes."""
        _, builder_list_attributes = ModelBuilder._base_builder_attributes()
        specs = ModelBuilder._builder_merge_attribute_specs()

        self.assertTrue(_ARRAY_BACKED_ATTRIBUTE_DTYPES)
        for name, warp_dtype in _ARRAY_BACKED_ATTRIBUTE_DTYPES.items():
            with self.subTest(attribute=name):
                self.assertIn(name, builder_list_attributes)
                self.assertIn(name, specs)
                self.assertEqual(specs[name].compaction_policy, "generic")
                scalar_dtype = getattr(warp_dtype, "_wp_scalar_type_", warp_dtype)
                self.assertIsNotNone(wp.dtype_to_numpy(scalar_dtype))


@wp.kernel
def count_mesh_aabb_hits_kernel(mesh_ids: wp.array[wp.uint64], hits: wp.array[wp.int32]):
    query = wp.mesh_query_aabb(mesh_ids[0], wp.vec3(-2.0, -2.0, -2.0), wp.vec3(2.0, 2.0, 2.0))
    face = wp.int32(0)
    while wp.mesh_query_aabb_next(query, face):
        wp.atomic_add(hits, 0, 1)


class TestReplicateMeshLifetime(unittest.TestCase):
    pass


def test_replicate_shared_mesh_survives_second_finalize(test: TestReplicateMeshLifetime, device):
    """Verify that finalizing a second model sharing mesh geometry keeps the first model's meshes alive.

    replicate() shares Mesh geometry objects by reference, so finalizing a second
    model built from the same source must not release the wp.Mesh whose id the
    first model stores in shape_source_ptr (regression test for a use-after-free).
    """
    # procedural grid mesh so no asset download is required
    n = 8
    gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n), np.linspace(-1.0, 1.0, n))
    vertices = np.stack([gx, gy, np.zeros_like(gx)], axis=-1).reshape(-1, 3)
    quad = np.arange(n * n).reshape(n, n)
    a, b, c, d = quad[:-1, :-1].ravel(), quad[:-1, 1:].ravel(), quad[1:, :-1].ravel(), quad[1:, 1:].ravel()
    indices = np.concatenate([np.stack([a, b, c], -1), np.stack([b, d, c], -1)]).ravel()
    face_count = len(indices) // 3

    source = ModelBuilder()
    source.add_shape_mesh(body=source.add_body(), mesh=Mesh(vertices, indices, compute_inertia=False))

    builder_1 = ModelBuilder()
    builder_1.replicate(source, 1)
    builder_2 = ModelBuilder()
    builder_2.replicate(source, 1)

    model_1 = builder_1.finalize(device=device)
    mesh_id = int(model_1.shape_source_ptr.numpy()[0])

    # finalizing a second model from the shared geometry must not invalidate model_1
    model_2 = builder_2.finalize(device=device)
    gc.collect()

    live_ids = {geo.mesh.id for geo in model_1.shape_source if getattr(geo, "mesh", None) is not None}
    test.assertIn(mesh_id, live_ids)

    hits = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(count_mesh_aabb_hits_kernel, dim=1, inputs=[model_1.shape_source_ptr, hits], device=device)
    test.assertEqual(hits.numpy()[0], face_count)

    # both models share the same finalized wp.Mesh for the shared geometry
    test.assertEqual(int(model_2.shape_source_ptr.numpy()[0]), mesh_id)


devices = get_test_devices()
add_function_test(
    TestReplicateMeshLifetime,
    "test_replicate_shared_mesh_survives_second_finalize",
    test_replicate_shared_mesh_survives_second_finalize,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestModelBuilderReplicateLabelPrefixes(unittest.TestCase):
    """Per-world label prefixes on the batched replication path."""

    @staticmethod
    def _source(root: str = "") -> ModelBuilder:
        """A two-link prototype whose labels are relative to ``root`` (empty names the root)."""
        builder = ModelBuilder()
        body = builder.add_link(xform=wp.transform(), label=root)
        builder.add_shape_box(body=body, label=f"{root}/box" if root else "box")
        return builder

    def test_each_world_gets_its_own_prefix(self):
        """Each replicated world's labels are prefixed with its own entry from label_prefixes."""
        scene = ModelBuilder()
        prefixes = [f"/World/envs/env_{index}/Robot" for index in range(3)]
        scene.replicate(self._source(), 3, label_prefixes=prefixes)

        self.assertEqual(scene.shape_label, [f"{prefix}/box" for prefix in prefixes])

    def test_a_none_prefix_leaves_that_world_alone(self):
        """A None entry in label_prefixes copies that world's labels unprefixed."""
        scene = ModelBuilder()
        scene.replicate(self._source("root"), 2, label_prefixes=["left", None])

        self.assertEqual(scene.shape_label, ["left/root/box", "root/box"])

    def test_prefix_count_must_match_world_count(self):
        """label_prefixes must have exactly one entry per replicated world."""
        with self.assertRaises(ValueError):
            ModelBuilder().replicate(self._source("root"), 2, label_prefixes=["only-one"])
