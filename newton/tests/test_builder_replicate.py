# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton import Model, ModelBuilder
from newton.actuators import ControllerPD
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
            ControllerPD,
            index=builder.joint_qd_start[child_joint],
            pos_index=builder.joint_q_start[child_joint],
            kp=100.0,
            kd=10.0,
        )
        builder.add_muscle([root, child], [wp.vec3(), wp.vec3(0.0, 0.0, 0.5)], 1.0, 1.0, 1.0, 1.0, 0.0)
        builder.body_color_groups = [np.asarray([root, child, fixed])]
        # Uneven group sizes exercise the balancing in the merge combine.
        builder.set_coloring([[0], [1, 2, 3]])

        builder._record_cable_group("cable", (root, child + 1), (root_joint, child_joint + 1))
        builder._record_cloth_group("cloth", (0, 3), (0, 1), (0, 1))
        builder._record_soft_group("soft", (0, 4), (0, 1))
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
        manually_verified = ("_shape_collision_filter_pairs", "actuator_entries", "custom_attributes")
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
            for field in ("indices", "pos_indices", "controller_args", "delay_args", "clamping_args"):
                self.assertEqual(getattr(expected_entry, field), getattr(actual_entry, field))

    def test_replicate_matches_add_world_loop(self):
        world_count = 4
        for use_coord_layout_targets in (False, True):
            with mock.patch("newton.use_coord_layout_targets", use_coord_layout_targets):
                source = self._make_source()
                for spacing in ((0.0, 0.0, 0.0), (2.0, 3.0, 0.0)):
                    with self.subTest(use_coord_layout_targets=use_coord_layout_targets, spacing=spacing):
                        expected = self._make_destination()
                        for offset in compute_world_offsets(world_count, spacing, expected.up_axis):
                            expected.add_world(source, wp.transform(offset, wp.quat_identity()))

                        actual = self._make_destination()
                        actual.replicate(source, world_count, spacing)

                        self.assert_builder_merge_state_equal(expected, actual)

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

    def test_replicate_rejects_mismatched_explicit_transforms(self):
        with self.assertRaisesRegex(ValueError, "xforms must contain 2 entries, got 1"):
            ModelBuilder().replicate(self._make_source(), 2, xforms=[wp.transform_identity()])

    def test_replicate_does_not_call_add_world(self):
        source = self._make_source()
        builder = self._make_destination()

        with mock.patch.object(builder, "add_world", side_effect=AssertionError("unexpected scalar merge")):
            builder.replicate(source, 2)

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
        source = ModelBuilder()
        body = source.add_body()
        source.add_joint_free(
            parent=-1, child=body, parent_xform=wp.transform((0.2, 0.3, 0.4), wp.quat_rpy(0.123, 0.456, 0.789))
        )

        replicated = ModelBuilder()
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

    def test_all_builder_lists_have_merge_metadata(self):
        list_attributes = {name for name, value in vars(ModelBuilder()).items() if isinstance(value, list)}
        specs = ModelBuilder._builder_merge_attribute_specs()
        self.assertEqual(set(specs), list_attributes)
        self.assertIs(specs["shape_body"], Model._CORE_ATTRIBUTE_SPECS["shape_body"])

        base_attributes, base_lists = ModelBuilder._base_builder_attributes()

        added = {"_missing_merge_metadata"}
        with (
            mock.patch.object(ModelBuilder, "_BASE_ATTRIBUTES", base_attributes | added),
            mock.patch.object(ModelBuilder, "_BASE_LIST_ATTRIBUTES", base_lists | added),
        ):
            with self.assertRaisesRegex(RuntimeError, "_missing_merge_metadata"):
                ModelBuilder._build_builder_merge_attribute_specs(False)

        removed = {"body_lock_inertia"}
        with (
            mock.patch.object(ModelBuilder, "_BASE_ATTRIBUTES", base_attributes - removed),
            mock.patch.object(ModelBuilder, "_BASE_LIST_ATTRIBUTES", base_lists - removed),
        ):
            with self.assertRaisesRegex(RuntimeError, "body_lock_inertia"):
                ModelBuilder._build_builder_merge_attribute_specs(False)

        with mock.patch.object(ModelBuilder, "_BASE_LIST_ATTRIBUTES", base_lists - removed):
            with self.assertRaisesRegex(RuntimeError, "body_lock_inertia"):
                ModelBuilder._build_builder_merge_attribute_specs(False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
