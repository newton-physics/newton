# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import tracemalloc
import unittest
from collections.abc import Callable
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton import Mesh, Model, ModelBuilder
from newton.actuators import ControllerPD
from newton.tests.unittest_utils import add_function_test, get_test_devices
from newton.utils import compute_world_offsets

# Runtime handles rebuilt by every finalize(); they carry a device pointer rather than model data.
_OPAQUE_MODEL_TYPES = (wp.Bvh, wp.HashGrid, wp.Mesh, wp.Volume)

# finalize() writes only part of each of these, leaving the rest as uninitialized memory that varies per run.
_UNINITIALIZED_MODEL_ATTRIBUTES = ("body_colors", "bvh_shape_enabled", "particle_colors")

# The leading component is downloaded whole, because scenes reference sibling mesh directories.
_ASSET_SCENES = (
    "anybotics_anymal_c/urdf/anymal.urdf",
    "anybotics_anymal_c/usd/anymal_c.usda",
    "anybotics_anymal_d/usd/anymal_d.usda",
    "apptronik_apollo/usd_structured/apptronik_apollo.usda",
    "booster_t1/usd_structured/T1.usda",
    "disneyresearch/dr_legs/usd/dr_legs.usda",
    "disneyresearch/dr_legs/usd/dr_legs_with_boxes.usda",
    "franka_emika_panda/urdf/fr3.urdf",
    "franka_emika_panda/urdf/fr3_franka_hand.urdf",
    "manipulation_objects/cup/model.usda",
    "manipulation_objects/pad/model.usda",
    "robotiq_2f85_v4/usd_structured/Dual_wrist_camera.usda",
    "shadow_hand/usd_structured/left_shadow_hand.usda",
    "style3d/avatars/Female.usd",
    "style3d/garments/Female_T_Shirt.usd",
    "unitree_g1/mjcf/g1_29dof_with_hand_rev_1_0.xml",
    "unitree_g1/usd/g1_isaac.usd",
    "unitree_g1/usd_structured/g1_29dof_with_hand_rev_1_0.usda",
    "unitree_go2/usd/go2.usda",
    "unitree_h1/mjcf/h1_with_hand.xml",
    "unitree_h1/urdf/h1.urdf",
    "unitree_h1/usd/h1_minimal.usda",
    "unitree_h1/usd_structured/h1.usda",
    "universal_robots_ur10/usd/ur10_instanceable.usda",
    "universal_robots_ur5e/usd_structured/ur5e.usda",
    "wonik_allegro/urdf/allegro_hand_description_left.urdf",
    "wonik_allegro/usd_structured/allegro_left.usda",
)

# Folders the rest of the suite already downloads, so the default run fetches nothing extra.
_DEFAULT_ASSET_SCENES = (
    "anybotics_anymal_d/usd/anymal_d.usda",
    "disneyresearch/dr_legs/usd/dr_legs.usda",
    "franka_emika_panda/urdf/fr3_franka_hand.urdf",
    "universal_robots_ur10/usd/ur10_instanceable.usda",
)

_MEMORY_SCENES = (
    "anybotics_anymal_d/usd/anymal_d.usda",
    "franka_emika_panda/urdf/fr3_franka_hand.urdf",
    "universal_robots_ur10/usd/ur10_instanceable.usda",
)

# Also the length of the scenarios' xforms and label_prefixes.
_REPLICATE_WORLD_COUNT = 4


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
    def _make_destination_with_prefix(source: ModelBuilder) -> ModelBuilder:
        """Destination carrying global entities, a pre-merged copy of ``source``, and one existing world."""
        builder = ModelBuilder()
        body = builder.add_body(label="global")
        builder.add_shape_sphere(body, radius=0.1, label="global_shape")
        builder.add_builder(source, label_prefix="prefix")
        builder.begin_world()
        existing = builder.add_body(label="existing")
        builder.add_shape_box(existing, hx=0.1, hy=0.1, hz=0.1, label="existing_shape")
        builder.end_world()
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

    def assert_model_values_equal(self, expected, actual, path: str) -> None:
        """Compare two finalized-model values, recursing through containers and objects that lack ``__eq__``."""
        self.assertIs(type(expected), type(actual), path)

        if isinstance(expected, _OPAQUE_MODEL_TYPES):
            return
        if isinstance(expected, wp.array):
            self.assertEqual(expected.dtype, actual.dtype, path)
            self.assertEqual(expected.shape, actual.shape, path)
            expected, actual = expected.numpy(), actual.numpy()

        if isinstance(expected, np.ndarray):
            self.assertEqual(expected.dtype, actual.dtype, path)
            if expected.dtype.kind == "f":
                np.testing.assert_allclose(actual, expected, err_msg=path)
            else:
                np.testing.assert_array_equal(actual, expected, err_msg=path)
        elif isinstance(expected, (float, np.floating)):
            np.testing.assert_allclose(actual, expected, err_msg=path)
        elif isinstance(expected, (bool, int, str, bytes, np.integer, np.bool_, set, frozenset, type(None))):
            self.assertEqual(expected, actual, path)
        elif isinstance(expected, (list, tuple)):
            for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
                self.assert_model_values_equal(expected_item, actual_item, f"{path}[{index}]")
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual), path)
            for key in expected:
                self.assert_model_values_equal(expected[key], actual[key], f"{path}[{key!r}]")
        elif isinstance(expected, type) or type(expected).__eq__ is not object.__eq__:
            self.assertEqual(expected, actual, path)
        elif hasattr(expected, "__dict__"):
            self.assert_model_values_equal(vars(expected), vars(actual), path)
        else:
            self.fail(f"{path}: unhandled type {type(expected).__name__}; add an explicit comparison")

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

    def assert_models_equal(self, expected: Model, actual: Model) -> None:
        """Compare every attribute of two finalized models."""
        self.assertEqual(set(vars(expected)), set(vars(actual)))
        for name in sorted(set(vars(expected)) - set(_UNINITIALIZED_MODEL_ATTRIBUTES)):
            with self.subTest(attribute=name):
                self.assert_model_values_equal(getattr(expected, name), getattr(actual, name), name)

    def test_replicate_and_finalize_matches_every_model_attribute(self):
        """Compare the whole model, over destination shapes, per-world placement, labels, and target layout."""
        xforms = [
            wp.transform((1.0, 2.0, 3.0), wp.quat_rpy(0.1, 0.2, 0.3)),
            wp.transform((-2.0, 1.0, 0.5), wp.quat_rpy(-0.2, 0.4, 0.1)),
            wp.transform((0.5, -1.5, 2.0), wp.quat_rpy(0.3, -0.1, 0.2)),
            wp.transform((-1.0, 0.5, -2.5), wp.quat_rpy(-0.1, -0.3, 0.4)),
        ]
        prefixes = [f"/World/envs/env_{index}/Robot" for index in range(_REPLICATE_WORLD_COUNT)]
        placed = {"xforms": xforms, "label_prefixes": prefixes}
        spaced = {"spacing": (2.0, 3.0, 0.0)}

        # Prefixes on half of these, so the unprefixed label path stays covered.
        scenarios = (
            ("empty/spacing", False, spaced),
            ("prefixed/xforms/labels", True, placed),
            ("empty/xforms/labels", False, placed),
            ("prefixed/spacing", True, spaced),
        )

        for label, prefixed_destination, replicate_kwargs in scenarios:
            with self.subTest(scenario=label), mock.patch("newton.use_coord_layout_targets", True):
                # The target layout selects the joint_target_q merge frequency, so build under the patch.
                source = self._make_source()

                def make_destination(prefixed=prefixed_destination, source=source) -> ModelBuilder:
                    return self._make_destination_with_prefix(source) if prefixed else ModelBuilder()

                expected_builder = make_destination()
                expected_builder.replicate(source, _REPLICATE_WORLD_COUNT, **replicate_kwargs)
                expected = expected_builder.finalize(device="cpu")

                unchanged_builder = make_destination()
                actual_builder = make_destination()
                actual = actual_builder.replicate_and_finalize(
                    source, _REPLICATE_WORLD_COUNT, device="cpu", **replicate_kwargs
                )

                self.assert_builder_merge_state_equal(unchanged_builder, actual_builder)
                self.assert_models_equal(expected, actual)

    def test_replicate_and_finalize_infers_row_shape_from_prefix(self):
        """tri_indices row width must come from the destination when the replicated source has no triangles."""
        source = ModelBuilder()
        for i in range(3):
            source.add_particle(wp.vec3(float(i), 0.0, 0.0), wp.vec3(), 1.0)

        def make_destination() -> ModelBuilder:
            builder = ModelBuilder()
            particles = [builder.add_particle(wp.vec3(0.0, float(i), 0.0), wp.vec3(), 1.0) for i in range(3)]
            builder.add_triangle(*particles)
            return builder

        expected_builder = make_destination()
        expected_builder.replicate(source, 2, spacing=(1.0, 0.0, 0.0))
        expected = expected_builder.finalize(device="cpu")

        actual = make_destination().replicate_and_finalize(source, 2, spacing=(1.0, 0.0, 0.0), device="cpu")

        self.assert_models_equal(expected, actual)

    def test_replicate_and_finalize_matches_legacy_target_layout(self):
        """Cover the DOF-shaped joint_target_q projection, which only runs while use_coord_layout_targets is
        off. Delete this test together with that flag; every other case runs the coordinate layout."""
        with mock.patch("newton.use_coord_layout_targets", False):
            source = self._make_source()
            # BALL and DISTANCE drop their quat-w padding at different offsets than FREE.
            spinner = source.add_link(label="spinner")
            source.add_joint_ball(parent=-1, child=spinner, label="ball")
            floater = source.add_link(label="floater")
            source.add_joint_distance(parent=-1, child=floater, label="distance")

            expected_builder = ModelBuilder()
            expected_builder.replicate(source, _REPLICATE_WORLD_COUNT, spacing=(2.0, 3.0, 0.0))
            with self.assertWarnsRegex(DeprecationWarning, "legacy DOF-shaped joint_target_q layout"):
                expected = expected_builder.finalize(device="cpu")

            with self.assertWarnsRegex(DeprecationWarning, "legacy DOF-shaped joint_target_q layout"):
                actual = ModelBuilder().replicate_and_finalize(
                    source, _REPLICATE_WORLD_COUNT, spacing=(2.0, 3.0, 0.0), device="cpu"
                )

        self.assert_models_equal(expected, actual)

    @staticmethod
    def _import_asset_scene(scene: str) -> ModelBuilder:
        """Import one newton-assets scene given its repository-relative path."""
        folder, _, relative_path = scene.partition("/")
        path = newton.utils.download_asset(folder) / relative_path
        builder = ModelBuilder()
        importers = {".urdf": builder.add_urdf, ".xml": builder.add_mjcf}
        importers.get(path.suffix, builder.add_usd)(str(path))
        return builder

    @staticmethod
    def _build_measuring_peaks(build: Callable[[], Model]) -> tuple[Model, dict[str, int]]:
        """Build a model, reporting peak bytes allocated through Python and through Warp.

        Warp allocates outside the Python allocator, so neither tracker sees the other's memory.
        """
        gc.collect()
        with wp.ScopedMemoryTracker("replicate", print=False) as tracker:
            tracker.clear()
            tracemalloc.start()
            try:
                model = build()
                peaks = {"python": tracemalloc.get_traced_memory()[1]}
            finally:
                tracemalloc.stop()
            peaks["warp"] = wp._src.context.runtime.core.wp_alloc_tracker_get_peak_bytes()
        return model, peaks

    def test_replicate_and_finalize_matches_asset_scenes(self):
        """Agree with separate calls on real assets. Set NEWTON_TEST_ALL_ASSETS to sweep every scene."""
        scenes = _ASSET_SCENES if os.environ.get("NEWTON_TEST_ALL_ASSETS") else _DEFAULT_ASSET_SCENES
        world_count = 2
        for scene in scenes:
            with self.subTest(asset=scene):
                source = self._import_asset_scene(scene)

                expected_builder = ModelBuilder()
                expected_builder.replicate(source, world_count)
                expected = expected_builder.finalize(device="cpu")

                actual = ModelBuilder().replicate_and_finalize(source, world_count, device="cpu")

                self.assert_models_equal(expected, actual)

    def test_replicate_and_finalize_allocates_no_more_at_batch_size(self):
        """Replicate one source covering every feature at a realistic batch size, and check the contiguous
        arrays and the suspended collector cost no more memory than the separate calls do."""
        source = self._make_source()
        for scene in _MEMORY_SCENES:
            source.add_builder(self._import_asset_scene(scene))

        # The contiguous-array path repays its fixed cost past roughly 32 worlds; below that it peaks higher.
        world_count = 64

        def separate() -> Model:
            destination = ModelBuilder()
            destination.replicate(source, world_count)
            return destination.finalize(device="cpu")

        def combined() -> Model:
            return ModelBuilder().replicate_and_finalize(source, world_count, device="cpu")

        # BVHs are built once per mesh, so without this they land on whichever path is measured first.
        ModelBuilder().replicate_and_finalize(source, 2, device="cpu")

        expected, separate_peaks = self._build_measuring_peaks(separate)
        actual, combined_peaks = self._build_measuring_peaks(combined)

        self.assert_models_equal(expected, actual)
        for allocator, separate_peak in separate_peaks.items():
            with self.subTest(peak=allocator):
                self.assertLessEqual(combined_peaks[allocator], separate_peak)

    def test_replicate_and_finalize_preserves_gc_state(self):
        """Leave the collector's enabled/disabled state as it was before the call."""
        source = self._make_source()

        for initially_enabled in (True, False):
            with self.subTest(initially_enabled=initially_enabled):
                if not initially_enabled:
                    gc.disable()
                try:
                    ModelBuilder().replicate_and_finalize(source, 2, device="cpu")
                    self.assertEqual(gc.isenabled(), initially_enabled)
                finally:
                    gc.enable()

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

        with mock.patch.object(builder, "_test_group_pair", wraps=builder._test_group_pair) as test_group_pair:
            model = builder.finalize(device="cpu")

        self.assertEqual(model.shape_contact_pair_count, 4 * 6)
        self.assertEqual(test_group_pair.call_count, 6)

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
        body = source.add_link()
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
