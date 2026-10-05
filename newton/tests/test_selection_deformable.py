# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for deformable family selection, label matching, and world layout."""

import re
import unittest

import numpy as np
import warp as wp

import newton
from newton.selection import (
    DeformableCurveView,
    DeformableSurfaceView,
    DeformableVolumeView,
)
from newton.tests._selection_deformable_test_utils import (
    _add_test_cable,
    _add_test_cloth,
    _add_test_soft_body,
    _replicated_model,
)


class TestDeformableFamilyViews(unittest.TestCase):
    """Select each geometric family without a family argument."""

    def test_curve_view_filters_shared_labels(self):
        """Select only curve bodies when all families share the same label."""
        builder = newton.ModelBuilder()
        _add_test_cable(builder, label="object")
        _add_test_cloth(builder, label="object")
        _add_test_soft_body(builder, label="object")
        scene = newton.ModelBuilder()
        scene.replicate(builder, 2)
        model = scene.finalize(device="cpu")

        view = newton.selection.DeformableCurveView(model, "*")

        self.assertEqual((view.family, view.labels, view.worlds), ("curve", ["object", "object"], [0, 1]))
        self.assertEqual(view.ranges("body"), [(0, 3), (3, 6)])
        self.assertEqual(view.deformable_object_ranges(), [(0, 1), (1, 2)])
        np.testing.assert_array_equal(view.world_ids.numpy(), [0, 1])
        self.assertEqual(view.get_body_transforms(model).shape, (2, 3))
        with self.assertRaisesRegex(AttributeError, "no particle elements"):
            view.ranges("particle")

    def test_particle_views_filter_shared_labels_and_isolate_writes(self):
        """Write only the masked deformable object in the requested family."""
        builder = newton.ModelBuilder()
        _add_test_cable(builder, label="object")
        _add_test_cloth(builder, label="object")
        _add_test_soft_body(builder, label="object")
        scene = newton.ModelBuilder()
        scene.replicate(builder, 2)
        model = scene.finalize(device="cpu")

        for view_type, family, ranges in (
            (newton.selection.DeformableSurfaceView, "surface", [(0, 4), (8, 12)]),
            (newton.selection.DeformableVolumeView, "volume", [(4, 8), (12, 16)]),
        ):
            with self.subTest(family=family):
                view = view_type(model, "*")
                self.assertEqual((view.family, view.labels, view.worlds), (family, ["object", "object"], [0, 1]))
                self.assertEqual(view.ranges("particle"), ranges)
                for attribute, getter, setter in (
                    ("particle_q", view.get_particle_positions, view.set_particle_positions),
                    ("particle_qd", view.get_particle_velocities, view.set_particle_velocities),
                ):
                    state = model.state()
                    expected = getattr(state, attribute).numpy().copy()
                    values = wp.array(np.full((2, 4, 3), 7.0, dtype=np.float32), dtype=wp.vec3, device="cpu")

                    setter(state, values, mask=[False, True])

                    start, end = ranges[1]
                    expected[start:end] = 7.0
                    np.testing.assert_array_equal(getattr(state, attribute).numpy(), expected)
                    np.testing.assert_array_equal(getter(state).numpy()[1], np.full((4, 3), 7.0))


class TestDeformableSelection(unittest.TestCase):
    """Match labels and partition selected deformable objects by world."""

    def test_compiled_regex_uses_shared_label_matching(self):
        """Deformable selection accepts the shared compiled-regex selector."""
        model = _replicated_model(2, device="cpu")

        view = DeformableSurfaceView(model, re.compile(r"/World/Cloth"))

        self.assertEqual((view.family, view.labels), ("surface", ["/World/Cloth", "/World/Cloth"]))

    def test_soft_view_over_global_objects(self):
        """Select global deformable objects (world -1) as a single-world view."""
        builder = newton.ModelBuilder()
        _add_test_soft_body(builder, label="/World/Soft")
        model = builder.finalize()

        view = DeformableVolumeView(model, "/World/Soft")
        self.assertEqual((view.count, view.world_count), (1, 1))
        self.assertEqual(view.deformable_object_ranges(), [(0, 1)])
        np.testing.assert_array_equal(view.deformable_object_boundaries.numpy(), [0, 1])
        np.testing.assert_array_equal(view.world_ids.numpy(), [-1])
        self.assertEqual(view.get_particle_positions(model.state()).shape, (1, 4))

    def test_pattern_matches_multiple_objects_per_world(self):
        """A wildcard pattern selects several deformable objects per world when counts stay equal."""
        sub = newton.ModelBuilder()
        _add_test_cloth(sub, label="/World/ClothA")
        _add_test_cloth(sub, label="/World/ClothB")
        scene = newton.ModelBuilder()
        scene.replicate(sub, 2)
        model = scene.finalize()

        view = DeformableSurfaceView(model, "/World/Cloth*")
        self.assertEqual((view.count, view.count_per_world), (4, 2))
        self.assertEqual(view.labels, ["/World/ClothA", "/World/ClothB"] * 2)
        self.assertEqual(view.ranges("triangle"), [(0, 2), (2, 4), (4, 6), (6, 8)])

    def test_varying_object_counts_across_worlds_remain_selectable(self):
        """Worlds may contribute different deformable object counts while retaining stable order."""
        first = newton.ModelBuilder()
        _add_test_cloth(first, label="/World/ClothA")
        second = newton.ModelBuilder()
        _add_test_cloth(second, label="/World/ClothA")
        _add_test_cloth(second, label="/World/ClothB")
        scene = newton.ModelBuilder()
        scene.add_world(first)
        scene.add_world(second)
        scene.add_world(newton.ModelBuilder())

        view = DeformableSurfaceView(scene.finalize(), "/World/Cloth*")

        self.assertEqual((view.count, view.world_count, view.count_per_world), (3, 3, None))
        self.assertEqual(view.worlds, [0, 1, 1])
        self.assertEqual(view.deformable_object_ranges(), [(0, 1), (1, 3), (3, 3)])
        np.testing.assert_array_equal(view.deformable_object_boundaries.numpy(), [0, 1, 3, 3])
        np.testing.assert_array_equal(view.world_ids.numpy(), [0, 1, 1])
        self.assertEqual(view.labels, ["/World/ClothA", "/World/ClothA", "/World/ClothB"])
        self.assertEqual(view.get_particle_positions(view.model.state()).shape, (3, 4))

    def test_deformable_object_ranges_slice_interleaved_objects_by_world(self):
        """Select each world's cables without gaps from intervening cloth and volume deformable objects."""
        builder = newton.ModelBuilder()
        builder.begin_world()
        _add_test_cloth(builder, label="cloth")
        _add_test_cable(builder, label="cable")
        builder.end_world()

        builder.begin_world()
        _add_test_cloth(builder, label="cloth")
        _add_test_cable(builder, label="cable_0")
        _add_test_cable(builder, label="cable_1")
        _add_test_soft_body(builder, label="volume")
        _add_test_cable(builder, label="cable_2")
        builder.end_world()

        builder.begin_world()
        _add_test_cloth(builder, label="cloth")
        _add_test_cable(builder, label="cable")
        builder.end_world()
        model = builder.finalize(device="cpu")

        cables = DeformableCurveView(model, "cable*")
        self.assertEqual(cables.labels, ["cable", "cable_0", "cable_1", "cable_2", "cable"])
        self.assertEqual(cables.deformable_object_ranges(), [(0, 1), (1, 4), (4, 5)])
        np.testing.assert_array_equal(cables.world_ids.numpy(), [0, 1, 1, 1, 2])
        np.testing.assert_array_equal(cables.deformable_object_boundaries.numpy(), [0, 1, 4, 5])

        deformable_object_index = 2
        self.assertEqual(cables.labels[deformable_object_index], "cable_1")
        self.assertEqual(cables.worlds[deformable_object_index], 1)
        self.assertEqual(cables.ranges("body")[deformable_object_index], (6, 9))
        self.assertEqual(cables.ranges("joint")[deformable_object_index], (6, 9))
        self.assertEqual(cables.bodies_per_deformable_object, 3)
        self.assertEqual(cables.elements_per_deformable_object("joint"), 3)
        self.assertEqual(cables.labels[3], "cable_2")

        world_id = 1
        start, end = cables.deformable_object_ranges()[world_id]
        self.assertEqual(cables.labels[start:end], ["cable_0", "cable_1", "cable_2"])

        cloths = DeformableSurfaceView(model, "cloth")
        self.assertEqual(cloths.deformable_object_ranges(), [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(cloths.ranges("particle"), [(0, 4), (4, 8), (12, 16)])
        self.assertEqual(cloths.ranges("triangle"), [(0, 2), (2, 4), (8, 10)])
        self.assertEqual(cloths.ranges("edge"), [(0, 5), (5, 10), (16, 21)])
        self.assertEqual(cloths.particles_per_deformable_object, 4)
        volumes = DeformableVolumeView(model, "volume")
        self.assertEqual(volumes.deformable_object_ranges(), [(0, 0), (0, 1), (1, 1)])
        self.assertEqual(volumes.ranges("particle"), [(8, 12)])
        self.assertEqual(volumes.ranges("tetrahedron"), [(0, 1)])
        self.assertEqual(volumes.particles_per_deformable_object, 4)

    def test_list_patterns_use_shared_label_matching(self):
        """A list of glob patterns follows the matching contract shared by selection views."""
        sub = newton.ModelBuilder()
        _add_test_cloth(sub, label="/World/ClothA")
        _add_test_cloth(sub, label="/World/ClothB")
        scene = newton.ModelBuilder()
        scene.replicate(sub, 2)
        model = scene.finalize()

        view = DeformableSurfaceView(model, ["/World/ClothA", "/World/ClothB"])
        self.assertEqual((view.count, view.count_per_world), (4, 2))
        self.assertEqual(view.labels, ["/World/ClothA", "/World/ClothB"] * 2)

    def test_selection_errors(self):
        """No match raises KeyError; ragged element counts and bad families raise ValueError."""
        builder = newton.ModelBuilder()
        _add_test_cloth(builder, label="/World/ClothA")  # 4 particles
        _add_test_cloth(
            builder,
            label="/World/ClothB",
            vertices=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
            ],
            indices=[0, 1, 2, 0, 2, 3, 1, 4, 2],
        )
        model = builder.finalize()

        with self.assertRaises(KeyError):
            DeformableSurfaceView(model, "/World/DoesNotExist")
        view = DeformableSurfaceView(model, "/World/Cloth*")
        self.assertEqual([end - start for start, end in view.ranges("particle")], [4, 5])
        np.testing.assert_array_equal(view.starts("particle").numpy(), [0, 4])
        with self.assertRaisesRegex(ValueError, "Varying particle counts.*ranges"):
            view.elements_per_deformable_object("particle")
        with self.assertRaisesRegex(ValueError, "Varying particle counts.*ranges"):
            view.get_particle_positions(model.state())
        with self.assertRaisesRegex(ValueError, "Varying particle counts.*ranges"):
            view.set_particle_positions(model.state(), wp.zeros((2, 4), dtype=wp.vec3))
        view = DeformableSurfaceView(model, "/World/ClothA")
        with self.assertRaisesRegex(AttributeError, "no body elements"):
            view.ranges("body")
        with self.assertRaises(ValueError):
            view.set_particle_positions(model.state(), wp.zeros((2, 4), dtype=wp.vec3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
