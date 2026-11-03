# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test deformable selection, state access, resets, and optional USD integration."""

import os
import re
import unittest
import weakref
from typing import Any

import numpy as np
import warp as wp

import newton
from newton.selection import ArticulationView, DeformableCurveView, DeformableSurfaceView, DeformableVolumeView
from newton.solvers import SolverVBD
from newton.solvers.experimental.coupled import SolverCoupled
from newton.tests._deformable_test_utils import _add_curve, _add_surface, _add_test_anchored_cable, _add_volume
from newton.tests._usd_deformable_test_utils import (
    _add_cable_curve,
    _add_cloth_mesh,
    _add_physics_attachment,
    _author_deformable_element_array,
    _bind_deformable_material,
    _deformable_stage,
)
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, get_cuda_test_devices, get_test_devices

_CABLE_PTS = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
_MIXED_ASSET = os.path.join(os.path.dirname(__file__), "assets", "deformables_mixed.usda")


def _replicated_model(world_count=3, device=None):
    """One cloth + one cable per world, replicated."""
    sub = newton.ModelBuilder()
    _add_test_cloth(sub, label="/World/Cloth")
    _add_test_cable(sub, label="/World/Cable")
    scene = newton.ModelBuilder()
    scene.replicate(sub, world_count)
    return scene.finalize() if device is None else scene.finalize(device=device)


def _irregular_model(device=None):
    """Three equal cloths and cables separated by unequal unrelated data."""
    builder = newton.ModelBuilder()
    _add_test_cloth(builder, label="selected_cloth_0")
    builder.add_particle(wp.vec3(0.0), wp.vec3(0.0), 1.0)
    _add_test_cloth(builder, label="selected_cloth_1")
    builder.add_particle(wp.vec3(0.0), wp.vec3(0.0), 1.0)
    builder.add_particle(wp.vec3(0.0), wp.vec3(0.0), 1.0)
    _add_test_cloth(builder, label="selected_cloth_2")

    _add_test_cable(builder, label="selected_cable_0")
    builder.add_link()
    _add_test_cable(builder, label="selected_cable_1")
    builder.add_link()
    builder.add_link()
    _add_test_cable(builder, label="selected_cable_2")
    return builder.finalize() if device is None else builder.finalize(device=device)


def _add_test_articulation(builder):
    root = builder.add_link(label="robot/root")
    root_joint = builder.add_joint_free(child=root, label="robot/root_joint")
    builder.add_articulation([root_joint], label="robot")


def _add_test_cable(builder, label="cable"):
    builder.add_rod(
        rod=newton.Rod(_CABLE_PTS, radius=0.02),
        label=label,
        body_frame_origin="com",
    )


def _add_test_cloth(builder, label="cloth", vertices=None, indices=None):
    if vertices is None:
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    if indices is None:
        indices = [0, 1, 2, 0, 2, 3]
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 2.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=vertices,
        indices=indices,
        density=1.0,
        label=label,
    )


def _add_test_soft_body(builder, label="soft"):
    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 3.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        indices=[0, 1, 2, 3],
        density=1.0,
        k_mu=100.0,
        k_lambda=100.0,
        k_damp=0.0,
        label=label,
    )


def _particle_property_model(device, *, irregular=False, requires_grad=False):
    """Create three worlds with distinct cloth and volume particle properties."""
    builder = newton.ModelBuilder()
    for world in range(3):
        builder.begin_world()
        for _ in range(world if irregular else 0):
            builder.add_particle(wp.vec3(0.0), wp.vec3(0.0), 100.0)
        _add_test_cloth(builder)
        _add_test_soft_body(builder)
        builder.particle_mass[-8:] = [mass * 2**world for mass in (0, 1, 2, 4, 8, 16, 32, 64)]
        builder.particle_flags[-8:] = [0, 1, 0, 1, 1, 0, 1, 0]
        builder.end_world()
    return builder.finalize(device=device, requires_grad=requires_grad)


@wp.kernel
def _sum_particle_properties(values: wp.array2d[wp.float32], loss: wp.array[float]):
    row, particle = wp.tid()
    wp.atomic_add(loss, 0, values[row, particle])


@wp.kernel
def _sum_first_component(values: wp.array2d[Any], loss: wp.array[float]):
    row, element = wp.tid()
    wp.atomic_add(loss, 0, values[row, element][0])


def _state_accessors(model, *, pattern="/World/*"):
    cloth = DeformableSurfaceView(model, pattern)
    cable = DeformableCurveView(model, pattern)
    return (
        (cloth.get_particle_positions, cloth.set_particle_positions, "particle_q"),
        (cloth.get_particle_velocities, cloth.set_particle_velocities, "particle_qd"),
        (cable.get_body_transforms, cable.set_body_transforms, "body_q"),
        (cable.get_body_velocities, cable.set_body_velocities, "body_qd"),
    )


def _reset_cloths(cloths, states, indices_world, mask_cloths, positions_default, velocities_default):
    cloths.set_mask_from_indices(mask_cloths, indices_world)
    for state in states:
        cloths.set_particle_positions(state, positions_default, mask=mask_cloths)
        cloths.set_particle_velocities(state, velocities_default, mask=mask_cloths)


class TestDeformableFamilyViews(unittest.TestCase):
    """Select each geometric family without a family argument."""

    def test_family_views_filter_shared_labels_and_isolate_writes(self):
        """Select each family, log a curve summary, and isolate writes with shared labels."""
        builder = newton.ModelBuilder()
        _add_test_cable(builder, label="object")
        _add_test_cloth(builder, label="object")
        _add_test_soft_body(builder, label="object")
        scene = newton.ModelBuilder()
        scene.replicate(builder, 2)
        model = scene.finalize(device="cpu")

        with self.assertLogs("newton", level="INFO") as captured:
            view = newton.selection.DeformableCurveView(model, "*", verbose=True)
        self.assertIn("DeformableCurveView '*' (curve): 2 deformable object(s)", captured.output[0])

        self.assertEqual((view.family, view.labels, view.worlds), ("curve", ["object", "object"], [0, 1]))
        self.assertEqual(view.ranges("body"), [(0, 3), (3, 6)])
        # Enum members name the same element kind as their lowercase names.
        self.assertEqual(view.ranges(newton.Model.AttributeFrequency.BODY), [(0, 3), (3, 6)])
        self.assertIs(view.starts(newton.Model.AttributeFrequency.JOINT), view.starts("joint"))
        self.assertEqual(view.element_count(newton.Model.AttributeFrequency.BODY), 3)
        self.assertEqual(view.deformable_object_ranges(), [(0, 1), (1, 2)])
        np.testing.assert_array_equal(view.worlds_device.numpy(), [0, 1])
        self.assertEqual(view.get_body_transforms(model).shape, (2, 3))
        for kind in ("particle", newton.Model.AttributeFrequency.PARTICLE):
            with self.assertRaisesRegex(ValueError, "record body, joint elements, not 'particle'"):
                view.ranges(kind)

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

    def test_soft_view_over_global_objects(self):
        """Select global deformable objects (world -1) as a single partition."""
        builder = newton.ModelBuilder()
        _add_test_soft_body(builder, label="/World/Soft")
        model = builder.finalize()

        view = DeformableVolumeView(model, "/World/Soft")
        self.assertEqual(view.count, 1)
        self.assertEqual(view.count_per_world, 1)
        self.assertEqual(view.deformable_object_ranges(), [(0, 1)])
        np.testing.assert_array_equal(view.deformable_object_boundaries.numpy(), [0, 1])
        np.testing.assert_array_equal(view.worlds_device.numpy(), [-1])
        self.assertEqual(view.get_particle_positions(model.state()).shape, (1, 4))

    def test_label_patterns_select_multiple_objects_per_world(self):
        """Match glob, list, and compiled-regex patterns against the same world layout."""
        prototype = newton.ModelBuilder()
        _add_test_cloth(prototype, label="/World/ClothA")
        _add_test_cloth(prototype, label="/World/ClothB")
        builder = newton.ModelBuilder()
        builder.replicate(prototype, 2)
        model = builder.finalize(device="cpu")
        for pattern in (
            "/World/Cloth*",
            ["/World/ClothA", "/World/ClothB"],
            re.compile(r"/World/Cloth[AB]"),
            [0, 1, 2, 3],
        ):
            with self.subTest(pattern=pattern):
                view = DeformableSurfaceView(model, pattern)
                self.assertEqual((view.count, view.count_per_world), (4, 2))
                self.assertEqual(view.labels, ["/World/ClothA", "/World/ClothB"] * 2)
                self.assertEqual(view.ranges("triangle"), [(0, 2), (2, 4), (4, 6), (6, 8)])

        view = DeformableSurfaceView(model, [0, 2])
        self.assertEqual(view.labels, ["/World/ClothA"] * 2)
        self.assertEqual(view.worlds, [0, 1])
        self.assertEqual(view.ranges("triangle"), [(0, 2), (4, 6)])

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

        self.assertEqual((view.count, view.count_per_world), (3, None))
        self.assertEqual(view.worlds, [0, 1, 1])
        self.assertEqual(view.deformable_object_ranges(), [(0, 1), (1, 3), (3, 3)])
        np.testing.assert_array_equal(view.deformable_object_boundaries.numpy(), [0, 1, 3, 3])
        np.testing.assert_array_equal(view.worlds_device.numpy(), [0, 1, 1])
        self.assertEqual(view.labels, ["/World/ClothA", "/World/ClothA", "/World/ClothB"])
        self.assertEqual(view.get_particle_positions(view.model.state()).shape, (3, 4))

        # Equal counts in the matching worlds do not hide the empty third world.
        single = DeformableSurfaceView(view.model, "/World/ClothA")
        self.assertIsNone(single.count_per_world)
        self.assertEqual(single.deformable_object_ranges(), [(0, 1), (1, 2), (2, 2)])

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
        np.testing.assert_array_equal(cables.worlds_device.numpy(), [0, 1, 1, 1, 2])
        np.testing.assert_array_equal(cables.deformable_object_boundaries.numpy(), [0, 1, 4, 5])

        deformable_object_index = 2
        self.assertEqual(cables.labels[deformable_object_index], "cable_1")
        self.assertEqual(cables.worlds[deformable_object_index], 1)
        self.assertEqual(cables.ranges("body")[deformable_object_index], (6, 9))
        self.assertEqual(cables.ranges("joint")[deformable_object_index], (6, 9))
        self.assertEqual(cables.body_count, 3)
        self.assertEqual(cables.element_count("joint"), 3)
        self.assertEqual(cables.labels[3], "cable_2")

        world_id = 1
        start, end = cables.deformable_object_ranges()[world_id]
        self.assertEqual(cables.labels[start:end], ["cable_0", "cable_1", "cable_2"])

        cloths = DeformableSurfaceView(model, "cloth")
        self.assertEqual(cloths.deformable_object_ranges(), [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(cloths.ranges("particle"), [(0, 4), (4, 8), (12, 16)])
        self.assertEqual(cloths.ranges("triangle"), [(0, 2), (2, 4), (8, 10)])
        self.assertEqual(cloths.ranges("edge"), [(0, 5), (5, 10), (16, 21)])
        self.assertEqual(cloths.particle_count, 4)
        volumes = DeformableVolumeView(model, "volume")
        self.assertEqual(volumes.deformable_object_ranges(), [(0, 0), (0, 1), (1, 1)])
        self.assertEqual(volumes.ranges("particle"), [(8, 12)])
        self.assertEqual(volumes.ranges("tetrahedron"), [(0, 1)])
        self.assertEqual(volumes.particle_count, 4)

    def test_selection_errors(self):
        """Report unequal counts without raising, but reject batched access and unavailable kinds."""
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
        self.assertIsNone(view.particle_count)
        self.assertIsNone(view.element_count("particle"))
        self.assertIsNone(view.element_count(newton.Model.AttributeFrequency.PARTICLE))
        with self.assertRaisesRegex(ValueError, "Varying particle counts.*ranges"):
            view.get_particle_positions(model.state())
        for getter in (view.get_particle_inv_masses, view.get_particle_flags):
            with self.subTest(getter=getter.__name__):
                with self.assertRaisesRegex(ValueError, "Varying particle counts.*ranges"):
                    getter(model)
        with self.assertRaisesRegex(ValueError, "Varying particle counts.*ranges"):
            view.set_particle_positions(model.state(), wp.zeros((2, 4), dtype=wp.vec3))
        view = DeformableSurfaceView(model, "/World/ClothA")
        for accessor in (view.element_count, view.ranges, view.starts):
            with self.subTest(accessor=accessor.__name__):
                for kind in ("body", newton.Model.AttributeFrequency.BODY):
                    with self.assertRaisesRegex(ValueError, "record particle, triangle, edge elements, not 'body'"):
                        accessor(kind)
                with self.assertRaisesRegex(TypeError, "kind must be"):
                    accessor(None)
        with self.assertRaises(ValueError):
            view.set_particle_positions(model.state(), wp.zeros((2, 4), dtype=wp.vec3))


class TestDeformableBuilderObjects(unittest.TestCase):
    """Select deformable objects recorded by native builder calls without requiring USD."""

    def test_native_constructors_are_selectable(self):
        """Select each native constructor's ranges from one mixed model."""
        builder = newton.ModelBuilder()
        _add_curve(builder)
        _add_curve(builder, label="closed", topology="closed")
        _add_curve(builder, label="graph", topology="graph")
        _add_surface(builder)
        _add_surface(builder, label="cloth_grid", grid=True)
        _add_volume(builder)
        _add_volume(builder, label="soft_grid", grid=True)
        model = builder.finalize(device="cpu")
        for view_type, label, ranges in (
            (DeformableCurveView, "curve_0", {"body": (0, 3), "joint": (0, 3)}),
            (DeformableCurveView, "closed", {"body": (3, 6), "joint": (3, 7)}),
            (DeformableCurveView, "graph", {"body": (6, 9), "joint": (7, 10)}),
            (DeformableSurfaceView, "surface_0", {"particle": (0, 4), "triangle": (0, 2), "edge": (0, 5)}),
            (DeformableSurfaceView, "cloth_grid", {"particle": (4, 8), "triangle": (2, 4), "edge": (5, 10)}),
            (DeformableVolumeView, "volume_0", {"particle": (8, 12), "tetrahedron": (0, 1)}),
            (DeformableVolumeView, "soft_grid", {"particle": (12, 20), "tetrahedron": (1, 6)}),
        ):
            with self.subTest(label=label):
                view = view_type(model, label)
                self.assertEqual((view.count, view.labels, view.worlds), (1, [label], [-1]))
                for kind, (start, end) in ranges.items():
                    self.assertEqual(view.ranges(kind), [(start, end)])
                    self.assertEqual(view.element_count(kind), end - start)
                if view_type is DeformableCurveView:
                    self.assertEqual(model.joint_type.numpy()[ranges["joint"][0]], newton.JointType.FREE)

    def test_default_and_edited_labels_survive_replication(self):
        """Select generated and application-edited labels in every copied world."""
        prototype = newton.ModelBuilder()
        _add_test_cable(prototype, label=None)
        _add_test_cloth(prototype, label=None)
        _add_test_soft_body(prototype, label=None)
        for edited in (False, True):
            if edited:
                prototype.curve_label[:] = ["/Template/Cable"]
                prototype.surface_label = ["/Template/Cloth"]
                prototype.volume_label[0] = "/Template/SoftBody"
            builder = newton.ModelBuilder()
            builder.replicate(prototype, 2)
            model = builder.finalize(device="cpu")
            for view_type, label in (
                (DeformableCurveView, "/Template/Cable" if edited else "curve_0"),
                (DeformableSurfaceView, "/Template/Cloth" if edited else "surface_0"),
                (DeformableVolumeView, "/Template/SoftBody" if edited else "volume_0"),
            ):
                with self.subTest(edited=edited, label=label):
                    view = view_type(model, label)
                    self.assertEqual((view.count, view.labels, view.worlds), (2, [label, label], [0, 1]))

    def test_mixed_native_deformables_replicate_with_offset_ranges(self):
        """Curve, surface, and volume deformable objects retain disjoint ranges after replication."""
        prototype = newton.ModelBuilder()

        # Curve first: three segment bodies, two rod joints, and a free root.
        _add_curve(prototype, label="curve", topology="graph")

        # Surface second: four particles, two triangles, and five cloth edges.
        _add_test_cloth(prototype, label="surface")

        # Volume last: eight particles and five tetrahedra.
        _add_volume(prototype, label="volume", grid=True)

        scene = newton.ModelBuilder()
        scene.replicate(prototype, 2)
        model = scene.finalize(device="cpu")
        state = model.state()

        curve = DeformableCurveView(model, "curve")
        self.assertEqual((curve.count, curve.worlds, curve.count_per_world), (2, [0, 1], 1))
        np.testing.assert_array_equal(curve.deformable_object_boundaries.numpy(), [0, 1, 2])
        self.assertEqual(curve.ranges("body"), [(0, 3), (3, 6)])
        self.assertEqual(curve.ranges("joint"), [(0, 3), (3, 6)])
        self.assertEqual(curve.get_body_transforms(state).shape, (2, 3))

        surface = DeformableSurfaceView(model, "surface")
        self.assertEqual((surface.count, surface.worlds, surface.count_per_world), (2, [0, 1], 1))
        np.testing.assert_array_equal(surface.deformable_object_boundaries.numpy(), [0, 1, 2])
        self.assertEqual(surface.ranges("particle"), [(0, 4), (12, 16)])
        self.assertEqual(surface.ranges("triangle"), [(0, 2), (14, 16)])
        self.assertEqual(surface.ranges("edge"), [(0, 5), (23, 28)])
        self.assertEqual(surface.get_particle_positions(state).shape, (2, 4))

        volume = DeformableVolumeView(model, "volume")
        self.assertEqual((volume.count, volume.worlds, volume.count_per_world), (2, [0, 1], 1))
        np.testing.assert_array_equal(volume.deformable_object_boundaries.numpy(), [0, 1, 2])
        self.assertEqual(volume.ranges("particle"), [(4, 12), (16, 24)])
        self.assertEqual(volume.ranges("tetrahedron"), [(0, 5), (5, 10)])
        self.assertEqual(volume.get_particle_positions(state).shape, (2, 8))

    def test_builder_built_prototype_clones_select_per_world(self):
        """A labeled soft body and rod built in a prototype and cloned per world with
        add_world stay selectable, with correctly offset ranges (the Isaac Lab pattern
        of building deformables in per-world builder hooks)."""
        proto = newton.ModelBuilder()
        _add_test_soft_body(proto, label="soft_proto")
        proto.add_rod(
            rod=newton.Rod([(0.0, 2.0, 1.0), (0.1, 2.0, 1.0), (0.2, 2.0, 1.0)], radius=0.02),
            label="cable_proto",
            wrap_in_articulation=True,
            body_frame_origin="com",
        )

        scene = newton.ModelBuilder()
        scene.add_world(proto)
        scene.add_world(proto)
        model = scene.finalize()
        state = model.state()

        soft = DeformableVolumeView(model, "soft_proto")
        self.assertEqual((soft.count, soft.worlds, soft.particle_count), (2, [0, 1], 4))
        (r0, r1) = soft.ranges("particle")
        self.assertEqual(r1[0] - r0[0], 4)
        self.assertNotEqual(r0, r1)

        cable = DeformableCurveView(model, "cable_proto")
        self.assertEqual((cable.count, cable.worlds, cable.body_count), (2, [0, 1], 2))
        self.assertEqual(cable.element_count("joint"), 2)

        # State access round-trips through the offset ranges.
        positions = soft.get_particle_positions(state)
        lifted = positions.numpy().copy()
        lifted[1, :, 2] += 3.0
        soft.set_particle_positions(state, wp.array(lifted, dtype=wp.vec3))
        np.testing.assert_allclose(soft.get_particle_positions(state).numpy(), lifted, atol=1e-6)

    def test_fixed_joint_collapse_selects_remaining_curve_bodies(self):
        """Select the remaining rod parts, or all parts when its fixed anchor is kept."""
        for keep_anchor, count in ((False, 1), (True, 2)):
            with self.subTest(keep_anchor=keep_anchor):
                builder = newton.ModelBuilder()
                _add_test_anchored_cable(builder)
                self.assertEqual((builder.curve_label, builder.curve_world), (["anchored_curve"], [-1]))

                builder.collapse_fixed_joints(joints_to_keep=["anchor"] if keep_anchor else [])

                self.assertEqual((builder.body_count, builder.joint_count), (count, count))
                self.assertEqual((builder.curve_label, builder.curve_world), (["anchored_curve"], [-1]))
                model = builder.finalize()
                self.assertEqual((model.body_count, model.joint_count), (count, count))
                view = DeformableCurveView(model, "anchored_curve")
                self.assertEqual(view.ranges("body"), [(0, count)])
                self.assertEqual(view.ranges("joint"), [(0, 1)])
                self.assertEqual(view.get_body_transforms(model.state()).shape, (1, count))

    def test_reduced_curve_ranges_survive_cloning(self):
        """Clone reduced rods and their references before or after collapse."""
        for cloning in ("replicate", "add_world"):
            for collapse_first in (False, True):
                with self.subTest(cloning=cloning, collapse_first=collapse_first):
                    prototype = newton.ModelBuilder()
                    _add_test_anchored_cable(prototype, label="cable")
                    prototype.add_custom_attribute(
                        newton.ModelBuilder.CustomAttribute(
                            name="curve_source",
                            dtype=wp.int32,
                            frequency=newton.Model.AttributeFrequency.CURVE,
                            references=newton.Model.AttributeFrequency.CURVE,
                            values={0: 0},
                        )
                    )
                    if collapse_first:
                        prototype.collapse_fixed_joints()
                    scene = newton.ModelBuilder()
                    if cloning == "replicate":
                        scene.replicate(prototype, 2)
                    else:
                        for _ in range(2):
                            scene.add_world(prototype)
                    if not collapse_first:
                        scene.collapse_fixed_joints()
                    model = scene.finalize(device="cpu")
                    view = DeformableCurveView(model, "cable")
                    self.assertEqual(view.labels, ["cable", "cable"])
                    self.assertEqual(view.worlds, [0, 1])
                    self.assertEqual(view.ranges("body"), [(0, 1), (1, 2)])
                    self.assertEqual(view.ranges("joint"), [(0, 1), (1, 2)])
                    np.testing.assert_array_equal(model.curve_source.numpy(), [0, 1])

    def test_reduced_curve_excludes_external_merge_parent(self):
        """Keep only the rod's own surviving bodies, including after repeated collapse."""
        builder = newton.ModelBuilder()
        parent = builder.add_link(label="robot/root")
        root = builder.add_joint_free(parent)
        bodies, joints = builder.add_rod(
            rod=newton.Rod(_CABLE_PTS[:3], radius=0.02),
            label="anchored_curve",
            wrap_in_articulation=False,
            body_frame_origin="com",
        )
        anchor = builder.add_joint_fixed(parent, bodies[0])
        builder.add_articulation([root, *joints, anchor], label="robot_with_cable")
        for _ in range(2):
            builder.collapse_fixed_joints()
            model = builder.finalize(device="cpu")
            view = DeformableCurveView(model, "anchored_curve")
            self.assertEqual(model.body_count, 2)
            self.assertEqual(model.body_label[0], "robot/root")
            self.assertEqual(view.ranges("body"), [(1, 2)])
            self.assertEqual(view.ranges("joint"), [(1, 2)])

    def test_different_reductions_keep_world_ranges(self):
        """Expose unequal surviving body counts without inventing a rectangular layout."""
        scene = newton.ModelBuilder()
        for keep_anchor in (False, True):
            prototype = newton.ModelBuilder()
            _add_test_anchored_cable(prototype, label="cable")
            prototype.collapse_fixed_joints(joints_to_keep=["anchor"] if keep_anchor else [])
            scene.add_world(prototype)
        scene.begin_world()
        scene.end_world()
        model = scene.finalize(device="cpu")
        view = DeformableCurveView(model, "cable")
        self.assertEqual(view.worlds, [0, 1])
        self.assertEqual(view.deformable_object_ranges(), [(0, 1), (1, 2), (2, 2)])
        self.assertEqual(view.ranges("body"), [(0, 1), (1, 3)])
        self.assertIsNone(view.body_count)
        self.assertIsNone(view.element_count("body"))
        self.assertIsNone(view.element_count(newton.Model.AttributeFrequency.BODY))
        with self.assertRaises(ValueError):
            view.get_body_transforms(model.state())


def test_reduced_rod_masked_writes_in_mixed_worlds(test, device):
    """Reset one reduced rod without changing other worlds, rigid bodies, cloth, or volume."""

    def reset(view, target, positions, velocities, mask):
        view.set_body_transforms(target, positions, mask=mask)
        view.set_body_velocities(target, velocities, mask=mask)

    prototype = newton.ModelBuilder()
    _add_test_articulation(prototype)
    _add_test_anchored_cable(prototype, label="cable")
    _add_test_cloth(prototype)
    _add_test_soft_body(prototype)
    prototype.collapse_fixed_joints()
    builder = newton.ModelBuilder()
    builder.replicate(prototype, 2)
    model = builder.finalize(device=device)
    state = model.state()
    cable = DeformableCurveView(model, "cable")
    articulation = ArticulationView(model, prototype.articulation_label[1])
    cloth = DeformableSurfaceView(model, "cloth")
    volume = DeformableVolumeView(model, "soft")
    before = {name: getattr(state, name).numpy().copy() for name in ("body_q", "body_qd", "particle_q", "particle_qd")}
    positions = cable.get_body_transforms(state).numpy().copy()
    positions[:, :, 2] += 1.0
    velocities = np.full_like(cable.get_body_velocities(state).numpy(), 2.0)
    positions_wp = wp.array(positions, dtype=wp.transform, device=device)
    velocities_wp = wp.array(velocities, dtype=wp.spatial_vector, device=device)
    mask = wp.array([False, True], dtype=wp.bool, device=device)

    # Warm kernels without changing the state under test.
    reset(cable, model.state(), positions_wp, velocities_wp, mask)
    if device.is_cuda:
        with wp.ScopedMempool(device, enable=False):
            with wp.ScopedCapture(device) as capture:
                reset(cable, state, positions_wp, velocities_wp, mask)
        for _ in range(2):
            wp.capture_launch(capture.graph)
    else:
        reset(cable, state, positions_wp, velocities_wp, mask)

    expected_q = before["body_q"].copy()
    expected_qd = before["body_qd"].copy()
    expected_q[3:4] = positions[1]
    expected_qd[3:4] = velocities[1]
    np.testing.assert_array_equal(state.body_q.numpy(), expected_q)
    np.testing.assert_array_equal(state.body_qd.numpy(), expected_qd)
    for name in ("particle_q", "particle_qd"):
        np.testing.assert_array_equal(getattr(state, name).numpy(), before[name])
    np.testing.assert_array_equal(
        articulation.get_link_transforms(state).numpy().reshape(2, 1, 7),
        cable.get_body_transforms(state).numpy(),
    )
    np.testing.assert_array_equal(
        articulation.get_link_velocities(state).numpy().reshape(2, 1, 6),
        cable.get_body_velocities(state).numpy(),
    )
    test.assertEqual(cloth.get_particle_positions(state).shape, (2, 4))
    test.assertEqual(volume.get_particle_positions(state).shape, (2, 4))


def test_sparse_rigid_and_uneven_deformable_masks(test, device):
    """Reset world 2 through sparse robot and uneven deformable selections."""
    world_0 = newton.ModelBuilder()
    _add_test_articulation(world_0)
    _add_test_cable(world_0)
    _add_test_cloth(world_0)
    world_1 = newton.ModelBuilder()
    _add_test_articulation(world_1)
    world_1.articulation_label[0] = "other"
    _add_test_soft_body(world_1)
    world_2 = newton.ModelBuilder()
    _add_test_articulation(world_2)
    for _ in range(2):
        _add_test_cable(world_2)
    for _ in range(2):
        _add_test_cloth(world_2)
    _add_test_soft_body(world_2)
    scene = newton.ModelBuilder()
    scene.add_world(world_0)
    scene.add_world(world_1)
    scene.add_world(world_2)

    model = scene.finalize(device=device)
    state = model.state()
    rigid = ArticulationView(model, "robot", homogeneous=False)
    all_rigid = ArticulationView(model, ["robot", "other"], homogeneous=False)
    cable = DeformableCurveView(model, "cable")
    cloth = DeformableSurfaceView(model, "cloth")
    soft = DeformableVolumeView(model, "soft")

    np.testing.assert_array_equal(rigid.world_ids.numpy(), [0, 2])
    test.assertEqual((rigid.world_count, rigid.count_per_world), (2, 1))
    values = wp.ones_like(rigid.get_dof_velocities(state))
    mask_worlds = wp.array([False, True], dtype=bool, device=device)
    rigid.set_dof_velocities(state, values, mask=mask_worlds)
    expected_rigid = np.zeros((3, 1, 6), dtype=np.float32)
    expected_rigid[2] = 1.0
    np.testing.assert_array_equal(all_rigid.get_dof_velocities(state).numpy(), expected_rigid)

    expected = {name: getattr(state, name).numpy().copy() for name in ("body_qd", "particle_qd", "joint_qd")}
    for view, kind, name, worlds, boundaries, setter in (
        (cable, "body", "body_qd", [0, 2, 2], [0, 1, 1, 3], cable.set_body_velocities),
        (cloth, "particle", "particle_qd", [0, 2, 2], [0, 1, 1, 3], cloth.set_particle_velocities),
        (soft, "particle", "particle_qd", [1, 2], [0, 0, 1, 2], soft.set_particle_velocities),
    ):
        with test.subTest(family=view.family):
            test.assertIsNone(view.count_per_world)
            test.assertEqual(view.worlds, worlds)
            np.testing.assert_array_equal(view.worlds_device.numpy(), worlds)
            np.testing.assert_array_equal(view.deformable_object_boundaries.numpy(), boundaries)
            array = getattr(state, name)
            shape = (view.count, view.element_count(kind), *expected[name].shape[1:])
            values = wp.array(np.full(shape, 2.0, dtype=np.float32), dtype=array.dtype, device=device)
            mask = wp.array([world == 2 for world in worlds], dtype=bool, device=device)
            setter(state, values, mask=mask)
            for world, (start, end) in zip(worlds, view.ranges(kind), strict=True):
                if world == 2:
                    expected[name][start:end] = 2.0
            for field, expected_values in expected.items():
                np.testing.assert_array_equal(getattr(state, field).numpy(), expected_values)


class TestDeformableAndArticulationViews(unittest.TestCase):
    """Rigid and deformable selections sharing finalized mixed models."""

    def test_rigid_cable_cloth_and_volume_survive_unrelated_collapse(self):
        """All selection families coexist after an unrelated fixed joint collapses."""
        builder = newton.ModelBuilder()
        collapsed_body = builder.add_link(label="collapsed_body")
        builder.add_joint_fixed(parent=-1, child=collapsed_body, label="collapse_me")
        _add_test_articulation(builder)
        cable_bodies, _cable_joints = builder.add_rod(
            rod=newton.Rod([(0.0, 2.0, 1.0), (0.1, 2.0, 1.0), (0.2, 2.0, 1.0)], radius=0.02),
            label="cable",
            body_frame_origin="com",
        )
        _add_test_cloth(builder)
        _add_test_soft_body(builder)

        body_count = builder.body_count
        joint_count = builder.joint_count
        builder.collapse_fixed_joints()
        self.assertEqual(builder.body_count, body_count - 1)
        self.assertEqual(builder.joint_count, joint_count - 1)

        model = builder.finalize()
        state = model.state()
        rigid = ArticulationView(model, "robot")
        cable = DeformableCurveView(model, "cable")
        cloth = DeformableSurfaceView(model, "cloth")
        soft = DeformableVolumeView(model, "soft")

        self.assertEqual(rigid.get_root_transforms(state).shape, (1, 1))
        self.assertEqual(cable.get_body_transforms(state).shape, (1, 2))
        self.assertEqual(cloth.get_particle_positions(state).shape, (1, 4))
        self.assertEqual(soft.get_particle_positions(state).shape, (1, 4))
        self.assertEqual(cable.ranges("body"), [(cable_bodies[0] - 1, cable_bodies[-1])])

        rigid_values = rigid.get_root_transforms(state).numpy().copy()
        cable_values = cable.get_body_transforms(state).numpy().copy()
        cloth_values = cloth.get_particle_positions(state).numpy().copy()
        soft_values = soft.get_particle_positions(state).numpy().copy()

        moved_rigid = rigid_values.copy()
        moved_rigid[..., 0] += 1.0
        rigid.set_root_transforms(state, wp.array(moved_rigid, dtype=wp.transform, device=model.device))
        np.testing.assert_array_equal(cable.get_body_transforms(state).numpy(), cable_values)
        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), cloth_values)
        np.testing.assert_array_equal(soft.get_particle_positions(state).numpy(), soft_values)

        moved_cable = cable_values.copy()
        moved_cable[..., 1] += 1.0
        cable.set_body_transforms(state, wp.array(moved_cable, dtype=wp.transform, device=model.device))
        np.testing.assert_allclose(rigid.get_root_transforms(state).numpy(), moved_rigid, atol=1e-6)
        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), cloth_values)
        np.testing.assert_array_equal(soft.get_particle_positions(state).numpy(), soft_values)

        moved_cloth = cloth_values.copy()
        moved_cloth[..., 2] += 1.0
        cloth.set_particle_positions(state, wp.array(moved_cloth, dtype=wp.vec3, device=model.device))
        np.testing.assert_allclose(cable.get_body_transforms(state).numpy(), moved_cable, atol=1e-6)
        np.testing.assert_array_equal(soft.get_particle_positions(state).numpy(), soft_values)

        moved_soft = soft_values.copy()
        moved_soft[..., 0] += 1.0
        soft.set_particle_positions(state, wp.array(moved_soft, dtype=wp.vec3, device=model.device))
        np.testing.assert_allclose(rigid.get_root_transforms(state).numpy(), moved_rigid, atol=1e-6)
        np.testing.assert_allclose(cable.get_body_transforms(state).numpy(), moved_cable, atol=1e-6)
        np.testing.assert_allclose(cloth.get_particle_positions(state).numpy(), moved_cloth, atol=1e-6)
        np.testing.assert_allclose(soft.get_particle_positions(state).numpy(), moved_soft, atol=1e-6)

    def test_views_update_disjoint_state(self):
        """Keep rigid, cloth, and volume view writes disjoint across replicated worlds."""
        prototype = newton.ModelBuilder()
        _add_test_articulation(prototype)
        _add_test_cloth(prototype)
        _add_test_soft_body(prototype)
        scene = newton.ModelBuilder()
        scene.replicate(prototype, 2)
        model = scene.finalize()
        state = model.state()

        rigid = ArticulationView(model, "robot")
        cloth = DeformableSurfaceView(model, "cloth")
        soft = DeformableVolumeView(model, "soft")
        self.assertEqual(rigid.get_root_transforms(state).shape, (2, 1))
        self.assertEqual(cloth.get_particle_positions(state).shape, (2, 4))
        self.assertEqual(soft.get_particle_positions(state).shape, (2, 4))

        rigid_before = rigid.get_root_transforms(state).numpy().copy()
        soft_before = soft.get_particle_positions(state).numpy().copy()
        cloth_values = cloth.get_particle_positions(state).numpy().copy()
        cloth_values[1, :, 2] += 2.0
        cloth.set_particle_positions(state, wp.array(cloth_values, dtype=wp.vec3, device=model.device))
        np.testing.assert_array_equal(rigid.get_root_transforms(state).numpy(), rigid_before)
        np.testing.assert_array_equal(soft.get_particle_positions(state).numpy(), soft_before)

        rigid_values = rigid_before.copy()
        rigid_values[0, 0, 0] += 1.0
        rigid.set_root_transforms(state, wp.array(rigid_values, dtype=wp.transform, device=model.device))
        np.testing.assert_allclose(rigid.get_root_transforms(state).numpy(), rigid_values, atol=1e-6)
        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), cloth_values)

    def test_rigid_view_coexists_with_reduced_curve_object(self):
        """Keep an unrelated rigid view valid alongside a partially collapsed rod."""
        builder = newton.ModelBuilder()
        _add_test_articulation(builder)
        _add_test_anchored_cable(builder)

        builder.collapse_fixed_joints()
        model = builder.finalize()

        rigid = ArticulationView(model, "robot")
        self.assertEqual(rigid.get_root_transforms(model).shape, (1, 1))
        cable = DeformableCurveView(model, "anchored_curve")
        self.assertEqual(cable.ranges("body"), [(1, 2)])


for _test in (
    test_reduced_rod_masked_writes_in_mixed_worlds,
    test_sparse_rigid_and_uneven_deformable_masks,
):
    add_function_test(TestDeformableAndArticulationViews, _test.__name__, _test, devices=get_test_devices())


def test_particle_property_snapshots(test, device):
    """Read selected Model properties without changing saved snapshots."""
    for irregular in (False, True):
        model = _particle_property_model(device, irregular=irregular)
        for view_type, label, row, flags in (
            (DeformableSurfaceView, "cloth", [0.0, 1.0, 0.5, 0.25], [0, 1, 0, 1]),
            (DeformableVolumeView, "soft", [0.125, 0.0625, 0.03125, 0.015625], [1, 0, 1, 0]),
        ):
            with test.subTest(device=device, irregular=irregular, family=label):
                view = view_type(model, label)
                inverse_masses = np.array(row, dtype=np.float32)[None, :] / np.array([1, 2, 4])[:, None]
                for getter, source, expected in (
                    (view.get_particle_inv_masses, model.particle_inv_mass, inverse_masses),
                    (view.get_particle_flags, model.particle_flags, np.tile(flags, (3, 1))),
                ):
                    with test.subTest(getter=getter.__name__):
                        original = wp.clone(source)
                        values = getter(model)
                        test.assertEqual(values.shape, view.get_particle_positions(model).shape)
                        test.assertEqual(values.shape, (3, 4))
                        test.assertIs(values.dtype, source.dtype)
                        np.testing.assert_array_equal(values.numpy(), expected)
                        saved = wp.clone(values)
                        source.zero_()
                        test.assertIs(getter(model), values)
                        np.testing.assert_array_equal(values.numpy(), np.zeros((3, 4)))
                        np.testing.assert_array_equal(saved.numpy(), expected)
                        wp.copy(source, original)
                        with test.assertRaises(AttributeError):
                            getter(model.state())


def test_particle_property_gradients(test, device):
    """Propagate inverse-mass gradients only to the selected family's particles."""
    for irregular in (False, True):
        model = _particle_property_model(device, irregular=irregular, requires_grad=True)
        for view_type, label, offset in (
            (DeformableSurfaceView, "cloth", 0),
            (DeformableVolumeView, "soft", 4),
        ):
            with test.subTest(device=device, irregular=irregular, family=label):
                view = view_type(model, label)
                loss = wp.zeros(1, device=device, requires_grad=True)
                with wp.Tape() as tape:
                    values = view.get_particle_inv_masses(model)
                    wp.launch(_sum_particle_properties, dim=values.shape, inputs=[values, loss], device=device)
                tape.backward(loss)
                expected = np.zeros(model.particle_count)
                for start in (0, 9, 19) if irregular else (0, 8, 16):
                    expected[start + offset : start + offset + 4] = 1.0
                np.testing.assert_array_equal(model.particle_inv_mass.grad.numpy(), expected)
                test.assertFalse(view.get_particle_flags(model).requires_grad)
                tape.zero()


def test_particle_property_capture(test, device):
    """Capture warmed Model reads and observe property changes on replay without allocations."""
    for irregular in (False, True):
        model = _particle_property_model(device, irregular=irregular)
        for view_type, label in ((DeformableSurfaceView, "cloth"), (DeformableVolumeView, "soft")):
            view = view_type(model, label)
            for getter, source in (
                (view.get_particle_inv_masses, model.particle_inv_mass),
                (view.get_particle_flags, model.particle_flags),
            ):
                with test.subTest(irregular=irregular, family=label, getter=getter.__name__):
                    original = wp.clone(source)
                    result = getter(model)
                    output = wp.clone(result)
                    expected = output.numpy().copy()
                    with wp.ScopedMempool(device, enable=False):
                        with wp.ScopedCapture(device) as capture:
                            wp.copy(output, getter(model))
                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(output.numpy(), expected)
                    source.zero_()
                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(output.numpy(), np.zeros((3, 4)))
                    test.assertIs(getter(model), result)
                    wp.copy(source, original)


def test_getters_release_discarded_states(test, device):
    """Release discarded state arrays and getter buffers while keeping the view usable."""
    for make_model, pattern in ((_replicated_model, "/World/*"), (_irregular_model, "selected_*")):
        model = make_model(device=device)
        for getter, _setter, attribute in _state_accessors(model, pattern=pattern):
            with test.subTest(device=device, pattern=pattern, getter=getter.__name__):
                refs = []
                for _ in range(3):
                    state = model.state(requires_grad=True)
                    result = getter(state)
                    expected = result.numpy().copy()
                    refs.extend(weakref.ref(obj) for obj in (state, getattr(state, attribute), result))
                    state_ref = weakref.ref(state)
                    del state
                    test.assertIsNone(state_ref())
                    np.testing.assert_array_equal(result.numpy(), expected)
                    del result
                test.assertTrue(all(ref() is None for ref in refs))
                live_state = model.state()
                test.assertTrue(np.isfinite(getter(live_state).numpy()).all())


def test_getters_follow_replaced_arrays(test, device):
    """Keep alternating states separate and release replaced arrays after callers drop their views."""
    for make_model, pattern in ((_replicated_model, "/World/*"), (_irregular_model, "selected_*")):
        model = make_model(device=device)
        for getter, _setter, attribute in _state_accessors(model, pattern=pattern):
            with test.subTest(device=device, pattern=pattern, getter=getter.__name__):
                states = [model.state(requires_grad=True) for _ in range(2)]
                for value, state in enumerate(states, 1):
                    array = getattr(state, attribute)
                    array.assign(np.full_like(array.numpy(), value))
                buffers = [getter(state) for state in states]
                for value, (state, buffer) in enumerate(zip(states, buffers, strict=True), 1):
                    test.assertIs(getter(state), buffer)
                    np.testing.assert_array_equal(buffer.numpy(), np.full_like(buffer.numpy(), value))

                old_array = weakref.ref(getattr(states[0], attribute))
                setattr(states[0], attribute, wp.zeros_like(getattr(states[0], attribute)))
                result = getter(states[0])
                np.testing.assert_array_equal(result.numpy(), np.zeros_like(result.numpy()))
                test.assertIs(getter(states[1]), buffers[1])
                np.testing.assert_array_equal(buffers[0].numpy(), np.ones_like(buffers[0].numpy()))
                del buffers[0]
                test.assertIsNone(old_array())


def test_getter_gradients(test, device):
    """Follow initial, newly enabled, and replaced source gradient buffers."""
    for make_model, pattern in ((_replicated_model, "/World/*"), (_irregular_model, "selected_*")):
        model = make_model(device=device)
        for getter, _setter, attribute in _state_accessors(model, pattern=pattern):
            for setup in ("initial", "enabled", "replaced"):
                with test.subTest(device=device, pattern=pattern, getter=getter.__name__, setup=setup):
                    state = model.state(requires_grad=setup != "enabled")
                    source = getattr(state, attribute)
                    getter(state)
                    if setup == "enabled":
                        source.requires_grad = True
                    elif setup == "replaced":
                        old_grad = weakref.ref(source.grad)
                        source.grad = wp.zeros_like(source, requires_grad=False)
                    loss = wp.zeros(1, device=device, requires_grad=True)
                    with wp.Tape() as tape:
                        values = getter(state)
                        wp.launch(_sum_first_component, dim=values.shape, inputs=[values, loss], device=device)
                    tape.backward(loss)

                    expected = np.zeros_like(source.numpy())
                    expected[:, 0] = 1.0
                    if make_model is _irregular_model:
                        # The fixture leaves unrelated particles/bodies between the selected objects.
                        gaps = [4, 9, 10] if attribute.startswith("particle") else [3, 7, 8]
                        expected[gaps, 0] = 0.0
                    np.testing.assert_array_equal(source.grad.numpy(), expected)
                    source.grad.zero_()
                    test.assertIs(getter(state), values)
                    if setup == "replaced":
                        test.assertIsNone(old_grad())
                    source.requires_grad = False
                    test.assertFalse(getter(state).requires_grad)


def test_getter_gradients_cuda_graph(test, device):
    """Replay getter gradients with warmed buffers and a fixed configuration."""

    def evaluate(getter, state, loss):
        with wp.Tape() as tape:
            result = getter(state)
            wp.launch(_sum_first_component, dim=result.shape, inputs=[result, loss], device=loss.device)
        tape.backward(loss)
        return result, tape

    for make_model, pattern in ((_replicated_model, "/World/*"), (_irregular_model, "selected_*")):
        model = make_model(device=device)
        for getter, _setter, attribute in _state_accessors(model, pattern=pattern):
            with test.subTest(pattern=pattern, getter=getter.__name__):
                state = model.state(requires_grad=True)
                getter(state)
                setattr(state, attribute, wp.clone(getattr(state, attribute), requires_grad=True))
                source = getattr(state, attribute)
                loss = wp.zeros(1, device=device, requires_grad=True)
                values, tape = evaluate(getter, state, loss)  # Warm the getter and backward kernels.
                expected = np.zeros_like(source.numpy())
                expected[:, 0] = 1.0
                if make_model is _irregular_model:
                    gaps = [4, 9, 10] if attribute.startswith("particle") else [3, 7, 8]
                    expected[gaps, 0] = 0.0
                with wp.ScopedMempool(device, enable=False):
                    with wp.ScopedCapture(device) as capture:
                        tape.zero()
                        loss.zero_()
                        captured, _ = evaluate(getter, state, loss)
                test.assertIs(captured, values)
                for _ in range(2):
                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(source.grad.numpy(), expected)


def test_irregular_getters_reject_short_state_arrays(test, device):
    """Check capacity before gathering irregular ranges, including after a cached read."""
    model = _irregular_model(device=device)
    for getter, _setter, attribute in _state_accessors(model, pattern="selected_*"):
        with test.subTest(device=device, getter=getter.__name__):
            state = model.state()
            original = getattr(state, attribute)
            getter(state)
            # Keep the full allocation alive so a missing guard cannot read freed memory.
            setattr(state, attribute, original[:-1])
            with test.assertRaisesRegex(ValueError, "at least"):
                getter(state)


def test_batched_state_round_trips(test, device):
    """Read and write all particle and body state types on CPU and CUDA."""
    model = _replicated_model(3, device=device)
    for getter, setter, attribute in _state_accessors(model):
        with test.subTest(device=device, attribute=attribute):
            view = getter.__self__
            test.assertEqual((view.count, view.count_per_world), (3, 1))
            test.assertEqual(view.worlds, [0, 1, 2])
            state = model.state()
            target = getattr(state, attribute)
            before = target.numpy().copy()
            values = getter(state)
            test.assertEqual(values.shape, (3, 4 if attribute.startswith("particle") else 3))
            test.assertEqual(values.device, model.device)
            changed = values.numpy().copy()
            if attribute.endswith("qd"):
                changed += np.arange(1, 4)[:, None, None]
            else:
                changed[..., 0] += np.arange(1, 4)[:, None]
            np.testing.assert_array_equal(target.numpy(), before)
            dtype = target.dtype
            if attribute == "particle_q":
                # Equivalent vec3 inputs must pass Newton's dtype validation.
                dtype = wp.types.vector(length=3, dtype=wp.float32)
            setter(state, wp.array(changed, dtype=dtype, device=device))
            np.testing.assert_allclose(getter(state).numpy(), changed, atol=1e-6)


def test_irregular_getter_cold_capture_keeps_buffers_separate(test, device):
    """Read eagerly before replaying independent first-use getter captures."""
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture of allocations requires the mempool")
    model = _irregular_model(device=device)
    state = model.state()
    warm_view = DeformableSurfaceView(model, "selected_cloth_*")
    expected = warm_view.get_particle_positions(state).numpy().copy()
    view = DeformableSurfaceView(model, "selected_cloth_*")

    with wp.ScopedCapture(device) as first:
        first_result = view.get_particle_positions(state)
    with wp.ScopedCapture(device) as second:
        second_result = view.get_particle_positions(state)
    # Independent graphs must not borrow storage allocated by an unreplayed graph.
    test.assertIsNot(first_result, second_result)

    eager_result = view.get_particle_positions(state)
    np.testing.assert_array_equal(eager_result.numpy(), expected)
    for capture, result, offset in ((second, second_result, 1.0), (first, first_result, 2.0)):
        values = expected + offset
        view.set_particle_positions(state, wp.array(values, dtype=wp.vec3, device=device))
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(result.numpy(), values)
        np.testing.assert_array_equal(view.get_particle_positions(state).numpy(), values)


def test_irregular_getters_capture_all_state_dtypes(test, device):
    """Internally staged vec3, transform, and spatial-vector reads replay in a CUDA graph."""
    model = _irregular_model(device=device)
    state = model.state()
    for getter, setter, _attribute in _state_accessors(model, pattern="selected_*"):
        with test.subTest(getter=getter.__name__):
            result = getter(state)  # warm up and allocate indexed staging
            with wp.ScopedCapture(device) as capture:
                captured_result = getter(state)
            test.assertIs(result, captured_result)

            expected = result.numpy().copy()
            expected[..., 0] += 0.25
            setter(state, wp.array(expected, dtype=result.dtype, device=device))
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(result.numpy(), expected, atol=1.0e-6)


class TestDeformableStateReads(unittest.TestCase):
    """Read regular and irregular state and property layouts, including captured gathers."""

    def test_regular_getter_returns_a_live_state_view(self):
        """A regular selection aliases state data like an ArticulationView getter."""
        model = _replicated_model(2, device="cpu")
        state = model.state()
        view = DeformableSurfaceView(model, "/World/Cloth")
        positions = view.get_particle_positions(state)
        moved = positions.numpy().copy()
        moved[..., 2] += 1.0

        view.set_particle_positions(state, wp.array(moved, dtype=wp.vec3, device=model.device))

        np.testing.assert_allclose(positions.numpy(), moved, atol=1.0e-6)

    def test_irregular_getter_reuses_internal_staging(self):
        """An indexed selection reuses its internally owned contiguous result."""
        model = _irregular_model(device="cpu")
        state = model.state()
        view = DeformableSurfaceView(model, "selected_cloth_*")
        first = view.get_particle_positions(state)
        moved = first.numpy().copy()
        moved[..., 2] += 1.0

        view.set_particle_positions(state, wp.array(moved, dtype=wp.vec3, device=model.device))
        second = view.get_particle_positions(state)

        self.assertIs(first, second)
        np.testing.assert_allclose(first.numpy(), moved, atol=1.0e-6)


for _test in (
    test_particle_property_snapshots,
    test_particle_property_gradients,
    test_getters_release_discarded_states,
    test_getters_follow_replaced_arrays,
    test_getter_gradients,
    test_irregular_getters_reject_short_state_arrays,
    test_batched_state_round_trips,
):
    add_function_test(TestDeformableStateReads, _test.__name__, _test, devices=get_test_devices())


for _test in (
    test_particle_property_capture,
    test_getter_gradients_cuda_graph,
    test_irregular_getter_cold_capture_keeps_buffers_separate,
    test_irregular_getters_capture_all_state_dtypes,
):
    add_function_test(TestDeformableStateReads, _test.__name__, _test, devices=get_cuda_test_devices())


def test_state_arrays_validate_before_access(test, device):
    """Reject invalid arrays in getters and setters without accessing their memory."""
    devices = get_test_devices()
    model = _replicated_model(3, device=device)
    for getter, setter, attribute in _state_accessors(model):
        state = model.state()
        original = getattr(state, attribute)
        values = wp.clone(getter(state))
        before = original.numpy().copy()
        invalid = [
            ("missing", None),
            ("not_warp", before),
            ("rank", original.reshape((1, original.size))),
            ("dtype", wp.zeros(original.size, dtype=wp.float32, device=device)),
            ("short", original[:-1]),
        ]
        invalid.extend(
            ("device", wp.zeros(original.size, dtype=original.dtype, device=other))
            for other in devices
            if other != device
        )
        for name, array in invalid:
            with test.subTest(device=device, attribute=attribute, array=name):
                setattr(state, attribute, array)
                # Inspect no returned alias: it may be unsafe without the guard.
                with test.assertRaisesRegex(ValueError, "state array"):
                    getter(state)
                # An all-false mask keeps the regression safe without the guard.
                with test.assertRaisesRegex(ValueError, "state array"):
                    setter(state, values, mask=[False] * 3)
                np.testing.assert_array_equal(original.numpy(), before)


def test_model_and_state_access_accepts_strided_storage(test, device):
    """Read and write padded, strided storage without changing gaps or masked rows."""
    for make_model, pattern in ((_replicated_model, "/World/*"), (_irregular_model, "selected_*")):
        model = make_model(device=device)
        for getter, setter, attribute in _state_accessors(model, pattern=pattern):
            for use_model in (False, True):
                with test.subTest(device=device, getter=getter.__name__, pattern=pattern, model=use_model):
                    state_untouched = model.state()
                    untouched = getattr(state_untouched, attribute).numpy().copy()
                    target = model if use_model else model.state()
                    original = getattr(target, attribute)
                    before = original.numpy().copy()
                    batched = getter(target).numpy().copy()
                    host = np.full((2 * (len(before) + 2), before.shape[-1]), -17.0, dtype=np.float32)
                    host[: 2 * len(before) : 2] = before
                    backing = wp.array(host, dtype=original.dtype, device=device)
                    setattr(target, attribute, backing[::2])
                    np.testing.assert_array_equal(getter(target).numpy(), batched)

                    values = batched.copy()
                    values[..., 0] += 1.0
                    setter(target, wp.array(values, dtype=original.dtype, device=device), mask=[True, False, True])
                    expected = host.copy()
                    view = getter.__self__
                    kind = "particle" if attribute.startswith("particle") else "body"
                    for row, (start, end) in enumerate(view.ranges(kind)):
                        if row != 1:
                            expected[2 * start : 2 * end : 2] = values[row]
                    np.testing.assert_array_equal(backing.numpy(), expected)
                    np.testing.assert_array_equal(getattr(state_untouched, attribute).numpy(), untouched)
                    setattr(target, attribute, original)


def test_state_capacity_only_covers_selected_ranges(test, device):
    """Accept arrays smaller than the model when the selected elements still fit."""
    model = _irregular_model(device=device)
    for getter, setter, attribute in _state_accessors(model, pattern="selected_*_0"):
        with test.subTest(device=device, getter=getter.__name__):
            state = model.state()
            values = wp.clone(getter(state))
            values.fill_(2.0)
            original = getattr(state, attribute)
            before = original.numpy().copy()
            required = values.shape[1]
            setattr(state, attribute, original[:required])
            setter(state, values)
            np.testing.assert_array_equal(getter(state).numpy(), values.numpy())
            expected = before.copy()
            expected[:required] = 2.0
            np.testing.assert_array_equal(original.numpy(), expected)


def test_host_values_validate_before_writing(test, device):
    """Reject missing or malformed host values without changing any rows, regardless of the mask."""
    model = _replicated_model(5, device=device)
    for getter, setter, attribute in _state_accessors(model):
        state = model.state()
        target = getattr(state, attribute)
        initial = wp.clone(target)
        before = target.numpy().copy()
        host = getter(state).numpy().copy()
        invalid = (
            ("missing", None),
            ("wrong_count", host[:-1]),
            ("wrong_components", host[..., :-1]),
            ("empty", []),
            ("ragged", [host[0].tolist(), host[1, :-1].tolist()]),
            ("not_numeric", np.full(host.shape, "invalid")),
            ("invalid_elements", np.full(host.shape, object())),
        )
        for name, values in invalid:
            for mask in (None, [True, False, True, False, True], [False] * 5):
                with test.subTest(device=device, setter=setter.__name__, values=name, mask=mask):
                    wp.copy(target, initial)
                    with test.assertRaisesRegex(ValueError, "values"):
                        setter(state, values, mask=mask)
                    np.testing.assert_array_equal(target.numpy(), before)


def test_indices_replace_mask_without_reordering_values(test, device):
    """Convert unsorted and repeated object indices into matching-row resets."""
    model = _replicated_model(4, device=device)
    cloth = DeformableSurfaceView(model, "/World/Cloth")
    backing = wp.ones(8, dtype=wp.bool, device=device)
    mask = backing[::2]
    cloth.set_mask_from_indices(mask, [2, np.int64(0), 2])
    np.testing.assert_array_equal(backing.numpy(), [True, True, False, True, True, True, False, True])
    state = model.state()
    before = state.particle_q.numpy().copy()
    defaults = cloth.get_particle_positions(model).numpy().copy()
    defaults[:, :, 2] += np.array([10, 20, 30, 40])[:, None]
    cloth.set_particle_positions(state, defaults, mask=mask)
    expected = before.copy()
    expected[:4] = defaults[0]
    expected[8:12] = defaults[2]
    np.testing.assert_array_equal(state.particle_q.numpy(), expected)
    cloth.set_mask_from_indices(mask, [])
    np.testing.assert_array_equal(backing.numpy(), [False, True] * 4)


def test_mask_conversion_validates_without_mutation(test, device):
    """Reject invalid host indices and array layouts before replacing a mask."""
    model = _replicated_model(3, device=device)
    view = DeformableSurfaceView(model, "/World/Cloth")
    mask = wp.array([True, False, True], dtype=wp.bool, device=device)
    for indices, error in (
        ([-0.2], TypeError),
        ([1.9], TypeError),
        (["1"], TypeError),
        ([True], TypeError),
        ([np.bool_(False)], TypeError),
        (None, TypeError),
        ([-1], ValueError),
        ([3], ValueError),
        ([0, 3], ValueError),
        (wp.zeros((1, 3), dtype=wp.int32, device=device), ValueError),
        (wp.zeros(1, dtype=wp.int64, device=device), ValueError),
    ):
        with test.subTest(device=device, indices=indices):
            with test.assertRaises(error):
                view.set_mask_from_indices(mask, indices)
            np.testing.assert_array_equal(mask.numpy(), [True, False, True])
    other = "cpu" if device.is_cuda else "cuda:0"
    if wp.is_cuda_available():
        with test.assertRaisesRegex(ValueError, "device"):
            view.set_mask_from_indices(mask, wp.zeros(1, dtype=wp.int32, device=other))
        np.testing.assert_array_equal(mask.numpy(), [True, False, True])

    for output in (None, [True] * 3):
        with test.assertRaisesRegex(TypeError, "caller-owned"):
            view.set_mask_from_indices(output, [0])
    for dtype, shape in ((wp.int32, (3,)), (wp.bool, (1, 3)), (wp.bool, (2,))):
        output = wp.ones(shape, dtype=dtype, device=device)
        before = output.numpy().copy()
        with test.assertRaisesRegex(ValueError, "mask"):
            view.set_mask_from_indices(output, [0])
        np.testing.assert_array_equal(output.numpy(), before)
    overlapping = wp.array(ptr=mask.ptr, shape=(3,), strides=(0,), dtype=wp.bool, device=device)
    with test.assertRaisesRegex(ValueError, "overlap"):
        view.set_mask_from_indices(overlapping, [0])
    np.testing.assert_array_equal(mask.numpy(), [True, False, True])


def test_mask_conversion_uses_view_rows_in_uneven_and_global_scenes(test, device):
    """Keep object membership independent of model-world partition sizes."""
    for global_only in (False, True):
        builder = newton.ModelBuilder()
        for count in (1, 0, 2):
            if not global_only:
                builder.begin_world()
            if count == 0:
                builder.add_particle(wp.vec3(0.0), wp.vec3(0.0), 1.0)
            for _ in range(count):
                _add_test_cable(builder)
            if not global_only:
                builder.end_world()
        model = builder.finalize(device=device)
        cables = DeformableCurveView(model, "cable")
        test.assertEqual(cables.worlds, [-1, -1, -1] if global_only else [0, 2, 2])
        mask = wp.zeros(3, dtype=wp.bool, device=device)
        indices = range(1, 3) if global_only else range(*cables.deformable_object_ranges()[2])
        cables.set_mask_from_indices(mask, indices)
        np.testing.assert_array_equal(mask.numpy(), [False, True, True])


def test_conversion_and_reset_replay_changing_membership(test, device):
    """Replay duplicate and padded device selectors without retaining old selections."""
    model = _replicated_model(3, device=device)
    for getter, setter, attribute in _state_accessors(model):
        view = setter.__self__
        state = model.state()
        initial = getter(state).numpy().copy()
        defaults = initial.copy()
        defaults[..., 0] += 10.0
        values = wp.array(defaults, dtype=getattr(state, attribute).dtype, device=device)
        indices = wp.array([2, 0, 2, -1], dtype=wp.int32, device=device)
        mask = wp.zeros(3, dtype=wp.bool, device=device)
        view.set_mask_from_indices(mask, indices)
        setter(state, values, mask=mask)
        with wp.ScopedMempool(device, enable=False):
            with wp.ScopedCapture(device) as capture:
                view.set_mask_from_indices(mask, indices)
                setter(state, values, mask=mask)
        for selected, expected_mask in (
            ([2, 0, 2, -1], [True, False, True]),
            ([1, -1, 3, 1], [False, True, False]),
            ([-1, 3, -1, 99], [False] * 3),
        ):
            setter(state, initial)
            indices.assign(np.array(selected, dtype=np.int32))
            wp.capture_launch(capture.graph)
            expected = initial.copy()
            expected[expected_mask] = defaults[expected_mask]
            np.testing.assert_array_equal(mask.numpy(), expected_mask)
            np.testing.assert_array_equal(getter(state).numpy(), expected)


def test_conversion_rejects_host_indices_during_capture(test, device):
    """Leave a caller-owned mask intact when host conversion is attempted in capture."""
    view = DeformableSurfaceView(_replicated_model(3, device=device), "/World/Cloth")
    mask = wp.ones(3, dtype=wp.bool, device=device)
    for pooled in (False, True):
        for indices in ([0], []):
            with wp.ScopedMempool(device, enable=pooled):
                with wp.ScopedCapture(device):
                    with test.assertRaisesRegex(RuntimeError, "before CUDA capture"):
                        view.set_mask_from_indices(mask, indices)
            np.testing.assert_array_equal(mask.numpy(), [True] * 3)


def test_masked_writes_keep_full_buffer_row_order(test, device):
    """Copy matching full-buffer rows and preserve every unselected row."""
    model = _replicated_model(4, device=device)
    for getter, setter, attribute in _state_accessors(model):
        state = model.state()
        target = getattr(state, attribute)
        initial = wp.clone(target)
        before = getter(state).numpy().copy()
        defaults = before.copy()
        for i in range(4):
            defaults[i, ..., 0] += 10.0 + i
        values = wp.array(defaults, dtype=target.dtype, device=device)
        strided = np.stack((defaults, defaults), axis=-1)[..., 0]
        host_inputs = (
            ("full", defaults),
            ("list", defaults.tolist()),
            ("float64", defaults.astype(np.float64)),
            ("strided", strided),
            ("flat", defaults.reshape(-1)),
            ("flat_list", defaults.reshape(-1).tolist()),
            ("rows", defaults.reshape(-1, defaults.shape[-1])),
            ("fortran", np.asfortranarray(defaults.reshape(defaults.shape[-1], -1))),
        )
        for selection in (None, [True, False, True, False], [False] * 4):
            with test.subTest(device=device, setter=setter.__name__, selection=selection):
                wp.copy(target, initial)
                expected = before.copy()
                chosen = np.ones(4, dtype=bool) if selection is None else selection
                expected[chosen] = defaults[chosen]

                # Every second entry belongs to the mask. Other entries are sentinels.
                backing_np = np.ones(8, dtype=bool)
                backing_np[::2] = chosen
                backing = wp.array(backing_np, dtype=wp.bool, device=device)
                mask = None if selection is None else backing[::2]
                setter(state, values, mask=mask)
                np.testing.assert_array_equal(getter(state).numpy(), expected)
                np.testing.assert_array_equal(values.numpy(), defaults)
                np.testing.assert_array_equal(backing.numpy(), backing_np)

                # Host inputs follow the same mapping outside capture.
                for name, host_values in host_inputs:
                    with test.subTest(host_layout=name):
                        # An earlier successful write must not hide a later no-op.
                        wp.copy(target, initial)
                        setter(state, host_values, mask=selection)
                        np.testing.assert_array_equal(getter(state).numpy(), expected)


def test_separate_masked_writes_preserve_gradients(test, device):
    """Keep both writes' gradients when one view updates separate states."""

    def differentiate(setter, states, values, masks, seeds, selectors):
        with wp.Tape() as tape:
            for i, (state, value, mask) in enumerate(zip(states, values, masks, strict=True)):
                if selectors is not None:
                    setter.__self__.set_mask_from_indices(mask, selectors[i])
                setter(state, value, mask=mask)
        tape.backward(grads=seeds)
        return tape

    model = _replicated_model(3, device=device)
    for convert_indices in (False, True):
        for getter, setter, attribute in _state_accessors(model):
            with test.subTest(device=device, setter=setter.__name__, convert_indices=convert_indices):
                states = [model.state(requires_grad=True) for _ in range(2)]
                targets = [getattr(state, attribute) for state in states]
                values = [
                    wp.array(getter(state).numpy(), dtype=target.dtype, device=device, requires_grad=True)
                    for state, target in zip(states, targets, strict=True)
                ]
                masks_np = ([True, False, True], [False, True, True])
                masks = [wp.array(mask, dtype=wp.bool, device=device) for mask in masks_np]
                selectors = (
                    [wp.array(rows, dtype=wp.int32, device=device) for rows in ([2, 0, 2], [1, 2, 1])]
                    if convert_indices
                    else None
                )
                seeds = {target: wp.full_like(target, i + 1) for i, target in enumerate(targets)}
                expected = [
                    np.broadcast_to((i + 1) * np.array(mask)[:, None, None], value.numpy().shape)
                    for i, (mask, value) in enumerate(zip(masks_np, values, strict=True))
                ]
                tape = differentiate(setter, states, values, masks, seeds, selectors)
                for value, gradient in zip(values, expected, strict=True):
                    np.testing.assert_array_equal(value.grad.numpy(), gradient)

                if device.is_cuda:
                    tape.zero()
                    with wp.ScopedCapture(device) as capture:
                        tape = differentiate(setter, states, values, masks, seeds, selectors)
                    for _ in range(2):
                        tape.zero()
                        wp.capture_launch(capture.graph)
                        for value, gradient in zip(values, expected, strict=True):
                            np.testing.assert_array_equal(value.grad.numpy(), gradient)


def test_masked_setters_capture_without_allocations(test, device):
    """Replay all setter types with changing masks and gradient-enabled values."""
    model = _replicated_model(3, device=device)
    for getter, setter, attribute in _state_accessors(model):
        for use_mask in (False, True):
            with test.subTest(setter=setter.__name__, use_mask=use_mask):
                state = model.state(requires_grad=True)
                initial = getter(state).numpy().copy()
                values = wp.array(initial, dtype=getattr(state, attribute).dtype, device=device, requires_grad=True)
                mask = wp.zeros(3, dtype=wp.bool, device=device) if use_mask else None
                setter(state, values, mask=mask)
                # No active tape: requires_grad alone must not trigger hidden storage.
                with wp.ScopedMempool(device, enable=False):
                    with wp.ScopedCapture(device) as capture:
                        setter(state, values, mask=mask)
                expected = initial.copy()
                for step, chosen in enumerate(([True, False, True], [False] * 3, [True] * 3)):
                    next_values = initial.copy()
                    next_values[..., 0] += step + 1
                    values.assign(next_values)
                    if mask is not None:
                        mask.assign(np.array(chosen, dtype=bool))
                    wp.capture_launch(capture.graph)
                    rows = chosen if use_mask else [True] * 3
                    expected[rows] = next_values[rows]
                    np.testing.assert_array_equal(getter(state).numpy(), expected)


def test_host_inputs_rejected_during_capture(test, device):
    """Reject host conversions independently of CUDA memory-pool settings."""
    model = _replicated_model(3, device=device)
    for getter, setter, attribute in _state_accessors(model):
        state = model.state()
        values = wp.clone(getter(state))
        host_values = values.numpy()
        before = getattr(state, attribute).numpy().copy()
        for pooled in (False, True):
            for value, mask in ((host_values, None), (values, [False] * 3), ([], None), (values, []), (None, None)):
                with test.subTest(setter=setter.__name__, pooled=pooled):
                    with wp.ScopedMempool(device, enable=pooled):
                        with wp.ScopedCapture(device):
                            with test.assertRaisesRegex(RuntimeError, "before CUDA capture"):
                                setter(state, value, mask=mask)
                    np.testing.assert_array_equal(getattr(state, attribute).numpy(), before)


class TestDeformableStateWrites(unittest.TestCase):
    """Validate masked writes, capture, and gradients through public setters."""

    def test_masks_validate_before_writing(self):
        """Reject malformed masks without changing the destination."""
        model = _replicated_model(3, device="cpu")
        for getter, setter, attribute in _state_accessors(model):
            state = model.state()
            before = getattr(state, attribute).numpy().copy()
            values = wp.clone(getter(state))
            invalid = [
                [0, 2],
                [0, 1, 0],
                [True, 1, False],
                ["", "yes", ""],
                [True],
                [[True, False, True]],
                [],
                wp.zeros(3, dtype=wp.int32, device="cpu"),
                wp.zeros((1, 3), dtype=wp.bool, device="cpu"),
            ]
            if wp.is_cuda_available():
                invalid.append(wp.zeros(3, dtype=wp.bool, device="cuda:0"))
            for mask in invalid:
                with self.subTest(setter=setter.__name__, mask=mask):
                    with self.assertRaisesRegex(ValueError, "mask"):
                        setter(state, values, mask=mask)
                    np.testing.assert_array_equal(getattr(state, attribute).numpy(), before)

    def test_all_false_masks_still_validate_values(self):
        """Validate full values and reject aliases even when nothing is selected."""
        model = _replicated_model(3, device="cpu")
        for getter, setter, attribute in _state_accessors(model):
            state = model.state()
            before = getattr(state, attribute).numpy().copy()
            live = getter(state)
            invalid = [
                (wp.zeros((2, live.shape[1]), dtype=live.dtype, device="cpu"), "shape"),
                (wp.zeros(live.shape, dtype=float, device="cpu"), "dtype"),
                (live, "overlap.*clone"),
            ]
            if wp.is_cuda_available():
                invalid.append((wp.zeros(live.shape, dtype=live.dtype, device="cuda:0"), "device"))
            for values, message in invalid:
                with self.subTest(setter=setter.__name__, message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        setter(state, values, mask=[False] * 3)
                    np.testing.assert_array_equal(getattr(state, attribute).numpy(), before)
            with self.assertRaisesRegex(ValueError, "overlap.*clone"):
                setter(state, live)
            setter(state, wp.clone(live))


for _test in (
    test_state_arrays_validate_before_access,
    test_model_and_state_access_accepts_strided_storage,
    test_state_capacity_only_covers_selected_ranges,
    test_host_values_validate_before_writing,
    test_indices_replace_mask_without_reordering_values,
    test_mask_conversion_validates_without_mutation,
    test_mask_conversion_uses_view_rows_in_uneven_and_global_scenes,
    test_masked_writes_keep_full_buffer_row_order,
    test_separate_masked_writes_preserve_gradients,
):
    add_function_test(TestDeformableStateWrites, _test.__name__, _test, devices=get_test_devices())


for _test in (
    test_conversion_and_reset_replay_changing_membership,
    test_conversion_rejects_host_indices_during_capture,
    test_masked_setters_capture_without_allocations,
    test_host_inputs_rejected_during_capture,
):
    add_function_test(TestDeformableStateWrites, _test.__name__, _test, devices=get_cuda_test_devices())


def test_partial_resets_update_both_active_states(test, device):
    """Preserve matching rows and untouched assets across changing reset sets."""
    prototype = newton.ModelBuilder()
    _add_test_cloth(prototype, label="cloth_left")
    _add_test_cloth(prototype, label="cloth_right")
    builder = newton.ModelBuilder()
    builder.replicate(prototype, 4)
    model = builder.finalize(device=device)
    for captured in (False, True) if device.is_cuda else (False,):
        with test.subTest(device=device, captured=captured):
            cloths = DeformableSurfaceView(model, "cloth_left")
            test.assertEqual(cloths.worlds, list(range(model.world_count)))
            test.assertEqual(cloths.ranges("particle"), [(0, 4), (8, 12), (16, 20), (24, 28)])
            states = [model.state(), model.state()]
            positions_np = cloths.get_particle_positions(model).numpy().copy()
            velocities_np = cloths.get_particle_velocities(model).numpy().copy()
            positions_np[:, :, 2] += np.array([10, 20, 30, 40])[:, None]
            velocities_np[:, :, 0] = np.array([1, 2, 3, 4])[:, None]
            positions_default = wp.array(positions_np, dtype=wp.vec3, device=device)
            velocities_default = wp.array(velocities_np, dtype=wp.vec3, device=device)
            indices_world = wp.array([2, 0], dtype=wp.int32, device=device)
            mask_cloths = wp.zeros(cloths.count, dtype=wp.bool, device=device)

            reset_args = (cloths, states, indices_world, mask_cloths, positions_default, velocities_default)
            _reset_cloths(*reset_args)
            if captured:
                with wp.ScopedCapture(device) as capture:
                    _reset_cloths(*reset_args)

            for step, (indices, selected) in enumerate((([2, 0], [0, 2]), ([1, -1], [1]), ([3, 2], [2, 3]))):
                # Fresh values expose stale mask bits that a second identical reset would hide.
                before = []
                for i, state in enumerate(states):
                    state.particle_q.fill_(100 + 10 * step + i)
                    state.particle_qd.fill_(-100 - 10 * step - i)
                    before.append((state.particle_q.numpy().copy(), state.particle_qd.numpy().copy()))
                indices_world.assign(np.array(indices, dtype=np.int32))
                if captured:
                    wp.capture_launch(capture.graph)
                else:
                    _reset_cloths(*reset_args)
                for state, (positions, velocities) in zip(states, before, strict=True):
                    # Each world holds four left-cloth particles, then four right-cloth particles.
                    positions.reshape(4, 8, 3)[selected, :4] = positions_np[selected]
                    velocities.reshape(4, 8, 3)[selected, :4] = velocities_np[selected]
                    np.testing.assert_array_equal(state.particle_q.numpy(), positions)
                    np.testing.assert_array_equal(state.particle_qd.numpy(), velocities)
                np.testing.assert_array_equal(positions_default.numpy(), positions_np)
                np.testing.assert_array_equal(velocities_default.numpy(), velocities_np)
                states.reverse()


class TestDeformableResets(unittest.TestCase):
    """Reset one asset in selected worlds without disturbing other state."""


add_function_test(
    TestDeformableResets,
    "test_partial_resets_update_both_active_states",
    test_partial_resets_update_both_active_states,
    devices=get_test_devices(),
)


def test_compact_cable_requires_its_generated_root(test, device):
    """Keep a complete USD cable, but omit its identity when the root is not selected."""
    stage = _deformable_stage()
    _add_cable_curve(stage, "/World/Cable", _CABLE_PTS)
    builder = newton.ModelBuilder()
    builder.add_rod(rod=newton.Rod(_CABLE_PTS, radius=0.01), label="background", body_frame_origin="com")
    result = builder.add_usd(stage, return_deformable_results=True)
    bodies, rod_joints = result["path_cable_map"]["/World/Cable"]
    test.assertEqual(rod_joints, [4, 5])
    builder.color()
    model = builder.finalize(device=device)
    cable = DeformableCurveView(model, "/World/Cable")
    test.assertEqual(cable.ranges("joint"), [(3, 6)])
    for include_root in (False, True):
        with test.subTest(device=device, include_root=include_root):
            joints = list(range(3, 6)) if include_root else rod_joints
            coupled = SolverCoupled(
                model,
                entries=[SolverCoupled.Entry(name="cable", solver=SolverVBD, bodies=bodies, joints=joints)],
            )
            compact = coupled.view("cable")
            test.assertEqual(compact.body_count, 3)
            # An omitted root is replaced by a FREE joint, not the original recorded joint.
            test.assertEqual(compact.joint_count, 3)
            test.assertEqual(compact.joint_type.numpy()[0], newton.JointType.FREE)
            test.assertEqual(compact.curve_count, int(include_root))
            test.assertEqual(compact.curve_label, ["/World/Cable"] if include_root else [])
            if include_root:
                selected = DeformableCurveView(compact, "/World/Cable")
                test.assertEqual(selected.ranges("body"), [(0, 3)])
                test.assertEqual(selected.ranges("joint"), [(0, 3)])
            test.assertEqual(cable.ranges("joint"), [(3, 6)])


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableObjects(unittest.TestCase):
    """Prim-path deformable object lookup on the finalized Model across lifecycle transformations."""

    def test_mixed_scene_objects_resolve_after_finalize(self):
        """Every family of the mixed scene resolves by prim path on the finalized Model."""
        builder = newton.ModelBuilder()
        builder.add_usd(_MIXED_ASSET)
        model = builder.finalize()

        cable = DeformableCurveView(model, "/World/CableA/sim")
        self.assertEqual(cable.count, 1)
        self.assertEqual(cable.body_count, 3)
        self.assertEqual(cable.element_count("joint"), 3)  # free root and two rod joints
        cloth = DeformableSurfaceView(model, "/World/Cloth/sim")
        self.assertEqual(cloth.particle_count, 4)
        self.assertEqual(cloth.ranges("triangle"), [(0, 2)])
        soft = DeformableVolumeView(model, "/World/Soft*/sim")
        self.assertEqual(soft.count, 2)
        soft_ranges = soft.ranges("particle")
        self.assertNotEqual(soft_ranges[0], soft_ranges[1])
        self.assertEqual(soft.element_count("tetrahedron"), 1)
        # No begin_world -> global deformable objects.
        self.assertEqual(cable.worlds, [-1])
        with self.assertRaises(KeyError):
            DeformableCurveView(model, "/World/DoesNotExist")

    def test_replicated_objects_select_per_world(self):
        """replicate() duplicates labels across worlds: one deformable object per world, ranges
        offset per world, and raw ranges come back in world order."""
        stage = _deformable_stage()
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _author_deformable_element_array(cloth.GetPrim(), "thicknesses", [0.001], "constant")
        _bind_deformable_material(stage, cloth.GetPrim(), "/World/ClothMat")
        sub = newton.ModelBuilder()
        sub.add_usd(stage)
        scene = newton.ModelBuilder()
        scene.replicate(sub, 3)
        model = scene.finalize()

        view = DeformableSurfaceView(model, "/World/Cloth")
        self.assertEqual((view.count, view.count_per_world), (3, 1))
        self.assertEqual(view.labels, ["/World/Cloth"] * 3)
        self.assertEqual(view.worlds, [0, 1, 2])
        self.assertEqual(view.ranges("particle"), [(4 * w, 4 * w + 4) for w in range(3)])
        self.assertEqual(list(view.starts("particle").numpy()), [0, 4, 8])

    def test_cable_objects_replicate_with_free_and_attached_roots(self):
        """Keep one selectable cable per world with either free or attached roots."""
        from pxr import UsdGeom, UsdPhysics

        for attached_point in (None, 0, len(_CABLE_PTS) - 1):
            with self.subTest(attached_point=attached_point):
                stage = _deformable_stage()
                _add_cable_curve(stage, "/World/Cable", _CABLE_PTS)
                if attached_point is not None:
                    plug = UsdGeom.Cube.Define(stage, "/World/Plug")
                    plug.CreateSizeAttr(0.1)
                    UsdPhysics.RigidBodyAPI.Apply(plug.GetPrim())
                    UsdPhysics.CollisionAPI.Apply(plug.GetPrim())
                    _add_physics_attachment(
                        stage,
                        "/World/Attachment",
                        src0="/World/Cable",
                        src1="/World/Plug",
                        type0="point",
                        indices0=[attached_point],
                        coords1=[_CABLE_PTS[attached_point]],
                    )

                source = newton.ModelBuilder()
                result = source.add_usd(stage, return_deformable_results=True)
                bodies, _ = result["path_cable_map"]["/World/Cable"]
                root = 0 if attached_point is None else 1
                if attached_point is not None:
                    (attachment,) = result["path_attachment_map"]["/World/Attachment"]
                    plug_body = result["path_body_map"]["/World/Plug"]
                    plug_joint = next(j for j, child in enumerate(source.joint_child) if child == plug_body)
                    self.assertEqual(source.joint_parent[attachment], plug_body)
                    self.assertEqual(source.joint_child[attachment], bodies[0 if attached_point == 0 else -1])
                    self.assertEqual(source.joint_articulation[attachment], source.joint_articulation[plug_joint])

                scene = newton.ModelBuilder()
                scene.replicate(source, 2)
                view = DeformableCurveView(scene.finalize(device="cpu"), "/World/Cable")
                self.assertEqual((view.count, view.worlds), (2, [0, 1]))
                self.assertEqual(
                    view.ranges("body"),
                    [(bodies[0] + w * source.body_count, bodies[-1] + 1 + w * source.body_count) for w in range(2)],
                )
                self.assertEqual(
                    view.ranges("joint"),
                    [(root + w * source.joint_count, root + 3 + w * source.joint_count) for w in range(2)],
                )

    def test_heterogeneous_worlds_resolve_with_world_tags(self):
        """Worlds holding different deformables each resolve with the right world tag."""
        cloth_stage = _deformable_stage()
        cloth = _add_cloth_mesh(cloth_stage, "/World/Cloth")
        _author_deformable_element_array(cloth.GetPrim(), "thicknesses", [0.001], "constant")
        _bind_deformable_material(cloth_stage, cloth.GetPrim(), "/World/ClothMat")
        cable_stage = _deformable_stage()
        _add_cable_curve(cable_stage, "/World/Cable", _CABLE_PTS)

        cloth_sub = newton.ModelBuilder()
        cloth_sub.add_usd(cloth_stage)
        cable_sub = newton.ModelBuilder()
        cable_sub.add_usd(cable_stage)
        scene = newton.ModelBuilder()
        scene.add_world(cloth_sub)  # world 0: cloth only
        scene.add_world(cable_sub)  # world 1: cable only
        model = scene.finalize()

        self.assertEqual(DeformableSurfaceView(model, "/World/Cloth").worlds, [0])
        self.assertEqual(DeformableCurveView(model, "/World/Cable").worlds, [1])

    def test_cable_object_survives_fixed_joint_collapse(self):
        """Cable body ranges ride the reindexing of collapse_fixed_joints onto the Model."""
        from pxr import UsdGeom, UsdPhysics

        stage = _deformable_stage()
        # Two rigid bodies joined by a fixed joint -> collapsed, reindexing all bodies;
        # these parse before the cable so the cable indices shift.
        for name in ("A", "B"):
            body = UsdGeom.Xform.Define(stage, f"/World/{name}")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Fix")
        fixed.CreateBody0Rel().SetTargets(["/World/A"])
        fixed.CreateBody1Rel().SetTargets(["/World/B"])
        _add_cable_curve(stage, "/World/Cable", _CABLE_PTS)

        builder = newton.ModelBuilder()
        builder.add_usd(stage, collapse_fixed_joints=True)
        model = builder.finalize()

        view = DeformableCurveView(model, "/World/Cable")
        ((b0, b1),) = view.ranges("body")
        self.assertEqual(b1 - b0, 3)
        self.assertTrue(all("/World/Cable" in model.body_label[b] for b in range(b0, b1)))

    def test_native_and_usd_ball_anchors_survive_collapse(self):
        """Keep native and USD rod membership aligned for the supported ball-anchor import."""
        stage = _deformable_stage()
        _add_cable_curve(stage, "/World/Cable", _CABLE_PTS[:3])
        _add_physics_attachment(
            stage,
            "/World/Anchor",
            src0="/World/Cable",
            type0="point",
            indices0=[0],
            coords1=[_CABLE_PTS[0]],
        )
        imported = newton.ModelBuilder()
        imported.add_usd(stage, collapse_fixed_joints=True)

        native = newton.ModelBuilder()
        bodies, joints = native.add_rod(
            rod=newton.Rod(_CABLE_PTS[:3], radius=0.02),
            label="/World/Cable",
            wrap_in_articulation=False,
            body_frame_origin="com",
        )
        anchor = native.add_joint_ball(
            -1,
            bodies[0],
            parent_xform=wp.transform(wp.vec3(*_CABLE_PTS[0]), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
        )
        native.add_articulation([*joints, anchor])
        native.collapse_fixed_joints()

        imported_model = imported.finalize(device="cpu")
        native_model = native.finalize(device="cpu")
        imported_view = DeformableCurveView(imported_model, "/World/Cable")
        native_view = DeformableCurveView(native_model, "/World/Cable")
        self.assertEqual(imported_view.ranges("body"), [(0, 2)])
        self.assertEqual(native_view.ranges("body"), [(0, 2)])
        self.assertEqual(imported_view.get_body_transforms(imported_model).shape, (1, 2))
        np.testing.assert_allclose(
            imported_view.get_body_transforms(imported_model).numpy(),
            native_view.get_body_transforms(native_model).numpy(),
            atol=1.0e-7,
        )
        # The imported root is recorded at construction. The separately added native anchor is not.
        self.assertEqual(imported_view.ranges("joint"), [(0, 2)])
        self.assertEqual(native_view.ranges("joint"), [(0, 1)])

    def test_welded_graph_ranges_survive_collapse_and_replication(self):
        """Select a complete welded graph after finalization, collapse, and world cloning."""
        from pxr import UsdGeom, UsdPhysics

        stage = _deformable_stage()
        _add_cable_curve(stage, "/World/Trunk", _CABLE_PTS)
        _add_cable_curve(stage, "/World/Branch", [(0.1, 0.0, 1.0), (0.1, 0.1, 1.0), (0.1, 0.2, 1.0)])
        _add_physics_attachment(
            stage,
            "/World/Junction",
            src0="/World/Branch",
            src1="/World/Trunk",
            type0="point",
            type1="point",
            indices0=[0],
            indices1=[1],
        )

        for collapse, body_start in ((False, 0), (True, 1)):
            if collapse:
                # This pair shifts the graph's indices when its fixed joint is removed.
                for name in ("A", "B"):
                    body = UsdGeom.Xform.Define(stage, f"/World/{name}")
                    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
                fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Fix")
                fixed.CreateBody0Rel().SetTargets(["/World/A"])
                fixed.CreateBody1Rel().SetTargets(["/World/B"])
            prototype = newton.ModelBuilder()
            result = prototype.add_usd(stage, collapse_fixed_joints=collapse, return_deformable_results=True)
            graph_label = result["path_cable_attrs"]["/World/Branch"]["graph_component"]

            for cloning in ("none", "add_world", "replicate"):
                scene = prototype
                copies = 1 if cloning == "none" else 2
                if cloning != "none":
                    scene = newton.ModelBuilder()
                    if cloning == "replicate":
                        scene.replicate(prototype, copies)
                    else:
                        for _ in range(copies):
                            scene.add_world(prototype)
                body_ranges = [
                    (body_start + world * prototype.body_count, body_start + 5 + world * prototype.body_count)
                    for world in range(copies)
                ]
                joint_ranges = [(5 * i, 5 * i + 5) for i in range(copies)]
                worlds = [-1] if cloning == "none" else [0, 1]
                # Share USD parsing and cloning setup across the device checks.
                for device in get_test_devices():
                    with self.subTest(collapse=collapse, cloning=cloning, device=device):
                        model = scene.finalize(device=device)
                        self.assertEqual(model.curve_count, copies)
                        self.assertEqual(model.curve_label, [graph_label] * copies)
                        self.assertEqual(model.curve_world.numpy().tolist(), worlds)
                        self.assertEqual(model.curve_body_start.numpy().tolist(), [start for start, _ in body_ranges])
                        self.assertEqual(model.curve_body_end.numpy().tolist(), [end for _, end in body_ranges])
                        self.assertEqual(model.curve_joint_start.numpy().tolist(), [start for start, _ in joint_ranges])
                        self.assertEqual(model.curve_joint_end.numpy().tolist(), [end for _, end in joint_ranges])

                        view = DeformableCurveView(model, graph_label)
                        self.assertEqual((view.count, view.body_count), (copies, 5))
                        self.assertEqual(view.worlds, worlds)
                        self.assertEqual(view.ranges("body"), body_ranges)
                        self.assertEqual(view.ranges("joint"), joint_ranges)
                        self.assertEqual(view.element_count("joint"), 5)
                        self.assertEqual(view.get_body_transforms(model).shape, (copies, 5))
                        joint_types = model.joint_type.numpy().tolist()
                        for start, end in joint_ranges:
                            self.assertEqual(
                                joint_types[start:end], [newton.JointType.FREE, *[newton.JointType.ROD] * 4]
                            )

                        # Only the graph label selects an object; the other source path is an import-map key.
                        other_path = next(path for path in result["path_cable_map"] if path != graph_label)
                        with self.assertRaises(KeyError):
                            DeformableCurveView(model, other_path)


add_function_test(
    TestUSDDeformableObjects,
    "test_compact_cable_requires_its_generated_root",
    test_compact_cable_requires_its_generated_root,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
