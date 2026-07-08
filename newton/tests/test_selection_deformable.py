# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for DeformableView: batched selection over imported deformable groups."""

import re
import unittest

import numpy as np
import warp as wp

import newton
from newton.selection import DeformableView
from newton.tests._usd_deformable_test_utils import _add_cable_curve, _add_cloth_mesh, _deformable_stage
from newton.tests.unittest_utils import USD_AVAILABLE

_CABLE_PTS = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]


def _replicated_model(world_count=3):
    """One cloth + one cable per world, replicated."""
    stage = _deformable_stage()
    _add_cloth_mesh(stage, "/World/Cloth")
    _add_cable_curve(stage, "/World/Cable", _CABLE_PTS)
    sub = newton.ModelBuilder()
    sub.add_usd(stage)
    scene = newton.ModelBuilder()
    scene.replicate(sub, world_count)
    return scene.finalize()


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestDeformableView(unittest.TestCase):
    """Label-pattern selection and batched state access over deformable groups."""

    def test_cloth_view_selects_and_batches_across_worlds(self):
        """A replicated cloth selects one group per world with batched particle state."""
        model = _replicated_model(3)
        state = model.state()

        view = DeformableView(model, "/World/Cloth", family="surface")
        self.assertEqual((view.count, view.world_count, view.count_per_world), (3, 3, 1))
        self.assertEqual(view.worlds, [0, 1, 2])
        self.assertEqual(view.particles_per_group, 4)

        positions = view.get_particle_positions(state)
        self.assertEqual(positions.shape, (3, 4))

        # Round-trip: lift each world's cloth by its world index and read it back.
        lifted = positions.numpy()
        for g in range(3):
            lifted[g, :, 2] += float(g + 1)
        view.set_particle_positions(state, wp.array(lifted, dtype=wp.vec3))
        np.testing.assert_allclose(view.get_particle_positions(state).numpy(), lifted, atol=1e-6)

        # Velocities go through the same path.
        velocities = np.full((3, 4, 3), 2.5, dtype=np.float32)
        view.set_particle_velocities(state, wp.array(velocities, dtype=wp.vec3))
        np.testing.assert_allclose(view.get_particle_velocities(state).numpy(), velocities, atol=1e-6)

    def test_cable_view_batches_body_transforms(self):
        """A replicated cable selects per world and round-trips its segment transforms."""
        model = _replicated_model(2)
        state = model.state()

        view = DeformableView(model, "/World/Cable", family="curve")
        self.assertEqual((view.count, view.bodies_per_group), (2, 3))

        transforms = view.get_body_transforms(state)
        self.assertEqual(transforms.shape, (2, 3))
        shifted = transforms.numpy()
        shifted[:, :, 1] += 5.0  # translate all segments in y
        view.set_body_transforms(state, wp.array(shifted, dtype=wp.transform))
        np.testing.assert_allclose(view.get_body_transforms(state).numpy(), shifted, atol=1e-6)

        velocities = view.get_body_velocities(state)
        self.assertEqual(velocities.shape, (2, 3))

    def test_soft_view_over_global_groups(self):
        """Groups outside any world (world -1) are selectable as a single-world view."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        tet = UsdGeom.TetMesh.Define(stage, "/World/Soft")
        tet.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 2.0)])
        tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        model = builder.finalize()

        view = DeformableView(model, "/World/Soft", family="volume")
        self.assertEqual((view.count, view.world_count), (1, 1))
        self.assertEqual(view.get_particle_positions(model.state()).shape, (1, 4))

    def test_pattern_matches_multiple_groups_per_world(self):
        """A wildcard pattern selects several groups per world when counts stay equal."""
        stage = _deformable_stage()
        _add_cloth_mesh(stage, "/World/ClothA")
        _add_cloth_mesh(stage, "/World/ClothB")
        sub = newton.ModelBuilder()
        sub.add_usd(stage)
        scene = newton.ModelBuilder()
        scene.replicate(sub, 2)
        model = scene.finalize()

        view = DeformableView(model, "/World/Cloth*", family="surface")
        self.assertEqual((view.count, view.count_per_world), (4, 2))
        self.assertEqual(view.labels, ["/World/ClothA", "/World/ClothB"] * 2)
        self.assertEqual(view.group_ids, [0, 1, 2, 3])
        self.assertEqual(view.ranges("triangle"), [(0, 2), (2, 4), (4, 6), (6, 8)])

    def test_compiled_regex_pattern_selects_by_fullmatch(self):
        """A compiled regular expression selects groups by fullmatch: alternation picks
        two labels per replicated world, and a partial match selects nothing."""
        stage = _deformable_stage()
        _add_cloth_mesh(stage, "/World/ClothA")
        _add_cloth_mesh(stage, "/World/ClothB")
        sub = newton.ModelBuilder()
        sub.add_usd(stage)
        scene = newton.ModelBuilder()
        scene.replicate(sub, 2)
        model = scene.finalize()

        view = DeformableView(model, re.compile(r"/World/Cloth(A|B)"), family="surface")
        self.assertEqual((view.count, view.count_per_world), (4, 2))
        self.assertEqual(view.labels, ["/World/ClothA", "/World/ClothB"] * 2)

        # fullmatch: a prefix of the label is not a match.
        with self.assertRaises(KeyError):
            DeformableView(model, re.compile(r"/World/Cloth"), family="surface")

    def test_view_round_trip_on_cpu(self):
        """The gather/scatter path works on a CPU-finalized model, not just CUDA."""
        stage = _deformable_stage()
        _add_cloth_mesh(stage, "/World/Cloth")
        sub = newton.ModelBuilder()
        sub.add_usd(stage)
        scene = newton.ModelBuilder()
        scene.replicate(sub, 2)
        model = scene.finalize(device="cpu")
        state = model.state()

        view = DeformableView(model, "/World/Cloth", family="surface")
        positions = view.get_particle_positions(state)
        self.assertTrue(positions.device.is_cpu)
        lifted = positions.numpy()
        lifted[..., 2] += 1.0
        view.set_particle_positions(state, wp.array(lifted, dtype=wp.vec3, device="cpu"))
        np.testing.assert_allclose(view.get_particle_positions(state).numpy(), lifted, atol=1e-6)

    def test_selection_errors(self):
        """No match raises KeyError; ragged element counts and bad families raise ValueError."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        _add_cloth_mesh(stage, "/World/ClothA")  # 4 particles
        big = UsdGeom.Mesh.Define(stage, "/World/ClothB")  # 5 particles -> ragged with A
        big.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0), (2.0, 0.0, 1.0)])
        big.CreateFaceVertexCountsAttr([3, 3, 3])
        big.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3, 1, 4, 2])
        big.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
        big.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")  # match the fixture: no gating warning
        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        model = builder.finalize()

        with self.assertRaises(KeyError):
            DeformableView(model, "/World/DoesNotExist", family="surface")
        with self.assertRaisesRegex(ValueError, "Varying particle counts"):
            DeformableView(model, "/World/Cloth*", family="surface")
        with self.assertRaisesRegex(ValueError, "Unknown deformable family"):
            DeformableView(model, "/World/ClothA", family="ropes")
        view = DeformableView(model, "/World/ClothA", family="surface")
        with self.assertRaisesRegex(AttributeError, "no body elements"):
            view.get_body_transforms(model.state())
        with self.assertRaises(ValueError):
            view.set_particle_positions(model.state(), wp.zeros((2, 4), dtype=wp.vec3))

    def test_indexed_partial_writes_touch_only_selected_groups(self):
        """indices= scatters into the selected groups only, from host and device index
        forms, and cable body velocities round-trip through an indexed write."""
        model = _replicated_model(3)
        state = model.state()

        cloth = DeformableView(model, "/World/Cloth", family="surface")
        before = cloth.get_particle_positions(state).numpy()
        moved = before[[1]].copy()
        moved[..., 2] += 5.0
        cloth.set_particle_positions(state, wp.array(moved, dtype=wp.vec3), indices=[1])
        after = cloth.get_particle_positions(state).numpy()
        np.testing.assert_array_equal(after[[0, 2]], before[[0, 2]])
        np.testing.assert_allclose(after[1], moved[0], atol=1e-6)

        cable = DeformableView(model, "/World/Cable", family="curve")
        velocities = np.zeros((1, cable.bodies_per_group, 6), dtype=np.float32)
        velocities[..., 3] = 2.0
        device_indices = wp.array([2], dtype=wp.int32, device=model.device)
        cable.set_body_velocities(state, wp.array(velocities, dtype=wp.spatial_vector), indices=device_indices)
        out = cable.get_body_velocities(state).numpy()
        np.testing.assert_allclose(out[2], velocities[0], atol=1e-6)
        np.testing.assert_array_equal(out[:2], np.zeros_like(out[:2]))

        with self.assertRaisesRegex(ValueError, "must be in"):
            cloth.set_particle_positions(state, wp.array(moved, dtype=wp.vec3), indices=[3])


class TestDeformableViewBuilderGroups(unittest.TestCase):
    """Groups recorded by labeled builder calls (no USD) are selectable through the view."""

    def test_builder_built_prototype_clones_select_per_world(self):
        """A labeled soft body and rod built in a prototype and cloned per world with
        add_world stay selectable, with correctly offset ranges (the Isaac Lab pattern
        of building deformables in per-world builder hooks)."""
        proto = newton.ModelBuilder()
        proto.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            indices=[0, 1, 2, 3],
            density=100.0,
            k_mu=1.0e4,
            k_lambda=1.0e4,
            k_damp=0.0,
            label="soft_proto",
        )
        proto.add_rod(
            positions=[(0.0, 2.0, 1.0), (0.1, 2.0, 1.0), (0.2, 2.0, 1.0)],
            radius=0.02,
            label="cable_proto",
            wrap_in_articulation=True,
            body_frame_origin="com",
        )

        scene = newton.ModelBuilder()
        scene.add_world(proto)
        scene.add_world(proto)
        model = scene.finalize()
        state = model.state()

        soft = DeformableView(model, "soft_proto", family="volume")
        self.assertEqual((soft.count, soft.worlds, soft.particles_per_group), (2, [0, 1], 4))
        (r0, r1) = soft.ranges("particle")
        self.assertEqual(r1[0] - r0[0], 4)
        self.assertNotEqual(r0, r1)

        cable = DeformableView(model, "cable_proto", family="curve")
        self.assertEqual((cable.count, cable.worlds, cable.bodies_per_group), (2, [0, 1], 2))
        self.assertEqual(cable.elements_per_group("joint"), 1)

        # State access round-trips through the offset ranges.
        positions = soft.get_particle_positions(state)
        lifted = positions.numpy()
        lifted[1, :, 2] += 3.0
        soft.set_particle_positions(state, wp.array(lifted, dtype=wp.vec3))
        np.testing.assert_allclose(soft.get_particle_positions(state).numpy(), lifted, atol=1e-6)

    def test_unlabeled_builder_deformables_record_no_group(self):
        """Deformables built without a label stay undiscoverable (no accidental groups)."""
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
            indices=[0, 1, 2, 0, 2, 3],
            density=0.1,
        )
        model = builder.finalize()
        with self.assertRaises(KeyError):
            DeformableView(model, "*", family="surface")


if __name__ == "__main__":
    unittest.main(verbosity=2)
