# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for DeformableView: batched selection over imported deformable groups."""

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

        view = DeformableView(model, "/World/Cloth", family="cloth")
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

        view = DeformableView(model, "/World/Cable", family="cable")
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

        view = DeformableView(model, "/World/Soft", family="soft")
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

        view = DeformableView(model, "/World/Cloth*", family="cloth")
        self.assertEqual((view.count, view.count_per_world), (4, 2))
        self.assertEqual(view.labels, ["/World/ClothA", "/World/ClothB"] * 2)

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
            DeformableView(model, "/World/DoesNotExist", family="cloth")
        with self.assertRaisesRegex(ValueError, "Varying particle counts"):
            DeformableView(model, "/World/Cloth*", family="cloth")
        with self.assertRaisesRegex(ValueError, "Unknown deformable family"):
            DeformableView(model, "/World/ClothA", family="ropes")
        view = DeformableView(model, "/World/ClothA", family="cloth")
        with self.assertRaisesRegex(AttributeError, "no body elements"):
            view.get_body_transforms(model.state())
        with self.assertRaises(ValueError):
            view.set_particle_positions(model.state(), wp.zeros((2, 4), dtype=wp.vec3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
