# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for mesh/heightfield broadphase AABB computation and bounding-sphere radius."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import GeoType, Heightfield, Mesh
from newton._src.geometry.utils import compute_shape_radius


class TestMeshShapeAABB(unittest.TestCase):
    """Verify that broadphase AABBs for mesh shapes are tight (not bounding-sphere)."""

    def _build_model_with_mesh_at(self, mesh, body_pos, body_quat=None, scale=None):
        """Helper: build a model with a single mesh shape on a free body."""
        if body_quat is None:
            body_quat = wp.quat_identity()
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(body_pos, body_quat))
        kwargs = {}
        if scale is not None:
            kwargs["scale"] = scale
        builder.add_shape_mesh(body=body, mesh=mesh, **kwargs)
        builder.add_ground_plane()
        model = builder.finalize()
        return model

    def _get_shape_aabb(self, model, shape_idx=0):
        """Run collision pipeline and return the world-space AABB for a shape."""
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=100)
        contacts = pipeline.contacts()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        pipeline.collide(state, contacts)
        lo = pipeline.narrow_phase.shape_aabb_lower.numpy()[shape_idx]
        hi = pipeline.narrow_phase.shape_aabb_upper.numpy()[shape_idx]
        return lo, hi

    def test_axis_aligned_mesh_tight_aabb(self):
        """Axis-aligned box mesh AABB should be close to the actual box extents."""
        hx, hy, hz = 0.2, 0.2, 0.05
        mesh = Mesh.create_box(hx, hy, hz, compute_inertia=True)
        pos = wp.vec3(0.0, 0.0, 1.0)
        model = self._build_model_with_mesh_at(mesh, pos)
        lo, hi = self._get_shape_aabb(model)

        margin = model.shape_margin.numpy()[0] + model.shape_gap.numpy()[0]

        np.testing.assert_allclose(lo[0], pos[0] - hx - margin, atol=1e-4)
        np.testing.assert_allclose(hi[0], pos[0] + hx + margin, atol=1e-4)
        np.testing.assert_allclose(lo[1], pos[1] - hy - margin, atol=1e-4)
        np.testing.assert_allclose(hi[1], pos[1] + hy + margin, atol=1e-4)
        np.testing.assert_allclose(lo[2], pos[2] - hz - margin, atol=1e-4)
        np.testing.assert_allclose(hi[2], pos[2] + hz + margin, atol=1e-4)

    def test_off_center_mesh_no_false_broadphase(self):
        """A mesh far above a ground plane must NOT produce a broadphase pair.

        This is the core scenario from the bug report: a flat table mesh at z=0.05
        should not have a bounding sphere so large that it overlaps with a gripper
        mesh at z=0.375.
        """
        table_half = (0.2, 0.2, 0.05)
        table_mesh = Mesh.create_box(*table_half, compute_inertia=True)
        builder = newton.ModelBuilder()

        table_body = builder.add_body(
            xform=wp.transform(wp.vec3(0, 0, table_half[2]), wp.quat_identity()),
        )
        builder.add_shape_mesh(body=table_body, mesh=table_mesh)

        gripper_half = (0.03, 0.02, 0.04)
        gripper_mesh = Mesh.create_box(*gripper_half, compute_inertia=True)
        gripper_body = builder.add_body(
            xform=wp.transform(wp.vec3(0, 0, 0.375), wp.quat_identity()),
        )
        builder.add_shape_mesh(body=gripper_body, mesh=gripper_mesh)

        builder.add_ground_plane()
        model = builder.finalize()

        pipeline = newton.CollisionPipeline(model, rigid_contact_max=200)
        contacts = pipeline.contacts()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        pipeline.collide(state, contacts)

        table_hi = pipeline.narrow_phase.shape_aabb_upper.numpy()[0]
        margin = model.shape_margin.numpy()[0] + model.shape_gap.numpy()[0]
        self.assertLess(
            table_hi[2],
            table_half[2] * 2 + margin + 0.01,
            "Table AABB upper-z should be close to 0.1, not inflated by bounding sphere",
        )

        gripper_lo = pipeline.narrow_phase.shape_aabb_lower.numpy()[1]
        self.assertGreater(gripper_lo[2], table_hi[2], "Gripper and table AABBs should not overlap in z")

    def test_rotated_mesh_aabb(self):
        """A rotated mesh AABB should still tightly bound the geometry."""
        hx, hy, hz = 1.0, 0.1, 0.1
        mesh = Mesh.create_box(hx, hy, hz, compute_inertia=True)

        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2.0)
        pos = wp.vec3(0.0, 0.0, 2.0)
        model = self._build_model_with_mesh_at(mesh, pos, rot)
        lo, hi = self._get_shape_aabb(model)

        margin = model.shape_margin.numpy()[0] + model.shape_gap.numpy()[0]

        # After 90° Z rotation: X-extent ≈ hy, Y-extent ≈ hx
        np.testing.assert_allclose(hi[0] - lo[0], 2 * hy + 2 * margin, atol=0.02)
        np.testing.assert_allclose(hi[1] - lo[1], 2 * hx + 2 * margin, atol=0.02)
        np.testing.assert_allclose(hi[2] - lo[2], 2 * hz + 2 * margin, atol=0.02)

    def test_nonuniform_scale_mesh_aabb(self):
        """Non-uniform scale should be baked into the local AABB by the builder."""
        hx, hy, hz = 1.0, 1.0, 1.0
        mesh = Mesh.create_box(hx, hy, hz, compute_inertia=True)
        sx, sy, sz = 2.0, 0.5, 3.0
        pos = wp.vec3(0.0, 0.0, 5.0)
        model = self._build_model_with_mesh_at(mesh, pos, scale=(sx, sy, sz))
        lo, hi = self._get_shape_aabb(model)

        margin = model.shape_margin.numpy()[0] + model.shape_gap.numpy()[0]

        np.testing.assert_allclose(lo[0], pos[0] - hx * sx - margin, atol=1e-4)
        np.testing.assert_allclose(hi[0], pos[0] + hx * sx + margin, atol=1e-4)
        np.testing.assert_allclose(lo[1], pos[1] - hy * sy - margin, atol=1e-4)
        np.testing.assert_allclose(hi[1], pos[1] + hy * sy + margin, atol=1e-4)
        np.testing.assert_allclose(lo[2], pos[2] - hz * sz - margin, atol=1e-4)
        np.testing.assert_allclose(hi[2], pos[2] + hz * sz + margin, atol=1e-4)


class TestHeightfieldBoundingSphere(unittest.TestCase):
    """Verify heightfield bounding sphere handles asymmetric Z ranges correctly."""

    def test_asymmetric_z_range(self):
        """Asymmetric Z range [0, 10]: radius must use max(|min_z|, |max_z|), not half-range."""
        nrow, ncol = 4, 4
        data = np.zeros((nrow, ncol), dtype=np.float32)
        hf = Heightfield(data=data, nrow=nrow, ncol=ncol, hx=3.0, hy=4.0, min_z=0.0, max_z=10.0)
        radius = compute_shape_radius(GeoType.HFIELD, (1.0, 1.0, 1.0), hf)

        # Old code used (max_z - min_z)/2 = 5, giving sqrt(9+16+25) ≈ 7.07.
        # Correct: max(|0|, |10|) = 10, giving sqrt(9+16+100) ≈ 11.18.
        expected = np.sqrt(3.0**2 + 4.0**2 + 10.0**2)
        self.assertAlmostEqual(radius, expected, places=5)

    def test_radius_bounds_all_vertices(self):
        """Every heightfield corner vertex must lie within the bounding sphere.

        Uses min_z < 0 so |min_z| > |max_z|, exercising the negative-Z branch.
        """
        nrow, ncol = 4, 4
        data = np.zeros((nrow, ncol), dtype=np.float32)
        hf = Heightfield(data=data, nrow=nrow, ncol=ncol, hx=3.0, hy=4.0, min_z=-8.0, max_z=2.0)
        radius = compute_shape_radius(GeoType.HFIELD, (1.0, 1.0, 1.0), hf)

        corners = np.array(
            [
                [-3.0, -4.0, -8.0],
                [3.0, -4.0, -8.0],
                [-3.0, 4.0, -8.0],
                [3.0, 4.0, -8.0],
                [-3.0, -4.0, 2.0],
                [3.0, -4.0, 2.0],
                [-3.0, 4.0, 2.0],
                [3.0, 4.0, 2.0],
            ]
        )
        dists = np.linalg.norm(corners, axis=1)
        self.assertGreaterEqual(radius, np.max(dists) - 1e-6)


class TestHeightfieldLocalAABB(unittest.TestCase):
    """Verify shape_collision_aabb_lower/upper are correct for heightfields."""

    def test_heightfield_local_aabb(self):
        """Heightfield local AABB should reflect hx, hy, min_z, max_z."""
        nrow, ncol = 10, 10
        data = np.zeros((nrow, ncol), dtype=np.float32)
        hf = Heightfield(data=data, nrow=nrow, ncol=ncol, hx=5.0, hy=3.0, min_z=0.0, max_z=2.0)

        builder = newton.ModelBuilder()
        builder.add_shape_heightfield(heightfield=hf)
        model = builder.finalize()

        lo = model.shape_collision_aabb_lower.numpy()[0]
        hi = model.shape_collision_aabb_upper.numpy()[0]

        np.testing.assert_allclose(lo, [-5.0, -3.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(hi, [5.0, 3.0, 2.0], atol=1e-5)

    def test_heightfield_local_aabb_with_scale(self):
        """Heightfield local AABB should incorporate scale."""
        nrow, ncol = 4, 4
        data = np.zeros((nrow, ncol), dtype=np.float32)
        hf = Heightfield(data=data, nrow=nrow, ncol=ncol, hx=2.0, hy=3.0, min_z=-1.0, max_z=4.0)

        builder = newton.ModelBuilder()
        builder.add_shape_heightfield(heightfield=hf, scale=(2.0, 0.5, 3.0))
        model = builder.finalize()

        lo = model.shape_collision_aabb_lower.numpy()[0]
        hi = model.shape_collision_aabb_upper.numpy()[0]

        np.testing.assert_allclose(lo, [-4.0, -1.5, -3.0], atol=1e-5)
        np.testing.assert_allclose(hi, [4.0, 1.5, 12.0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
