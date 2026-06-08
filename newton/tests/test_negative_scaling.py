# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for negative-scale (mirroring) support in the collision pipeline.

Symmetric primitives (sphere, box, capsule, cylinder, ellipsoid, plane) accept
negative scale components silently (treated as magnitudes). Cones absorb
negative radial components (rotational symmetry around the height axis) but
reject a negative height. Heightfields reject negative components. Mesh-class
shapes (mesh, convex hull) carry signed scale natively through the narrow
phase, with triangle winding, AABBs, and SDF/soft-contact heuristics all
corrected for ``det(scale) < 0``.
"""

import os
import unittest

import numpy as np
import warp as wp

import newton
from newton import GeoType, Mesh
from newton._src.geometry.utils import compute_shape_radius, transform_points
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal


def _build_mesh_pair(scale_a, scale_b, body_pos_a=(-0.5, 0.0, 1.0), body_pos_b=(0.5, 0.0, 1.0)):
    """Build a two-body model with axis-aligned cube meshes at the given scales."""
    cube = Mesh.create_box(0.5, 0.5, 0.5, compute_inertia=True)
    builder = newton.ModelBuilder()
    body_a = builder.add_body(xform=wp.transform(wp.vec3(*body_pos_a), wp.quat_identity()))
    builder.add_shape_mesh(body=body_a, mesh=cube, scale=scale_a)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(*body_pos_b), wp.quat_identity()))
    builder.add_shape_mesh(body=body_b, mesh=cube, scale=scale_b)
    builder.add_ground_plane()
    return builder.finalize()


def _run_pipeline(model):
    """Run a single collide() pass and return the contact arrays."""
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=200)
    contacts = pipeline.contacts()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    pipeline.collide(state, contacts)
    return pipeline, contacts


class TestBuilderNormalization(unittest.TestCase):
    """Symmetric primitives absorb negative scale components via abs() in the builder."""

    def test_sphere_negative_radius_absorbed(self):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        shape = builder.add_shape_sphere(body=body, radius=-2.0)
        scale = builder.shape_scale[shape]
        # Sphere stores radius only in scale[0]; the y/z components are unused.
        self.assertEqual(scale[0], 2.0)

    def test_box_negative_extents_absorbed(self):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        shape = builder.add_shape_box(body=body, hx=-1.0, hy=2.0, hz=-3.0)
        scale = builder.shape_scale[shape]
        self.assertEqual(tuple(scale), (1.0, 2.0, 3.0))

    def test_capsule_negative_absorbed(self):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        shape = builder.add_shape_capsule(body=body, radius=-0.3, half_height=-0.5)
        scale = builder.shape_scale[shape]
        np.testing.assert_allclose(scale[:2], (0.3, 0.5), rtol=1e-6)

    def test_cylinder_negative_absorbed(self):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        shape = builder.add_shape_cylinder(body=body, radius=-0.4, half_height=-0.6)
        scale = builder.shape_scale[shape]
        np.testing.assert_allclose(scale[:2], (0.4, 0.6), rtol=1e-6)

    def test_ellipsoid_negative_absorbed(self):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        shape = builder.add_shape_ellipsoid(body=body, rx=-1.0, ry=-2.0, rz=-3.0)
        scale = builder.shape_scale[shape]
        self.assertEqual(tuple(scale), (1.0, 2.0, 3.0))

    def test_generic_add_shape_plane_negative_absorbed(self):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        shape = builder.add_shape(
            body=body,
            type=GeoType.PLANE,
            scale=(-2.0, -3.0, 0.0),
        )
        scale = builder.shape_scale[shape]
        self.assertEqual(tuple(scale), (2.0, 3.0, 0.0))

    def test_cone_negative_radius_absorbed(self):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        shape = builder.add_shape_cone(body=body, radius=-0.5, half_height=1.0)
        scale = builder.shape_scale[shape]
        np.testing.assert_allclose((scale[0], scale[1]), (0.5, 1.0), rtol=1e-6)

    def test_cone_negative_half_height_rejected(self):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        with self.assertRaises(ValueError):
            builder.add_shape_cone(body=body, radius=0.5, half_height=-1.0)

    def test_heightfield_negative_rejected(self):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        with self.assertRaises(ValueError):
            builder.add_shape(
                body=body,
                type=GeoType.HFIELD,
                scale=(-1.0, 1.0, 1.0),
            )

    def test_mesh_negative_scale_preserved(self):
        cube = Mesh.create_box(0.5, 0.5, 0.5, compute_inertia=True)
        builder = newton.ModelBuilder()
        body = builder.add_body()
        shape = builder.add_shape_mesh(body=body, mesh=cube, scale=(-1.0, 1.0, 1.0))
        scale = builder.shape_scale[shape]
        self.assertEqual(tuple(scale), (-1.0, 1.0, 1.0))


class TestComputeShapeRadius(unittest.TestCase):
    """``compute_shape_radius`` must use magnitudes so envelopes stay positive."""

    def test_sphere_radius_uses_magnitude(self):
        # The builder normally absorbs the sign first; verify the radius helper
        # is independently safe so direct callers (e.g. importer warm-up paths)
        # cannot regress.
        self.assertEqual(compute_shape_radius(GeoType.SPHERE, (-2.0, -2.0, -2.0), None), 2.0)

    def test_box_radius_uses_magnitude(self):
        np.testing.assert_allclose(
            compute_shape_radius(GeoType.BOX, (-1.0, -2.0, -2.0), None),
            np.linalg.norm([1.0, 2.0, 2.0]),
        )

    def test_capsule_radius_uses_magnitude(self):
        self.assertEqual(
            compute_shape_radius(GeoType.CAPSULE, (-0.5, -1.0, 0.0), None),
            1.5,
        )

    def test_ellipsoid_radius_uses_magnitude(self):
        self.assertEqual(
            compute_shape_radius(GeoType.ELLIPSOID, (-1.0, -2.0, -3.0), None),
            3.0,
        )

    def test_plane_finite_with_signed_scale(self):
        # Mixed signs on a finite plane should still classify as finite, not infinite.
        radius = compute_shape_radius(GeoType.PLANE, (-2.0, 1.0, 0.0), None)
        np.testing.assert_allclose(radius, np.linalg.norm([2.0, 1.0, 0.0]) * 0.5)


class TestMeshMirrorContacts(unittest.TestCase):
    """Mesh-mesh narrow phase produces the expected contact count under mirroring."""

    def _count_contacts(self, contacts):
        return int(contacts.rigid_contact_count.numpy()[0])

    def test_unmirrored_baseline_generates_contacts(self):
        # Two unit cubes overlapping along x produce >0 contacts.
        model = _build_mesh_pair(scale_a=(1.0, 1.0, 1.0), scale_b=(1.0, 1.0, 1.0))
        _, contacts = _run_pipeline(model)
        self.assertGreater(self._count_contacts(contacts), 0)

    def test_single_axis_mirror_generates_contacts(self):
        # Mirror cube B along x: still a unit cube geometrically, contacts should
        # still form.
        model = _build_mesh_pair(scale_a=(1.0, 1.0, 1.0), scale_b=(-1.0, 1.0, 1.0))
        _, contacts = _run_pipeline(model)
        self.assertGreater(
            self._count_contacts(contacts),
            0,
            "Single-axis mirrored mesh should still produce contacts (winding swap fixes back-face cull)",
        )

    def test_handed_mirror_generates_contacts(self):
        # All three axes negated (det < 0): point-symmetric reflection.
        model = _build_mesh_pair(scale_a=(1.0, 1.0, 1.0), scale_b=(-1.0, -1.0, -1.0))
        _, contacts = _run_pipeline(model)
        self.assertGreater(self._count_contacts(contacts), 0)

    def test_two_axis_mirror_equivalent_to_rotation(self):
        # Two axes negated (det > 0): equivalent to a 180-degree rotation,
        # winding is preserved without a swap. Should still produce contacts.
        model = _build_mesh_pair(scale_a=(1.0, 1.0, 1.0), scale_b=(-1.0, -1.0, 1.0))
        _, contacts = _run_pipeline(model)
        self.assertGreater(self._count_contacts(contacts), 0)

    def test_mirrored_mesh_aabb_is_positive_volume(self):
        # Sanity: the broadphase AABB of a mirrored mesh must still be a proper
        # box (lower <= upper componentwise).
        model = _build_mesh_pair(scale_a=(-1.0, 1.0, -1.0), scale_b=(1.0, 1.0, 1.0))
        pipeline, _ = _run_pipeline(model)
        lo = pipeline.narrow_phase.shape_aabb_lower.numpy()[0]
        hi = pipeline.narrow_phase.shape_aabb_upper.numpy()[0]
        self.assertTrue(np.all(hi >= lo), f"AABB inverted under mirror: lo={lo}, hi={hi}")


class TestSharedMeshAcrossSignedScales(unittest.TestCase):
    """A single ``Mesh`` instance can be shared across shapes with different
    signed scales without re-baking. Mirroring is handled at kernel time by
    ``get_triangle_shape_from_mesh`` (winding swap when ``det(scale) < 0``).
    """

    def test_shared_mesh_signed_scales_both_collide(self):
        # One Mesh, two shapes with opposite-handed scales colliding with a
        # third reference cube. Both should produce contacts using the same
        # mesh data — no per-shape vertex baking is required.
        cube = Mesh.create_box(0.5, 0.5, 0.5, compute_inertia=True)
        builder = newton.ModelBuilder()

        body_pos = builder.add_body(xform=wp.transform(wp.vec3(-0.5, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_mesh(body=body_pos, mesh=cube, scale=(1.0, 1.0, 1.0))

        body_neg = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_mesh(body=body_neg, mesh=cube, scale=(-1.0, 1.0, 1.0))

        body_ref = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_mesh(body=body_ref, mesh=cube, scale=(1.0, 1.0, 1.0))

        builder.add_ground_plane()
        model = builder.finalize()

        # Crucial: the same source Mesh object is reused for all three shapes;
        # nothing about the loader/builder should clone or re-bake vertices.
        self.assertIs(builder.shape_source[0], cube)
        self.assertIs(builder.shape_source[1], cube)
        self.assertIs(builder.shape_source[2], cube)

        _, contacts = _run_pipeline(model)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


class TestUsdNegativeScale(unittest.TestCase):
    """USDA imports with authored negative scale follow builder semantics."""

    @staticmethod
    def _asset_path(name):
        return os.path.join(os.path.dirname(__file__), "assets", name)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_cube_with_negative_scale(self):
        builder = newton.ModelBuilder()
        result = builder.add_usd(self._asset_path("negative_scale_cube.usda"))

        shape_id = result["path_shape_map"]["/World/Body/collision_cube"]
        self.assertEqual(builder.shape_type[shape_id], GeoType.BOX)
        assert_np_equal(
            np.array(result["path_shape_scale"]["/World/Body/collision_cube"]),
            np.array([-1.0, 0.5, -1.5]),
            tol=1e-6,
        )
        assert_np_equal(np.array(builder.shape_scale[shape_id]), np.array([0.5, 0.25, 0.75]), tol=1e-6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_sphere_with_negative_scale(self):
        builder = newton.ModelBuilder()
        result = builder.add_usd(self._asset_path("negative_scale_sphere.usda"))

        shape_id = result["path_shape_map"]["/World/Body/collision_sphere"]
        self.assertEqual(builder.shape_type[shape_id], GeoType.SPHERE)
        assert_np_equal(
            np.array(result["path_shape_scale"]["/World/Body/collision_sphere"]),
            np.array([-1.0, -1.0, -1.0]),
            tol=1e-6,
        )
        assert_np_equal(np.array(builder.shape_scale[shape_id]), np.array([1.0, 0.0, 0.0]), tol=1e-6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_mesh_with_negative_scale_generates_contacts(self):
        builder = newton.ModelBuilder()
        result = builder.add_usd(self._asset_path("negative_scale_mesh_pair.usda"))

        mirrored_shape_id = result["path_shape_map"]["/World/MirroredBody/collision_mesh"]
        self.assertEqual(builder.shape_type[mirrored_shape_id], GeoType.MESH)
        assert_np_equal(
            np.array(result["path_shape_scale"]["/World/MirroredBody/collision_mesh"]),
            np.array([-1.0, 1.0, 1.0]),
            tol=1e-6,
        )
        self.assertLess(np.prod(np.array(builder.shape_scale[mirrored_shape_id])), 0.0)

        model = builder.finalize()
        _, contacts = _run_pipeline(model)
        self.assertGreater(
            int(contacts.rigid_contact_count.numpy()[0]),
            0,
            "USD-imported mirrored mesh should still generate contacts",
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_asymmetric_mesh_negative_scale_transform_matches_usd(self):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/World/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass_api.CreateMassAttr().Set(1.0)
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=np.float32,
        )
        mesh = UsdGeom.Mesh.Define(stage, "/World/Body/Collision")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        mesh_col_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        mesh_col_api.GetApproximationAttr().Set(UsdPhysics.Tokens.none)
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.CreatePointsAttr().Set([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in vertices])
        mesh.CreateFaceVertexCountsAttr().Set([3, 3, 3, 3])
        mesh.CreateFaceVertexIndicesAttr().Set([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
        mesh.AddScaleOp().Set(Gf.Vec3d(-1.0, 1.0, 1.0))

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)

        shape_id = result["path_shape_map"]["/World/Body/Collision"]
        imported_vertices = transform_points(
            builder.shape_source[shape_id].vertices,
            builder.shape_transform[shape_id],
            scale=builder.shape_scale[shape_id],
        )
        expected_vertices = vertices * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
        assert_np_equal(imported_vertices, expected_vertices, tol=1e-6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_asymmetric_mesh_negative_scale_from_parent_xform(self):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/World/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass_api.CreateMassAttr().Set(1.0)
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        parent = UsdGeom.Xform.Define(stage, "/World/Body/Parent")
        parent.AddScaleOp().Set(Gf.Vec3d(-1.0, 2.0, 3.0))

        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=np.float32,
        )
        mesh = UsdGeom.Mesh.Define(stage, "/World/Body/Parent/Collision")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        mesh_col_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        mesh_col_api.GetApproximationAttr().Set(UsdPhysics.Tokens.none)
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.CreatePointsAttr().Set([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in vertices])
        mesh.CreateFaceVertexCountsAttr().Set([3, 3, 3, 3])
        mesh.CreateFaceVertexIndicesAttr().Set([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)

        shape_path = "/World/Body/Parent/Collision"
        shape_id = result["path_shape_map"][shape_path]
        assert_np_equal(np.array(result["path_shape_scale"][shape_path]), np.array([-1.0, 2.0, 3.0]), tol=1e-6)

        imported_vertices = transform_points(
            builder.shape_source[shape_id].vertices,
            builder.shape_transform[shape_id],
            scale=builder.shape_scale[shape_id],
        )
        expected_vertices = vertices * np.array([-1.0, 2.0, 3.0], dtype=np.float32)
        assert_np_equal(imported_vertices, expected_vertices, tol=1e-6)


if __name__ == "__main__":
    unittest.main()
