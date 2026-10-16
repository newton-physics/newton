# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for compact convex support-map acceleration."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.geometry.support_function import (
    AcceleratedSupportMapDataProvider,
    GenericShapeData,
    SupportMapDataProvider,
    _has_verified_hull,
    pack_mesh_ptr,
    support_map_accelerated,
)
from newton._src.sim.builder import (
    _build_convex_support_acceleration,
    _convex_hull_adjacency,
    _deduplicate_convex_collision_mesh,
)


@wp.kernel
def _accelerated_support_kernel(
    mesh_id: wp.uint64,
    directions: wp.array[wp.vec3],
    shape_support_data: wp.array[wp.vec4i],
    support_lut: wp.array[int],
    support_vertex_offsets: wp.array[int],
    support_neighbors: wp.array[int],
    result: wp.array[wp.vec3],
):
    tid = wp.tid()
    shape = GenericShapeData()
    shape.shape_type = newton.GeoType.CONVEX_MESH
    shape.scale = wp.vec3(1.0)
    shape.auxiliary = pack_mesh_ptr(mesh_id)
    shape.center = wp.vec3(0.0)
    shape.shape_index = 0

    provider = AcceleratedSupportMapDataProvider()
    provider.shape_support_data = shape_support_data
    provider.support_lut = support_lut
    provider.support_vertex_offsets = support_vertex_offsets
    provider.support_neighbors = support_neighbors
    result[tid] = support_map_accelerated(shape, directions[tid], provider)


@wp.kernel
def _hull_validity_kernel(metadata: wp.array[wp.vec4i], result: wp.array[int]):
    shape = GenericShapeData()
    shape.shape_type = newton.GeoType.CONVEX_MESH
    shape.shape_index = wp.tid()
    provider = SupportMapDataProvider()
    provider.shape_support_data = metadata
    result[wp.tid()] = int(_has_verified_hull(shape, provider))


class TestConvexSupportAcceleration(unittest.TestCase):
    def test_nonpositive_resolution_disables_acceleration(self):
        """Keep legacy disabled metadata safe with empty acceleration buffers."""
        mesh = newton.Mesh.create_box(1.0, 1.0, 1.0, duplicate_vertices=False, compute_inertia=False)
        device = wp.get_device()
        mesh_id = mesh.finalize(device=device)
        directions = wp.array([(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        result = wp.empty(1, dtype=wp.vec3, device=device)
        empty = wp.empty(0, dtype=int, device=device)
        for packed in (-1, 0, 1 << 16):
            metadata = wp.array([(-1, -1, -1, packed)], dtype=wp.vec4i, device=device)
            wp.launch(
                _accelerated_support_kernel,
                dim=1,
                inputs=[mesh_id, directions, metadata, empty, empty, empty],
                outputs=[result],
                device=device,
            )
            self.assertEqual(float(result.numpy()[0, 0]), 1.0)
        metadata = wp.array([(-1, -1, -1, value) for value in (-1, 0, 32, 1 << 16, (1 << 16) | 32)], dtype=wp.vec4i)
        validity = wp.empty(5, dtype=int)
        wp.launch(_hull_validity_kernel, dim=5, inputs=[metadata], outputs=[validity])
        np.testing.assert_array_equal(validity.numpy(), [0, 0, 0, 1, 1])

    def test_hull_validation_checks_all_support_vertices(self):
        """Reject omitted extreme vertices while accepting unused interior support points."""
        mesh = newton.Mesh.create_box(1.0, 1.0, 1.0, duplicate_vertices=False, compute_inertia=False)
        for scale in (1.0e-3, 1.0, 1.0e3):
            for x, valid in ((0.0, True), (1.000003, False)):
                with self.subTest(scale=scale, x=x):
                    source = newton.Mesh(
                        compute_inertia=False,
                        vertices=np.vstack((mesh.vertices, [x, 0.0, 0.0])).astype(np.float32) * scale,
                        indices=mesh.indices,
                    )
                    self.assertEqual(_convex_hull_adjacency(source) is not None, valid)

    def test_hull_validation_does_not_weld_away_a_missing_small_face(self):
        """Keep a narrow opening visible when validating a nearly closed source mesh."""
        from scipy.spatial import ConvexHull

        mesh = newton.Mesh.create_box(1.0, 1.0, 1.0, duplicate_vertices=False, compute_inertia=False)
        vertices = np.asarray(mesh.vertices)
        vertices = vertices[~np.all(vertices == 1.0, axis=1)]
        bevel = np.ones((3, 3)) - np.eye(3) * 3.0e-7
        points = np.vstack((vertices, bevel)).astype(np.float32)
        faces = ConvexHull(points).simplices
        faces = faces[~np.all(faces >= len(vertices), axis=1)]
        self.assertIsNone(_convex_hull_adjacency(newton.Mesh(compute_inertia=False, vertices=points, indices=faces)))

    def test_hull_validation_rejects_incomplete_and_degenerate_surfaces(self):
        """Reject partial, duplicated, and flat source surfaces without changing their support points."""
        mesh = newton.Mesh.create_box(1.0, 1.0, 1.0, duplicate_vertices=False, compute_inertia=False)
        triangles = np.asarray(mesh.indices).reshape(-1, 3)
        for indices in (triangles[:-1], np.vstack((triangles, triangles[0]))):
            with self.subTest(triangles=len(indices)):
                self.assertIsNone(
                    _convex_hull_adjacency(newton.Mesh(compute_inertia=False, vertices=mesh.vertices, indices=indices))
                )
        flat = np.asarray(mesh.vertices).copy()
        flat[:, 2] = 0.0
        self.assertIsNone(_convex_hull_adjacency(newton.Mesh(compute_inertia=False, vertices=flat, indices=triangles)))

    def test_accelerated_support_matches_exhaustive_support_value(self):
        """Match exhaustive support values for accelerated convex meshes."""
        mesh = _deduplicate_convex_collision_mesh(
            newton.Mesh.create_sphere(
                0.5,
                num_latitudes=24,
                num_longitudes=48,
                compute_normals=False,
                compute_uvs=False,
                compute_inertia=False,
            )
        )
        acceleration = _build_convex_support_acceleration(mesh)
        self.assertIsNotNone(acceleration)
        lut, offsets, neighbors = acceleration

        rng = np.random.default_rng(123)
        directions = rng.normal(size=(32768, 3)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        device = wp.get_device()
        mesh_id = mesh.finalize(device=device)
        result = wp.empty(len(directions), dtype=wp.vec3, device=device)
        wp.launch(
            _accelerated_support_kernel,
            dim=len(directions),
            inputs=[
                mesh_id,
                wp.array(directions, dtype=wp.vec3, device=device),
                wp.array([(0, 0, 0, 32)], dtype=wp.vec4i, device=device),
                wp.array(lut, dtype=int, device=device),
                wp.array(offsets, dtype=int, device=device),
                wp.array(neighbors, dtype=int, device=device),
            ],
            outputs=[result],
            device=device,
        )

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        expected_dot = np.max(directions @ vertices.T, axis=1)
        actual_dot = np.einsum("ij,ij->i", result.numpy(), directions)
        np.testing.assert_allclose(actual_dot, expected_dot, rtol=2.0e-6, atol=2.0e-6)

    def test_nonconvex_topology_falls_back(self):
        """Reject acceleration data for unsupported nonconvex topology."""
        rng = np.random.default_rng(456)
        vertices = rng.normal(size=(300, 3)).astype(np.float32)
        vertices /= np.linalg.norm(vertices, axis=1)[:, None]
        # Arbitrary triangles through the point cloud are not supporting hull
        # faces, even though support mapping treats the point set as convex.
        indices = np.arange(300, dtype=np.int32).reshape(-1, 3)
        source = SimpleNamespace(vertices=vertices, indices=indices)
        self.assertIsNone(_build_convex_support_acceleration(source))

    def test_incomplete_hull_topology_falls_back(self):
        """Reject acceleration data when a connected convex hull has a missing face."""
        mesh = _deduplicate_convex_collision_mesh(
            newton.Mesh.create_sphere(
                0.5,
                num_latitudes=24,
                num_longitudes=48,
                compute_normals=False,
                compute_uvs=False,
                compute_inertia=False,
            )
        )
        incomplete = SimpleNamespace(
            vertices=mesh.vertices,
            indices=np.delete(np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3), 1, axis=0),
        )
        with self.assertWarnsRegex(RuntimeWarning, "complete closed hull topology"):
            self.assertIsNone(_build_convex_support_acceleration(incomplete))

    def test_small_hulls_do_not_allocate_acceleration(self):
        """Skip support acceleration for small convex hulls."""
        mesh = newton.Mesh.create_sphere(
            0.5,
            num_latitudes=8,
            num_longitudes=16,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        self.assertIsNone(_build_convex_support_acceleration(mesh))

    def test_pipeline_specializes_support_map_for_available_acceleration(self):
        """Specialize support mapping only when acceleration data exists."""

        def create_pipeline(num_latitudes, num_longitudes):
            builder = newton.ModelBuilder()
            mesh = newton.Mesh.create_sphere(
                0.5,
                num_latitudes=num_latitudes,
                num_longitudes=num_longitudes,
                compute_normals=False,
                compute_uvs=False,
                compute_inertia=False,
            )
            builder.add_shape_convex_hull(body=-1, mesh=mesh)
            model = builder.finalize(device="cpu")
            return newton.CollisionPipeline(model, broad_phase="explicit")

        small_pipeline = create_pipeline(8, 16)
        accelerated_pipeline = create_pipeline(24, 48)

        self.assertFalse(small_pipeline.narrow_phase.convex_support_acceleration)
        self.assertTrue(accelerated_pipeline.narrow_phase.convex_support_acceleration)

    def test_mesh_triangle_contacts_use_convex_support_acceleration(self):
        """Generate mesh-triangle contacts with accelerated convex support."""
        builder = newton.ModelBuilder()
        builder.add_shape_mesh(body=-1, mesh=newton.Mesh.create_box(0.5, compute_inertia=False))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.75)))
        convex = newton.Mesh.create_sphere(
            0.5,
            num_latitudes=24,
            num_longitudes=48,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        builder.add_shape_convex_hull(body=body, mesh=convex)
        model = builder.finalize()
        for reduce_contacts in (False, True):
            with self.subTest(reduce_contacts=reduce_contacts):
                pipeline = newton.CollisionPipeline(
                    model,
                    broad_phase="explicit",
                    reduce_contacts=reduce_contacts,
                )
                contacts = pipeline.contacts()

                self.assertTrue(pipeline.narrow_phase.convex_support_acceleration)
                pipeline.collide(model.state(), contacts)
                self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


if __name__ == "__main__":
    unittest.main()
