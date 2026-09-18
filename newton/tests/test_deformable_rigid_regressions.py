# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise mixed geometry, signed finite planes, and collision replay."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton._src.geometry.sdf_texture import SLOT_LINEAR, TextureSDFData
from newton._src.geometry.soft_contacts_sdf import (
    _closest_edge_plane,
    _closest_face_plane,
    eval_shape_sdf,
    optimize_edge_sdf,
    optimize_face_sdf,
)
from newton.tests.unittest_utils import add_function_test, configure_sdf_for_collision_shapes, get_test_devices


class TestDeformableRigidRegressions(unittest.TestCase):
    pass


@wp.kernel
def _sample_flat_sdf_features(sdfs: wp.array[TextureSDFData], scale: wp.vec3, distances: wp.array[float]):
    a = wp.vec3(0.2, 0.2, 0.5)
    b = wp.vec3(0.8, 0.2, 0.5)
    c = wp.vec3(0.5, 0.8, 0.5)
    _lower, phi, _grad = eval_shape_sdf(newton.GeoType.MESH, scale, a, 0, sdfs)
    distances[0] = phi
    _u, _x, phi, _grad = optimize_edge_sdf(newton.GeoType.MESH, scale, a, b, 0, sdfs, 24)
    distances[1] = phi
    _bary, _x, phi, _grad = optimize_face_sdf(newton.GeoType.MESH, scale, a, b, c, 0, sdfs, 24, 16)
    distances[2] = phi


def test_flat_sdf_preserves_separation(test, device):
    """Preserve signed separation at zero-gradient samples instead of creating phantom contacts."""
    # A constant trilinear cell models a quantized plateau or stationary SDF sample.
    # Coarse cloth edges/faces can minimize onto these cells far from the surface.
    for distance in (0.25, -0.25):
        texture = wp.Texture3D(
            np.full((2, 2, 2), distance, dtype=np.float32),
            filter_mode=wp.TextureFilterMode.LINEAR,
            address_mode=wp.TextureAddressMode.CLAMP,
            normalized_coords=False,
            device=device,
        )
        sdf = TextureSDFData()
        sdf.coarse_texture = texture
        sdf.subgrid_texture = texture
        sdf.subgrid_start_slots = wp.full((1, 1, 1), int(SLOT_LINEAR), dtype=wp.uint32, device=device)
        sdf.sdf_box_lower = wp.vec3(-1.0)
        sdf.sdf_box_upper = wp.vec3(1.0)
        sdf.inv_sdf_dx = wp.vec3(0.5)
        sdf.subgrid_size = 1
        sdf.subgrid_size_f = 1.0
        sdf.fine_to_coarse = 1.0
        sdfs = wp.array([sdf], dtype=TextureSDFData, device=device)
        distances = wp.empty(3, dtype=float, device=device)
        for scale in (wp.vec3(1.0), wp.vec3(2.0, 3.0, 4.0), wp.vec3(-2.0, 3.0, 4.0)):
            wp.launch(_sample_flat_sdf_features, dim=1, inputs=[sdfs, scale, distances], device=device)
            np.testing.assert_allclose(distances.numpy(), distance * min(abs(v) for v in scale), atol=1.0e-6)


@wp.kernel
def _sample_nonconvex_sdf_search(sdfs: wp.array[TextureSDFData], distances: wp.array2d[float]):
    it = wp.tid()
    p = wp.vec3(0.0)
    q = wp.vec3(1.0, 0.0, 0.0)
    _lower, phi, _grad = eval_shape_sdf(newton.GeoType.MESH, wp.vec3(1.0), p, 0, sdfs)
    distances[it, 0] = phi
    _u, _x, phi, _grad = optimize_edge_sdf(newton.GeoType.MESH, wp.vec3(1.0), p, q, 0, sdfs, 24)
    distances[it, 1] = phi
    _u, _x, phi, _grad = optimize_edge_sdf(newton.GeoType.MESH, wp.vec3(1.0), q, p, 0, sdfs, 24)
    distances[it, 2] = phi
    _bary, _x, phi, _grad = optimize_face_sdf(
        newton.GeoType.MESH,
        wp.vec3(1.0),
        wp.vec3(-0.5, -0.3, 0.0),
        wp.vec3(1.0, -0.3, 0.0),
        wp.vec3(-0.5, 0.6, 0.0),
        0,
        sdfs,
        it,
        16,
    )
    distances[it, 3] = phi


def test_nonconvex_sdf_search_retains_endpoints(test, device):
    """Retain a better endpoint and prevent uphill face iterations on a multi-basin mesh SDF."""
    sphere = newton.Mesh.create_sphere(0.2, num_latitudes=16, num_longitudes=32)
    points = np.asarray(sphere.vertices)
    mesh = newton.Mesh(
        np.concatenate((points + np.array((0.0, 0.0, 0.21)), points + np.array((0.75, 0.0, 0.3)))),
        np.concatenate((sphere.indices, np.asarray(sphere.indices) + len(points))),
    )
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.configure_sdf(force_sdf=True)
    builder.add_shape_mesh(body=-1, mesh=mesh, cfg=cfg)
    model = builder.finalize(device=device)
    distances = wp.empty((25, 4), dtype=float, device=device)
    wp.launch(_sample_nonconvex_sdf_search, dim=25, inputs=[model._texture_sdf_data, distances], device=device)
    values = distances.numpy()
    test.assertLess(values[0, 0], 0.02)
    for column in (1, 2):
        with test.subTest(edge_direction=column):
            test.assertLessEqual(float(np.max(values[:, column] - values[:, 0])), 1.0e-6)
    with test.subTest(feature="face"):
        test.assertLessEqual(float(np.max(np.diff(values[:, 3]))), 1.0e-6)


def test_soft_contact_workspace_storage(test, device):
    """Avoid unused SDF scratch and keep mesh provenance in final contact slots."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)
    builder.add_cloth_grid(
        pos=wp.vec3(-0.4, -0.4, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=2,
        dim_y=2,
        cell_x=0.4,
        cell_y=0.4,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
    test.assertEqual(pipeline._soft_sdf_fallback_tids.size, 0)
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh.create_box(0.5, 0.5, 0.5))
    configure_sdf_for_collision_shapes(builder)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
    test.assertEqual(pipeline._soft_sdf_fallback_tids.size, 0)
    test.assertIsNotNone(pipeline._soft_mesh_contact_data)
    contacts = pipeline.contacts()
    test.assertEqual(contacts._soft_contact_mesh_features.size, contacts.soft_contact_max)
    test.assertFalse(hasattr(pipeline._soft_mesh_contact_data, "candidates"))
    # Exercise the counter bound without allocating billions of candidate records.
    oversized = mock.MagicMock()
    oversized.__len__.return_value = np.iinfo(np.int32).max
    with (
        mock.patch("newton._src.sim.collide._build_soft_edge_rigid_contact_pairs", return_value=oversized),
        test.assertRaisesRegex(ValueError, "32-bit"),
    ):
        newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)


def test_soft_contact_accumulation_thread_counts(test, device):
    """Preserve coupled body and cloth updates when increasing contact accumulation lanes."""
    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.5)
    builder.add_cloth_grid(
        pos=wp.vec3(-0.4, -0.4, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=8,
        dim_y=8,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
    )
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.1)
    results = []
    for threads in (4, 128):
        solver = newton.solvers.SolverVBD(
            model,
            iterations=2,
            rigid_compliant_alm=True,
            rigid_body_particle_contact_buffer_size=1024,
        )
        solver._body_particle_contact_threads = threads
        state_in, state_out = model.state(), model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state_in, contacts)
        test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 128)
        solver.step(state_in, state_out, model.control(), contacts, 0.001)
        results.append((state_out.body_q.numpy(), state_out.body_qd.numpy(), state_out.particle_q.numpy()))
    for serial, parallel in zip(*results, strict=True):
        np.testing.assert_allclose(parallel, serial, rtol=1.0e-5, atol=1.0e-6)


@wp.kernel
def _plane_features(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    scale: wp.vec3,
    points: wp.array[wp.vec3],
    distances: wp.array[float],
    normals: wp.array[wp.vec3],
):
    _u, edge_point, edge_phi, edge_normal = _closest_edge_plane(scale, a, b)
    _bary, face_point, face_phi, face_normal = _closest_face_plane(scale, a, b, c)
    points[0] = edge_point
    distances[0] = edge_phi
    normals[0] = edge_normal
    points[1] = face_point
    distances[1] = face_phi
    normals[1] = face_normal


def test_finite_plane_feature_geometry(test, device):
    """Check signed clipping and outside closest features across scales and face orientations."""
    cases = (
        # Sloping penetration: the minimum lies on a footprint edge, not a soft vertex.
        (((-1.0, -1.0, -2.0), (3.0, -1.0, 0.0), (-1.0, 1.0, -2.0)), -1.55, (0.0, 0.0, 1.0)),
        (((-1.0, -1.0, 0.2), (3.0, -1.0, 0.2), (-1.0, 1.0, 0.2)), 0.2, (0.0, 0.0, 1.0)),
        # A vertical soft face outside the quad must use its corner normal, even below the sheet.
        (((0.2, 0.2, -1.0), (0.2, 0.2, 1.0), (0.3, 0.3, 0.0)), np.sqrt(0.02), (np.sqrt(0.5), np.sqrt(0.5), 0.0)),
        (((0.0, -1.0, -2.0), (0.0, 1.0, -2.0), (0.0, 0.0, 2.0)), -2.0, (0.0, 0.0, 1.0)),
    )
    for factor in (0.001, 1.0, 1000.0):
        for vertices, expected_phi, expected_normal in cases:
            with test.subTest(factor=factor, vertices=vertices):
                points = wp.empty(2, dtype=wp.vec3, device=device)
                distances = wp.empty(2, dtype=float, device=device)
                normals = wp.empty(2, dtype=wp.vec3, device=device)
                wp.launch(
                    _plane_features,
                    dim=1,
                    inputs=[
                        *(wp.vec3(*(factor * np.array(v))) for v in vertices),
                        wp.vec3(0.2 * factor, 0.2 * factor, 0.0),
                    ],
                    outputs=[points, distances, normals],
                    device=device,
                )
                test.assertAlmostEqual(float(distances.numpy()[1]) / factor, expected_phi, delta=2.0e-5)
                np.testing.assert_allclose(normals.numpy()[1], expected_normal, atol=2.0e-5)
                test.assertTrue(np.isfinite(points.numpy()).all())
    # Clip an edge crossing the footprint while both endpoints are outside and deeply penetrating.
    wp.launch(
        _plane_features,
        dim=1,
        inputs=[
            wp.vec3(-1.0, 0.0, -2.0),
            wp.vec3(1.0, 0.0, -1.0),
            wp.vec3(0.0, 1.0, -1.0),
            wp.vec3(0.2, 0.2, 0.0),
        ],
        outputs=[points, distances, normals],
        device=device,
    )
    test.assertAlmostEqual(float(distances.numpy()[0]), -1.55, delta=1.0e-6)
    np.testing.assert_allclose(points.numpy()[0], (-0.1, 0.0, -1.55), atol=1.0e-6)


def test_large_heightfield_task_contacts(test, device):
    """Preserve exact minima across split terrain scans, empty tasks, and graph replay."""
    builder = newton.ModelBuilder()
    data = np.zeros((33, 33), dtype=np.float32)
    data[12:20, 12:20] = 1.0
    builder.add_shape_heightfield(
        heightfield=newton.Heightfield(data=data, nrow=33, ncol=33, hx=1.0, hy=1.0, min_z=0.0, max_z=0.01)
    )
    for scale, offset, height in ((4.0, 0.0, 0.02), (0.05, 0.0, 0.02), (1.0, 10.0, 0.02), (4.0, 0.0, -0.02)):
        first = len(builder.particle_q)
        for x, y in ((-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)):
            builder.add_particle(wp.vec3(x * scale + offset, y * scale, height), wp.vec3(), 0.1, radius=0.0)
        builder.add_triangle(first, first + 1, first + 2)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.05)
    state = model.state()
    reference = pipeline.contacts()
    reference._soft_heightfield_work = None
    pipeline.collide(state, reference)

    def records(contacts):
        count = int(contacts.soft_contact_count.numpy()[0])
        indices = contacts.soft_contact_indices.numpy()[:count]
        order = np.lexsort(indices.T[::-1])
        return np.concatenate(
            [
                indices[order],
                contacts.soft_contact_barycentric.numpy()[:count][order],
                contacts.soft_contact_body_pos.numpy()[:count][order],
                contacts.soft_contact_normal.numpy()[:count][order],
            ],
            axis=1,
        )

    expected = records(reference)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    np.testing.assert_allclose(records(contacts), expected, atol=1.0e-6)
    counts, offsets, _winners = contacts._soft_heightfield_work
    np.testing.assert_array_equal(counts.numpy(), (4, 0, 0, 4, 0))
    np.testing.assert_array_equal(offsets.numpy(), (0, 4, 4, 4, 8))
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            pipeline.collide(state, contacts)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(records(contacts), expected, atol=1.0e-6)
    original = state.particle_q.numpy().copy()
    shifted = original.copy()
    shifted[:, 2] += 2.0
    state.particle_q.assign(shifted)
    if device.is_cuda:
        wp.capture_launch(capture.graph)
    else:
        pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0)
    np.testing.assert_array_equal(counts.numpy(), np.zeros(5, dtype=np.int64))
    state.particle_q.assign(original)
    if device.is_cuda:
        wp.capture_launch(capture.graph)
    else:
        pipeline.collide(state, contacts)
    np.testing.assert_allclose(records(contacts), expected, atol=1.0e-6)

    # As with SDF compaction, differentiable collision retains the serial replay path.
    grad_model = builder.finalize(device=device, requires_grad=True)
    grad_pipeline = newton.CollisionPipeline(
        grad_model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.05
    )
    test.assertIsNone(grad_pipeline.contacts()._soft_heightfield_work)


def test_particle_gradient_after_pipeline_reuse(test, device):
    """Preserve particle contact gradients when a later collision overwrites rigid bounds."""
    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_sphere(body=body, radius=1.0)
    builder.add_particle(pos=wp.vec3(1.0, 0.2, 0.0), vel=wp.vec3(), mass=1.0, radius=0.1)
    model = builder.finalize(device=device, requires_grad=True)
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.1)
    gradients = []
    for reuse in (False, True):
        state = model.state()
        contacts = pipeline.contacts()
        with wp.Tape() as tape:
            pipeline.collide(state, contacts)
        test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 1)
        if reuse:
            later_state = model.state()
            later_state.body_q.assign(np.array(((10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),), dtype=np.float32))
            pipeline.collide(later_state, pipeline.contacts())
        seed = wp.ones(contacts.soft_contact_body_pos.shape, dtype=wp.vec3, device=device)
        tape.backward(grads={contacts.soft_contact_body_pos: seed})
        gradients.append(state.particle_q.grad.numpy().copy())
    test.assertGreater(float(np.linalg.norm(gradients[0])), 0.5)
    np.testing.assert_allclose(gradients[1], gradients[0], atol=1.0e-6)


def test_finite_plane_penetrating_face(test, device):
    """Detect the finite footprint inside a penetrating face whose vertices are all outside."""
    builder = newton.ModelBuilder()
    builder.add_shape_plane(width=0.2, length=0.2)
    for point in ((-1.0, -1.0, -10.0), (3.0, -1.0, -10.0), (-1.0, 1.0, -10.0)):
        builder.add_particle(wp.vec3(*point), wp.vec3(), 0.1, radius=0.0)
    builder.add_triangle(0, 1, 2)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.05)
    state = model.state()
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    count = int(contacts.soft_contact_count.numpy()[0])
    test.assertEqual(count, 1)
    bary = contacts.soft_contact_barycentric.numpy()[0]
    point = bary @ state.particle_q.numpy()
    test.assertLessEqual(abs(float(point[0])), 0.10001)
    test.assertLessEqual(abs(float(point[1])), 0.10001)
    np.testing.assert_allclose(contacts.soft_contact_body_pos.numpy()[0], (point[0], point[1], 0.0), atol=1.0e-5)
    np.testing.assert_allclose(contacts.soft_contact_normal.numpy()[0], (0.0, 0.0, 1.0), atol=1.0e-6)


def test_mixed_mesh_edge_dispatch(test, device):
    """Keep mesh edge contacts when compact scheduling is enabled in a mixed scene."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)
    mesh_shape = builder.add_shape_mesh(body=-1, mesh=newton.Mesh.create_box(0.5, 0.5, 0.5))
    builder.add_cloth_grid(
        pos=wp.vec3(-0.4, -0.4, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=2,
        dim_y=2,
        cell_x=0.4,
        cell_y=0.4,
        mass=0.1,
    )
    configure_sdf_for_collision_shapes(builder)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.1)
    state = model.state()
    records = []
    for threshold in (10**9, 0):
        # Force compaction on this small fixture with room for every possible edge append.
        pipeline._soft_sdf_fallback_tids = wp.empty(
            max(len(pipeline.soft_edge_rigid_pairs), len(pipeline.soft_face_rigid_pairs)),
            dtype=wp.int32,
            device=device,
        )
        contacts = pipeline.contacts()
        with (
            mock.patch("newton._src.geometry.soft_contacts_sdf._SDF_COMPACTION_MIN_PAIRS", threshold),
            mock.patch("newton._src.geometry.soft_contacts_sdf._SDF_SPECIALIZATION_MIN_PAIRS_PER_GEO", 0),
        ):
            pipeline.collide(state, contacts)
        count = int(contacts.soft_contact_count.numpy()[0])
        indices = contacts.soft_contact_indices.numpy()[:count]
        shapes = contacts.soft_contact_shape.numpy()[:count]
        rows = np.flatnonzero((shapes == mesh_shape) & (indices[:, 1] >= 0) & (indices[:, 2] < 0))
        order = rows[np.lexsort((indices[rows, 1], indices[rows, 0]))]
        records.append((indices[order], contacts.soft_contact_body_pos.numpy()[order]))
    test.assertGreater(len(records[0][0]), 0)
    np.testing.assert_array_equal(records[1][0], records[0][0])
    np.testing.assert_allclose(records[1][1], records[0][1], atol=1.0e-6)


for device in get_test_devices():
    for fn in (
        test_soft_contact_accumulation_thread_counts,
        test_soft_contact_workspace_storage,
        test_particle_gradient_after_pipeline_reuse,
        test_finite_plane_penetrating_face,
        test_finite_plane_feature_geometry,
        test_large_heightfield_task_contacts,
    ):
        add_function_test(TestDeformableRigidRegressions, fn.__name__, fn, devices=[device])
    if device.is_cuda:
        add_function_test(
            TestDeformableRigidRegressions,
            test_nonconvex_sdf_search_retains_endpoints.__name__,
            test_nonconvex_sdf_search_retains_endpoints,
            devices=[device],
        )
        add_function_test(
            TestDeformableRigidRegressions,
            test_flat_sdf_preserves_separation.__name__,
            test_flat_sdf_preserves_separation,
            devices=[device],
        )
        add_function_test(
            TestDeformableRigidRegressions,
            test_mixed_mesh_edge_dispatch.__name__,
            test_mixed_mesh_edge_dispatch,
            devices=[device],
        )


if __name__ == "__main__":
    unittest.main()
