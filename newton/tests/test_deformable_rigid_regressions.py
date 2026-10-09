# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise mixed geometry, signed finite planes, and collision replay."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton._src.geometry import soft_contacts_mesh
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


def test_disconnected_mesh_contact_capacity(test, device):
    """Reserve default contact storage for every face near a particle, across disconnected patches."""
    base = newton.Mesh.create_box(0.02, 0.02, 0.02, compute_inertia=False)
    centers = np.array(
        (
            (0.06, 0.0, 0.0),
            (-0.06, 0.0, 0.0),
            (0.0, 0.06, 0.0),
            (0.0, -0.06, 0.0),
            (0.0, 0.0, 0.06),
            (0.0, 0.0, -0.06),
        ),
        dtype=np.float32,
    )
    vertices = np.concatenate([np.asarray(base.vertices) + center for center in centers])
    indices = np.concatenate(
        [np.asarray(base.indices) + component * len(base.vertices) for component in range(len(centers))]
    )

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh(vertices, indices, compute_inertia=False))
    builder.add_particle(wp.vec3(0.0), wp.vec3(0.0), mass=1.0, radius=0.0)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_gap=0.05, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.collide(state, contacts)

    # Detection reports every face within the band, which must fit the default capacity.
    raw_count = int(contacts.soft_contact_count.numpy()[0])
    test.assertGreater(raw_count, len(centers))
    test.assertLessEqual(raw_count, contacts.soft_contact_max)
    # The feature filter keeps one contact per patch.
    soft_contacts_mesh.filter_soft_mesh_contacts(model, state, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), len(centers))


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
    depth: float,
    points: wp.array[wp.vec3],
    distances: wp.array[float],
    normals: wp.array[wp.vec3],
):
    _u, edge_point, edge_phi, edge_normal = _closest_edge_plane(scale, a, b, depth)
    _bary, face_point, face_phi, face_normal = _closest_face_plane(scale, a, b, c, depth)
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
                        1.0e10,
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
            1.0e10,
        ],
        outputs=[points, distances, normals],
        device=device,
    )
    test.assertAlmostEqual(float(distances.numpy()[0]), -1.55, delta=1.0e-6)
    np.testing.assert_allclose(points.numpy()[0], (-0.1, 0.0, -1.55), atol=1.0e-6)

    # A finite sheet only penetrates within its slab depth, like the per-particle sdf_plane.
    bounded_cases = (
        # Entirely below the slab: the unsigned sheet distance from below, never a penetration.
        ((-1.0, -1.0, -0.3), (3.0, -1.0, -0.3), (-1.0, 1.0, -0.3), np.hypot(0.9, 0.3), 0.3, (0.0, 0.0, -1.0)),
        # Sloping through the slab bottom: clip the signed minimum at the slab depth.
        ((0.0, -0.05, 0.5), (0.0, 0.05, -1.0), (0.05, 0.0, 0.5), -0.1, -0.1, (0.0, 0.0, 1.0)),
    )
    for a, b, c, edge_phi, face_phi, face_normal in bounded_cases:
        with test.subTest(vertices=(a, b, c)):
            wp.launch(
                _plane_features,
                dim=1,
                inputs=[wp.vec3(*a), wp.vec3(*b), wp.vec3(*c), wp.vec3(0.2, 0.2, 0.0), 0.1],
                outputs=[points, distances, normals],
                device=device,
            )
            test.assertAlmostEqual(float(distances.numpy()[0]), edge_phi, delta=1.0e-6)
            test.assertAlmostEqual(float(distances.numpy()[1]), face_phi, delta=1.0e-6)
            np.testing.assert_allclose(normals.numpy()[1], face_normal, atol=1.0e-6)


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
    """Detect the finite footprint inside a face whose vertices are all outside, within the slab only."""
    for depth, expected_count in ((0.02, 1), (10.0, 0)):
        builder = newton.ModelBuilder()
        builder.add_shape_plane(width=0.2, length=0.2)
        for point in ((-1.0, -1.0, -depth), (3.0, -1.0, -depth), (-1.0, 1.0, -depth)):
            builder.add_particle(wp.vec3(*point), wp.vec3(), 0.1, radius=0.0)
        builder.add_triangle(0, 1, 2)
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.05)
        state = model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        count = int(contacts.soft_contact_count.numpy()[0])
        with test.subTest(depth=depth):
            test.assertEqual(count, expected_count)
            if expected_count == 0:
                continue
            bary = contacts.soft_contact_barycentric.numpy()[0]
            point = bary @ state.particle_q.numpy()
            test.assertLessEqual(abs(float(point[0])), 0.10001)
            test.assertLessEqual(abs(float(point[1])), 0.10001)
            np.testing.assert_allclose(
                contacts.soft_contact_body_pos.numpy()[0], (point[0], point[1], 0.0), atol=1.0e-5
            )
            np.testing.assert_allclose(contacts.soft_contact_normal.numpy()[0], (0.0, 0.0, 1.0), atol=1.0e-6)


def test_cloth_under_finite_plane_shelf(test, device):
    """A finite plane is a sheet: cloth lying well below a shelf must not be pulled up through it."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    shelf = builder.add_shape_plane(
        body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()), width=0.5, length=0.5
    )
    builder.add_cloth_grid(
        pos=wp.vec3(-0.4, -0.4, 0.01),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=8,
        dim_y=8,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
        particle_radius=0.005,
    )
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01, enable_rigid_soft_full_surface_contact=True)
    solver = newton.solvers.SolverVBD(model, iterations=10, rigid_compliant_alm=True)
    state_0, state_1, control, contacts = model.state(), model.state(), model.control(), pipeline.contacts()
    pipeline.collide(state_0, contacts)
    count = int(contacts.soft_contact_count.numpy()[0])
    test.assertEqual(int((contacts.soft_contact_shape.numpy()[:count] == shelf).sum()), 0)
    for _ in range(30):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    test.assertLess(float(state_0.particle_q.numpy()[:, 2].max()), 0.1)


def test_cloth_settles_on_heightfield_stairs(test, device):
    """Full-surface heightfield contacts must not push cloth sideways off steep step risers."""
    half_extent, samples = 1.5, 61
    x = np.linspace(-half_extent, half_extent, samples)
    heights = np.floor((x + half_extent) / 0.3) * 0.1
    builder = newton.ModelBuilder()
    builder.add_shape_heightfield(
        heightfield=newton.Heightfield(
            data=np.tile(heights[None, :], (samples, 1)).astype(np.float32),
            nrow=samples,
            ncol=samples,
            hx=half_extent,
            hy=half_extent,
        )
    )
    builder.add_cloth_grid(
        pos=wp.vec3(-1.0, -1.0, 1.2),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=16,
        dim_y=16,
        cell_x=2.0 / 16,
        cell_y=2.0 / 16,
        mass=0.05,
        particle_radius=0.01,
    )
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01, enable_rigid_soft_full_surface_contact=True)
    solver = newton.solvers.SolverVBD(model, iterations=10, rigid_compliant_alm=True)
    state_0, state_1, control, contacts = model.state(), model.state(), model.control(), pipeline.contacts()
    for _ in range(900):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 600.0)
        state_0, state_1 = state_1, state_0
    q = state_0.particle_q.numpy()
    # The per-particle path settles near mean x = -0.1; sliding off the stairs drives it past -0.5.
    test.assertGreater(float(q[:, 0].mean()), -0.4)
    test.assertTrue(np.all(np.abs(q[:, :2]) < half_extent))
    test.assertGreater(float(q[:, 2].min()), 0.09)


def test_convex_hull_edges_use_collision_numbering(test, device):
    """Convex hulls of unwelded meshes must build and match MESH ridge contacts for any vertex order."""
    box = newton.Mesh.create_box(0.2, 0.2, 0.2, compute_inertia=False)
    vertices, indices = np.asarray(box.vertices), np.asarray(box.indices).reshape(-1)
    ridge_up = wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.25 * np.pi))
    for seed in range(20):
        order = np.random.default_rng(seed).permutation(len(vertices))
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        edge_contacts = []
        for add_shape in ("add_shape_mesh", "add_shape_convex_hull"):
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            mesh = newton.Mesh(vertices[order], inverse[indices], compute_inertia=False)
            getattr(builder, add_shape)(body=-1, mesh=mesh, xform=ridge_up)
            # The ridge passes under a single 0.2 m cloth cell, between soft vertices.
            builder.add_cloth_grid(
                pos=wp.vec3(-0.3, -0.3, 0.2835),
                rot=wp.quat_identity(),
                vel=wp.vec3(),
                dim_x=3,
                dim_y=3,
                cell_x=0.2,
                cell_y=0.2,
                mass=0.1,
                particle_radius=0.0,
            )
            model = builder.finalize(device=device)
            pipeline = newton.CollisionPipeline(
                model, soft_contact_gap=0.02, enable_rigid_soft_full_surface_contact=True
            )
            contacts = pipeline.contacts()
            pipeline.collide(model.state(), contacts)
            count = int(contacts.soft_contact_count.numpy()[0])
            contact_indices = contacts.soft_contact_indices.numpy()[:count]
            edge_contacts.append(int(np.sum((contact_indices[:, 1] >= 0) & (contact_indices[:, 2] < 0))))
        with test.subTest(seed=seed):
            test.assertGreater(edge_contacts[0], 0)
            test.assertEqual(edge_contacts[1], edge_contacts[0])


def test_identical_meshes_share_contact_precomputation(test, device):
    """Distinct Mesh objects with identical geometry, e.g. replicated by importers, are processed once."""
    sphere = newton.Mesh.create_sphere(0.2, num_latitudes=8, num_longitudes=8, compute_inertia=False)
    builder = newton.ModelBuilder()
    for world in range(4):
        copy = newton.Mesh(np.array(sphere.vertices), np.array(sphere.indices), compute_inertia=False)
        builder.add_shape_mesh(body=-1, mesh=copy, xform=wp.transform(wp.vec3(float(world), 0.0, 0.0)))
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5, -0.5, 0.3),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=4,
        dim_y=4,
        cell_x=1.0,
        cell_y=0.25,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    feature_data = mock.Mock(wraps=soft_contacts_mesh._mesh_feature_data)
    components = mock.Mock(wraps=soft_contacts_mesh._max_faces_near_point)
    with (
        mock.patch.object(soft_contacts_mesh, "_mesh_feature_data", feature_data),
        mock.patch.object(soft_contacts_mesh, "_max_faces_near_point", components),
    ):
        newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
    test.assertEqual(feature_data.call_count, 1)
    reference_calls = components.call_count

    builder = newton.ModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=sphere)
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5, -0.5, 0.3),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=1,
        dim_y=1,
        cell_x=1.0,
        cell_y=1.0,
        mass=0.1,
    )
    components.reset_mock()
    with mock.patch.object(soft_contacts_mesh, "_max_faces_near_point", components):
        newton.CollisionPipeline(builder.finalize(device=device), enable_rigid_soft_full_surface_contact=True)
    test.assertEqual(reference_calls, components.call_count)


def _pinched_pad_contact_normals(device, depth, offset=(0.0, 0.0, 0.0), yaw=0.0):
    """Collide a cube-mesh pad with a cloth edge running ``depth`` behind its inner (+Y) face.

    The soft edge's endpoints lie outside the pad, so only full-surface edge contacts can act.
    Returns the normals of the filtered contacts in the pad frame.
    """
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    origin = wp.vec3(*offset)
    half = 0.03
    vertices = np.array(
        [[x, y, z] for z in (-half, half) for y in (-half, half) for x in (-half, half)], dtype=np.float32
    )
    faces = np.array(
        [
            [0, 2, 1],
            [1, 2, 3],
            [4, 5, 6],
            [5, 7, 6],
            [0, 1, 4],
            [1, 5, 4],
            [2, 6, 3],
            [3, 6, 7],
            [0, 4, 2],
            [2, 4, 6],
            [1, 3, 5],
            [3, 7, 5],
        ],
        dtype=np.int32,
    )
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    pad = builder.add_body(xform=wp.transform(origin, rotation))
    builder.add_shape_mesh(pad, mesh=newton.Mesh(vertices, faces.reshape(-1), compute_inertia=False))
    builder.add_cloth_grid(
        pos=origin + wp.quat_rotate(rotation, wp.vec3(-0.12, half - depth, 0.0)),
        rot=rotation,
        vel=wp.vec3(),
        dim_x=1,
        dim_y=1,
        cell_x=0.24,
        cell_y=0.06,
        mass=0.1,
        particle_radius=0.01,
    )
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_gap=0.01, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.collide(state, contacts)
    # Keep the pairs the solver applies forces to.
    soft_contacts_mesh.filter_soft_mesh_contacts(model, state, contacts)
    count = int(contacts.soft_contact_count.numpy()[0])
    normals = contacts.soft_contact_normal.numpy()[:count]
    inverse = wp.quat_inverse(rotation)
    return np.array([wp.quat_rotate(inverse, wp.vec3(*n)) for n in normals]).reshape(-1, 3)


def test_full_surface_skips_meshes_without_particle_collision(test, device):
    """Precompute mesh features only for meshes that collide with particles."""
    builder = newton.ModelBuilder()
    colliding = builder.add_shape_mesh(body=-1, mesh=newton.Mesh.create_box(0.5, 0.5, 0.1, compute_inertia=False))
    visual = builder.add_shape_mesh(
        body=-1,
        mesh=newton.Mesh.create_sphere(0.2, compute_inertia=False),
        cfg=newton.ModelBuilder.ShapeConfig(has_particle_collision=False),
    )
    builder.add_cloth_grid(
        pos=wp.vec3(-0.3, -0.3, 0.12),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=2,
        dim_y=2,
        cell_x=0.3,
        cell_y=0.3,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    feature_data = mock.Mock(wraps=soft_contacts_mesh._mesh_feature_data)
    with mock.patch.object(soft_contacts_mesh, "_mesh_feature_data", feature_data):
        pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
    test.assertEqual(feature_data.call_count, 1)
    # The visual mesh keeps per-particle pairs so enabling COLLIDE_PARTICLES later still works.
    particle_shapes = set(pipeline.soft_rigid_contact_pairs.numpy()[:, 1].tolist())
    test.assertIn(visual, particle_shapes)
    test.assertNotIn(colliding, particle_shapes)
    for pairs in (pipeline.soft_edge_rigid_pairs, pipeline.soft_face_rigid_pairs):
        test.assertEqual(len(pairs), 0)


def test_mesh_pad_pinch_pushes_toward_nearest_exit(test, device):
    """A soft edge pinched inside a mesh pad keeps contacts that push it out through the nearest face.

    Covers pairs inside the contact band (0.02 m here), penetrations beyond it that a closing
    gripper produces, and poses away from the origin where the crossing point's inside/outside
    sign is only roundoff.
    """
    cases = [(depth, (0.0, 0.0, 0.0), 0.0) for depth in (0.005, 0.015, 0.025)]
    rng = np.random.default_rng(3)
    cases += [(0.005, tuple(rng.uniform(-50.0, 50.0, 3)), float(rng.uniform(0.0, 2.0 * np.pi))) for _ in range(6)]
    for depth, offset, yaw in cases:
        with test.subTest(depth=depth, offset=offset, yaw=yaw):
            normals = _pinched_pad_contact_normals(device, depth, offset, yaw)
            test.assertGreater(len(normals), 0)
            # The inner face is the nearest exit; the top and bottom faces are farther than any depth.
            test.assertTrue(np.any(normals[:, 1] > 0.99), normals)
            test.assertTrue(np.all(np.abs(normals[:, 2]) < 1.0e-3), normals)


def test_dat_budget_counts_full_surface_mesh_queries(test, device):
    """DAT bounds rigid motion by the soft query radius even when meshes are the only soft colliders."""
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5, -0.5, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=4,
        dim_y=4,
        cell_x=0.25,
        cell_y=0.25,
        mass=0.05,
        particle_radius=5.0e-3,
    )
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.4), wp.quat_identity()))
    builder.add_shape_mesh(body, mesh=newton.Mesh.create_sphere(0.25, num_latitudes=8, num_longitudes=8))
    builder.color()
    model = builder.finalize(device=device)
    for full_surface in (False, True):
        with test.subTest(full_surface=full_surface):
            pipeline = newton.CollisionPipeline(
                model, soft_contact_gap=0.1, enable_rigid_soft_full_surface_contact=full_surface
            )
            solver = newton.solvers.SolverVBD(
                model,
                iterations=1,
                rigid_compliant_alm=True,
                rigid_soft_enable_dat=True,
                collision_pipeline=pipeline,
            )
            test.assertAlmostEqual(solver._rigid_soft_query_radius_min, 0.105, places=6)


def test_separated_mesh_shells_keep_default_capacity(test, device):
    """Shells that no particle can touch together must not multiply the default contact capacity."""
    box = newton.Mesh.create_box(0.05, 0.05, 0.05, compute_inertia=False)
    vertices, indices = np.asarray(box.vertices), np.asarray(box.indices).reshape(-1)

    def default_capacity(shell_count):
        offsets = [np.array([0.3 * (k % 4), 0.3 * (k // 4), 0.0]) for k in range(shell_count)]
        mesh = newton.Mesh(
            np.concatenate([vertices + offset for offset in offsets]),
            np.concatenate([indices + k * len(vertices) for k in range(shell_count)]),
            compute_inertia=False,
        )
        builder = newton.ModelBuilder()
        builder.add_shape_mesh(body=-1, mesh=mesh)
        builder.add_cloth_grid(
            pos=wp.vec3(-0.1, -0.1, 0.03),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=16,
            dim_y=16,
            cell_x=0.075,
            cell_y=0.075,
            mass=0.01,
            particle_radius=0.005,
        )
        model = builder.finalize(device=device)
        return newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True).soft_contact_max

    # No particle can be near faces of two shells, so each particle reserves at most one
    # shell's faces, however many shells the mesh has.
    particles = 17 * 17
    shell_faces = len(indices) // 3
    for shell_count in (1, 16):
        test.assertLessEqual(default_capacity(shell_count), particles * shell_faces)


def _cloth_over_mesh_contacts(device, gap=0.02):
    """Collide a perturbed cloth draped over a box mesh's top face and edges."""
    rng = np.random.default_rng(7)
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh.create_box(0.2, 0.2, 0.05, compute_inertia=False))
    builder.add_cloth_grid(
        pos=wp.vec3(-0.24, -0.24, 0.055),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=16,
        dim_y=16,
        cell_x=0.03,
        cell_y=0.03,
        mass=0.1,
        particle_radius=0.005,
    )
    model = builder.finalize(device=device)
    state = model.state()
    q = state.particle_q.numpy()
    state.particle_q.assign(q + rng.normal(scale=0.3 * gap, size=q.shape).astype(np.float32))
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_gap=gap, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return model, state, pipeline, contacts


def _soft_contact_records(contacts):
    count = int(contacts.soft_contact_count.numpy()[0])
    rows = np.concatenate(
        (
            contacts._soft_contact_mesh_features.numpy()[:count],
            contacts.soft_contact_shape.numpy()[:count, None],
        ),
        axis=1,
    )
    return {tuple(row) for row in rows.tolist()}, count


def test_mesh_detection_reports_every_feature_pair(test, device):
    """Detection keeps redundant feature pairs; the solver-side filter selects a subset of them."""
    model, state, _pipeline, contacts = _cloth_over_mesh_contacts(device)
    raw, raw_count = _soft_contact_records(contacts)
    soft_contacts_mesh.filter_soft_mesh_contacts(model, state, contacts)
    kept, kept_count = _soft_contact_records(contacts)
    test.assertEqual(len(raw), raw_count)
    test.assertGreater(kept_count, 0)
    test.assertLess(kept_count, raw_count)
    test.assertTrue(kept <= raw)
    # Filtered records carry geometry evaluated for their own features.
    indices = contacts.soft_contact_indices.numpy()[:kept_count]
    test.assertTrue(np.all(indices[:, 0] >= 0))
    normals = contacts.soft_contact_normal.numpy()[:kept_count]
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-5)


def test_mesh_contact_filter_runs_once_per_detection(test, device):
    """Contacts reused across substeps are not filtered again at other positions."""
    model, state, pipeline, contacts = _cloth_over_mesh_contacts(device)
    raw_count = int(contacts.soft_contact_count.numpy()[0])
    soft_contacts_mesh.filter_soft_mesh_contacts(model, state, contacts)
    kept, kept_count = _soft_contact_records(contacts)
    moved = model.state()
    moved.particle_q.assign(state.particle_q.numpy() + np.float32(0.01))
    soft_contacts_mesh.filter_soft_mesh_contacts(model, moved, contacts)
    test.assertEqual(_soft_contact_records(contacts), (kept, kept_count))
    # A new detection reports every pair again.
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), raw_count)


def test_mesh_evaluation_skips_particle_contacts(test, device):
    """Mesh evaluation must not overwrite per-particle contacts against meshes it does not handle."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh.create_box(0.1, 0.1, 0.1, compute_inertia=False))
    late = builder.add_shape_mesh(
        body=-1,
        mesh=newton.Mesh.create_box(0.1, 0.1, 0.1, compute_inertia=False),
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        cfg=newton.ModelBuilder.ShapeConfig(has_particle_collision=False),
    )
    builder.add_particle(wp.vec3(1.0, 0.0, 0.105), wp.vec3(0.0), mass=1.0, radius=0.0)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_gap=0.02, enable_rigid_soft_full_surface_contact=True
    )
    flags = model.shape_flags.numpy()
    flags[late] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
    model.shape_flags.assign(flags)
    contacts = pipeline.contacts()
    # Stale mesh records from an earlier detection may share slots with per-particle contacts.
    contacts._soft_contact_mesh_features.fill_(wp.vec3i(0, 0, 0))
    pipeline.collide(model.state(), contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 1)
    test.assertEqual(int(contacts.soft_contact_shape.numpy()[0]), late)
    test.assertEqual(int(contacts.soft_contact_particle.numpy()[0]), 0)
    np.testing.assert_allclose(contacts.soft_contact_normal.numpy()[0], (0.0, 0.0, 1.0), atol=1.0e-5)


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
        test_disconnected_mesh_contact_capacity,
        test_soft_contact_accumulation_thread_counts,
        test_soft_contact_workspace_storage,
        test_particle_gradient_after_pipeline_reuse,
        test_finite_plane_penetrating_face,
        test_cloth_under_finite_plane_shelf,
        test_cloth_settles_on_heightfield_stairs,
        test_convex_hull_edges_use_collision_numbering,
        test_identical_meshes_share_contact_precomputation,
        test_full_surface_skips_meshes_without_particle_collision,
        test_mesh_pad_pinch_pushes_toward_nearest_exit,
        test_dat_budget_counts_full_surface_mesh_queries,
        test_separated_mesh_shells_keep_default_capacity,
        test_finite_plane_feature_geometry,
        test_large_heightfield_task_contacts,
        test_mesh_detection_reports_every_feature_pair,
        test_mesh_contact_filter_runs_once_per_detection,
        test_mesh_evaluation_skips_particle_contacts,
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
