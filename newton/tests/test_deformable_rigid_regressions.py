# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise mixed geometry, signed finite planes, and collision replay."""

import sys
import unittest
import warnings
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
from newton.solvers.experimental.coupled import SolverCoupled
from newton.tests.unittest_utils import (
    StdOutCapture,
    add_function_test,
    configure_sdf_for_collision_shapes,
    get_test_devices,
)
from newton.tests.utils.soft_contacts import evaluate_soft_contact_force_mask


class TestDeformableRigidRegressions(unittest.TestCase):
    pass


def _contact_force_observables(solver, contacts):
    """Allocate force output on both the branch and CI's merge with main."""
    # Main renamed observable kinds and requires the contact buffer at allocation.
    if hasattr(solver, "ObservableKind"):
        return solver.observables({solver.ObservableKind.CONTACT_F}, contacts=contacts)
    return solver.observables({newton.solvers.SolverObservableFlags.CONTACT_F})


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
    """Allow explicit capacity for dense contacts across disconnected patches."""
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
        model,
        broad_phase="nxn",
        soft_contact_gap=0.05,
        enable_rigid_soft_full_surface_contact=True,
        # Reserve every face plus a recovery row for this particle's dense patch.
        soft_contact_max=len(indices) // 3 + 1,
    )
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.collide(state, contacts)

    # Detection reports every face within the band, which must fit the requested capacity.
    raw_count = int(contacts.soft_contact_count.numpy()[0])
    test.assertGreater(raw_count, len(centers))
    test.assertLessEqual(raw_count, contacts.soft_contact_max)
    # The feature filter keeps one contact per patch.
    mask = evaluate_soft_contact_force_mask(model, state, contacts)
    test.assertEqual(len(_force_contact_rows(contacts, mask)), len(centers))
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), raw_count)


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
        body_qd = state_out.body_qd.numpy()
        results.append(
            (
                state_out.body_q.numpy(),
                body_qd[:, :3],
                state_out.particle_q.numpy(),
                body_qd[:, 3:],
            )
        )
    for serial, parallel in zip(results[0][:3], results[1][:3], strict=True):
        np.testing.assert_allclose(parallel, serial, rtol=1.0e-5, atol=1.0e-6)
    # Different FP32 accumulation orders perturb the nearly cancelling torques.
    # Reconstructing angular velocity over 1 ms amplifies this pose roundoff;
    # 5e-6 rad/s corresponds to only 5e-9 radians over the step.
    np.testing.assert_allclose(results[1][3], results[0][3], rtol=1.0e-5, atol=5.0e-6)


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


def test_mesh_degenerate_faces_do_not_own_contacts(test, device):
    """Keep valid edge and vertex contact forces when a degenerate triangle precedes their owner."""
    for invalid in ([], [0, 1, 1], [0, 1, 3]):
        with test.subTest(invalid=invalid):
            builder = newton.ModelBuilder(gravity=wp.vec3())
            mesh = newton.Mesh(
                [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0.5, 0, 0)],
                [*invalid, 0, 1, 2],
                compute_inertia=False,
            )
            builder.add_shape_mesh(body=-1, mesh=mesh, cfg=builder.ShapeConfig(margin=0.0, gap=0.0))
            for p in ((0.5, -0.005, 0.005), (-0.005, -0.005, 0.005)):
                builder.add_particle(wp.vec3(*p), wp.vec3(), 1.0, radius=0.01)
            model = builder.finalize(device=device)
            pipeline = newton.CollisionPipeline(
                model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.01
            )
            state, contacts = model.state(), pipeline.contacts()
            pipeline.collide(state, contacts)
            count = int(contacts.soft_contact_count.numpy()[0])
            mask = evaluate_soft_contact_force_mask(model, state, contacts).numpy()[:count]
            particles = contacts.soft_contact_particle.numpy()[:count][mask]
            np.testing.assert_array_equal(np.sort(particles), (0, 1))


def test_mesh_face_contact_uses_maximum_radius(test, device):
    """Detect the TV contacts whose maximum corner radius activates the VBD force law."""
    builder = newton.ModelBuilder(gravity=wp.vec3())
    builder.add_shape_mesh(
        body=-1,
        mesh=newton.Mesh.create_box(0.01, 0.01, 0.01),
        cfg=builder.ShapeConfig(margin=0.0, gap=0.0),
    )
    for p, radius in zip(((0, 0, 0.06), (1, 0, 0.06), (0, 1, 0.06)), (0.0, 0.2, 0.0), strict=True):
        builder.add_particle(wp.vec3(*p), wp.vec3(), 1.0, radius=radius)
    builder.add_triangle(0, 1, 2)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.01)
    state, contacts = model.state(), pipeline.contacts()
    pipeline.collide(state, contacts)
    count = int(contacts.soft_contact_count.numpy()[0])
    features = contacts._soft_contact_mesh_features.numpy()[:count]
    mask = evaluate_soft_contact_force_mask(model, state, contacts).numpy()[:count]
    tv = np.flatnonzero(((features[:, 0] & 7) == 1) & mask)
    test.assertGreater(len(tv), 0)
    bary = contacts.soft_contact_barycentric.numpy()[tv]
    ids = contacts.soft_contact_indices.numpy()[tv]
    x = np.sum(state.particle_q.numpy()[ids] * bary[:, :, None], axis=1)
    delta = x - contacts.soft_contact_body_pos.numpy()[tv]
    separation = np.sum(delta * contacts.soft_contact_normal.numpy()[tv], axis=1)
    test.assertTrue(np.all(separation < 0.2))


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
    for _ in range(600):
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
    with (
        mock.patch.object(soft_contacts_mesh, "_mesh_feature_data", feature_data),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error", newton.exceptions.NewtonDeprecationWarning)
        pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
    test.assertEqual(feature_data.call_count, 1)
    # Identical instances share the actual adjacency spans, not just feature construction.
    offsets = pipeline._soft_mesh_contact_data.adjacency[0].numpy()
    np.testing.assert_array_equal(offsets, np.zeros_like(offsets))


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
    mask = evaluate_soft_contact_force_mask(model, state, contacts)
    rows = _force_contact_rows(contacts, mask)
    features_np = contacts._soft_contact_mesh_features.numpy()[rows, 0]
    recovery_np = (features_np & int(soft_contacts_mesh._MESH_FEATURE_RECOVERY)) != 0
    # Eq. (14) also permits wireframe EE forces; only recovery rows promise the nearest exit.
    normals = contacts.soft_contact_normal.numpy()[rows[recovery_np]]
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


def test_mesh_pad_pinch_recovery_uses_nearest_exit(test, device):
    """A soft edge pinched inside a mesh pad keeps recovery contacts toward the nearest face.

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
                model,
                soft_contact_gap=0.1,
                enable_rigid_soft_full_surface_contact=full_surface,
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
    """Budget added shells by rigid features rather than all particle-face combinations."""
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

    # Each triangulated box adds eight vertex and eighteen edge query sources. It must not
    # multiply the existing cloth workload by the number of shells.
    test.assertEqual(default_capacity(16) - default_capacity(1), 8 * (8 + 18) * 15)


def test_mesh_contact_capacity(test, device):
    """Bound high-poly storage and honor explicit contact capacity."""
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh.create_sphere(0.18, num_latitudes=16, num_longitudes=32))
    builder.add_cloth_grid(
        pos=wp.vec3(-1.0, -1.0, 0.05),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=16,
        dim_y=16,
        cell_x=0.125,
        cell_y=0.125,
        mass=0.1,
        particle_radius=0.0,
    )
    model = builder.finalize(device=device)

    def pipeline(**kwargs):
        return newton.CollisionPipeline(
            model, soft_contact_gap=0.05, enable_rigid_soft_full_surface_contact=True, **kwargs
        )

    default = pipeline().soft_contact_max
    # Keep a generous memory ceiling independent of the exact triangulation.
    # The former nearby-face multiplier exceeds this for this fixture.
    test.assertLess(default, 30000)
    test.assertGreater(default, 0)
    for capacity in (0, 123, default * 2):
        test.assertEqual(pipeline(soft_contact_max=capacity).soft_contact_max, capacity)


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


def _force_contact_rows(contacts, mask=None):
    count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
    if mask is None:
        return np.arange(count)
    return np.flatnonzero(mask.numpy()[:count])


def _soft_contact_records(contacts, *, mask=None):
    indices = _force_contact_rows(contacts, mask)
    rows = np.concatenate(
        (
            contacts._soft_contact_mesh_features.numpy()[indices],
            contacts.soft_contact_shape.numpy()[indices, None],
        ),
        axis=1,
    )
    return {tuple(row) for row in rows.tolist()}, len(indices)


def test_mesh_detection_reports_every_feature_pair(test, device):
    """Detection keeps redundant feature pairs; the solver-side filter selects a subset of them."""
    model, state, _pipeline, contacts = _cloth_over_mesh_contacts(device)
    raw, raw_count = _soft_contact_records(contacts)
    mask = evaluate_soft_contact_force_mask(model, state, contacts)
    kept, kept_count = _soft_contact_records(contacts, mask=mask)
    test.assertEqual(len(raw), raw_count)
    test.assertGreater(kept_count, 0)
    test.assertLess(kept_count, raw_count)
    test.assertTrue(kept <= raw)
    # Filtered records carry geometry evaluated for their own features.
    indices = contacts.soft_contact_indices.numpy()[_force_contact_rows(contacts, mask)]
    test.assertTrue(np.all(indices[:, 0] >= 0))
    normals = contacts.soft_contact_normal.numpy()[_force_contact_rows(contacts, mask)]
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-5)


def test_mesh_contact_filter_tracks_current_positions(test, device):
    """Reconsider rejected and accepted candidates after motion without re-detection."""
    model, state, pipeline, contacts = _cloth_over_mesh_contacts(device)
    raw_count = int(contacts.soft_contact_count.numpy()[0])
    mask = evaluate_soft_contact_force_mask(model, state, contacts)
    initial = mask.numpy().copy()
    moved = model.state()
    moved.particle_q.assign(state.particle_q.numpy() + np.float32(0.01))
    mask = evaluate_soft_contact_force_mask(model, moved, contacts)
    current = mask.numpy()
    test.assertTrue(np.any(current & ~initial))
    test.assertTrue(np.any(initial & ~current))
    mask = evaluate_soft_contact_force_mask(model, state, contacts)
    np.testing.assert_array_equal(mask.numpy(), initial)
    # A new detection reports every pair again.
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), raw_count)


def test_mesh_force_mask_preserves_detection(test, device):
    """Keep every detected row intact while refreshing force selection on graph replay."""
    model, state, pipeline, contacts = _cloth_over_mesh_contacts(device)
    fields = (
        "soft_contact_count",
        "soft_contact_particle",
        "soft_contact_indices",
        "soft_contact_shape",
        "soft_contact_barycentric",
        "soft_contact_body_pos",
        "soft_contact_body_vel",
        "soft_contact_normal",
        "_soft_contact_mesh_features",
        "_soft_contact_mesh_params",
    )
    count = int(contacts.soft_contact_count.numpy()[0])
    original = {name: getattr(contacts, name).numpy().copy() for name in fields}
    mask = evaluate_soft_contact_force_mask(model, state, contacts)
    kept = _force_contact_rows(contacts, mask)
    test.assertGreater(len(kept), 0)
    test.assertLess(len(kept), count)
    for name in fields:
        np.testing.assert_array_equal(getattr(contacts, name).numpy(), original[name])

    # Replay the filter against changed positions, keeping the same detection rows.
    moved = model.state()
    moved.particle_q.assign(state.particle_q.numpy() + np.float32(0.01))
    expected = _soft_contact_records(contacts, mask=mask)
    mask = evaluate_soft_contact_force_mask(model, moved, contacts)
    moved_expected = _soft_contact_records(contacts, mask=mask)
    test.assertNotEqual(moved_expected, expected)

    def filter_only():
        evaluate_soft_contact_force_mask(model, moved, contacts, out=mask)

    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            filter_only()
    for shift, records in ((0.0, expected), (0.01, moved_expected), (0.0, expected)):
        moved.particle_q.assign(state.particle_q.numpy() + np.float32(shift))
        if device.is_cuda:
            wp.capture_launch(capture.graph)
        else:
            filter_only()
        test.assertEqual(_soft_contact_records(contacts, mask=mask), records)
        for name in fields:
            np.testing.assert_array_equal(getattr(contacts, name).numpy(), original[name])

    mask = wp.empty(contacts.soft_contact_max, dtype=bool, device=device)

    def detect_and_filter():
        pipeline.collide(state, contacts)
        evaluate_soft_contact_force_mask(model, state, contacts, out=mask)

    detect_and_filter()
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            detect_and_filter()
    initial_q = state.particle_q.numpy().copy()
    for dz in (1.0, 0.0):
        state.particle_q.assign(initial_q + np.array((0.0, 0.0, dz), dtype=np.float32))
        if device.is_cuda:
            wp.capture_launch(capture.graph)
        else:
            detect_and_filter()
        if dz:
            test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0)
            test.assertFalse(np.any(mask.numpy()))
        else:
            test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), count)
            test.assertEqual(_soft_contact_records(contacts, mask=mask), expected)


def test_mesh_force_mask_preserves_geometry_gradients(test, device):
    """Backpropagate through detected geometry even after force-mask computation."""
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.1, 0.1, 0.05, compute_inertia=False))
    builder.add_particle(wp.vec3(0.02, 0.01, 0.055), wp.vec3(0.0), 1.0, radius=0.01)
    model = builder.finalize(device=device, requires_grad=True)
    pipeline = newton.CollisionPipeline(
        model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.02, soft_contact_max=128
    )
    gradients = []
    for filtered in (False, True):
        state, contacts = model.state(), pipeline.contacts()
        with wp.Tape() as tape:
            pipeline.collide(state, contacts)
        if filtered:
            evaluate_soft_contact_force_mask(model, state, contacts)
        # All raw geometry remains differentiable, including rows rejected for OGC forces.
        seed = wp.ones(contacts.soft_contact_max, dtype=wp.vec3, device=device)
        tape.backward(grads={contacts.soft_contact_body_pos: seed})
        gradients.append(state.particle_q.grad.numpy().copy())
    test.assertGreater(np.linalg.norm(gradients[0]), 1.0)
    np.testing.assert_allclose(gradients[1], gradients[0], atol=1.0e-6)


def test_mesh_force_mask_keeps_non_mesh_contacts(test, device):
    """Leave analytic-shape rows enabled and rigid-rigid rows unchanged."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_ground_plane()
    builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.1, 0.1, 0.05, compute_inertia=False))
    body = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.05), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_particle(wp.vec3(0.2, 0.0, 0.005), wp.vec3(0.0), 1.0, radius=0.01)
    builder.add_particle(wp.vec3(0.103, 0.104, 0.055), wp.vec3(0.0), 1.0, radius=0.01)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.02)
    state = model.state()
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    begin = int(contacts._soft_contact_mesh_state.numpy()[0])
    test.assertGreater(begin, 0)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    fields = (
        "soft_contact_count",
        "soft_contact_indices",
        "soft_contact_shape",
        "soft_contact_barycentric",
        "soft_contact_normal",
        "rigid_contact_count",
        "rigid_contact_point0",
    )
    original = {name: getattr(contacts, name).numpy().copy() for name in fields}
    mask = evaluate_soft_contact_force_mask(model, state, contacts)
    test.assertTrue(np.all(mask.numpy()[:begin]))
    for name in fields:
        np.testing.assert_array_equal(getattr(contacts, name).numpy(), original[name])


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


def _box_patch_contacts(
    device,
    points,
    triangles,
    *,
    capacity=512,
    reverse_faces=False,
    corner_shift=0,
    inactive_particles=(),
    color=False,
    dynamic=False,
):
    """Detect a small soft patch against a box without solver-side filtering."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    mesh = newton.Mesh.create_box(0.1, 0.1, 0.05, compute_inertia=dynamic)
    if reverse_faces or corner_shift:
        faces = np.roll(np.asarray(mesh.indices).reshape(-1, 3), corner_shift, axis=1)
        if reverse_faces:
            faces = faces[::-1]
        mesh = newton.Mesh(mesh.vertices, faces.reshape(-1))
    body = builder.add_body() if dynamic else -1
    builder.add_shape_mesh(body, mesh=mesh, cfg=builder.ShapeConfig(margin=0.0, gap=0.0))
    if triangles:
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=points,
            indices=np.asarray(triangles).reshape(-1).tolist(),
            density=1.0,
            particle_radius=0.01,
        )
    else:
        for point in points:
            builder.add_particle(wp.vec3(*point), wp.vec3(0.0), mass=1.0, radius=0.01)
    if color:
        builder.color()
    model = builder.finalize(device=device)
    if inactive_particles:
        flags = model.particle_flags.numpy()
        flags[list(inactive_particles)] &= ~int(newton.ParticleFlags.ACTIVE)
        model.particle_flags.assign(flags)
    state = model.state()
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        enable_rigid_soft_full_surface_contact=True,
        soft_contact_gap=0.02,
        soft_contact_max=capacity,
    )
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return model, state, contacts


def test_mesh_endpoint_pairs_filter_only_in_solver(test, device):
    """Keep both directed vertex queries, but remove endpoint EE records in the solver."""
    # The rigid top corner is closest to soft vertex 0. VT and TV are separate
    # querying-vertex contributions, not duplicates of the same directed query.
    model, state, contacts = _box_patch_contacts(
        device,
        [(0.103, 0.104, 0.055), (0.15, 0.104, 0.055), (0.103, 0.15, 0.055)],
        [(0, 1, 2)],
    )
    rows = _force_contact_rows(contacts)
    families = contacts._soft_contact_mesh_features.numpy()[rows, 0] & 7
    weights = contacts.soft_contact_barycentric.numpy()[rows]
    test.assertTrue(np.any((families == 1) & (weights.max(axis=1) == 1.0)))
    test.assertTrue(np.any((families == 2) & (weights.max(axis=1) == 1.0)))
    mask = evaluate_soft_contact_force_mask(model, state, contacts)
    rows = _force_contact_rows(contacts, mask)
    count = len(rows)
    families = contacts._soft_contact_mesh_features.numpy()[rows, 0] & 7
    test.assertEqual(count, 2)
    np.testing.assert_array_equal(np.sort(families), (0, 1))
    np.testing.assert_allclose(contacts.soft_contact_body_pos.numpy()[rows], [(0.1, 0.1, 0.05)] * 2, atol=1.0e-6)
    np.testing.assert_allclose(contacts.soft_contact_barycentric.numpy()[rows], [(1.0, 0.0, 0.0)] * 2)
    normal = np.array((0.003, 0.004, 0.005)) / np.linalg.norm((0.003, 0.004, 0.005))
    np.testing.assert_allclose(contacts.soft_contact_normal.numpy()[rows], [normal] * 2, atol=1.0e-6)


def test_mesh_ee_uses_edge_offset_region(test, device):
    """Keep interior EE contacts regardless of adjacent-face slopes (OGC Eq. 14)."""
    for rigid_height, soft_height in ((0.5, 0.01), (0.0, -0.49)):
        with test.subTest(rigid_height=rigid_height, soft_height=soft_height):
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            builder.add_shape_mesh(
                -1,
                mesh=newton.Mesh(
                    [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, rigid_height)],
                    [0, 1, 2],
                    compute_inertia=False,
                ),
                cfg=builder.ShapeConfig(margin=0.0, gap=0.0),
            )
            builder.add_cloth_mesh(
                pos=wp.vec3(0.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0),
                vertices=[(0.0, -1.0, 0.01), (0.0, 1.0, 0.01), (1.0, 0.0, soft_height)],
                indices=[0, 1, 2],
                density=1.0,
                particle_radius=0.0,
            )
            model = builder.finalize(device=device)
            pipeline = newton.CollisionPipeline(
                model,
                enable_rigid_soft_full_surface_contact=True,
                soft_contact_gap=0.02,
                soft_contact_max=64,
            )
            state, contacts = model.state(), pipeline.contacts()
            pipeline.collide(state, contacts)
            count = int(contacts.soft_contact_count.numpy()[0])
            features_np = contacts._soft_contact_mesh_features.numpy()[:count]
            edge_indices_np = model.edge_indices.numpy()
            rigid_edge_indices_np = contacts._soft_contact_mesh_data.rigid_features[2].numpy()
            rows = [
                i
                for i, feature in enumerate(features_np)
                if (int(feature[0]) & 7) == 2
                and set(edge_indices_np[feature[1], 2:]) == {0, 1}
                and set(rigid_edge_indices_np[feature[2], 1:]) == {0, 1}
            ]
            test.assertEqual(len(rows), 1)
            row = rows[0]
            np.testing.assert_allclose(contacts.soft_contact_barycentric.numpy()[row], (0.5, 0.5, 0.0), atol=1.0e-6)
            test.assertTrue(evaluate_soft_contact_force_mask(model, state, contacts).numpy()[row])


def test_mesh_tv_filter_replay(test, device):
    """Replay detection and target-feature filtering after contacts disappear and return."""
    model, state, _contacts = _box_patch_contacts(
        device,
        [(0.103, 0.104, 0.055), (0.15, 0.104, 0.055), (0.103, 0.15, 0.055)],
        [(0, 1, 2)],
    )
    pipeline = newton.CollisionPipeline(
        model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.02, soft_contact_max=512
    )
    contacts = pipeline.contacts()

    mask = wp.empty(contacts.soft_contact_max, dtype=bool, device=device)

    def detect_and_filter():
        pipeline.collide(state, contacts)
        evaluate_soft_contact_force_mask(model, state, contacts, out=mask)

    detect_and_filter()
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            detect_and_filter()
    original = state.particle_q.numpy().copy()
    for displacement, expected_families in ((0.0, [0, 1]), (1.0, []), (0.0, [0, 1])):
        state.particle_q.assign(original + np.array((0.0, 0.0, displacement), dtype=np.float32))
        if device.is_cuda:
            wp.capture_launch(capture.graph)
        else:
            detect_and_filter()
        rows = _force_contact_rows(contacts, mask)

        families = contacts._soft_contact_mesh_features.numpy()[rows, 0] & 7
        np.testing.assert_array_equal(np.sort(families), expected_families)


def test_mesh_tv_shared_soft_vertex_has_one_owner(test, device):
    """Keep one TV row per soft target vertex, not per incident triangle or position."""
    points = [(0.103, 0.104, 0.055), (0.15, 0.104, 0.055), (0.15, 0.15, 0.055), (0.103, 0.15, 0.055)]
    for triangles in ([(0, 1, 2), (0, 2, 3)], [(0, 2, 3), (0, 1, 2)], [(2, 3, 0), (1, 2, 0)]):
        # Coincident but disconnected patches must each retain their own TV row.
        for copies in (1, 2):
            with test.subTest(triangles=triangles, copies=copies):
                faces = [tuple(v + 4 * copy for v in tri) for copy in range(copies) for tri in triangles]
                model, state, contacts = _box_patch_contacts(device, points * copies, faces)
                mask = evaluate_soft_contact_force_mask(model, state, contacts)
                rows = _force_contact_rows(contacts, mask)

                features = contacts._soft_contact_mesh_features.numpy()[rows]
                tv = rows[np.flatnonzero((features[:, 0] & 7) == 1)]
                test.assertEqual(len(tv), copies)
                indices = contacts.soft_contact_indices.numpy()[tv]
                weights = contacts.soft_contact_barycentric.numpy()[tv]
                targets = indices[np.arange(copies), np.argmax(weights, axis=1)]
                np.testing.assert_array_equal(np.sort(targets), np.arange(copies) * 4)


def test_mesh_tv_checks_soft_target_block(test, device):
    """Test TV against the target soft feature, including neighbors across triangle boundaries."""
    # The rigid corner projects to c inside this tilted triangle. Its source-side
    # rigid cone rejects c-r=(-.001,-.001,.004), but a target face interior needs
    # no feature-neighbor check (OGC Algorithm 1).
    c = np.array((0.099, 0.099, 0.054))
    u = np.array((0.002, -0.002, 0.0))
    v = np.array((0.002, 0.002, 0.001))
    model, state, contacts = _box_patch_contacts(device, [c + u, c - u + v, c - u - v], [(0, 1, 2)])
    mask = evaluate_soft_contact_force_mask(model, state, contacts)
    rows = _force_contact_rows(contacts, mask)

    features = contacts._soft_contact_mesh_features.numpy()[rows]
    tv = rows[np.flatnonzero((features[:, 0] & 7) == 1)]
    test.assertEqual(len(tv), 1)
    bary = contacts.soft_contact_barycentric.numpy()[tv[0]]
    test.assertTrue(np.all(bary > 0.0))
    np.testing.assert_allclose(bary @ state.particle_q.numpy(), c, atol=1.0e-6)

    # For the shared edge, the second triangle slopes toward the query. The
    # first triangle returns the edge, but its neighboring opposite vertex
    # makes that edge fail the target block test: dot((0,0,-.005),(0,-.01,-.01))>0.
    # For the shared vertex, the added neighbor similarly points toward the query.
    cases = (
        ([(0.095, 0.1, 0.055), (0.105, 0.1, 0.055), (0.1, 0.11, 0.055), (0.1, 0.09, 0.045)], [(0, 1, 2), (1, 0, 3)]),
        (
            [(0.103, 0.104, 0.055), (0.15, 0.104, 0.055), (0.103, 0.15, 0.055), (0.101, 0.102, 0.055)],
            [(0, 1, 2), (0, 3, 1)],
        ),
    )
    for points, triangles in cases:
        model, state, contacts = _box_patch_contacts(device, points, triangles)
        rows = _force_contact_rows(contacts)

        features = contacts._soft_contact_mesh_features.numpy()[rows]
        test.assertTrue(np.any(((features[:, 0] & 7) == 1) & (features[:, 1] == 0)))
        mask = evaluate_soft_contact_force_mask(model, state, contacts)
        rows = _force_contact_rows(contacts, mask)

        features = contacts._soft_contact_mesh_features.numpy()[rows]
        test.assertFalse(np.any(((features[:, 0] & 7) == 1) & (features[:, 1] == 0)))


def test_mesh_tv_shared_soft_edge_has_one_owner(test, device):
    """Apply one contact when a rigid corner projects onto two triangles' shared edge."""
    # Both triangles return the identical point (0.1, 0.1, 0.055), with equal
    # weights on particles 0 and 2. Counting both doubles the same contact force.
    points = [(0.095, 0.095, 0.055), (0.105, 0.095, 0.055), (0.105, 0.105, 0.055), (0.095, 0.105, 0.055)]
    for triangles in ([(0, 1, 2), (0, 2, 3)], [(0, 2, 3), (0, 1, 2)]):
        model, state, contacts = _box_patch_contacts(device, points, triangles)
        mask = evaluate_soft_contact_force_mask(model, state, contacts)
        rows = _force_contact_rows(contacts, mask)

        families = contacts._soft_contact_mesh_features.numpy()[rows, 0] & 7
        tv = rows[np.flatnonzero(families == 1)]
        test.assertEqual(len(tv), 1)
        indices = contacts.soft_contact_indices.numpy()[tv[0]]
        weights = contacts.soft_contact_barycentric.numpy()[tv[0]]
        force_weights = np.zeros(4)
        np.add.at(force_weights, indices, weights)
        np.testing.assert_allclose(force_weights, (0.5, 0.0, 0.5, 0.0), atol=1.0e-6)


def test_mesh_tv_owner_includes_fixed_triangle(test, device):
    """Give a shared TV feature one owner even when its first incident triangle is fixed."""
    for points in (
        # Closest target is shared vertex 0.
        [(0.103, 0.104, 0.055), (0.15, 0.104, 0.055), (0.15, 0.15, 0.055), (0.103, 0.15, 0.055)],
        # Closest target is shared edge 0--2.
        [(0.095, 0.095, 0.055), (0.105, 0.095, 0.055), (0.105, 0.105, 0.055), (0.095, 0.105, 0.055)],
    ):
        model, state, contacts = _box_patch_contacts(
            device, points, [(0, 1, 2), (0, 2, 3)], inactive_particles=(0, 1, 2)
        )
        mask = evaluate_soft_contact_force_mask(model, state, contacts)
        rows = _force_contact_rows(contacts, mask)

        features = contacts._soft_contact_mesh_features.numpy()[rows]
        tv = features[(features[:, 0] & 7) == 1]
        test.assertEqual(len(tv), 1)
        test.assertEqual(int(tv[0, 1]), 0)


def test_vbd_preserves_unfiltered_mesh_contacts(test, device):
    """Keep raw detection through solver refreshes and report forces in its row order."""
    Frequency = newton.solvers.SolverBase.CollisionFrequencyType
    Slot = newton.solvers.SolverBase.CollisionSlot
    for mode in (None, Frequency.PRE_INIT, Frequency.PRE_POST_INIT, Frequency.ITERATIONS):
        with test.subTest(mode=mode):
            model, state, external_contacts = _box_patch_contacts(
                device,
                [(0.103, 0.104, 0.055), (0.15, 0.104, 0.055), (0.103, 0.15, 0.055)],
                [(0, 1, 2)],
                color=True,
            )
            pipeline = None
            if mode is not None:
                pipeline = newton.CollisionPipeline(
                    model,
                    enable_rigid_soft_full_surface_contact=True,
                    soft_contact_gap=0.02,
                    soft_contact_max=512,
                    contact_matching="latest",
                )
            solver = newton.solvers.SolverVBD(
                model,
                iterations=2,
                collision_pipeline=pipeline,
                collision_frequency_type={Slot.RIGID: mode} if mode is not None else None,
            )
            contacts = solver.contacts if mode is not None else external_contacts
            before = _soft_contact_records(external_contacts)
            observations = _contact_force_observables(solver, contacts)
            state_out = model.state()

            solver.step(
                state,
                state_out,
                model.control(),
                None if mode is not None else contacts,
                1.0e-4,
                observables=observations,
            )
            if device.is_cuda:
                # Allocate the control outside capture; a real caller also reuses it.
                control = model.control()
                with wp.ScopedCapture(device=device) as capture:
                    solver.step(
                        state,
                        state_out,
                        control,
                        None if mode is not None else contacts,
                        1.0e-4,
                        observables=observations,
                    )
                wp.capture_launch(capture.graph)
            raw, raw_count = _soft_contact_records(contacts)
            test.assertGreater(raw_count, 2)
            if mode is None:
                test.assertEqual((raw, raw_count), before)
            mask = evaluate_soft_contact_force_mask(model, state_out, contacts)
            mapping = _force_contact_rows(contacts, mask)
            test.assertEqual(len(mapping), 2)
            forces = observations.contact_f.numpy()[contacts.rigid_contact_max :]
            test.assertTrue(np.all(np.linalg.norm(forces[mapping], axis=1) > 0.0))
            excluded = np.ones(contacts.soft_contact_max, dtype=bool)
            excluded[mapping] = False
            np.testing.assert_array_equal(forces[excluded], 0.0)


def test_mesh_force_mask_matches_compacted_solve(test, device):
    """Match explicit row removal for particle forces, rigid reactions, and force reporting."""
    for dynamic in (False, True):
        for gather in (False, True):
            with test.subTest(dynamic=dynamic, gather=gather):
                model, initial, contacts = _box_patch_contacts(
                    device,
                    [(0.103, 0.104, 0.055), (0.15, 0.104, 0.055), (0.103, 0.15, 0.055)],
                    [(0, 1, 2)],
                    color=True,
                    dynamic=dynamic,
                )
                mask = evaluate_soft_contact_force_mask(model, initial, contacts)
                rows = _force_contact_rows(contacts, mask)
                test.assertEqual(len(rows), 2)
                test.assertLess(len(rows), int(contacts.soft_contact_count.numpy()[0]))

                # Independent control: physically remove rejected rows. This buffer has no
                # mesh metadata or mask; every stored row is consumed normally.
                compact = newton.Contacts(contacts.rigid_contact_max, contacts.soft_contact_max, device=device)
                compact.soft_contact_count.assign(np.array([len(rows)], dtype=np.int32))
                for name in (
                    "soft_contact_particle",
                    "soft_contact_indices",
                    "soft_contact_barycentric",
                    "soft_contact_shape",
                    "soft_contact_body_pos",
                    "soft_contact_body_vel",
                    "soft_contact_normal",
                ):
                    target = getattr(compact, name)
                    values = target.numpy()
                    values[: len(rows)] = getattr(contacts, name).numpy()[rows]
                    target.assign(values)
                results = []
                for selected in (contacts, compact):
                    solver = newton.solvers.SolverVBD(model, iterations=3)
                    solver._particle_contact_use_gather = gather
                    state, state_out = model.state(), model.state()
                    observations = _contact_force_observables(solver, selected)
                    solver.step(state, state_out, model.control(), selected, 1.0e-4, observables=observations)
                    results.append(
                        (
                            state_out.particle_q.numpy(),
                            state_out.body_q.numpy() if dynamic else np.empty((0, 7)),
                            observations.contact_f.numpy()[selected.rigid_contact_max :],
                            solver.body_particle_contact_penalty_k.numpy(),
                        )
                    )
                np.testing.assert_allclose(results[0][0], results[1][0], atol=1.0e-7, rtol=1.0e-6)
                np.testing.assert_allclose(results[0][1], results[1][1], atol=1.0e-7, rtol=1.0e-6)
                np.testing.assert_allclose(results[0][2][rows], results[1][2][: len(rows)], atol=1.0e-6, rtol=1.0e-5)
                np.testing.assert_allclose(results[0][3][rows], results[1][3][: len(rows)], atol=1.0e-6)
                rejected = ~mask.numpy()
                np.testing.assert_array_equal(results[0][2][rejected], 0.0)
                test.assertGreater(np.linalg.norm(results[0][0] - initial.particle_q.numpy()), 0.0)
                if dynamic:
                    test.assertGreater(np.linalg.norm(results[0][1] - initial.body_q.numpy()), 0.0)


def test_coupled_vbd_mesh_force_mask_replay(test, device):
    """Copy raw geometry and mesh provenance through coupled VBD after re-detection."""
    model, state, _ = _box_patch_contacts(
        device,
        [(0.103, 0.104, 0.055), (0.15, 0.104, 0.055), (0.103, 0.15, 0.055)],
        [(0, 1, 2)],
        color=True,
    )
    pipeline = newton.CollisionPipeline(
        model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.02, soft_contact_max=512
    )
    contacts, output = pipeline.contacts(), model.state()
    coupled = SolverCoupled(
        model,
        entries=[
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda view: newton.solvers.SolverVBD(view, iterations=2),
                particles=list(range(model.particle_count)),
            )
        ],
    )
    control = model.control()
    initial = state.particle_q.numpy().copy()

    def step():
        pipeline.collide(state, contacts)
        coupled.step(state, output, control, contacts, 1.0e-4)

    step()
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            step()
    for shift in (0.0, 1.0, 0.0):
        state.particle_q.assign(initial + np.array((0.0, 0.0, shift), dtype=np.float32))
        if device.is_cuda:
            wp.capture_launch(capture.graph)
        else:
            step()
        count = int(contacts.soft_contact_count.numpy()[0])
        entry = coupled._entry_contact_buffers["vbd"]
        test.assertEqual(int(entry.soft_contact_count.numpy()[0]), count)
        mapping = coupled._entry_soft_contact_src_to_dst["vbd"].numpy()[:count]
        test.assertTrue(np.all(mapping >= 0))
        np.testing.assert_array_equal(
            entry.soft_contact_indices.numpy()[mapping], contacts.soft_contact_indices.numpy()[:count]
        )
        np.testing.assert_array_equal(
            entry._soft_contact_mesh_features.numpy()[mapping], contacts._soft_contact_mesh_features.numpy()[:count]
        )
        test.assertIs(entry._soft_contact_mesh_data, contacts._soft_contact_mesh_data)
        test.assertTrue(np.isfinite(output.particle_q.numpy()).all())
        if shift:
            test.assertEqual(count, 0)
        else:
            test.assertGreater(count, 2)
            mask = evaluate_soft_contact_force_mask(model, state, contacts)
            test.assertEqual(len(_force_contact_rows(contacts, mask)), 2)


def test_mesh_edge_recovery_shared_triangle_boundary(test, device):
    """Recover each box chord once, including crossings on a face triangulation diagonal."""
    # Edge 0--1 crosses the bottom and top faces. At x=y=0 both crossings
    # lie on shared triangle edges: neither a missed entry nor two owners is valid.
    # The one chord midpoint is z=0; an extra almost-zero-depth row at z=0.05
    # is not another chord. The off-center diagonal also needs an exact third
    # barycentric weight: reconstructing it as 1-u-v produces -1.49e-8 instead
    # of zero on one triangle and can give the crossing two owners.
    # Reordering faces and cycling corners exercises all shared-feature slots.
    cases = (
        ((0.0, 0.0, -0.08), (0.0, 0.0, 0.08), [0.5]),
        ((-0.099, -0.099, -0.08), (-0.099, -0.099, 0.08), [0.5]),
        ((-0.005, 0.0, -0.08), (-0.005, 0.0, 0.08), [0.5]),
        ((0.005, 0.0, -0.08), (0.005, 0.0, 0.08), [0.5]),
        ((-0.16, -0.16, -0.08), (0.16, 0.16, 0.08), [0.5]),
        # An edge along a face has no interior chord, even if the mesh sign
        # query labels its zero-distance midpoint as inside.
        ((0.1, 0.0, -0.08), (0.1, 0.0, 0.08), []),
        ((0.11, 0.0, -0.08), (0.11, 0.0, 0.08), []),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.08), []),
    )
    for p, q, expected in cases:
        for permutation in range(6):
            with test.subTest(p=p, q=q, permutation=permutation):
                model, state, contacts = _box_patch_contacts(
                    device,
                    [p, q, (0.3, 0.3, 0.08)],
                    [(0, 1, 2)],
                    reverse_faces=permutation >= 3,
                    corner_shift=permutation % 3,
                )
                mask = evaluate_soft_contact_force_mask(model, state, contacts)
                rows = _force_contact_rows(contacts, mask)

                families = contacts._soft_contact_mesh_features.numpy()[rows, 0] & 7
                indices = contacts.soft_contact_indices.numpy()[rows]
                target = (families == 3) & (np.min(indices[:, :2], axis=1) == 0) & (np.max(indices[:, :2], axis=1) == 1)
                test.assertEqual(int(target.sum()), len(expected))
                if expected:
                    weights = contacts.soft_contact_barycentric.numpy()[rows][target][0]
                    np.testing.assert_allclose(weights, (0.5, 0.5, 0.0), atol=1.0e-6)


def test_mesh_filter_preserves_overflow(test, device):
    """Leave an overflowing buffer intact instead of hiding its incomplete detection."""
    points = [(0.02, 0.02, 0.055), (-0.02, -0.02, 0.055)]
    capture = StdOutCapture()
    capture.begin()
    try:
        model, state, contacts = _box_patch_contacts(
            device,
            points,
            [],
            capacity=1,
        )
        wp.synchronize()
    finally:
        output = capture.end()
    required = int(contacts._soft_contact_mesh_data.contact_count.numpy()[0])
    test.assertGreater(required, contacts.soft_contact_max + 1)
    # Warp's native printf capture is unreliable on Windows (see its print tests).
    if sys.platform != "win32":
        test.assertEqual(output.count("Mesh soft-contact capacity exceeded"), 1)
        test.assertIn(f"soft_contact_max=1, required={required}", output)
        test.assertIn(f"Increase soft_contact_max to at least {required}", output)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max + 1)
    features = contacts._soft_contact_mesh_features.numpy().copy()
    positions = contacts.soft_contact_body_pos.numpy().copy()
    for _ in range(2):
        mask = evaluate_soft_contact_force_mask(model, state, contacts)
        test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
        np.testing.assert_array_equal(contacts._soft_contact_mesh_features.numpy(), features)
        np.testing.assert_array_equal(contacts.soft_contact_body_pos.numpy(), positions)
    test.assertTrue(np.all(mask.numpy()))

    # The printed requirement must fit a complete rerun of the same collision pass.
    capture.begin()
    try:
        _, _, complete = _box_patch_contacts(device, points, [], capacity=required)
        test.assertEqual(int(complete.soft_contact_count.numpy()[0]), required)
    finally:
        output = capture.end()
    if sys.platform != "win32":
        test.assertNotIn("Mesh soft-contact capacity exceeded", output)


def _add_bvh_dat_patch(builder, points):
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=points,
        indices=[0, 1, 2],
        density=1.0,
        particle_radius=0.0,
    )


def test_mesh_complete_rigid_indices_and_reuse(test, device):
    """Keep complete zero-weight corners; clear stale mesh IDs before analytic reuse."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    mesh = newton.Mesh.create_box(0.1, 0.1, 0.05)
    mesh_shape = builder.add_shape_mesh(-1, mesh=mesh)
    plane = builder.add_shape_plane()
    _add_bvh_dat_patch(builder, [(-0.095, -0.095, 0.055), (0.15, -0.15, 0.055), (0.15, 0.15, 0.055)])
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        enable_rigid_soft_full_surface_contact=True,
        soft_contact_gap=0.02,
        soft_contact_max=512,
    )
    contacts, state = pipeline.contacts(), model.state()
    pipeline.collide(state, contacts)
    count = int(contacts.soft_contact_count.numpy()[0])
    soft = contacts.soft_contact_indices.numpy()[:count]
    rigid = contacts.soft_contact_rigid_indices.numpy()[:count]
    shapes = contacts.soft_contact_shape.numpy()[:count]
    families = {(int(np.sum(s >= 0)), int(np.sum(r >= 0))) for s, r in zip(soft, rigid, strict=True)}
    test.assertTrue({(1, 3), (2, 2), (3, 1)} <= families)
    for corners in rigid[shapes == mesh_shape]:
        slots = corners[corners >= 0]
        test.assertGreater(len(slots), 0)
        test.assertTrue(np.all(slots < len(mesh.indices)))
        test.assertEqual(len(set(np.asarray(mesh.indices)[slots])), len(slots))
    # One triangle's closest point can lie on its edge/vertex; it still needs all
    # three rigid corners for curved-motion truncation.
    vt = rigid[np.sum(soft >= 0, axis=1) == 1]
    test.assertTrue(np.all(vt[:, 1:] == vt[:, :1] + np.array((1, 2))))
    flags = model.shape_flags.numpy()
    flags[mesh_shape] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
    model.shape_flags.assign(flags)
    state.particle_q.assign(state.particle_q.numpy() - np.array((0.0, 0.0, 0.05), dtype=np.float32))
    pipeline.collide(state, contacts)
    count = int(contacts.soft_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    test.assertTrue(np.all(contacts.soft_contact_shape.numpy()[:count] == plane))
    test.assertTrue(np.all(contacts.soft_contact_rigid_indices.numpy() == -1))
    contacts.soft_contact_rigid_indices.fill_(wp.vec3i(1, 2, 3))
    contacts.clear_buffers = True
    contacts.clear()
    test.assertTrue(np.all(contacts.soft_contact_rigid_indices.numpy() == -1))

    # Deep interior recovery rows pair with a nearest surface point, not a complete primitive.
    flags[mesh_shape] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
    model.shape_flags.assign(flags)
    state.particle_q.assign(np.array([(-0.01, -0.01, 0.0), (0.01, -0.01, 0.0), (0.0, 0.01, 0.0)], dtype=np.float32))
    pipeline.collide(state, contacts)
    count = int(contacts.soft_contact_count.numpy()[0])
    features_np = contacts._soft_contact_mesh_features.numpy()[:count, 0]
    recovery_np = (features_np >= 0) & ((features_np & int(soft_contacts_mesh._MESH_FEATURE_RECOVERY)) != 0)
    test.assertTrue(np.any(recovery_np))
    test.assertTrue(np.all(contacts.soft_contact_rigid_indices.numpy()[:count][recovery_np] == -1))


def test_mesh_complete_bvh_query_coverage(test, device):
    """Keep all edges, fixed geometry, short edges and slivers without crossing worlds."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    meshes = []
    for _ in range(2):
        builder.begin_world()
        mesh = newton.Mesh.create_box(0.1, 0.1, 0.05)
        mesh._collision_edges = np.empty((0, 2), dtype=np.int32)
        meshes.append(mesh)
        builder.add_shape_mesh(-1, mesh=mesh)
        _add_bvh_dat_patch(builder, [(-0.095, -0.095, 0.055), (0.15, -0.15, 0.055), (0.15, 0.15, 0.055)])
        builder.end_world()
    model = builder.finalize(device=device)
    model.particle_flags.zero_()
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        enable_rigid_soft_full_surface_contact=True,
        soft_contact_gap=0.02,
        soft_contact_max=512,
    )
    test.assertEqual(len(pipeline._soft_mesh_contact_data.rigid_features[2]), 36)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    count = int(contacts.soft_contact_count.numpy()[0])
    soft = contacts.soft_contact_indices.numpy()[:count]
    rigid = contacts.soft_contact_rigid_indices.numpy()[:count]
    shapes = contacts.soft_contact_shape.numpy()[:count]
    worlds = model.particle_world.numpy()[soft[:, 0]]
    np.testing.assert_array_equal(worlds, model.shape_world.numpy()[shapes])
    for world in (0, 1):
        families = {
            (int(np.sum(s >= 0)), int(np.sum(r >= 0)))
            for s, r in zip(soft[worlds == world], rigid[worlds == world], strict=True)
        }
        test.assertTrue({(1, 3), (2, 2), (3, 1)} <= families)

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    mesh = newton.Mesh([(-0.0005, 0.0, 0.0), (0.0005, 0.0, 0.0), (0.0, -0.0001, 0.0)], [0, 1, 2], compute_inertia=False)
    builder.add_shape_mesh(-1, mesh=mesh, cfg=builder.ShapeConfig(margin=0.0, gap=0.0))
    _add_bvh_dat_patch(builder, [(0.0, -0.0005, 0.0001), (0.0, 0.0005, 0.0001), (0.001, 0.0005, 0.0001)])
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        enable_rigid_soft_full_surface_contact=True,
        soft_contact_gap=0.0002,
        soft_contact_max=512,
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    count = int(contacts.soft_contact_count.numpy()[0])
    soft = contacts.soft_contact_indices.numpy()[:count]
    rigid = contacts.soft_contact_rigid_indices.numpy()[:count]
    matching = [
        i
        for i, (s, r) in enumerate(zip(soft, rigid, strict=True))
        if set(s[s >= 0]) == {0, 1} and set(r[r >= 0]) == {0, 1}
    ]
    test.assertTrue(matching, "the perpendicular 1 mm edges are only 0.1 mm apart")
    for i in matching:
        np.testing.assert_allclose(contacts.soft_contact_barycentric.numpy()[i], (0.5, 0.5, 0.0), atol=1.0e-6)
        np.testing.assert_allclose(contacts.soft_contact_body_pos.numpy()[i], (0.0, 0.0, 0.0), atol=1.0e-7)

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_shape_mesh(
        -1,
        mesh=newton.Mesh([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0e-7, 0.0)], [0, 1, 2], compute_inertia=False),
        cfg=builder.ShapeConfig(margin=0.0, gap=0.0),
    )
    builder.add_particle(pos=wp.vec3(0.25, 2.5e-8, 1.0e-5), vel=wp.vec3(0.0), mass=1.0, radius=0.0)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        enable_rigid_soft_full_surface_contact=True,
        soft_contact_gap=1.0e-4,
        soft_contact_max=512,
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 1)
    np.testing.assert_array_equal(contacts.soft_contact_rigid_indices.numpy()[0], (0, 1, 2))


for device in get_test_devices():
    for fn in (
        test_mesh_complete_rigid_indices_and_reuse,
        test_mesh_complete_bvh_query_coverage,
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
        test_mesh_pad_pinch_recovery_uses_nearest_exit,
        test_dat_budget_counts_full_surface_mesh_queries,
        test_separated_mesh_shells_keep_default_capacity,
        test_mesh_contact_capacity,
        test_finite_plane_feature_geometry,
        test_mesh_degenerate_faces_do_not_own_contacts,
        test_mesh_face_contact_uses_maximum_radius,
        test_large_heightfield_task_contacts,
        test_mesh_detection_reports_every_feature_pair,
        test_mesh_contact_filter_tracks_current_positions,
        test_mesh_force_mask_preserves_detection,
        test_mesh_force_mask_preserves_geometry_gradients,
        test_mesh_force_mask_keeps_non_mesh_contacts,
        test_mesh_evaluation_skips_particle_contacts,
        test_mesh_endpoint_pairs_filter_only_in_solver,
        test_mesh_ee_uses_edge_offset_region,
        test_mesh_tv_filter_replay,
        test_mesh_tv_shared_soft_vertex_has_one_owner,
        test_mesh_tv_checks_soft_target_block,
        test_mesh_tv_shared_soft_edge_has_one_owner,
        test_mesh_tv_owner_includes_fixed_triangle,
        test_vbd_preserves_unfiltered_mesh_contacts,
        test_mesh_force_mask_matches_compacted_solve,
        test_coupled_vbd_mesh_force_mask_replay,
        test_mesh_edge_recovery_shared_triangle_boundary,
        test_mesh_filter_preserves_overflow,
    ):
        add_function_test(
            TestDeformableRigidRegressions,
            fn.__name__,
            fn,
            devices=[device],
            check_output=fn is not test_mesh_filter_preserves_overflow,
        )
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
