# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the BVH full-surface rigid-soft contact backend."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.soft_contacts_bvh import (
    RIGID_SOFT_BVH_CONTACT_EE,
    RIGID_SOFT_BVH_CONTACT_FV,
    RIGID_SOFT_BVH_CONTACT_VF,
)
from newton._src.geometry.tri_mesh_collision import TriMeshCollisionDetector
from newton.tests.unittest_utils import (
    add_function_test,
    configure_sdf_for_collision_shapes,
    get_cuda_test_devices,
    get_test_devices,
)


@wp.kernel
def _sum_contact_geometry(
    count: wp.array[wp.int32],
    barycentric: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    loss: wp.array[float],
):
    """Build a scalar loss that exercises differentiable BVH emission geometry."""
    tid = wp.tid()
    if tid < count[0]:
        wp.atomic_add(loss, 0, barycentric[tid][0] + 0.1 * normal[tid][0])


def _tetra_mesh() -> newton.Mesh:
    """Return an outward-wound unit tetrahedron with six full edges."""
    return newton.Mesh(
        np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float32),
        np.asarray((0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3), dtype=np.int32),
        compute_inertia=False,
    )


def _add_two_triangle_cloth(builder: newton.ModelBuilder, *, z_offset: float = 0.0) -> None:
    """Add disconnected triangles exercising FV near the origin and EE across the bottom edge."""
    vertices = [
        wp.vec3(-0.12, -0.12, 0.03 + z_offset),
        wp.vec3(0.12, -0.12, 0.03 + z_offset),
        wp.vec3(0.0, 0.12, 0.03 + z_offset),
        wp.vec3(0.45, -0.10, 0.03 + z_offset),
        wp.vec3(0.45, 0.10, 0.03 + z_offset),
        wp.vec3(0.65, 0.0, 0.25 + z_offset),
    ]
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=vertices,
        indices=[0, 1, 2, 3, 4, 5],
        density=0.1,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
        particle_radius=0.0,
    )


def _add_single_cloth_triangle(builder: newton.ModelBuilder, vertices) -> None:
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(*point) for point in vertices],
        indices=[0, 1, 2],
        density=0.1,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
        particle_radius=0.0,
    )


def _build_scene(
    device,
    *,
    scale=(1.0, 1.0, 1.0),
    soft_contact_max=None,
    rigid_soft_bvh_candidate_max=None,
    z_offset=0.0,
    convex=False,
    dynamic_rigid=False,
):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    rigid_body = -1
    if dynamic_rigid:
        rigid_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
    add_shape = builder.add_shape_convex_hull if convex else builder.add_shape_mesh
    add_shape(body=rigid_body, mesh=_tetra_mesh(), scale=scale)
    _add_two_triangle_cloth(builder, z_offset=z_offset)
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=False,
        broad_phase="nxn",
        rigid_contact_max=64,
        max_triangle_pairs=64,
        soft_contact_max=soft_contact_max,
        rigid_soft_bvh_candidate_max=rigid_soft_bvh_candidate_max,
        soft_contact_gap=0.06,
        enable_rigid_soft_full_surface_contact=True,
    )
    return model, pipeline


def _closest_point_triangle(point, a, b, c):
    """NumPy Ericson closest point, returning point and barycentrics."""
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.array((1.0, 0.0, 0.0))
    bp = point - b
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, np.array((0.0, 1.0, 0.0))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.array((1.0 - v, v, 0.0))
    cp = point - c
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, np.array((0.0, 0.0, 1.0))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.array((1.0 - w, 0.0, w))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.array((0.0, 1.0 - w, w))
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    bary = np.array((1.0 - v - w, v, w))
    return bary[0] * a + bary[1] * b + bary[2] * c, bary


def _closest_segment_segment(p0, p1, q0, q1):
    """Return segment parameters and distance for a non-symbolic brute-force oracle."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    d, e = np.dot(u, w), np.dot(v, w)
    denom = a * c - b * b
    s = 0.0 if denom <= 1.0e-14 else np.clip((b * e - c * d) / denom, 0.0, 1.0)
    t = (b * s + e) / c if c > 1.0e-14 else 0.0
    if t < 0.0:
        t = 0.0
        s = np.clip(-d / a, 0.0, 1.0) if a > 1.0e-14 else 0.0
    elif t > 1.0:
        t = 1.0
        s = np.clip((b - d) / a, 0.0, 1.0) if a > 1.0e-14 else 0.0
    return s, t, np.linalg.norm((p0 + s * u) - (q0 + t * v))


def _brute_pair_sets(model, gap):
    soft = model.particle_q.numpy()
    radii = model.particle_radius.numpy()
    triangles = model.tri_indices.numpy()
    soft_edges = model.edge_indices.numpy()[:, 2:4]
    mesh = model.shape_source[0]
    scale = model.shape_scale.numpy()[0]
    rigid = np.asarray(mesh.vertices) * scale
    rigid_triangles = np.asarray(mesh.indices).reshape(-1, 3)
    rigid_edges = np.asarray(mesh.edges)
    margin = float(model.shape_margin.numpy()[0])

    vf = set()
    for particle, point in enumerate(soft):
        threshold = gap + margin + radii[particle]
        for face, tri in enumerate(rigid_triangles):
            closest, _ = _closest_point_triangle(point, *rigid[tri])
            if np.linalg.norm(point - closest) < threshold:
                vf.add((particle, face))

    fv = set()
    for vertex, point in enumerate(rigid):
        for tri_index, tri in enumerate(triangles):
            closest, _ = _closest_point_triangle(point, *soft[tri])
            threshold = gap + margin + max(radii[tri])
            if np.linalg.norm(point - closest) < threshold:
                fv.add((tri_index, vertex))

    edge_lookup = {tuple(sorted(map(int, edge))): edge_index for edge_index, edge in enumerate(rigid_edges)}
    ee = set()
    for rigid_edge_index, (rv0, rv1) in enumerate(rigid_edges):
        for soft_edge_index, (sv0, sv1) in enumerate(soft_edges):
            _, _, distance = _closest_segment_segment(rigid[rv0], rigid[rv1], soft[sv0], soft[sv1])
            threshold = gap + margin + max(radii[sv0], radii[sv1])
            if distance < threshold:
                ee.add((soft_edge_index, rigid_edge_index))
    return vf, fv, ee, edge_lookup


def _brute_pair_distances(model, pairs):
    """Return sorted geometric distances for the three independently enumerated pair sets."""
    soft = model.particle_q.numpy()
    triangles = model.tri_indices.numpy()
    soft_edges = model.edge_indices.numpy()[:, 2:4]
    mesh = model.shape_source[0]
    rigid = np.asarray(mesh.vertices) * model.shape_scale.numpy()[0]
    rigid_triangles = np.asarray(mesh.indices).reshape(-1, 3)
    rigid_edges = np.asarray(mesh.edges)
    vf, fv, ee = pairs
    vf_distances = [
        np.linalg.norm(soft[particle] - _closest_point_triangle(soft[particle], *rigid[rigid_triangles[face]])[0])
        for particle, face in vf
    ]
    fv_distances = [
        np.linalg.norm(rigid[vertex] - _closest_point_triangle(rigid[vertex], *soft[triangles[tri]])[0])
        for tri, vertex in fv
    ]
    ee_distances = [
        _closest_segment_segment(
            rigid[rigid_edges[rigid_edge, 0]],
            rigid[rigid_edges[rigid_edge, 1]],
            soft[soft_edges[soft_edge, 0]],
            soft[soft_edges[soft_edge, 1]],
        )[2]
        for soft_edge, rigid_edge in ee
    ]
    return tuple(np.sort(values) for values in (vf_distances, fv_distances, ee_distances))


def _actual_pair_sets(pipeline):
    count = min(int(pipeline.rigid_soft_bvh_candidate_count.numpy()[0]), pipeline.rigid_soft_bvh_candidate_max)
    candidates = pipeline._rigid_soft_bvh_candidates.numpy()[:count]
    vertex_table = pipeline.rigid_soft_bvh_rigid_vertex_table.numpy()
    edge_table = pipeline.rigid_soft_bvh_rigid_edge_table.numpy()
    rigid_edges = np.asarray(pipeline.model.shape_source[0].edges)
    edge_lookup = {tuple(sorted(map(int, edge))): edge_index for edge_index, edge in enumerate(rigid_edges)}
    vf, fv, ee = set(), set(), set()
    for family, soft_feature, _shape, rigid_feature in candidates:
        if family == RIGID_SOFT_BVH_CONTACT_VF:
            vf.add((int(soft_feature), int(rigid_feature)))
        elif family == RIGID_SOFT_BVH_CONTACT_FV:
            fv.add((int(soft_feature), int(vertex_table[rigid_feature, 1])))
        elif family == RIGID_SOFT_BVH_CONTACT_EE:
            local_edge = tuple(sorted(map(int, edge_table[rigid_feature, 1:3])))
            ee.add((int(soft_feature), edge_lookup[local_edge]))
    return vf, fv, ee


def _brute_multiworld_pair_sets(model, gap):
    """Brute-force all world-compatible VF/FV/EE pairs, retaining rigid shape identity."""
    soft = model.particle_q.numpy()
    radii = model.particle_radius.numpy()
    particle_world = model.particle_world.numpy()
    triangles = model.tri_indices.numpy()
    soft_edges = model.edge_indices.numpy()[:, 2:4]
    shape_world = model.shape_world.numpy()
    shape_margin = model.shape_margin.numpy()
    expected = (set(), set(), set())

    def compatible(soft_world, rigid_world):
        return soft_world < 0 or rigid_world < 0 or soft_world == rigid_world

    for shape, mesh in enumerate(model.shape_source):
        if mesh is None:
            continue
        rigid = np.asarray(mesh.vertices) * model.shape_scale.numpy()[shape]
        rigid_triangles = np.asarray(mesh.indices).reshape(-1, 3)
        rigid_edges = np.asarray(mesh.edges)
        for particle, point in enumerate(soft):
            if not compatible(particle_world[particle], shape_world[shape]):
                continue
            threshold = gap + shape_margin[shape] + radii[particle]
            for face, tri in enumerate(rigid_triangles):
                closest, _ = _closest_point_triangle(point, *rigid[tri])
                if np.linalg.norm(point - closest) < threshold:
                    expected[0].add((particle, shape, face))
        for tri_index, tri in enumerate(triangles):
            if not compatible(particle_world[tri[0]], shape_world[shape]):
                continue
            threshold = gap + shape_margin[shape] + max(radii[tri])
            for vertex, point in enumerate(rigid):
                closest, _ = _closest_point_triangle(point, *soft[tri])
                if np.linalg.norm(point - closest) < threshold:
                    expected[1].add((tri_index, shape, vertex))
        for soft_edge_index, (sv0, sv1) in enumerate(soft_edges):
            if not compatible(particle_world[sv0], shape_world[shape]):
                continue
            threshold = gap + shape_margin[shape] + max(radii[sv0], radii[sv1])
            for rigid_edge_index, (rv0, rv1) in enumerate(rigid_edges):
                distance = _closest_segment_segment(rigid[rv0], rigid[rv1], soft[sv0], soft[sv1])[2]
                if distance < threshold:
                    expected[2].add((soft_edge_index, shape, rigid_edge_index))
    return expected


def test_backend_selection_and_partition(test, device):
    """BVH is the validated default, partitions mesh rows, and keeps SDF explicit."""
    model, pipeline = _build_scene(device)
    test.assertEqual(pipeline.full_surface_mesh_backend, "bvh")
    test.assertEqual(pipeline.soft_contact_pair_count, 0, "BVH replaces particle-mesh contact pairs")
    test.assertEqual(len(pipeline.soft_edge_rigid_pairs), 0, "BVH replaces full-surface SDF contact pairs")
    test.assertEqual(len(pipeline.soft_face_rigid_pairs), 0, "BVH replaces full-surface SDF contact pairs")
    test.assertEqual(len(pipeline.rigid_soft_bvh_soft_particle_rigid_shape_pairs), model.particle_count)
    test.assertEqual(len(pipeline.rigid_soft_bvh_rigid_vertex_table), 4)
    test.assertEqual(len(pipeline.rigid_soft_bvh_rigid_edge_table), 6)
    with test.assertRaises(ValueError):
        newton.CollisionPipeline(model, full_surface_mesh_backend="invalid")
    with test.assertRaises(ValueError):
        newton.CollisionPipeline(
            model,
            enable_rigid_soft_full_surface_contact=True,
            full_surface_mesh_backend="sdf",
        )
    model.shape_source[0] = None
    with test.assertRaisesRegex(ValueError, "requires every mesh and convex-mesh shape.*shape_0.*index 0"):
        newton.CollisionPipeline(
            model,
            enable_rigid_soft_full_surface_contact=True,
            full_surface_mesh_backend="bvh",
        )
    empty_model = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0)).finalize(device=device)
    empty_pipeline = newton.CollisionPipeline(empty_model, enable_rigid_soft_full_surface_contact=True)
    test.assertEqual(empty_pipeline.rigid_soft_bvh_candidate_max, 0)
    test.assertFalse(empty_pipeline.rigid_soft_bvh_requires_soft_feature_bvhs)


def test_rigid_feature_tables_scaled_and_mirrored(test, device):
    """Keep per-instance vertex and edge normals outward under signed scaling."""
    model, pipeline = _build_scene(device, scale=(-1.0, 2.0, 0.5))
    vertices = pipeline.rigid_soft_bvh_rigid_vertex_table.numpy()
    edges = pipeline.rigid_soft_bvh_rigid_edge_table.numpy()
    vertex_normals = pipeline.rigid_soft_bvh_rigid_vertex_normal.numpy()
    edge_normals = pipeline.rigid_soft_bvh_rigid_edge_outward.numpy()
    test.assertEqual(vertices.shape, (4, 2))
    test.assertEqual(edges.shape, (6, 3))
    test.assertTrue(np.array_equal(np.sort(vertices[:, 1]), np.arange(4)))
    test.assertEqual({tuple(sorted(edge[1:])) for edge in edges}, {tuple(sorted(e)) for e in _tetra_mesh().edges})
    test.assertTrue(np.all(np.isfinite(vertex_normals)))
    test.assertTrue(np.all(np.isfinite(edge_normals)))
    test.assertTrue(np.allclose(np.linalg.norm(vertex_normals, axis=1), 1.0, atol=1.0e-5))
    test.assertTrue(np.allclose(np.linalg.norm(edge_normals, axis=1), 1.0, atol=1.0e-5))
    scaled_vertices = np.asarray(model.shape_source[0].vertices) * model.shape_scale.numpy()[0]
    center = scaled_vertices.mean(axis=0)
    for row, normal in zip(vertices, vertex_normals, strict=True):
        test.assertGreater(float(np.dot(normal, scaled_vertices[row[1]] - center)), 0.0)
    for row, normal in zip(edges, edge_normals, strict=True):
        midpoint = 0.5 * (scaled_vertices[row[1]] + scaled_vertices[row[2]])
        test.assertGreater(float(np.dot(normal, midpoint - center)), 0.0)


def test_bvh_emits_rigid_mesh_surface_velocity(test, device):
    """Interpolate scaled rigid-mesh vertex velocities for VF, FV, and EE rows."""
    scale = np.asarray((1.5, 0.75, 2.0), dtype=np.float32)
    model, pipeline = _build_scene(device, scale=scale)
    mesh_velocity = np.asarray((0.2, -0.4, 0.6), dtype=np.float32)
    model.shape_source[0].mesh.velocities.assign(np.tile(mesh_velocity, (4, 1)))

    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    pair_sets = _actual_pair_sets(pipeline)
    test.assertTrue(all(pair_sets), "scene must exercise VF, FV, and EE velocity interpolation")
    count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
    expected = mesh_velocity * scale
    test.assertGreater(count, 0)
    test.assertTrue(np.allclose(contacts.soft_contact_body_vel.numpy()[:count], expected, atol=1.0e-6))


def test_mutable_shape_flag_is_not_frozen_at_construction(test, device):
    """A mesh disabled at construction can participate after COLLIDE_PARTICLES is enabled."""
    model, _unused_pipeline = _build_scene(device)
    flags = model.shape_flags.numpy()
    flags[0] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
    model.shape_flags.assign(flags)
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=False,
        broad_phase="nxn",
        rigid_contact_max=64,
        max_triangle_pairs=64,
        soft_contact_gap=0.06,
        enable_rigid_soft_full_surface_contact=True,
    )
    test.assertGreater(pipeline.rigid_soft_bvh_candidate_max, 0)
    pipeline.collide(model.state(), pipeline.contacts())
    test.assertEqual(int(pipeline.rigid_soft_bvh_candidate_count.numpy()[0]), 0)
    flags[0] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
    model.shape_flags.assign(flags)
    pipeline.collide(model.state(), pipeline.contacts())
    test.assertGreater(int(pipeline.rigid_soft_bvh_candidate_count.numpy()[0]), 0)


def test_convex_mesh_uses_bvh_backend(test, device):
    """Convex-mesh shapes use the same no-SDF VF/FV/EE backend as triangle meshes."""
    model, pipeline = _build_scene(device, convex=True)
    test.assertEqual(pipeline.soft_contact_pair_count, 0)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    actual = _actual_pair_sets(pipeline)
    expected = _brute_pair_sets(model, 0.06)[:3]
    test.assertEqual(actual, expected)
    test.assertTrue(all(expected_family for expected_family in expected))


def _thin_feature_model(device, rigid_vertices, rigid_indices, soft_vertices):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    mesh = newton.Mesh(
        np.asarray(rigid_vertices, dtype=np.float32),
        np.asarray(rigid_indices, dtype=np.int32),
        compute_inertia=False,
        is_solid=False,
    )
    cfg = builder.ShapeConfig(margin=0.0)
    builder.add_shape_mesh(body=-1, mesh=mesh, cfg=cfg)
    _add_single_cloth_triangle(builder, soft_vertices)
    builder.color()
    return builder.finalize(device=device)


def _assert_particle_pass_misses_bvh_family(test, model, family):
    particle_pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=False,
        broad_phase="nxn",
        rigid_contact_max=16,
        max_triangle_pairs=16,
        soft_contact_gap=0.05,
    )
    particle_contacts = particle_pipeline.contacts()
    particle_pipeline.collide(model.state(), particle_contacts)
    test.assertEqual(int(particle_contacts.soft_contact_count.numpy()[0]), 0)

    bvh = newton.CollisionPipeline(
        model,
        reduce_contacts=False,
        broad_phase="nxn",
        rigid_contact_max=16,
        max_triangle_pairs=16,
        soft_contact_gap=0.05,
        enable_rigid_soft_full_surface_contact=True,
    )
    bvh.collide(model.state(), bvh.contacts())
    actual = _actual_pair_sets(bvh)
    test.assertEqual(len(actual[0]), 0, "soft vertices must remain outside the query band")
    test.assertGreater(len(actual[family]), 0)


def test_open_mesh_uses_winding_as_one_sided_sign(test, device):
    """Treat an open rigid triangle's positive-winding side as its exterior."""
    model = _thin_feature_model(
        device,
        rigid_vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        rigid_indices=(0, 1, 2),
        soft_vertices=((0.2, 0.2, 0.02), (2.0, 0.0, 2.0), (0.0, 2.0, 2.0)),
    )
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=False,
        broad_phase="nxn",
        rigid_contact_max=16,
        max_triangle_pairs=16,
        soft_contact_gap=0.05,
        enable_rigid_soft_full_surface_contact=True,
    )

    for height, expected_separation_sign in ((0.02, 1.0), (-0.02, -1.0)):
        state = model.state()
        positions = state.particle_q.numpy()
        positions[0, 2] = height
        state.particle_q.assign(positions)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
        indices = contacts.soft_contact_indices.numpy()[:count]
        rows = np.flatnonzero((indices[:, 0] == 0) & (indices[:, 1] < 0))
        test.assertEqual(len(rows), 1)
        row = int(rows[0])
        normal = contacts.soft_contact_normal.numpy()[row]
        rigid_point = contacts.soft_contact_body_pos.numpy()[row]
        test.assertTrue(np.allclose(normal, (0.0, 0.0, 1.0), atol=1.0e-6))
        separation = float(np.dot(normal, positions[0] - rigid_point))
        test.assertGreater(expected_separation_sign * separation, 0.0)


def test_closed_mesh_interior_uses_nearest_outward_normal(test, device):
    """Orient a deeply interior VF row with the nearest closed-mesh face normal."""
    model, pipeline = _build_scene(device)
    pipeline.set_collision_detection_range(soft_contact_gap=0.11)
    state = model.state()
    positions = state.particle_q.numpy()
    positions[0] = (0.1, 0.2, 0.3)
    positions[1:] = ((2.0, 0.0, 2.0), (0.0, 2.0, 2.0), (2.0, 2.0, 2.0), (3.0, 2.0, 2.0), (2.0, 3.0, 2.0))
    state.particle_q.assign(positions)
    pipeline.refit_soft_surface_bvh(state)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
    indices = contacts.soft_contact_indices.numpy()[:count]
    rows = np.flatnonzero((indices[:, 0] == 0) & (indices[:, 1] < 0))
    test.assertEqual(len(rows), 1)
    row = int(rows[0])
    normal = contacts.soft_contact_normal.numpy()[row]
    rigid_point = contacts.soft_contact_body_pos.numpy()[row]
    test.assertTrue(np.allclose(normal, (-1.0, 0.0, 0.0), atol=1.0e-6))
    test.assertAlmostEqual(float(np.dot(normal, positions[0] - rigid_point)), -0.1, places=5)


def test_coincident_parallel_ee_uses_rigid_outward_normal(test, device):
    """Use the rigid edge's outward fallback for an exactly coincident parallel EE pair."""
    model = _thin_feature_model(
        device,
        rigid_vertices=((-0.2, 0.0, 0.0), (0.2, 0.0, 0.0), (-0.2, 0.2, 0.0)),
        rigid_indices=(0, 1, 2),
        soft_vertices=((-0.2, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.4, 0.2)),
    )
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=False,
        broad_phase="nxn",
        rigid_contact_max=16,
        max_triangle_pairs=16,
        soft_contact_gap=0.05,
        enable_rigid_soft_full_surface_contact=True,
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    candidate_count = min(
        int(pipeline.rigid_soft_bvh_candidate_count.numpy()[0]), pipeline.rigid_soft_bvh_candidate_max
    )
    candidates = pipeline._rigid_soft_bvh_candidates.numpy()[:candidate_count]
    rigid_edges = pipeline.rigid_soft_bvh_rigid_edge_table.numpy()
    soft_edges = model.edge_indices.numpy()[:, 2:4]
    output_rows = contacts.soft_contact_tids.numpy()[:candidate_count]
    matched_rows = []
    for candidate, output_row in zip(candidates, output_rows, strict=True):
        family, soft_edge, _shape, rigid_edge = map(int, candidate)
        if (
            family == RIGID_SOFT_BVH_CONTACT_EE
            and set(map(int, soft_edges[soft_edge])) == {0, 1}
            and set(map(int, rigid_edges[rigid_edge, 1:])) == {0, 1}
        ):
            matched_rows.append(int(output_row))
    test.assertEqual(len(matched_rows), 1)
    test.assertGreaterEqual(matched_rows[0], 0)
    normal = contacts.soft_contact_normal.numpy()[matched_rows[0]]
    test.assertTrue(np.all(np.isfinite(normal)))
    test.assertTrue(np.allclose(normal, (0.0, 0.0, 1.0), atol=1.0e-6))


def test_thin_feature_fv_catches_particle_only_miss(test, device):
    """A rigid vertex near a coarse soft-face interior is invisible to particle-only contact."""
    model = _thin_feature_model(
        device,
        rigid_vertices=((0.0, 0.0, 0.0), (-0.01, 0.0, 0.0), (0.0, -0.01, 0.0)),
        rigid_indices=(0, 1, 2),
        soft_vertices=((-0.2, -0.2, 0.03), (0.2, -0.2, 0.03), (0.0, 0.2, 0.03)),
    )
    _assert_particle_pass_misses_bvh_family(test, model, 1)


def test_thin_feature_ee_catches_particle_only_miss(test, device):
    """Crossing edge interiors are detected when every soft vertex is outside the query band."""
    model = _thin_feature_model(
        device,
        rigid_vertices=((-0.2, 0.0, 0.0), (0.2, 0.0, 0.0), (-0.2, -0.02, 0.0)),
        rigid_indices=(0, 1, 2),
        soft_vertices=((0.0, -0.2, 0.03), (0.0, 0.2, 0.03), (0.02, 0.2, 0.03)),
    )
    _assert_particle_pass_misses_bvh_family(test, model, 2)


def test_bvh_pair_sets_match_brute_force(test, device):
    """VF, FV, and EE BVH candidates exactly match brute-force primitive enumeration."""
    model, pipeline = _build_scene(device)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    expected = _brute_pair_sets(model, 0.06)[:3]
    actual = _actual_pair_sets(pipeline)
    test.assertTrue(all(expected_family for expected_family in expected), "scene must exercise VF, FV, and EE")
    test.assertEqual(actual, expected)

    total = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
    indices = contacts.soft_contact_indices.numpy()[:total]
    bary = contacts.soft_contact_barycentric.numpy()[:total]
    body_pos = contacts.soft_contact_body_pos.numpy()[:total]
    normals = contacts.soft_contact_normal.numpy()[:total]
    test.assertEqual(total, sum(len(family) for family in expected))
    test.assertTrue(np.all(np.isfinite(body_pos)))
    test.assertTrue(np.all(np.isfinite(normals)))
    test.assertTrue(np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-5))
    actual_distances = ([], [], [])
    particles = model.particle_q.numpy()
    for row, weights in zip(indices, bary, strict=True):
        valid = row >= 0
        test.assertAlmostEqual(float(np.sum(weights[valid])), 1.0, places=5)
        test.assertTrue(np.all(np.abs(weights[~valid]) < 1.0e-6))
    for row, weights, rigid_point in zip(indices, bary, body_pos, strict=True):
        valid = row >= 0
        soft_point = sum(weights[j] * particles[row[j]] for j in range(3) if valid[j])
        family_slot = {1: 0, 3: 1, 2: 2}[int(np.count_nonzero(valid))]
        actual_distances[family_slot].append(np.linalg.norm(soft_point - rigid_point))
    expected_distances = _brute_pair_distances(model, expected)
    for actual_family, expected_family in zip(actual_distances, expected_distances, strict=True):
        test.assertTrue(np.allclose(np.sort(actual_family), expected_family, atol=2.0e-5))

    # Map each candidate thread back to its emitted row through the replay table and verify the
    # rigid-to-soft normal, including the outward flip used for points inside a closed mesh.
    candidate_count = int(pipeline.rigid_soft_bvh_candidate_count.numpy()[0])
    candidates = pipeline._rigid_soft_bvh_candidates.numpy()[:candidate_count]
    emitted_rows = contacts.soft_contact_tids.numpy()[:candidate_count]
    rigid_vertices = np.asarray(model.shape_source[0].vertices) * model.shape_scale.numpy()[0]
    rigid_triangles = np.asarray(model.shape_source[0].indices).reshape(-1, 3)
    vertex_normals = pipeline.rigid_soft_bvh_rigid_vertex_normal.numpy()
    edge_normals = pipeline.rigid_soft_bvh_rigid_edge_outward.numpy()
    for candidate, output_index_raw in zip(candidates, emitted_rows, strict=True):
        family, _soft_feature, _shape, rigid_feature = map(int, candidate)
        output_index = int(output_index_raw)
        row = indices[output_index]
        valid = row >= 0
        soft_point = sum(bary[output_index, j] * particles[row[j]] for j in range(3) if valid[j])
        delta = soft_point - body_pos[output_index]
        if family == RIGID_SOFT_BVH_CONTACT_VF:
            tri = rigid_triangles[rigid_feature]
            outward = np.cross(
                rigid_vertices[tri[1]] - rigid_vertices[tri[0]], rigid_vertices[tri[2]] - rigid_vertices[tri[0]]
            )
        elif family == RIGID_SOFT_BVH_CONTACT_FV:
            outward = vertex_normals[rigid_feature]
        else:
            outward = edge_normals[rigid_feature]
        outward = outward / np.linalg.norm(outward)
        expected_normal = outward if np.linalg.norm(delta) < 1.0e-10 else delta / np.linalg.norm(delta)
        if np.dot(expected_normal, outward) < 0.0:
            expected_normal = -expected_normal
        test.assertTrue(np.allclose(normals[output_index], expected_normal, atol=2.0e-5))


def test_bvh_overflow_is_detectable_and_guarded(test, device):
    """An undersized candidate/output capacity reports attempted pairs without corrupting writes."""
    _model, independently_sized = _build_scene(device, soft_contact_max=1)
    test.assertEqual(
        independently_sized.rigid_soft_bvh_candidate_max,
        4 * independently_sized._rigid_soft_bvh_query_seed_count,
    )
    test.assertNotEqual(independently_sized.rigid_soft_bvh_candidate_max, independently_sized.soft_contact_max)

    for capacity in (0, 1):
        model, pipeline = _build_scene(
            device,
            soft_contact_max=capacity,
            rigid_soft_bvh_candidate_max=capacity,
        )
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)
        attempted = int(pipeline.rigid_soft_bvh_candidate_count.numpy()[0])
        test.assertGreater(attempted, pipeline.rigid_soft_bvh_candidate_max)
        test.assertLessEqual(min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max), capacity)
        test.assertTrue(np.all(np.isfinite(contacts.soft_contact_body_pos.numpy())))
        test.assertTrue(np.all(np.isfinite(contacts.soft_contact_normal.numpy())))


def test_bvh_refit_updates_detection(test, device):
    """A newly near soft surface is missed until explicit refit updates its stale BVH bounds."""
    model, pipeline = _build_scene(device, z_offset=5.0)
    state = model.state()
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    test.assertEqual(int(pipeline.rigid_soft_bvh_candidate_count.numpy()[0]), 0)
    moved = state.particle_q.numpy() - np.array((0.0, 0.0, 5.0), dtype=np.float32)
    state.particle_q.assign(moved)
    pipeline.collide(state, contacts)
    stale_sets = _actual_pair_sets(pipeline)
    test.assertGreater(len(stale_sets[0]), 0, "VF reads current particles against the rigid mesh BVH")
    test.assertEqual(len(stale_sets[1]), 0, "FV must still see the deliberately stale soft-triangle BVH")
    test.assertEqual(len(stale_sets[2]), 0, "EE must still see the deliberately stale soft-edge BVH")
    pipeline.refit_soft_surface_bvh(state)
    pipeline.collide(state, contacts)
    current_sets = _actual_pair_sets(pipeline)
    test.assertGreater(len(current_sets[1]), 0)
    test.assertGreater(len(current_sets[2]), 0)


def test_bvh_graph_capture(test, device):
    """The soft-BVH refit and fixed-capacity contact path replay under CUDA graph capture."""
    model, pipeline = _build_scene(device)
    state = model.state()
    contacts = pipeline.contacts()
    pipeline.refit_soft_surface_bvh(state)
    pipeline.collide(state, contacts)
    baseline = int(pipeline.rigid_soft_bvh_candidate_count.numpy()[0])
    baseline_candidates = pipeline._rigid_soft_bvh_candidates.numpy()[:baseline].copy()
    baseline_contact_count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
    baseline_geometry = (
        contacts.soft_contact_indices.numpy()[:baseline_contact_count].copy(),
        contacts.soft_contact_barycentric.numpy()[:baseline_contact_count].copy(),
        contacts.soft_contact_body_pos.numpy()[:baseline_contact_count].copy(),
        contacts.soft_contact_normal.numpy()[:baseline_contact_count].copy(),
    )
    with wp.ScopedCapture(device) as capture:
        pipeline.refit_soft_surface_bvh(state)
        pipeline.collide(state, contacts)
    for _ in range(2):
        wp.capture_launch(capture.graph)
        test.assertEqual(int(pipeline.rigid_soft_bvh_candidate_count.numpy()[0]), baseline)
        test.assertTrue(np.array_equal(pipeline._rigid_soft_bvh_candidates.numpy()[:baseline], baseline_candidates))
        count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
        test.assertEqual(count, baseline_contact_count)
        replay_geometry = (
            contacts.soft_contact_indices.numpy()[:count],
            contacts.soft_contact_barycentric.numpy()[:count],
            contacts.soft_contact_body_pos.numpy()[:count],
            contacts.soft_contact_normal.numpy()[:count],
        )
        test.assertTrue(np.array_equal(replay_geometry[0], baseline_geometry[0]))
        for replay, expected in zip(replay_geometry[1:], baseline_geometry[1:], strict=True):
            test.assertTrue(np.allclose(replay, expected, atol=1.0e-7))


def test_bvh_multiworld_grouping(test, device):
    """Multi-world BVH candidates exactly match brute force, including one global rigid mesh."""
    world_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    world_builder.add_shape_mesh(body=-1, mesh=_tetra_mesh())
    _add_two_triangle_cloth(world_builder)
    world_builder.color()

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    global_shape = builder.add_shape_mesh(body=-1, mesh=_tetra_mesh())
    builder.add_world(world_builder)
    builder.add_world(world_builder)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        rigid_contact_max=64,
        max_triangle_pairs=64,
        soft_contact_gap=0.06,
        enable_rigid_soft_full_surface_contact=True,
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    count = min(int(pipeline.rigid_soft_bvh_candidate_count.numpy()[0]), pipeline.rigid_soft_bvh_candidate_max)
    candidates = pipeline._rigid_soft_bvh_candidates.numpy()[:count]
    shape_world = model.shape_world.numpy()
    particle_world = model.particle_world.numpy()
    triangles = model.tri_indices.numpy()
    edges = model.edge_indices.numpy()
    observed_shape_worlds = set()
    actual = (set(), set(), set())
    vertex_table = pipeline.rigid_soft_bvh_rigid_vertex_table.numpy()
    edge_table = pipeline.rigid_soft_bvh_rigid_edge_table.numpy()
    shape_edge_lookup = {
        shape: {tuple(sorted(map(int, edge))): edge_index for edge_index, edge in enumerate(mesh.edges)}
        for shape, mesh in enumerate(model.shape_source)
        if mesh is not None
    }
    for family, soft_feature, shape, _rigid_feature in candidates:
        if family == RIGID_SOFT_BVH_CONTACT_VF:
            soft_world = particle_world[soft_feature]
        elif family == RIGID_SOFT_BVH_CONTACT_FV:
            soft_world = particle_world[triangles[soft_feature, 0]]
        else:
            soft_world = particle_world[edges[soft_feature, 2]]
        rigid_world = shape_world[shape]
        test.assertTrue(rigid_world < 0 or soft_world < 0 or rigid_world == soft_world)
        observed_shape_worlds.add(int(rigid_world))
        if family == RIGID_SOFT_BVH_CONTACT_VF:
            actual[0].add((int(soft_feature), int(shape), int(_rigid_feature)))
        elif family == RIGID_SOFT_BVH_CONTACT_FV:
            actual[1].add((int(soft_feature), int(shape), int(vertex_table[_rigid_feature, 1])))
        else:
            local_edge = tuple(sorted(map(int, edge_table[_rigid_feature, 1:3])))
            actual[2].add((int(soft_feature), int(shape), shape_edge_lookup[int(shape)][local_edge]))
    test.assertIn(-1, observed_shape_worlds)
    test.assertIn(0, observed_shape_worlds)
    test.assertIn(1, observed_shape_worlds)
    test.assertEqual(int(shape_world[global_shape]), -1)
    test.assertEqual(actual, _brute_multiworld_pair_sets(model, 0.06))


def test_bvh_matches_soft_self_feature_families(test, device):
    """Rigid-soft VF/FV/EE pairs match the corresponding pairs between two identical soft triangles."""
    # Rigid-soft model: soft triangle A is offset 3 cm from rigid triangle B.
    rigid_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    rigid_triangle = newton.Mesh(
        np.asarray(((-0.2, -0.2, 0.0), (0.2, -0.2, 0.0), (0.0, 0.2, 0.0)), dtype=np.float32),
        np.asarray((0, 1, 2), dtype=np.int32),
        compute_inertia=False,
        is_solid=False,
    )
    rigid_builder.add_shape_mesh(body=-1, mesh=rigid_triangle)
    rigid_builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.03),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(-0.2, -0.2, 0.0), wp.vec3(0.2, -0.2, 0.0), wp.vec3(0.0, 0.2, 0.0)],
        indices=[0, 1, 2],
        density=0.1,
        particle_radius=0.0,
    )
    rigid_builder.color()
    rigid_model = rigid_builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        rigid_model,
        broad_phase="nxn",
        rigid_contact_max=16,
        max_triangle_pairs=16,
        soft_contact_gap=0.05,
        enable_rigid_soft_full_surface_contact=True,
    )
    pipeline.collide(rigid_model.state(), pipeline.contacts())
    rigid_sets = _actual_pair_sets(pipeline)

    # Soft-self model: component A is the soft triangle above and component B is its rigid reference.
    self_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    vertices = [
        wp.vec3(-0.2, -0.2, 0.03),
        wp.vec3(0.2, -0.2, 0.03),
        wp.vec3(0.0, 0.2, 0.03),
        wp.vec3(-0.2, -0.2, 0.0),
        wp.vec3(0.2, -0.2, 0.0),
        wp.vec3(0.0, 0.2, 0.0),
    ]
    self_builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=vertices,
        indices=[0, 1, 2, 3, 4, 5],
        density=0.1,
        particle_radius=0.0,
    )
    self_builder.color()
    self_model = self_builder.finalize(device=device)
    detector = TriMeshCollisionDetector(self_model, init_collision_info=True, topological_contact_filter_threshold=0)
    detector.vertex_triangle_collision_detection(0.05)
    detector.edge_edge_collision_detection(0.05)

    info = detector.collision_info
    vt_values = info.vertex_colliding_triangles.numpy()
    vt_offsets = info.vertex_colliding_triangles_offsets.numpy()
    vt_counts = info.vertex_colliding_triangles_count.numpy()
    self_vf, self_fv = set(), set()
    for vertex in range(6):
        for collision in range(min(int(vt_counts[vertex]), int(vt_offsets[vertex + 1] - vt_offsets[vertex]))):
            tri = int(vt_values[2 * (vt_offsets[vertex] + collision) + 1])
            if vertex < 3 and tri == 1:
                self_vf.add((vertex, 0))
            elif vertex >= 3 and tri == 0:
                self_fv.add((0, vertex - 3))

    edge_indices = self_model.edge_indices.numpy()[:, 2:4]
    component_a_edges = {
        edge: tuple(sorted(map(int, endpoints))) for edge, endpoints in enumerate(edge_indices) if np.all(endpoints < 3)
    }
    component_b_edges = {
        edge: tuple(sorted(int(v) - 3 for v in endpoints))
        for edge, endpoints in enumerate(edge_indices)
        if np.all(endpoints >= 3)
    }
    local_b = {tuple(sorted(map(int, endpoints))): local for local, endpoints in enumerate(rigid_triangle.edges)}
    ee_values = info.edge_colliding_edges.numpy()
    ee_offsets = info.edge_colliding_edges_offsets.numpy()
    ee_counts = info.edge_colliding_edges_count.numpy()
    self_ee = set()
    for edge_a, _endpoints_a in component_a_edges.items():
        for collision in range(min(int(ee_counts[edge_a]), int(ee_offsets[edge_a + 1] - ee_offsets[edge_a]))):
            edge_b = int(ee_values[2 * (ee_offsets[edge_a] + collision) + 1])
            if edge_b in component_b_edges:
                self_ee.add((edge_a, local_b[component_b_edges[edge_b]]))

    # Convert soft edge ids to the rigid-soft model's edge ids by endpoint set.
    rigid_soft_edges = rigid_model.edge_indices.numpy()[:, 2:4]
    soft_edge_by_endpoints = {
        tuple(sorted(map(int, endpoints))): edge for edge, endpoints in enumerate(rigid_soft_edges)
    }
    self_ee_converted = {
        (soft_edge_by_endpoints[component_a_edges[edge_a]], rigid_edge) for edge_a, rigid_edge in self_ee
    }
    test.assertEqual(rigid_sets[0], self_vf)
    test.assertEqual(rigid_sets[1], self_fv)
    test.assertEqual(rigid_sets[2], self_ee_converted)


def test_bvh_and_sdf_backends_are_behaviorally_equivalent(test, device):
    """Provisioned box mesh produces the same nearest surface gap with BVH and SDF backends."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh.create_box(0.5, 0.5, 0.5))
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(-0.1, -0.1, 0.53), wp.vec3(0.1, -0.1, 0.53), wp.vec3(0.0, 0.1, 0.53)],
        indices=[0, 1, 2],
        density=0.1,
        particle_radius=0.0,
    )
    builder.color()
    configure_sdf_for_collision_shapes(builder)
    model = builder.finalize(device=device)
    separations = []
    for backend in ("bvh", "sdf"):
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            rigid_contact_max=32,
            max_triangle_pairs=32,
            soft_contact_gap=0.05,
            enable_rigid_soft_full_surface_contact=True,
            full_surface_mesh_backend=backend,
        )
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)
        count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
        test.assertGreater(count, 0)
        indices = contacts.soft_contact_indices.numpy()[:count]
        bary = contacts.soft_contact_barycentric.numpy()[:count]
        body = contacts.soft_contact_body_pos.numpy()[:count]
        normals = contacts.soft_contact_normal.numpy()[:count]
        particles = model.particle_q.numpy()
        distances = []
        for corners, weights, rigid_point, normal in zip(indices, bary, body, normals, strict=True):
            soft_point = sum(weights[j] * particles[corners[j]] for j in range(3) if corners[j] >= 0)
            distances.append(abs(float(np.dot(normal, soft_point - rigid_point))))
        separations.append(min(distances))
    test.assertAlmostEqual(separations[0], 0.03, delta=2.0e-3)
    test.assertAlmostEqual(separations[1], 0.03, delta=5.0e-3)
    test.assertAlmostEqual(separations[0], separations[1], delta=5.0e-3)


def test_bvh_differentiable_replay(test, device):
    """Multiple candidates discovered by one BVH owner replay through distinct differentiable slots."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_shape_mesh(body=-1, mesh=_tetra_mesh())
    _add_two_triangle_cloth(builder)
    builder.color()
    model = builder.finalize(device=device, requires_grad=True)
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=False,
        broad_phase="nxn",
        rigid_contact_max=64,
        max_triangle_pairs=64,
        soft_contact_gap=0.06,
        enable_rigid_soft_full_surface_contact=True,
        requires_grad=True,
    )
    state = model.state(requires_grad=True)
    contacts = pipeline.contacts()
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)
    with wp.Tape() as tape:
        pipeline.collide(state, contacts)
        wp.launch(
            _sum_contact_geometry,
            dim=contacts.soft_contact_max,
            inputs=[contacts.soft_contact_count, contacts.soft_contact_barycentric, contacts.soft_contact_normal],
            outputs=[loss],
            device=device,
        )
    tape.backward(loss)
    gradient = state.particle_q.grad.numpy()
    test.assertTrue(np.all(np.isfinite(gradient)))
    test.assertGreater(float(np.linalg.norm(gradient)), 0.0)


def test_vbd_iteration_schedule_refits_bvh(test, device):
    """Every rigid collision pass already scheduled by VBD refits the soft-feature BVHs."""
    model, pipeline = _build_scene(device, dynamic_rigid=True)
    calls = []
    original_refit = pipeline.refit_soft_surface_bvh

    def counted_refit(state):
        calls.append(state.particle_q)
        original_refit(state)

    pipeline.refit_soft_surface_bvh = counted_refit
    frequency = newton.solvers.SolverBase.CollisionFrequencyType
    solver = newton.solvers.SolverVBD(
        model,
        pipeline=pipeline,
        iterations=3,
        collision_frequency=[1, 1],
        collision_frequency_type=[frequency.ITERATIONS, frequency.NONE],
    )
    state_in = model.state()
    state_out = model.state()
    solver.step(state_in, state_out, control=None, contacts=None, dt=1.0 / 60.0)
    test.assertEqual(len(calls), 3, "pre-init plus iteration 1 and iteration 2 must each refit")


class TestRigidSoftBvh(unittest.TestCase):
    pass


for _name, _function in (
    ("test_backend_selection_and_partition", test_backend_selection_and_partition),
    ("test_rigid_feature_tables_scaled_and_mirrored", test_rigid_feature_tables_scaled_and_mirrored),
    ("test_bvh_emits_rigid_mesh_surface_velocity", test_bvh_emits_rigid_mesh_surface_velocity),
    ("test_mutable_shape_flag_is_not_frozen_at_construction", test_mutable_shape_flag_is_not_frozen_at_construction),
    ("test_convex_mesh_uses_bvh_backend", test_convex_mesh_uses_bvh_backend),
    ("test_thin_feature_fv_catches_particle_only_miss", test_thin_feature_fv_catches_particle_only_miss),
    ("test_thin_feature_ee_catches_particle_only_miss", test_thin_feature_ee_catches_particle_only_miss),
    ("test_open_mesh_uses_winding_as_one_sided_sign", test_open_mesh_uses_winding_as_one_sided_sign),
    ("test_closed_mesh_interior_uses_nearest_outward_normal", test_closed_mesh_interior_uses_nearest_outward_normal),
    ("test_coincident_parallel_ee_uses_rigid_outward_normal", test_coincident_parallel_ee_uses_rigid_outward_normal),
    ("test_bvh_pair_sets_match_brute_force", test_bvh_pair_sets_match_brute_force),
    ("test_bvh_refit_updates_detection", test_bvh_refit_updates_detection),
):
    add_function_test(TestRigidSoftBvh, _name, _function, devices=get_test_devices())

for _name, _function in (
    ("test_bvh_overflow_is_detectable_and_guarded", test_bvh_overflow_is_detectable_and_guarded),
    ("test_bvh_graph_capture", test_bvh_graph_capture),
    ("test_bvh_multiworld_grouping", test_bvh_multiworld_grouping),
    ("test_bvh_matches_soft_self_feature_families", test_bvh_matches_soft_self_feature_families),
    ("test_bvh_and_sdf_backends_are_behaviorally_equivalent", test_bvh_and_sdf_backends_are_behaviorally_equivalent),
    ("test_bvh_differentiable_replay", test_bvh_differentiable_replay),
    ("test_vbd_iteration_schedule_refits_bvh", test_vbd_iteration_schedule_refits_bvh),
):
    add_function_test(TestRigidSoftBvh, _name, _function, devices=get_cuda_test_devices(mode="basic"))


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
