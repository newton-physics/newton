# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the BVH full-surface rigid-soft contact back-end (soft_contacts_bvh).

Organization (each section carries its own registrations):

1. Shared reference helpers -- independent numpy geometry and record utilities.
2. Kernel unit tests -- each kernel launched directly on hand-computable geometry.
3. Integration: detection results -- pipeline records vs independent references.
4. Integration: gradients and the shared record stream -- differentiable replay.
5. Integration: flags, configuration, and API lifecycle.
6. Integration: diagnostics and capture -- overflow warnings, graph capture.
"""

import unittest
import warnings
from collections import Counter

import numpy as np
import warp as wp

import newton
from newton import GeoType
from newton._src.geometry.flags import ParticleFlags, ShapeFlags
from newton.tests.unittest_utils import (
    add_function_test,
    get_cuda_test_devices,
    get_test_devices,
)

soft_devices = get_test_devices()


class TestBvhFullSurfaceSoftContact(unittest.TestCase):
    pass


# ---------------------------------------------------------------------------
# Shared reference helpers
# Independent numpy geometry, record splitting/matching, and scene builders.
# ---------------------------------------------------------------------------


def _np_tri_closest_point(a, b, c, p):
    """Reference closest point on triangle (a, b, c) to p; returns (cp, bary)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.array([1.0, 0.0, 0.0])
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0.0 and d4 <= d3:
        return b, np.array([0.0, 1.0, 0.0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.array([1.0 - v, v, 0.0])
    cp_ = p - c
    d5, d6 = ab @ cp_, ac @ cp_
    if d6 >= 0.0 and d5 <= d6:
        return c, np.array([0.0, 0.0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.array([1.0 - w, 0.0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.array([0.0, 1.0 - w, w])
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return a + ab * v + ac * w, np.array([1.0 - v - w, v, w])


def _np_seg_seg_closest(p0, p1, q0, q1):
    """Reference closest points between segments; returns (s, t, dist)."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-12 and e <= 1e-12:
        return 0.0, 0.0, float(np.linalg.norm(r))
    if a <= 1e-12:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= 1e-12:
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    dist = float(np.linalg.norm((p0 + d1 * s) - (q0 + d2 * t)))
    return float(s), float(t), dist


def _np_quat_rotate(q, v):
    """Rotate vectors ``v`` (..., 3) by quaternion ``q`` = (x, y, z, w)."""
    qv, w = np.asarray(q[:3]), float(q[3])
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _np_unique_mesh_features(mesh):
    """Independent reference of the rigid feature tables: unique vertex positions (canonical
    1e-7 folding, restricted to face-referenced vertices) and unique edge endpoint positions."""
    verts = np.asarray(mesh.vertices, np.float64)
    tris = np.asarray(mesh.indices, np.int32).reshape(-1, 3)
    _, canon = np.unique(np.round(verts * 1e7).astype(np.int64), axis=0, return_inverse=True)
    canon = canon.ravel()
    canon_pos = {}
    for v_id in range(len(verts)):
        canon_pos.setdefault(int(canon[v_id]), verts[v_id])
    used = sorted({int(c) for c in canon[tris].ravel()})
    vertex_pos = np.array([canon_pos[c] for c in used])
    edge_keys = set()
    for t in tris:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            ca, cb = int(canon[t[a]]), int(canon[t[b]])
            edge_keys.add((min(ca, cb), max(ca, cb)))
    edge_pos = np.array([[canon_pos[ca], canon_pos[cb]] for ca, cb in sorted(edge_keys)])
    return vertex_pos, edge_pos


def _split_bvh_families(model, contacts, n_legacy_records):
    """Split emitted records into (VT, EE, TV) family dicts after the first ``n_legacy_records``.

    Returns per-family lists of ``(key, x_soft_world, x_rigid_world)``: VT keyed by particle id,
    EE by soft edge endpoints, TV by soft triangle corners. ``body_pos`` is body-local; for shapes
    with ``shape_body == -1`` (all scenes here) the body frame IS the world frame regardless of
    the shape transform, so it is world-space directly.
    """
    total = int(contacts.soft_contact_count.numpy()[0])
    idx = contacts.soft_contact_indices.numpy()[:total]
    bary = contacts.soft_contact_barycentric.numpy()[:total]
    body_pos = contacts.soft_contact_body_pos.numpy()[:total]
    q = model.particle_q.numpy()
    vt, ee, tv = [], [], []
    for i in range(n_legacy_records, total):
        c = idx[i]
        if c[2] >= 0:
            x_soft = bary[i, 0] * q[c[0]] + bary[i, 1] * q[c[1]] + bary[i, 2] * q[c[2]]
            tv.append(((int(c[0]), int(c[1]), int(c[2])), x_soft, body_pos[i]))
        elif c[1] >= 0:
            x_soft = bary[i, 0] * q[c[0]] + bary[i, 1] * q[c[1]]
            ee.append(((int(c[0]), int(c[1])), x_soft, body_pos[i]))
        else:
            vt.append((int(c[0]), q[c[0]], body_pos[i]))
    return vt, ee, tv


def _assert_family_matches(test, actual, expected, threshold, band=None, tol=1e-4):
    """Match actual records against a brute-force reference, tolerant only at the accept border.

    ``actual``: list of (key, distance); ``expected``: dict key -> sorted distances. Every expected
    pair strictly inside the threshold band must be reported; every actual record must appear in
    the reference within ``tol``. Pairs within ``band`` (default: ``tol``, never narrower) of the
    threshold may differ (float order). ``threshold`` must be an upper bound on the reference's
    per-pair accept bound ``gap + shape_margin + r_soft`` -- see ``_parity_threshold``.
    """
    if band is None:
        band = tol
    remaining = {k: sorted(v) for k, v in expected.items()}
    for key, d_act in actual:
        cands = remaining.get(key, [])
        match = next((j for j, d in enumerate(cands) if abs(d - d_act) <= tol), None)
        test.assertIsNotNone(match, f"unexpected record {key} at d={d_act}")
        cands.pop(match)
    for key, dists in remaining.items():
        for d in dists:
            test.assertGreater(d, threshold - band, f"missing record {key} at d={d}")


def _assert_bvh_only_records(test, pipeline):
    """Assert no non-BVH pass can emit records, and return the exact record base offset.

    ``_split_bvh_families`` skips leading *records*, while the pipeline exposes launch *pair*
    counts (each pair emits at most one record). The two coincide exactly when the counts are
    zero -- which BVH-only scenes guarantee: the mesh shape is excluded from the legacy particle
    pairs (VT replaces it) and, under the 'bvh' back-end, from the SDF edge/face pairs too.
    """
    test.assertEqual(pipeline.soft_contact_pair_count, 0)
    n_ef_pairs = len(pipeline.soft_edge_rigid_pairs) + len(pipeline.soft_face_rigid_pairs)
    test.assertEqual(n_ef_pairs, 0)
    return pipeline.soft_contact_pair_count + n_ef_pairs


def _parity_threshold(model, gap):
    """Upper bound on the kernels' per-pair accept bound ``gap + shape_margin + r_soft``."""
    margin_max = float(model.shape_margin.numpy().max()) if model.shape_count else 0.0
    radius_max = float(model.particle_radius.numpy().max()) if model.particle_count else 0.0
    return gap + margin_max + radius_max


def _build_cloth_over_mesh_box(device, particle_z=0.105, provision_sdf=False, dim=6, world_builder=None):
    """A cloth grid hovering 5 mm above a 0.25x0.25x0.1 half-extent mesh box (no SDF by default).

    The cloth (0.6 m span) overhangs the box footprint so its edges cross above the box's top
    edges and its interior covers the top corners -- all three feature families fire.
    """
    builder = world_builder if world_builder is not None else newton.ModelBuilder()
    box = newton.Mesh.create_box(0.25, 0.25, 0.1)
    cfg = newton.ModelBuilder.ShapeConfig()
    if provision_sdf:
        cfg.configure_sdf(force_sdf=True)
    builder.add_shape_mesh(-1, mesh=box, cfg=cfg)
    half = dim * 0.1 / 2.0
    builder.add_cloth_grid(
        pos=wp.vec3(-half, -half, particle_z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim,
        dim_y=dim,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
        particle_radius=0.01,
    )
    if world_builder is not None:
        return box
    return builder.finalize(device=device), box


def _bvh_brute_force_reference(model, box_mesh, gap):
    """Brute-force O(n*m) reference over all three feature families, honoring world compatibility.

    Returns dicts key -> list of distances for (VT, EE, TV), using the same accept test as the
    kernels: ``d < gap + shape_margin + r_soft``.
    """
    q = model.particle_q.numpy().astype(np.float64)
    radius = model.particle_radius.numpy()
    pw = model.particle_world.numpy()
    sw = model.shape_world.numpy()
    stype = model.shape_type.numpy()
    s_margin = model.shape_margin.numpy()
    xforms = model.shape_transform.numpy()
    tri = model.tri_indices.numpy() if model.tri_count else np.empty((0, 3), np.int32)
    edges = model.edge_indices.numpy()[:, 2:4] if model.edge_count else np.empty((0, 2), np.int32)
    rigid_verts_local, rigid_edges_local = _np_unique_mesh_features(box_mesh)
    tri_faces = np.asarray(box_mesh.indices, np.int32).reshape(-1, 3)
    verts_local = np.asarray(box_mesh.vertices, np.float64)

    vt, ee, tv = {}, {}, {}
    scales = model.shape_scale.numpy()
    mesh_shapes = [s for s in range(model.shape_count) if stype[s] in (int(GeoType.MESH), int(GeoType.CONVEX_MESH))]
    for s in mesh_shapes:
        # Static shapes: world = rotate(scale * local) + translate (full shape transform + scale).
        pos, quat = xforms[s][:3], xforms[s][3:]
        scale = scales[s]

        def to_world(x):
            return _np_quat_rotate(quat, x * scale) + pos  # noqa: B023

        r_verts = to_world(rigid_verts_local)
        r_edges = to_world(rigid_edges_local)
        for p in range(len(q)):
            if pw[p] != -1 and sw[s] != -1 and pw[p] != sw[s]:
                continue
            threshold = gap + s_margin[s] + radius[p]
            for f in tri_faces:
                cp, _ = _np_tri_closest_point(*(to_world(verts_local[f[k]]) for k in range(3)), q[p])
                d = float(np.linalg.norm(q[p] - cp))
                if d < threshold:
                    vt.setdefault(p, []).append(d)
        for rv in r_verts:
            for t in range(len(tri)):
                if pw[tri[t, 0]] != -1 and sw[s] != -1 and pw[tri[t, 0]] != sw[s]:
                    continue
                cp, bary = _np_tri_closest_point(q[tri[t, 0]], q[tri[t, 1]], q[tri[t, 2]], rv)
                r_soft = float(bary @ radius[tri[t]])
                d = float(np.linalg.norm(cp - rv))
                if d < gap + s_margin[s] + r_soft:
                    tv.setdefault(tuple(int(v) for v in tri[t]), []).append(d)
        for re_ in r_edges:
            for e in range(len(edges)):
                if pw[edges[e, 0]] != -1 and sw[s] != -1 and pw[edges[e, 0]] != sw[s]:
                    continue
                _, _, d = _np_seg_seg_closest(re_[0], re_[1], q[edges[e, 0]], q[edges[e, 1]])
                r_soft = float(max(radius[edges[e, 0]], radius[edges[e, 1]]))
                if d < gap + s_margin[s] + r_soft:
                    ee.setdefault((int(edges[e, 0]), int(edges[e, 1])), []).append(d)
    return vt, ee, tv


def _run_bvh_pipeline(model, gap=0.01, **kwargs):
    """Construct a BVH-backend full-surface pipeline, refit, collide; returns (pipeline, contacts)."""
    kwargs.setdefault("full_surface_mesh_backend", "bvh")
    pipeline = newton.CollisionPipeline(
        model, soft_contact_gap=gap, enable_rigid_soft_full_surface_contact=True, **kwargs
    )
    contacts = pipeline.contacts()
    state = model.state()
    if pipeline._full_surface_bvh_needs_detector:
        pipeline.refit_soft_contact_bvh(state)  # lazily creates the shared detector
    pipeline.collide(state, contacts)
    wp.synchronize_device(wp.get_device(model.device))
    return pipeline, contacts


# ---------------------------------------------------------------------------
# Kernel unit tests
# Each kernel launched directly on hand-computable geometry (tetrahedron / cube); no CollisionPipeline involved.
# ---------------------------------------------------------------------------


def _unit_rigid_tetrahedron(device, velocity=None):
    """One rigid corner tetrahedron A=(0,0,0), B=(1,0,0), C=(0,1,0), D=(0,0,1), at identity.

    Faces wound outward: face 0 = bottom (z=0, normal -z), face 1 = y=0 (normal -y),
    face 2 = x=0 (normal -x), face 3 = slant (normal (1,1,1)/sqrt(3)). Four faces, four vertices
    and six edges make every feature-index assertion non-trivial. Returns a dict of every
    hand-built shape-side kernel input, the feature tables from ``_mesh_feature_data`` (shape
    column prepended), and the keep-alive mesh objects.
    """
    from newton._src.geometry.soft_contacts_bvh import _mesh_feature_data  # noqa: PLC0415

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
    faces = np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], np.int32)
    mesh = newton.Mesh(vertices, faces)
    mesh_id = mesh.finalize(device=device)
    if velocity is not None:
        mesh.mesh.velocities.assign(np.tile(np.asarray(velocity, np.float32), (len(vertices), 1)))
    v_table, v_normals, e_table, e_outward = _mesh_feature_data(mesh)

    def _with_shape(table):
        return np.column_stack((np.zeros(len(table), np.int32), table))

    return {
        "mesh": mesh,  # keep-alive: owns the finalized wp.Mesh
        "mesh_id": mesh_id,
        "body_q": wp.array(np.zeros((1, 7), np.float32), dtype=wp.transform, device=device),
        "shape_transform": wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
        "shape_body": wp.array([-1], dtype=wp.int32, device=device),
        "shape_flags": wp.array([int(ShapeFlags.COLLIDE_PARTICLES)], dtype=wp.int32, device=device),
        "shape_scale": wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3, device=device),
        "shape_source_ptr": wp.array([mesh_id], dtype=wp.uint64, device=device),
        "shape_world": wp.array([-1], dtype=wp.int32, device=device),  # global: full-tree queries
        "shape_margin": wp.array([0.0], dtype=float, device=device),
        "vertex_table": wp.array(_with_shape(v_table), dtype=wp.vec2i, device=device),
        "vertex_normals": wp.array(v_normals, dtype=wp.vec3, device=device),
        "edge_table": wp.array(_with_shape(e_table), dtype=wp.vec3i, device=device),
        "edge_outward": wp.array(e_outward, dtype=wp.vec3, device=device),
        "v_table_np": v_table,
        "e_table_np": e_table,
    }


def _soft_particles(device, positions, radius=0.0, inactive=()):
    """Hand-built soft particle arrays; ``inactive`` lists particle indices with ACTIVE cleared."""
    positions = np.asarray(positions, np.float32)
    flags = np.full(len(positions), int(ParticleFlags.ACTIVE), np.int32)
    for i in inactive:
        flags[i] = 0
    return (
        wp.array(positions, dtype=wp.vec3, device=device),
        wp.array(np.full(len(positions), radius, np.float32), dtype=float, device=device),
        wp.array(flags, dtype=wp.int32, device=device),
    )


def _feature_bvh(device, lowers, uppers):
    """A one-group BVH over hand-given AABBs, with a full-tree-only group-roots array."""
    lo = wp.array(np.asarray(lowers, np.float32), dtype=wp.vec3, device=device)
    hi = wp.array(np.asarray(uppers, np.float32), dtype=wp.vec3, device=device)
    bvh = wp.Bvh(lo, hi)
    roots = wp.full(1, -1, dtype=wp.int32, device=device)  # unused: global-world shapes query the full tree
    return bvh, roots, lo, hi


def _candidates(device, capacity=8):
    return wp.zeros(1, dtype=wp.int32, device=device), wp.zeros(capacity, dtype=wp.vec4i, device=device)


def test_bvh_kernel_feature_tables(test, device):
    """_mesh_feature_data on a cube: exact angle-weighted vertex normals and edge outward dirs.

    On a cube every corner meets three faces at 90 degrees each, so the angle-weighted vertex
    normal is exactly the normalized corner diagonal sign(v)/sqrt(3) regardless of how the faces
    are triangulated. Edge outward dirs: a cube edge averages its two face normals
    ((n1+n2)/sqrt(2)); a face-diagonal edge lies in one face and keeps that face's normal.
    """
    from newton._src.geometry.soft_contacts_bvh import _mesh_feature_data  # noqa: PLC0415

    mesh = newton.Mesh.create_box(0.5, 0.5, 0.5)  # duplicated vertices: canonical folding must kick in
    v_table, v_normals, e_table, e_outward = _mesh_feature_data(mesh)
    test.assertEqual(len(v_table), 8)  # 24 stored vertices fold to 8 corners
    test.assertEqual(len(e_table), 18)  # 12 cube edges + 6 face diagonals

    indices = np.asarray(mesh.indices, np.int32)
    verts = np.asarray(mesh.vertices, np.float64)
    for row in range(8):
        corner = verts[indices[v_table[row]]]
        expected = np.sign(corner) / np.sqrt(3.0)
        test.assertTrue(np.allclose(v_normals[row], expected, atol=1e-6), f"corner {corner}")

    for row in range(18):
        p0 = verts[indices[e_table[row, 0]]]
        p1 = verts[indices[e_table[row, 1]]]
        shared = np.isclose(p0, p1) & np.isclose(np.abs(p0), 0.5)
        if shared.sum() == 2:  # cube edge: two shared boundary coordinates -> mean of 2 face normals
            expected = np.where(shared, np.sign(p0), 0.0) / np.sqrt(2.0)
        else:  # face diagonal: one shared boundary coordinate -> that face's normal
            test.assertEqual(int(shared.sum()), 1)
            expected = np.where(shared, np.sign(p0), 0.0)
        test.assertTrue(np.allclose(e_outward[row], expected, atol=1e-6), f"edge {p0}-{p1}")


def test_bvh_kernel_detect_vt(test, device):
    """detect_bvh_candidates_vt on the tetrahedron: the closest face is identified (index 2).

    P0 sits 0.05 off the x=0 face interior (accept, threshold 0.1; the other faces are 0.255+
    away, so exactly one candidate naming face 2); P1 sits 0.2 off (reject). Both flags gate
    per thread.
    """
    from newton._src.geometry.soft_contacts_bvh import BVH_CANDIDATE_VT, detect_bvh_candidates_vt  # noqa: PLC0415

    rig = _unit_rigid_tetrahedron(device)
    particle_q, particle_radius, particle_flags = _soft_particles(device, [[-0.05, 0.25, 0.25], [-0.2, 0.25, 0.25]])
    pairs = wp.array(np.array([[0, 0], [1, 0]], np.int32), dtype=wp.vec2i, device=device)
    count, cands = _candidates(device)
    inputs = [
        pairs,
        particle_q,
        particle_radius,
        particle_flags,
        rig["body_q"],
        rig["shape_transform"],
        rig["shape_body"],
        rig["shape_flags"],
        rig["shape_scale"],
        rig["shape_source_ptr"],
        rig["shape_margin"],
        0.1,  # gap
        8,
    ]
    wp.launch(detect_bvh_candidates_vt, dim=2, inputs=inputs, outputs=[count, cands], device=device)
    test.assertEqual(int(count.numpy()[0]), 1)
    test.assertTrue(np.all(cands.numpy()[0] == [int(BVH_CANDIDATE_VT), 0, 0, 2]))  # the x=0 face

    # Inactive particle: rejected despite being in range.
    _, _, inactive_flags = _soft_particles(device, [[-0.05, 0.25, 0.25], [-0.2, 0.25, 0.25]], inactive=(0,))
    count.zero_()
    inputs[3] = inactive_flags
    wp.launch(detect_bvh_candidates_vt, dim=2, inputs=inputs, outputs=[count, cands], device=device)
    test.assertEqual(int(count.numpy()[0]), 0)

    # COLLIDE_PARTICLES off: rejected.
    inputs[3] = particle_flags
    inputs[7] = wp.array([0], dtype=wp.int32, device=device)
    count.zero_()
    wp.launch(detect_bvh_candidates_vt, dim=2, inputs=inputs, outputs=[count, cands], device=device)
    test.assertEqual(int(count.numpy()[0]), 0)


def test_bvh_kernel_detect_tv(test, device):
    """detect_bvh_candidates_tv: a soft triangle over the tet's apex selects vertex D (id 3).

    The soft triangle ((-0.5,-0.5), (0.5,-0.5), (0,0.5)) at z=1.05 hovers 0.05 over the apex
    D=(0,0,1) only; A is 1.05 below it and B/C are >= 0.5 away laterally. Exactly one TV
    candidate, and its table row must resolve to vertex id 3 among the four.
    """
    from newton._src.geometry.soft_contacts_bvh import BVH_CANDIDATE_TV, detect_bvh_candidates_tv  # noqa: PLC0415

    rig = _unit_rigid_tetrahedron(device)
    soft_verts = [[-0.5, -0.5, 1.05], [0.5, -0.5, 1.05], [0.0, 0.5, 1.05]]
    particle_q, particle_radius, particle_flags = _soft_particles(device, soft_verts)
    tri_indices = wp.array(np.array([[0, 1, 2]], np.int32), dtype=wp.int32, ndim=2, device=device)
    lo = np.min(soft_verts, axis=0)
    hi = np.max(soft_verts, axis=0)
    bvh, roots, _l, _h = _feature_bvh(device, [lo], [hi])
    count, cands = _candidates(device)
    wp.launch(
        detect_bvh_candidates_tv,
        dim=len(rig["v_table_np"]),
        inputs=[
            rig["vertex_table"],
            particle_q,
            particle_radius,
            particle_flags,
            tri_indices,
            rig["body_q"],
            rig["shape_transform"],
            rig["shape_body"],
            rig["shape_flags"],
            rig["shape_scale"],
            rig["shape_source_ptr"],
            rig["shape_world"],
            rig["shape_margin"],
            bvh.id,
            roots,
            0,  # world_count (unused: global shape -> full-tree query)
            0.1,  # gap
            0.0,  # max_particle_radius
            8,
        ],
        outputs=[count, cands],
        device=device,
    )
    test.assertEqual(int(count.numpy()[0]), 1)
    cand = cands.numpy()[0]
    test.assertEqual(int(cand[0]), int(BVH_CANDIDATE_TV))
    test.assertEqual(int(cand[1]), 0)  # soft triangle 0
    # The rigid feature is the apex's table row: its face-vertex index must point at vertex id 3.
    row = int(cand[3])
    test.assertEqual(int(np.asarray(rig["mesh"].indices)[rig["v_table_np"][row]]), 3)

    # Fully inactive soft triangle: rejected; one active corner: accepted.
    for inactive, expected in (((0, 1, 2), 0), ((0, 1), 1)):
        _, _, flags = _soft_particles(device, soft_verts, inactive=inactive)
        count.zero_()
        wp.launch(
            detect_bvh_candidates_tv,
            dim=len(rig["v_table_np"]),
            inputs=[
                rig["vertex_table"],
                particle_q,
                particle_radius,
                flags,
                tri_indices,
                rig["body_q"],
                rig["shape_transform"],
                rig["shape_body"],
                rig["shape_flags"],
                rig["shape_scale"],
                rig["shape_source_ptr"],
                rig["shape_world"],
                rig["shape_margin"],
                bvh.id,
                roots,
                0,
                0.1,
                0.0,
                8,
            ],
            outputs=[count, cands],
            device=device,
        )
        test.assertEqual(int(count.numpy()[0]), expected, f"inactive={inactive}")


def test_bvh_kernel_detect_ee(test, device):
    """detect_bvh_candidates_ee: a soft edge crossing under rigid edge AB; only AB (of six) accepts.

    Soft segment (0.5,-0.3,-0.05)-(0.5,0.3,-0.05) crosses 0.05 below AB=(0,0,0)-(1,0,0). Closest
    approaches of the other five tet edges are 0.15 (BC), ~0.39 (BD) and >= 0.5 (AC, AD, CD).
    """
    from newton._src.geometry.soft_contacts_bvh import BVH_CANDIDATE_EE, detect_bvh_candidates_ee  # noqa: PLC0415

    rig = _unit_rigid_tetrahedron(device)
    soft_verts = [[0.5, -0.3, -0.05], [0.5, 0.3, -0.05]]
    particle_q, particle_radius, particle_flags = _soft_particles(device, soft_verts)
    edge_indices = wp.array(np.array([[-1, -1, 0, 1]], np.int32), dtype=wp.int32, ndim=2, device=device)
    lo = np.min(soft_verts, axis=0)
    hi = np.max(soft_verts, axis=0)
    bvh, roots, _l, _h = _feature_bvh(device, [lo], [hi])
    count, cands = _candidates(device)
    wp.launch(
        detect_bvh_candidates_ee,
        dim=len(rig["e_table_np"]),
        inputs=[
            rig["edge_table"],
            particle_q,
            particle_radius,
            particle_flags,
            edge_indices,
            rig["body_q"],
            rig["shape_transform"],
            rig["shape_body"],
            rig["shape_flags"],
            rig["shape_scale"],
            rig["shape_source_ptr"],
            rig["shape_world"],
            rig["shape_margin"],
            bvh.id,
            roots,
            0,
            1.0e-5,  # edge_edge_parallel_epsilon
            0.1,  # gap
            0.0,  # max_particle_radius
            8,
        ],
        outputs=[count, cands],
        device=device,
    )
    test.assertEqual(int(count.numpy()[0]), 1)
    cand = cands.numpy()[0]
    test.assertEqual(int(cand[0]), int(BVH_CANDIDATE_EE))
    test.assertEqual(int(cand[1]), 0)  # soft edge 0
    # The accepted rigid edge must be AB: endpoints at vertex ids {0, 1}.
    row = int(cand[3])
    mesh_indices = np.asarray(rig["mesh"].indices)
    endpoint_ids = {int(mesh_indices[rig["e_table_np"][row, 0]]), int(mesh_indices[rig["e_table_np"][row, 1]])}
    test.assertEqual(endpoint_ids, {0, 1})


def test_bvh_kernel_emit(test, device):
    """emit_bvh_contacts: every record field verified against hand-computed values, per family.

    The rigid tetrahedron with uniform surface velocity (1,2,3); soft features placed so each
    family yields one candidate with dyadic-rational geometry and a DISTINCT expected normal:
      VT: P=(-0.05,0.25,0.25) off the x=0 face -> body_pos (0,0.25,0.25), bary (1,0,0), normal -x
      TV: apex D under the soft triangle      -> body_pos (0,0,1), bary (1/4,1/4,1/2), normal +z
      EE: crossing 0.05 UNDER edge AB          -> body_pos (0.5,0,0), bary (1/2,1/2,0), normal -z
        (EE reference dir: cross((1,0,0),(0,1,0)) = +z, flipped by AB's outward (0,-1,-1)/sqrt(2)
        to -z, which agrees with the closest-point direction diff = (0,0,-0.05).)
    """
    from newton._src.geometry.soft_contacts_bvh import (  # noqa: PLC0415
        BVH_CANDIDATE_EE,
        BVH_CANDIDATE_TV,
        BVH_CANDIDATE_VT,
        emit_bvh_contacts,
    )

    velocity = (1.0, 2.0, 3.0)
    rig = _unit_rigid_tetrahedron(device, velocity=velocity)
    # Soft particles: 0 = VT particle; 1-3 = TV triangle (over the apex); 4-5 = EE endpoints.
    particle_q, _radius, _flags = _soft_particles(
        device,
        [
            [-0.05, 0.25, 0.25],
            [-0.5, -0.5, 1.05],
            [0.5, -0.5, 1.05],
            [0.0, 0.5, 1.05],
            [0.5, -0.3, -0.05],
            [0.5, 0.3, -0.05],
        ],
    )
    tri_indices = wp.array(np.array([[1, 2, 3]], np.int32), dtype=wp.int32, ndim=2, device=device)
    edge_indices = wp.array(np.array([[-1, -1, 4, 5]], np.int32), dtype=wp.int32, ndim=2, device=device)

    mesh_indices = np.asarray(rig["mesh"].indices)
    vertex_row_apex = next(r for r in range(4) if mesh_indices[rig["v_table_np"][r]] == 3)
    edge_row_ab = next(
        r
        for r in range(6)
        if {int(mesh_indices[rig["e_table_np"][r, 0]]), int(mesh_indices[rig["e_table_np"][r, 1]])} == {0, 1}
    )
    cand_np = np.array(
        [
            [int(BVH_CANDIDATE_VT), 0, 0, 2],  # particle 0 vs the x=0 face
            [int(BVH_CANDIDATE_TV), 0, 0, vertex_row_apex],  # soft tri 0 vs the apex D
            [int(BVH_CANDIDATE_EE), 0, 0, edge_row_ab],  # soft edge 0 vs rigid edge AB
        ],
        np.int32,
    )
    count = wp.array([3], dtype=wp.int32, device=device)
    cands = wp.array(cand_np, dtype=wp.vec4i, device=device)

    n_max = 8
    out_count = wp.zeros(1, dtype=wp.int32, device=device)
    tids = wp.full(3, -1, dtype=wp.int32, device=device)
    out_particle = wp.full(n_max, -7, dtype=wp.int32, device=device)
    out_indices = wp.zeros(n_max, dtype=wp.vec3i, device=device)
    out_bary = wp.zeros(n_max, dtype=wp.vec3, device=device)
    out_shape = wp.full(n_max, -7, dtype=wp.int32, device=device)
    out_body_pos = wp.zeros(n_max, dtype=wp.vec3, device=device)
    out_body_vel = wp.zeros(n_max, dtype=wp.vec3, device=device)
    out_normal = wp.zeros(n_max, dtype=wp.vec3, device=device)
    wp.launch(
        emit_bvh_contacts,
        dim=3,
        inputs=[
            count,
            cands,
            3,
            particle_q,
            tri_indices,
            edge_indices,
            rig["body_q"],
            rig["shape_transform"],
            rig["shape_body"],
            rig["shape_scale"],
            rig["shape_source_ptr"],
            rig["vertex_table"],
            rig["vertex_normals"],
            rig["edge_table"],
            rig["edge_outward"],
            1.0e-5,
            0,  # tid_base
            n_max,
        ],
        outputs=[
            out_count,
            tids,
            out_particle,
            out_indices,
            out_bary,
            out_shape,
            out_body_pos,
            out_body_vel,
            out_normal,
        ],
        device=device,
    )
    test.assertEqual(int(out_count.numpy()[0]), 3)
    test.assertEqual(sorted(tids.numpy().tolist()), [0, 1, 2])  # replay memo: one row per thread

    # Records land in racy order; identify each family by its corner signature.
    idx = out_indices.numpy()[:3]
    rows = {"vt": None, "tv": None, "ee": None}
    for i in range(3):
        if idx[i][2] >= 0:
            rows["tv"] = i
        elif idx[i][1] >= 0:
            rows["ee"] = i
        else:
            rows["vt"] = i
    expected = {
        "vt": {
            "particle": 0,
            "corners": [0, -1, -1],
            "bary": [1.0, 0.0, 0.0],
            "body_pos": [0.0, 0.25, 0.25],
            "normal": [-1.0, 0.0, 0.0],
        },
        "tv": {
            "particle": -1,
            "corners": [1, 2, 3],
            "bary": [0.25, 0.25, 0.5],
            "body_pos": [0.0, 0.0, 1.0],
            "normal": [0.0, 0.0, 1.0],
        },
        "ee": {
            "particle": -1,
            "corners": [4, 5, -1],
            "bary": [0.5, 0.5, 0.0],
            "body_pos": [0.5, 0.0, 0.0],
            "normal": [0.0, 0.0, -1.0],
        },
    }
    for fam, exp in expected.items():
        i = rows[fam]
        test.assertIsNotNone(i, f"{fam} record missing")
        test.assertEqual(int(out_particle.numpy()[i]), exp["particle"], fam)
        test.assertTrue(np.all(idx[i] == exp["corners"]), fam)
        test.assertTrue(np.allclose(out_bary.numpy()[i], exp["bary"], atol=1e-6), fam)
        test.assertTrue(np.allclose(out_body_pos.numpy()[i], exp["body_pos"], atol=1e-6), fam)
        test.assertEqual(int(out_shape.numpy()[i]), 0, fam)
        test.assertTrue(np.allclose(out_normal.numpy()[i], exp["normal"], atol=1e-6), fam)
        # Uniform surface velocity reaches body_vel through all three interpolation paths.
        test.assertTrue(np.allclose(out_body_vel.numpy()[i], velocity, atol=1e-6), fam)


add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_kernel_feature_tables",
    test_bvh_kernel_feature_tables,
    devices=soft_devices,
)
add_function_test(
    TestBvhFullSurfaceSoftContact, "test_bvh_kernel_detect_vt", test_bvh_kernel_detect_vt, devices=soft_devices
)
add_function_test(
    TestBvhFullSurfaceSoftContact, "test_bvh_kernel_detect_tv", test_bvh_kernel_detect_tv, devices=soft_devices
)
add_function_test(
    TestBvhFullSurfaceSoftContact, "test_bvh_kernel_detect_ee", test_bvh_kernel_detect_ee, devices=soft_devices
)
add_function_test(TestBvhFullSurfaceSoftContact, "test_bvh_kernel_emit", test_bvh_kernel_emit, devices=soft_devices)

# ---------------------------------------------------------------------------
# Integration: detection results
# Pipeline-level record correctness against independent references (parity, regressions, normals, velocities, cross-stack consistency).
# ---------------------------------------------------------------------------


def test_bvh_vertex_over_rigid_face(test, device):
    """A soft vertex over a rigid face interior: the VT record matches the legacy record exactly.

    Completes the feature-pair trio (vertex/face, face/vertex, edge/edge). Unlike the other two,
    this configuration IS visible to the per-particle path -- the assertion is agreement: the BVH
    back-end replaces the legacy closest-point record for mesh shapes, so its VT record must
    reproduce the legacy closest point and normal, losing nothing on the covered family.
    """
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.2, 0.2, 0.1))
    builder.add_particle(pos=wp.vec3(0.05, 0.0, 0.105), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
    model = builder.finalize(device=device)

    _, contacts = _run_bvh_pipeline(model, gap=0.01)
    n = int(contacts.soft_contact_count.numpy()[0])
    test.assertEqual(n, 1, "one particle over one face interior: exactly one VT record")
    idx = contacts.soft_contact_indices.numpy()[0]
    test.assertTrue(np.all(idx == [0, -1, -1]))
    test.assertEqual(int(contacts.soft_contact_particle.numpy()[0]), 0, "VT keeps the particle-only view")
    bvh_pos = contacts.soft_contact_body_pos.numpy()[0]
    bvh_normal = contacts.soft_contact_normal.numpy()[0]
    test.assertTrue(np.allclose(bvh_pos, [0.05, 0.0, 0.1], atol=1e-5), "closest point is the face foot")
    test.assertTrue(np.allclose(bvh_normal, [0.0, 0.0, 1.0], atol=1e-5), "normal is the face normal")

    # The legacy per-particle path sees this family too and must agree record-for-record.
    legacy = newton.CollisionPipeline(model, soft_contact_gap=0.01)
    c2 = legacy.contacts()
    legacy.collide(model.state(), c2)
    test.assertEqual(int(c2.soft_contact_count.numpy()[0]), 1)
    test.assertTrue(np.allclose(c2.soft_contact_body_pos.numpy()[0], bvh_pos, atol=1e-5))
    test.assertTrue(np.allclose(c2.soft_contact_normal.numpy()[0], bvh_normal, atol=1e-5))


def test_bvh_edge_across_rigid_edge(test, device):
    """A soft edge crossing a rigid box edge with all soft vertices clear: EE emits, per-particle misses.

    The soft segment dips within the query radius of the box's top edge only near the crossing;
    both endpoints are farther from the whole box than the radius, so the per-particle path (and
    the VT pass) report nothing -- exactly the tunneling family the EE query exists for.
    """
    builder = newton.ModelBuilder()
    box = newton.Mesh.create_box(0.2, 0.2, 0.1)
    builder.add_shape_mesh(-1, mesh=box)
    # Quad strip so finalize creates soft edges; the tested edge runs (0, 0.3, 0.05) -> (0, 0.1, 0.16),
    # passing ~5 mm over the rigid edge {(x, 0.2, 0.1)} at its midpoint.
    v0, v1 = np.array([0.0, 0.3, 0.05]), np.array([0.0, 0.1, 0.16])
    side = np.array([0.3, 0.0, 0.0])
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(*v0), wp.vec3(*v1), wp.vec3(*(v0 + side)), wp.vec3(*(v1 + side))],
        indices=[0, 1, 2, 2, 1, 3],
        density=1.0,
        particle_radius=0.0,
    )
    model = builder.finalize(device=device)
    test.assertGreater(model.edge_count, 0)

    _, contacts = _run_bvh_pipeline(model, gap=0.01)
    total = int(contacts.soft_contact_count.numpy()[0])
    idx = contacts.soft_contact_indices.numpy()[:total]
    ee = idx[(idx[:, 1] >= 0) & (idx[:, 2] < 0)]
    test.assertGreater(len(ee), 0, "EE pass must catch the edge-over-edge crossing")
    test.assertTrue(np.any((ee == [0, 1, -1]).all(axis=1) | (ee == [1, 0, -1]).all(axis=1)))
    # All soft vertices are clear of the box: no particle (VT) records at all.
    test.assertEqual(int(((idx[:total, 1] < 0) & (idx[:total, 0] >= 0)).sum()), 0)

    # The per-particle path (flag off) misses the crossing entirely.
    flat = newton.CollisionPipeline(model, soft_contact_gap=0.01)
    c2 = flat.contacts()
    flat.collide(model.state(), c2)
    test.assertEqual(int(c2.soft_contact_count.numpy()[0]), 0)


def test_bvh_face_over_rigid_vertex(test, device):
    """A soft face hovering over a rigid box corner with all its vertices clear: TV emits.

    The rigid corner (0.2, 0.2, 0.1) sits ~5 mm under the triangle's interior (its centroid, on a
    slightly slanted plane) while every triangle vertex is clear of the box: two hang laterally
    off the footprint, the third floats 15 mm above the top face -- the face-over-vertex tunneling
    family, invisible to both the per-particle path and the VT pass.
    """
    builder = newton.ModelBuilder()
    box = newton.Mesh.create_box(0.2, 0.2, 0.1)
    builder.add_shape_mesh(-1, mesh=box)
    for p in [(0.05, 0.05, 0.115), (0.35, 0.2, 0.1), (0.2, 0.35, 0.1)]:
        builder.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
    builder.add_triangle(0, 1, 2)
    model = builder.finalize(device=device)

    _, contacts = _run_bvh_pipeline(model, gap=0.01)
    total = int(contacts.soft_contact_count.numpy()[0])
    idx = contacts.soft_contact_indices.numpy()[:total]
    tv = idx[idx[:, 2] >= 0]
    test.assertGreater(len(tv), 0, "TV pass must catch the face-over-vertex configuration")
    particle_view = contacts.soft_contact_particle.numpy()[:total][idx[:, 2] >= 0]
    test.assertTrue(np.all(particle_view == -1), "face records carry no particle-only view")
    body_pos = contacts.soft_contact_body_pos.numpy()[:total][idx[:, 2] >= 0]
    test.assertTrue(
        np.any(np.linalg.norm(body_pos - np.array([0.2, 0.2, 0.1]), axis=1) < 1e-5),
        "the record's rigid point must be the box corner",
    )
    test.assertEqual(int(((idx[:, 1] < 0) & (idx[:, 0] >= 0)).sum()), 0, "no VT records: vertices are clear")

    flat = newton.CollisionPipeline(model, soft_contact_gap=0.01)
    c2 = flat.contacts()
    flat.collide(model.state(), c2)
    test.assertEqual(int(c2.soft_contact_count.numpy()[0]), 0)


def test_bvh_brute_force_parity(test, device):
    """BVH kernels report the identical (feature pair, distance) sets as an O(n*m) reference."""
    gap = 0.01
    model, box = _build_cloth_over_mesh_box(device)
    pipeline, contacts = _run_bvh_pipeline(model, gap=gap)
    vt_act, ee_act, tv_act = _split_bvh_families(model, contacts, _assert_bvh_only_records(test, pipeline))
    test.assertGreater(len(vt_act), 0)
    test.assertGreater(len(ee_act), 0)
    test.assertGreater(len(tv_act), 0)

    vt_exp, ee_exp, tv_exp = _bvh_brute_force_reference(model, box, gap)
    radius = model.particle_radius.numpy()
    threshold = gap + float(radius.max())
    _assert_family_matches(test, [(k, float(np.linalg.norm(xs - xr))) for k, xs, xr in vt_act], vt_exp, threshold)
    _assert_family_matches(test, [(k, float(np.linalg.norm(xs - xr))) for k, xs, xr in ee_act], ee_exp, threshold)
    _assert_family_matches(test, [(k, float(np.linalg.norm(xs - xr))) for k, xs, xr in tv_act], tv_exp, threshold)


def test_bvh_brute_force_parity_multi_world(test, device):
    """Two spatially-overlapping worlds: records never cross worlds and match the filtered reference."""
    gap = 0.01

    def _sub():
        b = newton.ModelBuilder()
        _build_cloth_over_mesh_box(None, world_builder=b)
        return b

    builder = newton.ModelBuilder()
    builder.add_world(_sub())
    builder.add_world(_sub())
    model = builder.finalize(device=device)
    box = newton.Mesh.create_box(0.25, 0.25, 0.1)

    pipeline, contacts = _run_bvh_pipeline(model, gap=gap)
    vt_act, ee_act, tv_act = _split_bvh_families(model, contacts, _assert_bvh_only_records(test, pipeline))

    # No cross-world records: every soft feature's world matches its shape's world.
    total = int(contacts.soft_contact_count.numpy()[0])
    idx = contacts.soft_contact_indices.numpy()[:total]
    shape = contacts.soft_contact_shape.numpy()[:total]
    pw = model.particle_world.numpy()
    sw = model.shape_world.numpy()
    for i in range(total):
        p_world, s_world = int(pw[idx[i, 0]]), int(sw[shape[i]])
        test.assertTrue(p_world == -1 or s_world == -1 or p_world == s_world)

    # The worlds overlap in space, so unfiltered detection would double every pair set.
    vt_exp, ee_exp, tv_exp = _bvh_brute_force_reference(model, box, gap)
    threshold = _parity_threshold(model, gap)
    _assert_family_matches(test, [(k, float(np.linalg.norm(xs - xr))) for k, xs, xr in vt_act], vt_exp, threshold)
    _assert_family_matches(test, [(k, float(np.linalg.norm(xs - xr))) for k, xs, xr in ee_act], ee_exp, threshold)
    _assert_family_matches(test, [(k, float(np.linalg.norm(xs - xr))) for k, xs, xr in tv_act], tv_exp, threshold)


def test_bvh_brute_force_parity_transformed(test, device):
    """Parity holds for a rotated, translated, non-uniformly scaled and a mirrored mesh shape."""
    family_totals = [0, 0, 0]
    for xform, scale in (
        (
            wp.transform(wp.vec3(0.3, -0.2, 0.05), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.15)),
            (2.0, 0.5, 1.5),
        ),
        (wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), (-1.0, 1.0, 1.0)),
    ):
        gap = 0.02
        builder = newton.ModelBuilder()
        box = newton.Mesh.create_box(0.25, 0.25, 0.1)
        builder.add_shape_mesh(-1, xform=xform, mesh=box, scale=wp.vec3(*scale))
        # Cloth plane near the transformed box's top. Deliberately NOT rotation-corrected: in
        # the rotated non-uniform case the plane cuts through the box, so several particles sit
        # INSIDE it -- the only coverage of the penetrated (sign-check-failed) VT branch.
        top_z = float(xform.p[2]) + 0.1 * abs(scale[2])
        builder.add_cloth_grid(
            pos=wp.vec3(float(xform.p[0]) - 0.3, float(xform.p[1]) - 0.3, top_z + 0.005),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=6,
            dim_y=6,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            particle_radius=0.01,
        )
        model = builder.finalize(device=device)
        pipeline, contacts = _run_bvh_pipeline(model, gap=gap)
        vt_act, ee_act, tv_act = _split_bvh_families(model, contacts, _assert_bvh_only_records(test, pipeline))
        for f, act in enumerate((vt_act, ee_act, tv_act)):
            family_totals[f] += len(act)

        vt_exp, ee_exp, tv_exp = _bvh_brute_force_reference(model, box, gap)
        threshold = _parity_threshold(model, gap)
        for act, exp in ((vt_act, vt_exp), (ee_act, ee_exp), (tv_act, tv_exp)):
            _assert_family_matches(
                test, [(k, float(np.linalg.norm(xs - xr))) for k, xs, xr in act], exp, threshold, tol=2e-4
            )
    # Every family must fire somewhere across the two cases (a silently-empty family would make
    # its parity check vacuous; the rotated case legitimately has zero TV records).
    for f, total in enumerate(family_totals):
        test.assertGreater(total, 0, f"family {('VT', 'EE', 'TV')[f]} never fired across the cases")


def test_bvh_contact_normals(test, device):
    """Emitted normals: unit length; closest-point direction on the primary branch, and the
    rigid feature's outward normal on the sign-check-failed (penetrated) fallback branch."""
    gap = 0.01

    def _records(model):
        _, contacts = _run_bvh_pipeline(model, gap=gap)
        n = int(contacts.soft_contact_count.numpy()[0])
        idx = contacts.soft_contact_indices.numpy()[:n]
        normals = contacts.soft_contact_normal.numpy()[:n]
        test.assertTrue(np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5))
        return idx, normals

    # VT: a particle over the top face interior -> normal is the face normal (+z).
    b = newton.ModelBuilder()
    b.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.2, 0.2, 0.1))
    b.add_particle(pos=wp.vec3(0.05, 0.0, 0.105), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
    idx, normals = _records(b.finalize(device=device))
    test.assertGreater(len(idx), 0)
    test.assertTrue(np.all(normals[:, 2] > 0.99), "VT normal over a face interior must be the face normal")

    # TV: soft face over the box corner -> normal points up out of the corner, along the face plane
    # normal (the corner sits under the triangle interior).
    b = newton.ModelBuilder()
    b.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.2, 0.2, 0.1))
    for p in [(0.05, 0.05, 0.115), (0.35, 0.2, 0.1), (0.2, 0.35, 0.1)]:
        b.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
    b.add_triangle(0, 1, 2)
    idx, normals = _records(b.finalize(device=device))
    tv_mask = idx[:, 2] >= 0
    test.assertGreater(int(tv_mask.sum()), 0)
    test.assertTrue(np.all(normals[tv_mask][:, 2] > 0.9), "TV normal at the corner must point up to the face")

    # EE: soft edge crossing the rigid top edge from above -> normal is ~+z at the crossing.
    b = newton.ModelBuilder()
    b.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.2, 0.2, 0.1))
    v0, v1 = np.array([0.0, 0.3, 0.05]), np.array([0.0, 0.1, 0.16])
    side = np.array([0.3, 0.0, 0.0])
    b.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(*v0), wp.vec3(*v1), wp.vec3(*(v0 + side)), wp.vec3(*(v1 + side))],
        indices=[0, 1, 2, 2, 1, 3],
        density=1.0,
        particle_radius=0.0,
    )
    idx, normals = _records(b.finalize(device=device))
    ee01 = ((idx[:, 0] == 0) & (idx[:, 1] == 1)) | ((idx[:, 0] == 1) & (idx[:, 1] == 0))
    ee01 &= idx[:, 2] < 0
    test.assertGreater(int(ee01.sum()), 0)
    # Two skew edges separate along their mutual perpendicular: cross(e_rigid, e_soft), oriented
    # away from the box (positive z here).
    expected = np.cross([1.0, 0.0, 0.0], v1 - v0)
    expected /= np.linalg.norm(expected)
    if expected[2] < 0.0:
        expected = -expected
    test.assertTrue(
        np.all(normals[ee01] @ expected > 0.999),
        f"EE normal must be the oriented mutual perpendicular {np.round(expected, 3)}",
    )

    # Fallback branch: a particle behind the +x side face's plane (resting on the top face near
    # the edge) gets that side record's SIGN-CHECK-FAILED normal -- the rigid face normal (+x),
    # by contract, not the closest-point direction (which points -x-ish).
    b = newton.ModelBuilder()
    b.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.2, 0.2, 0.1))
    b.add_particle(pos=wp.vec3(0.198, 0.0, 0.105), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
    idx, normals = _records(b.finalize(device=device))
    test.assertEqual(len(idx), 2)  # top-face record + side-face record from one VT thread
    side = int(np.argmax(normals[:, 0]))
    test.assertTrue(np.allclose(normals[side], [1.0, 0.0, 0.0], atol=1e-5), "fallback emits the face normal")
    test.assertTrue(np.allclose(normals[1 - side], [0.0, 0.0, 1.0], atol=1e-5))


def test_bvh_normals_under_scale(test, device):
    """Face normals stay outward under non-uniform and mirrored shape scale."""
    for scale in ((2.0, 0.5, 1.5), (-1.0, 1.0, 1.0)):
        builder = newton.ModelBuilder()
        builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.2, 0.2, 0.1), scale=wp.vec3(*scale))
        top = 0.1 * abs(scale[2])
        builder.add_particle(
            pos=wp.vec3(0.05 * abs(scale[0]), 0.0, top + 0.005), vel=wp.vec3(0.0), mass=0.1, radius=0.0
        )
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(
            model, soft_contact_gap=0.01, enable_rigid_soft_full_surface_contact=True, full_surface_mesh_backend="bvh"
        )
        contacts = pipeline.contacts()
        state = model.state()
        pipeline.collide(state, contacts)  # particle-only soft side: VT needs no soft BVH refit
        n = int(contacts.soft_contact_count.numpy()[0])
        test.assertGreater(n, 0)
        normals = contacts.soft_contact_normal.numpy()[:n]
        test.assertTrue(np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5))
        test.assertTrue(np.all(normals[:, 2] > 0.99), f"top-face normal must stay +z under scale {scale}")


def test_bvh_body_attached_mesh_and_surface_velocity(test, device):
    """A body-attached mesh collides at its body pose, and mesh surface velocities reach body_vel."""
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_mesh(body, mesh=newton.Mesh.create_box(0.2, 0.2, 0.1))
    builder.add_particle(pos=wp.vec3(0.55, 0.0, 0.105), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
    model = builder.finalize(device=device)

    # Give the mesh a uniform nonzero surface velocity (finalize zero-fills it).
    wp_mesh = model.shape_source[0].mesh
    vel = np.tile(np.array([1.0, 2.0, 3.0], np.float32), (len(wp_mesh.points), 1))
    wp_mesh.velocities.assign(vel)

    pipeline = newton.CollisionPipeline(
        model, soft_contact_gap=0.01, enable_rigid_soft_full_surface_contact=True, full_surface_mesh_backend="bvh"
    )
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.collide(state, contacts)  # particle-only soft side: VT needs no soft BVH refit
    n = int(contacts.soft_contact_count.numpy()[0])
    test.assertGreater(n, 0, "particle sits 5 mm over the body-posed box top")

    # body_pos is body-local: mapping through body_q must land on the box surface under the particle.
    body_q = state.body_q.numpy()[0]
    body_pos = contacts.soft_contact_body_pos.numpy()[:n]
    world = _np_quat_rotate(body_q[3:], body_pos) + body_q[:3]
    test.assertTrue(np.allclose(world[:, 2], 0.1, atol=1e-5), "contact points must lie on the top face")
    # body_vel carries the mesh surface velocity (body-local; identity rotation here).
    body_vel = contacts.soft_contact_body_vel.numpy()[:n]
    test.assertTrue(np.allclose(body_vel, [1.0, 2.0, 3.0], atol=1e-5))


def test_bvh_rigid_vs_soft_consistency(test, device):
    """Rigid-soft BVH queries match the soft self-contact detector on an identical kinematic mesh.

    Scene A: cloth over a *rigid* mesh box (BVH back-end). Scene B: the same cloth over the same
    box added as a *soft* kinematic mesh, detected by the self-contact detector. Per-feature
    contact counts and minimum distances must agree -- one geometric ground truth for both stacks.
    """
    gap = 0.02
    box = newton.Mesh.create_box(0.25, 0.25, 0.1, duplicate_vertices=False)
    n_box_verts = len(box.vertices)

    def _add_cloth(b):
        # Spans past the box footprint so cloth edges cross above the box's top edges (EE) and
        # the interior covers the corners (TV). Cell size 0.15 keeps every non-adjacent
        # cloth-cloth feature pair beyond the 0.02 query radius, so the self-contact detector in
        # scene B reports only cloth<->box pairs.
        b.add_cloth_grid(
            pos=wp.vec3(-0.3, -0.3, 0.105),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=4,
            dim_y=4,
            cell_x=0.15,
            cell_y=0.15,
            mass=0.1,
            particle_radius=0.0,
        )

    builder_a = newton.ModelBuilder()
    builder_a.add_shape_mesh(-1, mesh=box)
    _add_cloth(builder_a)
    model_a = builder_a.finalize(device=device)

    builder_b = newton.ModelBuilder()
    # Box first: its particles occupy [0, n_box_verts) so cloth indices are offset uniformly.
    builder_b.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(*v) for v in box.vertices],
        indices=box.indices.tolist(),
        density=1.0,
        particle_radius=0.0,
    )
    _add_cloth(builder_b)
    model_b = builder_b.finalize(device=device)

    # Scene A detection.
    pipe_a, contacts_a = _run_bvh_pipeline(model_a, gap=gap)
    vt_a, ee_a, tv_a = _split_bvh_families(model_a, contacts_a, _assert_bvh_only_records(test, pipe_a))

    # Scene B detection: pure self-contact at the same query radius.
    pipe_b = newton.CollisionPipeline(model_b)
    pipe_b.init_soft_self_contact(margin=gap, gap=0.0, topological_filter_threshold=0)
    contacts_b = pipe_b.contacts()
    state_b = model_b.state()
    pipe_b.refit_soft_contact_bvh(state_b)
    pipe_b.collide(state_b, contacts_b, soft_self_contact=True)
    wp.synchronize_device(wp.get_device(model_b.device))
    data = contacts_b.soft_self_contact_data

    # VT consistency: per cloth particle, count of rigid faces in range == count of box triangles
    # the detector reports for the corresponding scene-B vertex, and min distances agree.
    q_a = model_a.particle_q.numpy()
    counts_b = data.vertex_colliding_triangles_count.numpy()
    min_dist_b = data.vertex_colliding_triangles_min_dist.numpy()
    test.assertGreater(len(vt_a), 0, "scene drifted: no VT records to compare")
    vt_counts_a, vt_min_a = {}, {}
    for p, xs, xr in vt_a:
        vt_counts_a[p] = vt_counts_a.get(p, 0) + 1
        d = float(np.linalg.norm(xs - xr))
        vt_min_a[p] = min(vt_min_a.get(p, np.inf), d)
    for p in range(model_a.particle_count):
        vb = p + n_box_verts  # the same cloth vertex in scene B
        test.assertEqual(vt_counts_a.get(p, 0), int(counts_b[vb]), f"VT count mismatch at particle {p}")
        if vt_counts_a.get(p, 0):
            test.assertAlmostEqual(vt_min_a[p], float(min_dist_b[vb]), places=4)

    # EE consistency: scene-A records keyed by cloth-edge endpoint geometry must equal the
    # detector's box-edge -> cloth-edge pairs (extracted from the box side of scene B).
    def _edge_key(x0, x1):
        k0 = tuple(np.round(x0 * 1e6).astype(np.int64))
        k1 = tuple(np.round(x1 * 1e6).astype(np.int64))
        return (min(k0, k1), max(k0, k1))

    ee_keys_a = Counter()
    ee_dists_a = {}
    for (sv0, sv1), xs, xr in ee_a:
        key = _edge_key(q_a[sv0], q_a[sv1])
        ee_keys_a[key] += 1
        ee_dists_a.setdefault(key, []).append(float(np.linalg.norm(xs - xr)))

    edges_b = model_b.edge_indices.numpy()[:, 2:4]
    q_b = model_b.particle_q.numpy()
    pairs = data.edge_colliding_edges.numpy().reshape(-1, 2)
    box_edge_ids = {e for e in range(len(edges_b)) if (edges_b[e] < n_box_verts).all()}
    ee_keys_b = Counter()
    ee_dists_b = {}
    for e0, e1 in pairs:
        if e0 < 0 or e1 < 0:
            continue
        if int(e0) in box_edge_ids and int(e1) not in box_edge_ids:
            key = _edge_key(q_b[edges_b[e1, 0]], q_b[edges_b[e1, 1]])
            ee_keys_b[key] += 1
            # Pair distance recomputed independently (the detector stores only per-edge minima).
            _, _, d = _np_seg_seg_closest(
                q_b[edges_b[e0, 0]], q_b[edges_b[e0, 1]], q_b[edges_b[e1, 0]], q_b[edges_b[e1, 1]]
            )
            ee_dists_b.setdefault(key, []).append(d)
    test.assertGreater(sum(ee_keys_a.values()), 0)
    test.assertEqual(ee_keys_a, ee_keys_b)
    for key, dists_a in ee_dists_a.items():
        test.assertTrue(
            np.allclose(sorted(dists_a), sorted(ee_dists_b[key]), atol=1e-4),
            f"EE pair distances disagree for cloth edge {key}",
        )

    # TV consistency, per rigid vertex: scene-A face records name their rigid vertex through
    # body_pos, so group by it and compare against the detector's per-vertex counts (a total-only
    # comparison would let over/under-reports on different vertices cancel).
    test.assertGreater(len(tv_a), 0, "scene drifted: no TV records to compare")

    def _pos_key(x):
        return tuple(np.round(np.asarray(x) * 1e6).astype(np.int64))

    tv_counts_a = Counter(_pos_key(xr) for _corners, _xs, xr in tv_a)
    tv_counts_b = Counter()
    for v in range(n_box_verts):
        if counts_b[v]:
            tv_counts_b[_pos_key(q_b[v])] += int(counts_b[v])
    test.assertEqual(tv_counts_a, tv_counts_b)


def test_bvh_vs_sdf_backend_coverage(test, device):
    """'bvh' and 'sdf' back-ends agree on which soft particles are in contact (same provisioned scene)."""
    gap = 0.01
    model, _box = _build_cloth_over_mesh_box(device, provision_sdf=True)

    _, c_bvh = _run_bvh_pipeline(model, gap=gap)
    p_sdf, c_sdf = _run_bvh_pipeline(model, gap=gap, full_surface_mesh_backend="sdf")

    n_bvh = int(c_bvh.soft_contact_count.numpy()[0])
    idx_bvh = c_bvh.soft_contact_indices.numpy()[:n_bvh]
    # BVH: particles named by VT records; SDF: by the legacy particle records (first pair_count).
    bvh_particles = set(idx_bvh[(idx_bvh[:, 1] < 0) & (idx_bvh[:, 0] >= 0), 0].tolist())
    n_sdf = int(c_sdf.soft_contact_count.numpy()[0])
    idx_sdf = c_sdf.soft_contact_indices.numpy()[:n_sdf]
    sdf_particles = set(idx_sdf[(idx_sdf[:, 1] < 0) & (idx_sdf[:, 0] >= 0), 0].tolist())
    test.assertGreater(len(bvh_particles), 0)
    test.assertEqual(bvh_particles, sdf_particles)
    # Both back-ends produce full-surface (edge/face) coverage beyond the particle records.
    test.assertGreater(int((idx_bvh[:, 1] >= 0).sum()), 0)
    test.assertGreater(int((idx_sdf[:, 1] >= 0).sum()), 0)
    test.assertGreater(p_sdf.soft_contact_pair_count, 0)  # legacy pairs kept under 'sdf'


add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_vertex_over_rigid_face",
    test_bvh_vertex_over_rigid_face,
    devices=soft_devices,
)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_edge_across_rigid_edge",
    test_bvh_edge_across_rigid_edge,
    devices=soft_devices,
)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_face_over_rigid_vertex",
    test_bvh_face_over_rigid_vertex,
    devices=soft_devices,
)
add_function_test(
    TestBvhFullSurfaceSoftContact, "test_bvh_brute_force_parity", test_bvh_brute_force_parity, devices=soft_devices
)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_brute_force_parity_multi_world",
    test_bvh_brute_force_parity_multi_world,
    devices=soft_devices,
)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_brute_force_parity_transformed",
    test_bvh_brute_force_parity_transformed,
    devices=soft_devices,
)
add_function_test(
    TestBvhFullSurfaceSoftContact, "test_bvh_contact_normals", test_bvh_contact_normals, devices=soft_devices
)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_normals_under_scale",
    test_bvh_normals_under_scale,
    devices=soft_devices,
)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_body_attached_mesh_and_surface_velocity",
    test_bvh_body_attached_mesh_and_surface_velocity,
    devices=soft_devices,
)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_rigid_vs_soft_consistency",
    test_bvh_rigid_vs_soft_consistency,
    devices=soft_devices,
)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_vs_sdf_backend_coverage",
    test_bvh_vs_sdf_backend_coverage,
    devices=get_cuda_test_devices(),  # SDF provisioning requires CUDA texture SDFs
)

# ---------------------------------------------------------------------------
# Integration: gradients and the shared record stream
# Differentiable-replay correctness under multi-emission and nonzero legacy/SDF stream offsets.
# ---------------------------------------------------------------------------


def _emit_records_with_tape(device, rig, particles, tri_rows, edge_rows, candidate):
    """Launch emit_bvh_contacts on one hand-written candidate under a wp.Tape.

    Returns ``(tape, particle_q, out_bary, out_normal, out_body_pos, out_body_vel)`` with
    gradients enabled on the particle positions and on the outputs under test.
    """
    from newton._src.geometry.soft_contacts_bvh import emit_bvh_contacts  # noqa: PLC0415

    particle_q = wp.array(np.asarray(particles, np.float32), dtype=wp.vec3, requires_grad=True, device=device)
    tri_indices = wp.array(np.asarray(tri_rows, np.int32), dtype=wp.int32, ndim=2, device=device)
    edge_indices = wp.array(np.asarray(edge_rows, np.int32), dtype=wp.int32, ndim=2, device=device)
    count = wp.array([1], dtype=wp.int32, device=device)
    cands = wp.array(np.asarray([candidate], np.int32), dtype=wp.vec4i, device=device)
    out_count = wp.zeros(1, dtype=wp.int32, device=device)
    tids = wp.full(1, -1, dtype=wp.int32, device=device)
    out_particle = wp.zeros(1, dtype=wp.int32, device=device)
    out_indices = wp.zeros(1, dtype=wp.vec3i, device=device)
    out_bary = wp.zeros(1, dtype=wp.vec3, requires_grad=True, device=device)
    out_shape = wp.zeros(1, dtype=wp.int32, device=device)
    out_body_pos = wp.zeros(1, dtype=wp.vec3, requires_grad=True, device=device)
    out_body_vel = wp.zeros(1, dtype=wp.vec3, requires_grad=True, device=device)
    out_normal = wp.zeros(1, dtype=wp.vec3, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(
            emit_bvh_contacts,
            dim=1,
            inputs=[
                count,
                cands,
                1,
                particle_q,
                tri_indices,
                edge_indices,
                rig["body_q"],
                rig["shape_transform"],
                rig["shape_body"],
                rig["shape_scale"],
                rig["shape_source_ptr"],
                rig["vertex_table"],
                rig["vertex_normals"],
                rig["edge_table"],
                rig["edge_outward"],
                1.0e-5,
                0,  # tid_base
                1,
            ],
            outputs=[
                out_count,
                tids,
                out_particle,
                out_indices,
                out_bary,
                out_shape,
                out_body_pos,
                out_body_vel,
                out_normal,
            ],
            device=device,
        )
    return tape, particle_q, out_bary, out_normal, out_body_pos, out_body_vel


def _seeded_grad(test, tape, particle_q, out_array, seed, device):
    """Backward with ``seed`` on ``out_array``; returns particle gradients (tape zeroed first)."""
    tape.zero()
    tape.backward(grads={out_array: wp.array(np.asarray([seed], np.float32), dtype=wp.vec3, device=device)})
    return particle_q.grad.numpy().copy()


def test_bvh_kernel_gradients_wrt_particle_pos(test, device):
    """Gradients of every emitted output w.r.t. the soft particle positions.

    All expected values are hand-derived (and cross-checked by finite differences offline), with
    the nonzero blocks placed strictly inside smooth regions (no feature-region or branch
    switches). Blocks are asserted in full, so their zero entries double as leakage canaries.
    EE is the only family whose body-side outputs depend on the soft feature, so its scene hosts
    the nonzero body_pos/body_vel blocks; the VT counterparts live in the multi-emission and
    body-transform tests.

    VT (edge-feature closest point): particle p = (0.5, -0.03, -0.04) near tet edge AB.
      normal: cp = (p_x, 0, 0), diff = (0, p_y, p_z), d = 0.05, n = (0, -0.6, -0.8); the normal
      rotates with p: J = (I - n n^T)/d on the (y, z) block = [[12.8, -9.6], [-9.6, 7.2]].
      bary is the literal (1, 0, 0) with no adjoint path at all; its all-zero gradient is an
      out-of-graph canary against a future edit wiring it to inputs without gradient coverage.

    TV, penetrated (interior projection): tet translated so vertex A sits at (1/4, 1/4, -1/20)
      under the soft triangle (0,0,0), (1,0,0), (0,1,0):
      bary = (1/2, 1/4, 1/4); the Gram-system derivative couples corners: d bary1 =
      s0 (-1/2, 0, h), s1 (-1/4, 0, -h), s2 (-1/4, 0, 0) with h = 0.05 (z-terms from the plane
      tilting under corner motion).
      normal: the tet pierces the soft triangle, the sign check fails, and the normal is the
      constant fallback reference -(1,1,1)/sqrt(3): the zero gradient documents that the
      non-smooth branch is off-graph by design.
      body_pos/body_vel are the rigid vertex's own position and velocity -- constants with no
      adjoint path to the particles; their zero assertions are out-of-graph canaries.

    TV, outward side: the soft triangle shifted to start at (-1/4, -1/4, -1/20) puts vertex
      A = origin on its outward side, in the smooth branch: diff = cp - A is exactly the plane
      normal, so n = -normalize(cross(e1, e2)) and only out-of-plane corner motion tilts it.
      Seeding e_i gives grads s0 (0,0,-1), s_i (0,0,+1), third corner 0 -- every in-plane column
      is an in-graph zero.

    EE (perpendicular crossing): soft edge (0.5, -/+0.3, -0.05) under rigid edge AB, with
      distinct per-vertex mesh velocities (a uniform field has vB - vA = 0 and would hide a
      broken interpolation adjoint); all expectations are endpoint-order invariant.
      bary: t = 1/2, dt/d sv = (0, -5/6, -/+5/36) (the z-term from the common perpendicular
      rotating as the soft edge tilts).
      normal: diff stays the common perpendicular, n = normalize(diff) = (0, 0, -1); tilting an
      endpoint out of plane swings n_y at rate 1/|e_s| = 1/0.6: seed (0,1,0) gives grads
      s0 (0, 0, -5/3), s1 (0, 0, 5/3); seed (1,0,0) is an in-graph zero block (n_x is stationary
      under every endpoint motion). Composes the closest_point_edge_edge (s, t) adjoints with
      the normalize adjoint.
      body_pos = (s, 0, 0) and body_vel = (1-s) vA + s vB interpolate along the rigid edge, and
      the station s moves with the soft edge: shifting either endpoint by dx moves the t = 1/2
      crossing by dx/2, while y and z motions change t and the separation but not s:
      d s / d endpoint = (1/2, 0, 0) for BOTH endpoints, and seeding body_vel by (1,1,1) gives
      ((vB - vA) . seed) * (1/2, 0, 0) = (6, 0, 0) per endpoint.
    """
    from newton._src.geometry.soft_contacts_bvh import (  # noqa: PLC0415
        BVH_CANDIDATE_EE,
        BVH_CANDIDATE_TV,
        BVH_CANDIDATE_VT,
    )

    # --- VT: normal gradient at an edge-feature closest point ---
    rig = _unit_rigid_tetrahedron(device)
    tape, q, bary, normal, _body_pos, _body_vel = _emit_records_with_tape(
        device, rig, [[0.5, -0.03, -0.04]], [[0, 0, 0]], [[-1, -1, 0, 0]], [int(BVH_CANDIDATE_VT), 0, 0, 0]
    )
    test.assertTrue(np.allclose(normal.numpy()[0], [0.0, -0.6, -0.8], atol=1e-6))
    grad = _seeded_grad(test, tape, q, normal, (0.0, 1.0, 0.0), device)
    test.assertTrue(np.allclose(grad[0], [0.0, 12.8, -9.6], atol=1e-3), f"got {grad[0]}")
    grad = _seeded_grad(test, tape, q, normal, (0.0, 0.0, 1.0), device)
    test.assertTrue(np.allclose(grad[0], [0.0, -9.6, 7.2], atol=1e-3), f"got {grad[0]}")
    grad = _seeded_grad(test, tape, q, bary, (1.0, 1.0, 1.0), device)  # out-of-graph canary
    test.assertTrue(np.allclose(grad, 0.0, atol=1e-8), f"got {grad}")

    # --- TV, penetrated: bary blocks; fallback normal and body-side constants as canaries ---
    rig = _unit_rigid_tetrahedron(device, velocity=(1.0, 2.0, 3.0))
    rig["shape_transform"] = wp.array(
        [wp.transform(wp.vec3(0.25, 0.25, -0.05), wp.quat_identity())], dtype=wp.transform, device=device
    )
    mesh_indices = np.asarray(rig["mesh"].indices)
    vertex_row_a = next(r for r in range(4) if mesh_indices[rig["v_table_np"][r]] == 0)
    tape, q, bary, normal, body_pos, body_vel = _emit_records_with_tape(
        device,
        rig,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0, 1, 2]],
        [[-1, -1, 0, 1]],
        [int(BVH_CANDIDATE_TV), 0, 0, vertex_row_a],
    )
    test.assertTrue(np.allclose(bary.numpy()[0], [0.5, 0.25, 0.25], atol=1e-6))
    grad = _seeded_grad(test, tape, q, bary, (0.0, 1.0, 0.0), device)
    expected = [[-0.5, 0.0, 0.05], [-0.25, 0.0, -0.05], [-0.25, 0.0, 0.0]]
    test.assertTrue(np.allclose(grad, expected, atol=1e-4), f"got {grad}")
    grad = _seeded_grad(test, tape, q, bary, (1.0, 0.0, 0.0), device)
    expected = [[0.5, 0.5, -0.1], [0.25, 0.25, 0.05], [0.25, 0.25, 0.05]]
    test.assertTrue(np.allclose(grad, expected, atol=1e-4), f"got {grad}")
    # Penetrated: the sign check fails and the normal is the constant fallback reference (vertex
    # A's angle-weighted normal) -- the non-smooth branch carries no gradient by design
    # (out-of-graph zero).
    test.assertTrue(np.allclose(normal.numpy()[0], -np.ones(3) / np.sqrt(3.0), atol=1e-6))
    grad = _seeded_grad(test, tape, q, normal, (1.0, 1.0, 1.0), device)
    test.assertTrue(np.allclose(grad, 0.0, atol=1e-8), f"got {grad}")
    # TV body_pos/body_vel are the rigid vertex's own constants: out-of-graph canaries.
    test.assertTrue(np.allclose(body_pos.numpy()[0], [0.25, 0.25, -0.05], atol=1e-6))
    test.assertTrue(np.allclose(body_vel.numpy()[0], [1.0, 2.0, 3.0], atol=1e-6))
    grad = _seeded_grad(test, tape, q, body_pos, (1.0, 1.0, 1.0), device)
    test.assertTrue(np.allclose(grad, 0.0, atol=1e-8), f"got {grad}")
    grad = _seeded_grad(test, tape, q, body_vel, (1.0, 1.0, 1.0), device)
    test.assertTrue(np.allclose(grad, 0.0, atol=1e-8), f"got {grad}")

    # --- TV: normal gradient with vertex A on the soft triangle's outward side ---
    rig = _unit_rigid_tetrahedron(device)
    tape, q, bary, normal, _body_pos, _body_vel = _emit_records_with_tape(
        device,
        rig,
        [[-0.25, -0.25, -0.05], [0.75, -0.25, -0.05], [-0.25, 0.75, -0.05]],
        [[0, 1, 2]],
        [[-1, -1, 0, 1]],
        [int(BVH_CANDIDATE_TV), 0, 0, vertex_row_a],
    )
    test.assertTrue(np.allclose(bary.numpy()[0], [0.5, 0.25, 0.25], atol=1e-6))
    test.assertTrue(np.allclose(normal.numpy()[0], [0.0, 0.0, -1.0], atol=1e-6))
    grad = _seeded_grad(test, tape, q, normal, (1.0, 0.0, 0.0), device)
    expected = [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    test.assertTrue(np.allclose(grad, expected, atol=1e-4), f"got {grad}")
    grad = _seeded_grad(test, tape, q, normal, (0.0, 1.0, 0.0), device)
    expected = [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    test.assertTrue(np.allclose(grad, expected, atol=1e-4), f"got {grad}")

    # --- EE: bary gradient of the crossing parameter, w.r.t. the soft edge endpoints ---
    rig = _unit_rigid_tetrahedron(device)
    rig["mesh"].mesh.velocities.assign(
        np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], np.float32)
    )
    edge_row_ab = next(
        r
        for r in range(6)
        if {int(mesh_indices[rig["e_table_np"][r, 0]]), int(mesh_indices[rig["e_table_np"][r, 1]])} == {0, 1}
    )
    tape, q, bary, normal, body_pos, body_vel = _emit_records_with_tape(
        device,
        rig,
        [[0.5, -0.3, -0.05], [0.5, 0.3, -0.05]],
        [[0, 1, 1]],
        [[-1, -1, 0, 1]],
        [int(BVH_CANDIDATE_EE), 0, 0, edge_row_ab],
    )
    test.assertTrue(np.allclose(bary.numpy()[0], [0.5, 0.5, 0.0], atol=1e-6))
    grad = _seeded_grad(test, tape, q, bary, (0.0, 1.0, 0.0), device)
    expected = [[0.0, -5.0 / 6.0, -5.0 / 36.0], [0.0, -5.0 / 6.0, 5.0 / 36.0]]
    test.assertTrue(np.allclose(grad, expected, atol=1e-3), f"got {grad}")

    # --- EE: normal gradient through the common perpendicular ---
    test.assertTrue(np.allclose(normal.numpy()[0], [0.0, 0.0, -1.0], atol=1e-6))
    grad = _seeded_grad(test, tape, q, normal, (0.0, 1.0, 0.0), device)
    expected = [[0.0, 0.0, -5.0 / 3.0], [0.0, 0.0, 5.0 / 3.0]]
    test.assertTrue(np.allclose(grad, expected, atol=1e-3), f"got {grad}")
    grad = _seeded_grad(test, tape, q, normal, (1.0, 0.0, 0.0), device)  # in-graph zero block
    test.assertTrue(np.allclose(grad, 0.0, atol=1e-6), f"got {grad}")

    # --- EE: body_pos and body_vel move with the crossing parameter s ---
    test.assertTrue(np.allclose(body_pos.numpy()[0], [0.5, 0.0, 0.0], atol=1e-6))
    test.assertTrue(np.allclose(body_vel.numpy()[0], [1.0, 2.0, 3.0], atol=1e-6))
    grad = _seeded_grad(test, tape, q, body_pos, (1.0, 0.0, 0.0), device)
    expected = [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
    test.assertTrue(np.allclose(grad, expected, atol=1e-4), f"got {grad}")
    grad = _seeded_grad(test, tape, q, body_pos, (0.0, 1.0, 1.0), device)  # in-graph zero rows
    test.assertTrue(np.allclose(grad, 0.0, atol=1e-6), f"got {grad}")
    grad = _seeded_grad(test, tape, q, body_vel, (1.0, 1.0, 1.0), device)  # (vB - vA) . seed = 12
    expected = [[6.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
    test.assertTrue(np.allclose(grad, expected, atol=1e-3), f"got {grad}")


def test_bvh_kernel_gradients_wrt_body_transform(test, device):
    """Gradients of every emitted output w.r.t. the rigid BODY pose (the other differentiable input).

    All other gradient tests use static shapes (body -1), where body_q never enters the math. A
    body-attached tet at translation b with identity rotation, particle p over the bottom face
    interior at u = p - b = (1/4, 1/4, -1/20); per-vertex mesh velocities vA = 0, vB = (2,0,0),
    vC = (0,2,0) make the face velocity field (2x, 2y, 0) in in-plane coordinates (x, y).
    Every output is seeded against every taped input (particle position and body pose):

    body_pos = ((p-b)_x, (p-b)_y, 0), seed (1,1,1), effective seed on the local point s = (1,1,0):
      d/d particle = (1,1,0); d/d translation = (-1,-1,0); d/d q_v = -2 u x s = (-0.1, 0.1, 0);
      d/d w = 4 w (u . s) = 2.0.
    body_vel = (2x, 2y, 0) through the barycentric interpolation adjoint; the effective local
      seed doubles to s = (2,2,0), scaling every body_pos block by 2: d/d particle = (2,2,0),
      d/d translation = (-2,-2,0), d/d q_v = (-0.2, 0.2, 0), d/d w = 4.0. The forward value
      (0.5, 0.5, 0) also pins the barycentric-convention agreement between triangle_closest_point
      and wp.mesh_eval_velocity, which the uniform-velocity emit test cannot distinguish.
    normal = R(q) (0,0,-1), seed (1,0,0):
      d/d particle = 0 and d/d translation = 0 are IN-GRAPH zeros (the face-interior projection
      lets diff change only along the normal, which the normalize adjoint kills) -- leakage
      canaries through live adjoints; d/d q_v = 2 n x s = (0,-2,0); d/d w = 4 w (n . s) = 0.
    bary: the VT branch writes the literal (1,0,0) -- no adjoint path; out-of-graph canary.

    The q_v parts are the convention-independent rotation generators at identity
    (d(R(q)^T u)/d q_v = 2 [u]x for the inverse rotation, d(R(q) v)/d q_v = -2 [v]x for the
    forward one). The w parts are GAUGE components of Warp's raw-quaternion formula
    ((2w^2 - 1) v + ...): they point along the unit-norm constraint and carry no physical
    rotation; consumers project them out, but the raw adjoint must produce them.
    """
    from newton._src.geometry.soft_contacts_bvh import BVH_CANDIDATE_VT  # noqa: PLC0415

    b = np.array([0.1, 0.2, 0.3], np.float32)
    rig = _unit_rigid_tetrahedron(device)
    rig["mesh"].mesh.velocities.assign(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], np.float32)
    )
    rig["shape_body"] = wp.array([0], dtype=wp.int32, device=device)
    body_q = wp.array(
        np.array([[b[0], b[1], b[2], 0.0, 0.0, 0.0, 1.0]], np.float32),
        dtype=wp.transform,
        requires_grad=True,
        device=device,
    )
    rig["body_q"] = body_q

    p_world = b + np.array([0.25, 0.25, -0.05], np.float32)
    tape, q, out_bary, out_normal, out_body_pos, out_body_vel = _emit_records_with_tape(
        device, rig, [p_world], [[0, 0, 0]], [[-1, -1, 0, 0]], [int(BVH_CANDIDATE_VT), 0, 0, 0]
    )
    test.assertTrue(np.allclose(out_body_vel.numpy()[0], [0.5, 0.5, 0.0], atol=1e-6))

    def seeded_grads(out, seed):
        tape.zero()
        tape.backward(grads={out: wp.array(np.asarray([seed], np.float32), dtype=wp.vec3, device=device)})
        return q.grad.numpy()[0].copy(), body_q.grad.numpy()[0].copy()

    grad_particle, grad_body = seeded_grads(out_body_pos, (1.0, 1.0, 1.0))
    test.assertTrue(np.allclose(grad_particle, [1.0, 1.0, 0.0], atol=1e-4), f"got {grad_particle}")
    test.assertTrue(np.allclose(grad_body[:3], [-1.0, -1.0, 0.0], atol=1e-4), f"got {grad_body[:3]}")
    test.assertTrue(np.allclose(grad_body[3:], [-0.1, 0.1, 0.0, 2.0], atol=1e-4), f"got {grad_body[3:]}")

    grad_particle, grad_body = seeded_grads(out_body_vel, (1.0, 1.0, 1.0))
    test.assertTrue(np.allclose(grad_particle, [2.0, 2.0, 0.0], atol=1e-4), f"got {grad_particle}")
    test.assertTrue(np.allclose(grad_body[:3], [-2.0, -2.0, 0.0], atol=1e-4), f"got {grad_body[:3]}")
    test.assertTrue(np.allclose(grad_body[3:], [-0.2, 0.2, 0.0, 4.0], atol=1e-4), f"got {grad_body[3:]}")

    # d/particle and d/translation are IN-GRAPH zeros: the projection and normalize adjoints run
    # and vanish (leakage detectors), unlike the out-of-graph bary canary below.
    grad_particle, grad_body = seeded_grads(out_normal, (1.0, 0.0, 0.0))
    test.assertTrue(np.allclose(grad_particle, 0.0, atol=1e-6), f"got {grad_particle}")
    test.assertTrue(np.allclose(grad_body[:3], 0.0, atol=1e-6), f"got {grad_body[:3]}")
    test.assertTrue(np.allclose(grad_body[3:], [0.0, -2.0, 0.0, 0.0], atol=1e-4), f"got {grad_body[3:]}")

    # VT bary is the literal (1, 0, 0): out-of-graph canary.
    grad_particle, grad_body = seeded_grads(out_bary, (1.0, 1.0, 1.0))
    test.assertTrue(np.allclose(grad_particle, 0.0, atol=1e-8), f"got {grad_particle}")
    test.assertTrue(np.allclose(grad_body, 0.0, atol=1e-8), f"got {grad_body}")


def test_bvh_multi_emission_gradients(test, device):
    """Backward gradients are correct when one BVH query thread produces several records.

    One particle near the box's +x top edge yields two VT records from a single detection thread:
    the top face (closest point straight below, Jacobian diag(1,1,0)) and the +x side face
    (closest point clamped to the top edge running along y, Jacobian diag(0,1,0)). Seeding each
    record's ``body_pos`` adjoint separately must route through its own row -- the exact failure
    mode of a per-thread replay memo shared by multiple emissions.
    """
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.2, 0.2, 0.1))
    builder.add_particle(pos=wp.vec3(0.198, 0.0, 0.105), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
    model = builder.finalize(device=device, requires_grad=True)

    pipeline = newton.CollisionPipeline(
        model, soft_contact_gap=0.01, enable_rigid_soft_full_surface_contact=True, full_surface_mesh_backend="bvh"
    )
    contacts = pipeline.contacts()
    state = model.state()

    tape = wp.Tape()
    with tape:
        pipeline.collide(state, contacts)

    n = int(contacts.soft_contact_count.numpy()[0])
    test.assertEqual(n, 2, "expected exactly the top-face and side-face records")
    body_pos = contacts.soft_contact_body_pos.numpy()[:n]
    # Identify rows by the rigid closest point: on the top face (x < 0.2) vs on its +x edge (x = 0.2).
    top_row = int(np.argmin(body_pos[:, 0]))
    side_row = 1 - top_row
    test.assertTrue(np.allclose(body_pos[top_row], [0.198, 0.0, 0.1], atol=1e-5))
    test.assertTrue(np.allclose(body_pos[side_row], [0.2, 0.0, 0.1], atol=1e-5))

    def _grad_for(rows):
        tape.zero()
        seed = np.zeros((contacts.soft_contact_max, 3), dtype=np.float32)
        for r in rows:
            seed[r] = (1.0, 1.0, 1.0)
        tape.backward(grads={contacts.soft_contact_body_pos: wp.array(seed, dtype=wp.vec3, device=device)})
        return state.particle_q.grad.numpy()[0].copy()

    # Top face: body_pos = (px, py, 0.1) -> d body_pos / d particle = diag(1, 1, 0).
    test.assertTrue(np.allclose(_grad_for([top_row]), [1.0, 1.0, 0.0], atol=1e-5))
    # Side face: closest point slides along the top edge -> diag(0, 1, 0).
    test.assertTrue(np.allclose(_grad_for([side_row]), [0.0, 1.0, 0.0], atol=1e-5))
    # Both rows together accumulate.
    test.assertTrue(np.allclose(_grad_for([top_row, side_row]), [1.0, 2.0, 0.0], atol=1e-5))


def test_bvh_mixed_scene_stream_offsets(test, device):
    """BVH records coexist correctly with legacy + SDF records in the shared stream (tid_base != 0).

    Every other BVH scene is mesh-only, which forces ``tid_base = 0`` and leaves the shared
    replay-tids offset unexercised. Here an analytic sphere adds nonzero legacy particle pairs and
    SDF edge/face pairs ahead of the BVH passes. Asserts: the offsets are genuinely nonzero, both
    shapes emit, the mesh shape's records still match the brute-force reference exactly, and
    gradients through a BVH record remain exact -- a wrong ``tid_base`` would push the emit
    kernel's replay-memo writes out of bounds (guarded -> memo lost -> zero gradients).
    """
    gap = 0.01
    builder = newton.ModelBuilder()
    box = newton.Mesh.create_box(0.25, 0.25, 0.1)
    builder.add_shape_mesh(-1, mesh=box)
    # Analytic sphere under the cloth's overhang corner: within range of a few particles.
    sphere_shape = builder.add_shape_sphere(
        -1, xform=wp.transform(wp.vec3(0.25, 0.25, 0.0), wp.quat_identity()), radius=0.09
    )
    builder.add_cloth_grid(
        pos=wp.vec3(-0.3, -0.3, 0.105),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=6,
        dim_y=6,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
        particle_radius=0.01,
    )
    model = builder.finalize(device=device, requires_grad=True)

    pipeline = newton.CollisionPipeline(
        model, soft_contact_gap=gap, enable_rigid_soft_full_surface_contact=True, full_surface_mesh_backend="bvh"
    )
    # The sphere puts legacy particle pairs AND SDF edge/face pairs ahead of the BVH passes.
    test.assertGreater(pipeline.soft_contact_pair_count, 0)
    test.assertGreater(len(pipeline.soft_edge_rigid_pairs), 0)
    test.assertGreater(len(pipeline.soft_face_rigid_pairs), 0)

    contacts = pipeline.contacts()
    state = model.state()
    pipeline.refit_soft_contact_bvh(state)
    tape = wp.Tape()
    with tape:
        pipeline.collide(state, contacts)

    n = int(contacts.soft_contact_count.numpy()[0])
    idx = contacts.soft_contact_indices.numpy()[:n]
    bary = contacts.soft_contact_barycentric.numpy()[:n]
    body_pos = contacts.soft_contact_body_pos.numpy()[:n]
    shape = contacts.soft_contact_shape.numpy()[:n]
    q = model.particle_q.numpy()

    # Both shapes emit: the sphere via the legacy/SDF passes, the mesh via the BVH passes.
    test.assertGreater(int((shape == sphere_shape).sum()), 0, "sphere must emit legacy/SDF records")
    test.assertGreater(int((shape == 0).sum()), 0, "mesh must emit BVH records")

    # The mesh shape's records (all BVH: mesh shapes are excluded from legacy/SDF under 'bvh')
    # still match the brute-force reference exactly, despite sharing the stream.
    vt_act, ee_act, tv_act = [], [], []
    for i in range(n):
        if shape[i] != 0:
            continue
        c = idx[i]
        if c[2] >= 0:
            x_soft = bary[i, 0] * q[c[0]] + bary[i, 1] * q[c[1]] + bary[i, 2] * q[c[2]]
            tv_act.append(((int(c[0]), int(c[1]), int(c[2])), x_soft, body_pos[i]))
        elif c[1] >= 0:
            x_soft = bary[i, 0] * q[c[0]] + bary[i, 1] * q[c[1]]
            ee_act.append(((int(c[0]), int(c[1])), x_soft, body_pos[i]))
        else:
            vt_act.append((int(c[0]), q[c[0]], body_pos[i]))
    for act in (vt_act, ee_act, tv_act):
        test.assertGreater(len(act), 0)
    vt_exp, ee_exp, tv_exp = _bvh_brute_force_reference(model, box, gap)
    threshold = _parity_threshold(model, gap)
    for act, exp in ((vt_act, vt_exp), (ee_act, ee_exp), (tv_act, tv_exp)):
        _assert_family_matches(test, [(k, float(np.linalg.norm(xs - xr))) for k, xs, xr in act], exp, threshold)

    # Gradients through a BVH record stay exact with a nonzero tid_base. Pick a VT record whose
    # particle sits over the box top interior (body_pos on z = 0.1, away from edges).
    vt_rows = [
        i
        for i in range(n)
        if shape[i] == 0 and idx[i][1] < 0 and abs(body_pos[i][2] - 0.1) < 1e-5 and abs(body_pos[i][0]) < 0.2
    ]
    test.assertGreater(len(vt_rows), 0)
    row = vt_rows[0]
    particle = int(idx[row][0])
    seed = np.zeros((contacts.soft_contact_max, 3), dtype=np.float32)
    seed[row] = (1.0, 1.0, 1.0)
    tape.backward(grads={contacts.soft_contact_body_pos: wp.array(seed, dtype=wp.vec3, device=device)})
    grad = state.particle_q.grad.numpy()[particle]
    test.assertTrue(np.allclose(grad, [1.0, 1.0, 0.0], atol=1e-5), f"got {grad}")

    # And a legacy (sphere) record row still routes its own gradient (slots are not aliased).
    tape.zero()
    sphere_rows = np.flatnonzero(shape == sphere_shape)
    seed = np.zeros((contacts.soft_contact_max, 3), dtype=np.float32)
    seed[sphere_rows[0]] = (1.0, 1.0, 1.0)
    tape.backward(grads={contacts.soft_contact_body_pos: wp.array(seed, dtype=wp.vec3, device=device)})
    sphere_particle = int(idx[sphere_rows[0]][0])
    test.assertGreater(float(np.abs(state.particle_q.grad.numpy()[sphere_particle]).sum()), 0.0, "legacy replay intact")


add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_kernel_gradients_wrt_particle_pos",
    test_bvh_kernel_gradients_wrt_particle_pos,
    devices=soft_devices,
)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_kernel_gradients_wrt_body_transform",
    test_bvh_kernel_gradients_wrt_body_transform,
    devices=soft_devices,
)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_multi_emission_gradients",
    test_bvh_multi_emission_gradients,
    devices=soft_devices,
)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_mixed_scene_stream_offsets",
    test_bvh_mixed_scene_stream_offsets,
    devices=soft_devices,
)

# ---------------------------------------------------------------------------
# Integration: flags, configuration, and API lifecycle
# Mutable flag gating, backend/knob validation, triangle-less mode, refit/rebuild and deprecation contracts.
# ---------------------------------------------------------------------------


def test_bvh_flag_gating(test, device):
    """VT skips inactive particles; TV/EE skip a soft feature only when NO corner is active."""

    def _deactivate(model, particles):
        flags = model.particle_flags.numpy()
        for p in particles:
            flags[p] &= ~int(ParticleFlags.ACTIVE)
        model.particle_flags.assign(flags)

    # VT: two particles over the face; deactivating one removes exactly its records.
    b = newton.ModelBuilder()
    b.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.2, 0.2, 0.1))
    b.add_particle(pos=wp.vec3(-0.05, 0.0, 0.105), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
    b.add_particle(pos=wp.vec3(0.05, 0.0, 0.105), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
    model = b.finalize(device=device)
    _deactivate(model, [0])
    _, contacts = _run_bvh_pipeline(model)
    n = int(contacts.soft_contact_count.numpy()[0])
    idx = contacts.soft_contact_indices.numpy()[:n]
    test.assertGreater(n, 0)
    test.assertTrue(np.all(idx[:, 0] != 0), "inactive particle must emit no VT records")

    # TV: partially active soft triangle still collides; fully inactive one is skipped.
    def _corner_tri_model():
        b = newton.ModelBuilder()
        b.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.2, 0.2, 0.1))
        for p in [(0.05, 0.05, 0.115), (0.35, 0.2, 0.1), (0.2, 0.35, 0.1)]:
            b.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
        b.add_triangle(0, 1, 2)
        return b.finalize(device=device)

    model = _corner_tri_model()
    _deactivate(model, [0])  # one corner inactive: feature still takes forces via the others
    _, contacts = _run_bvh_pipeline(model)
    n = int(contacts.soft_contact_count.numpy()[0])
    test.assertGreater(n, 0, "a partially active soft triangle must still collide")

    model = _corner_tri_model()
    _deactivate(model, [0, 1, 2])
    _, contacts = _run_bvh_pipeline(model)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0, "a fully inactive feature is skipped")


def test_bvh_runtime_collide_particles_toggle(test, device):
    """A mesh disabled at construction joins the BVH back-end when COLLIDE_PARTICLES is enabled later.

    The shape mask deliberately ignores the mutable flag (kernels check it per thread), so the
    feature tables exist either way and runtime off -> on -> off transitions just work.
    """
    from newton._src.geometry.flags import ShapeFlags  # noqa: PLC0415

    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.has_particle_collision = False  # disabled at construction
    builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.2, 0.2, 0.1), cfg=cfg)
    for p in [(0.05, 0.05, 0.115), (0.35, 0.2, 0.1), (0.2, 0.35, 0.1)]:
        builder.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
    builder.add_triangle(0, 1, 2)
    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(
        model, soft_contact_gap=0.01, enable_rigid_soft_full_surface_contact=True, full_surface_mesh_backend="bvh"
    )
    test.assertGreater(pipeline._full_surface_bvh_thread_count, 0, "tables must exist for the disabled mesh")
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.refit_soft_contact_bvh(state)

    def _collide_count():
        pipeline.collide(state, contacts)
        return int(contacts.soft_contact_count.numpy()[0])

    test.assertEqual(_collide_count(), 0, "disabled shape must not collide")

    flags = model.shape_flags.numpy()
    flags[0] |= int(ShapeFlags.COLLIDE_PARTICLES)
    model.shape_flags.assign(flags)
    test.assertGreater(_collide_count(), 0, "enabling COLLIDE_PARTICLES at runtime must activate the back-end")

    flags[0] &= ~int(ShapeFlags.COLLIDE_PARTICLES)
    model.shape_flags.assign(flags)
    test.assertEqual(_collide_count(), 0, "disabling again must deactivate it")


def test_bvh_vt_only_without_triangles(test, device):
    """A particle-only soft body (no triangles): the BVH back-end runs VT-only, silently."""
    builder = newton.ModelBuilder()
    box = newton.Mesh.create_box(0.2, 0.2, 0.1)
    builder.add_shape_mesh(-1, mesh=box)
    for x in (-0.05, 0.0, 0.05):
        builder.add_particle(pos=wp.vec3(x, 0.0, 0.105), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
    model = builder.finalize(device=device)
    test.assertEqual(model.tri_count, 0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # silently: no warning may be raised
        pipeline = newton.CollisionPipeline(
            model, soft_contact_gap=0.01, enable_rigid_soft_full_surface_contact=True, full_surface_mesh_backend="bvh"
        )
    test.assertIsNone(pipeline._soft_contact_detector)
    test.assertEqual(len(pipeline._full_surface_bvh_rigid_vertex_table), 0)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    total = int(contacts.soft_contact_count.numpy()[0])
    test.assertGreater(total, 0)
    idx = contacts.soft_contact_indices.numpy()[:total]
    test.assertTrue(np.all(idx[:, 1] < 0), "VT-only: no edge/face records")
    with test.assertRaises(ValueError):
        pipeline.refit_soft_contact_bvh(model.state())


def test_bvh_sdf_backend_keeps_raise(test, device):
    """The 'sdf' back-end preserves the unprovisioned-mesh raise; 'bvh' removes it."""
    model, _box = _build_cloth_over_mesh_box(device, provision_sdf=False)
    with test.assertRaises(ValueError):
        newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True, full_surface_mesh_backend="sdf")
    newton.CollisionPipeline(
        model, enable_rigid_soft_full_surface_contact=True, full_surface_mesh_backend="bvh"
    )  # no raise
    with test.assertRaises(ValueError):
        newton.CollisionPipeline(model, full_surface_mesh_backend="nope")


def test_bvh_contact_headroom(test, device):
    """full_surface_bvh_contact_headroom validates and scales the default soft_contact_max."""
    model, _box = _build_cloth_over_mesh_box(device)
    with test.assertRaises(ValueError):
        newton.CollisionPipeline(
            model,
            enable_rigid_soft_full_surface_contact=True,
            full_surface_mesh_backend="bvh",
            full_surface_bvh_contact_headroom=-1,
        )
    p0 = newton.CollisionPipeline(
        model,
        enable_rigid_soft_full_surface_contact=True,
        full_surface_mesh_backend="bvh",
        full_surface_bvh_contact_headroom=0,
    )
    p7 = newton.CollisionPipeline(
        model,
        enable_rigid_soft_full_surface_contact=True,
        full_surface_mesh_backend="bvh",
        full_surface_bvh_contact_headroom=7,
    )
    threads = p0._full_surface_bvh_thread_count
    test.assertGreater(threads, 0)
    test.assertEqual(p7.soft_contact_max - p0.soft_contact_max, 7 * threads)
    # In this mesh-only scene ALL capacity comes from the headroom, so headroom 0 means a
    # zero-capacity stream: collide runs but emits nothing.
    test.assertEqual(p0.soft_contact_max, 0)
    c0 = p0.contacts()
    state = model.state()
    p0.refit_soft_contact_bvh(state)
    p0.collide(state, c0)
    test.assertEqual(int(c0.soft_contact_count.numpy()[0]), 0)


def test_bvh_refit_api(test, device):
    """refit/rebuild_soft_contact_bvh work for both owners; the old name warns and forwards."""
    model, _box = _build_cloth_over_mesh_box(device)
    state = model.state()

    # collide() without a prior refit: the BVHs were never built -- it warns, then builds them
    # from the current state so detection still works.
    p0 = newton.CollisionPipeline(
        model, soft_contact_gap=0.01, enable_rigid_soft_full_surface_contact=True, full_surface_mesh_backend="bvh"
    )
    c0 = p0.contacts()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p0.collide(state, c0)
    test.assertTrue(any("refit_soft_contact_bvh" in str(w.message) for w in caught))
    test.assertGreater(int(c0.soft_contact_count.numpy()[0]), 0)

    # BVH back-end owner (no self-contact configured): the first refit lazily builds the
    # detector (inside the helper), and a full rebuild must reproduce identical detection on the
    # same state (rebuild is a distinct path: trees reconstructed from scratch, not refitted).
    pipeline, contacts = _run_bvh_pipeline(model)
    n_before = int(contacts.soft_contact_count.numpy()[0])
    idx_before = contacts.soft_contact_indices.numpy()[:n_before]
    idx_before = idx_before[np.lexsort(idx_before.T[::-1])]  # row-wise order (emission order races)
    pipeline.rebuild_soft_contact_bvh(state)
    pipeline.collide(state, contacts)
    n_after = int(contacts.soft_contact_count.numpy()[0])
    test.assertEqual(n_after, n_before)
    idx_after = contacts.soft_contact_indices.numpy()[:n_after]
    idx_after = idx_after[np.lexsort(idx_after.T[::-1])]
    test.assertTrue(np.array_equal(idx_after, idx_before), "rebuild must not change detection")

    # Self-contact owner on a flag-off pipeline.
    p2 = newton.CollisionPipeline(model)
    with test.assertRaises(ValueError):
        p2.refit_soft_contact_bvh(state)  # no detector yet
    p2.init_soft_self_contact(topological_filter_threshold=0)
    p2.refit_soft_contact_bvh(state)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p2.refit_soft_self_contact_bvh(state.particle_q)
        p2.refit_soft_self_contact_bvh(state.particle_q, rebuild=True)  # the rebuild branch forwards too
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    test.assertEqual(len(deprecations), 2, "each alias call must warn (both branches)")


# ---------------------------------------------------------------------------
# Kernel-level unit tests: each kernel launched directly on hand-computable
# geometry (a single right triangle / a cube), no CollisionPipeline involved.
# ---------------------------------------------------------------------------


def test_bvh_refit_follows_motion(test, device):
    """Detection follows refit: a moved soft feature is missed with stale BVHs and found after refit."""
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.2, 0.2, 0.1))
    # TV-only scene (lone triangle), constructed FAR from the box.
    for p in [(0.05, 0.05, 1.115), (0.35, 0.2, 1.1), (0.2, 0.35, 1.1)]:
        builder.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
    builder.add_triangle(0, 1, 2)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model, soft_contact_gap=0.01, enable_rigid_soft_full_surface_contact=True, full_surface_mesh_backend="bvh"
    )
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.refit_soft_contact_bvh(state)
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0)

    # Drop the triangle to hover over the box corner; without a refit the TV pass queries stale
    # triangle AABBs a meter away and must stay silent.
    q = state.particle_q.numpy()
    q[:, 2] -= 1.0
    state.particle_q.assign(q)
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0, "stale BVHs must miss (documented contract)")

    pipeline.refit_soft_contact_bvh(state)
    pipeline.collide(state, contacts)
    test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0, "refit must expose the moved feature")


add_function_test(TestBvhFullSurfaceSoftContact, "test_bvh_flag_gating", test_bvh_flag_gating, devices=soft_devices)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_runtime_collide_particles_toggle",
    test_bvh_runtime_collide_particles_toggle,
    devices=soft_devices,
)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_vt_only_without_triangles",
    test_bvh_vt_only_without_triangles,
    devices=soft_devices,
)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_sdf_backend_keeps_raise",
    test_bvh_sdf_backend_keeps_raise,
    devices=soft_devices,
)
add_function_test(
    TestBvhFullSurfaceSoftContact, "test_bvh_contact_headroom", test_bvh_contact_headroom, devices=soft_devices
)
add_function_test(TestBvhFullSurfaceSoftContact, "test_bvh_refit_api", test_bvh_refit_api, devices=soft_devices)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_refit_follows_motion",
    test_bvh_refit_follows_motion,
    devices=soft_devices,
)

# ---------------------------------------------------------------------------
# Integration: diagnostics and capture
# Overflow counters + printed warnings, and CUDA graph-capture parity.
# ---------------------------------------------------------------------------


def test_bvh_overflow(test, device):
    """Deliberate overflow: counters stay diagnostic, records stay valid, and warnings print.

    Under detect + emit the CANDIDATE counter carries the attempted count (the record stream then
    fills exactly to capacity); the record counter can only overflow through the legacy pass, so a
    second, analytic-shape phase covers that diagnostic branch.
    """
    from newton.tests.unittest_utils import StdOutCapture  # noqa: PLC0415

    # Phase 1: BVH back-end overflow (candidate counter + candidate warning).
    model, _box = _build_cloth_over_mesh_box(device)
    pipeline = newton.CollisionPipeline(
        model,
        soft_contact_gap=0.01,
        enable_rigid_soft_full_surface_contact=True,
        full_surface_mesh_backend="bvh",
        soft_contact_max=2,
    )
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.refit_soft_contact_bvh(state)
    capture = StdOutCapture()
    capture.begin()
    try:
        pipeline.collide(state, contacts)
        wp.synchronize_device(wp.get_device(device))
    finally:
        output = capture.end()
    attempted = int(pipeline._full_surface_bvh_candidate_count.numpy()[0])
    test.assertGreater(attempted, len(pipeline._full_surface_bvh_candidates))
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
    shape = contacts.soft_contact_shape.numpy()[: contacts.soft_contact_max]
    test.assertTrue(np.all(shape >= 0), "stored records stay valid, no corruption")
    test.assertIn(f"BVH soft contact candidate buffer overflowed {attempted} > 2", output)

    # Phase 2: record-stream overflow branch -- only the legacy pass can push the RECORD counter
    # past capacity (the BVH emit stage fills exactly to capacity), so use an analytic shape.
    b = newton.ModelBuilder()
    b.add_shape_sphere(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.1)
    for x in (-0.02, 0.0, 0.02):
        for y in (-0.02, 0.0, 0.02):
            b.add_particle(pos=wp.vec3(x, y, 0.105), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
    model = b.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01, soft_contact_max=2)
    contacts = pipeline.contacts()
    capture = StdOutCapture()
    capture.begin()
    try:
        pipeline.collide(model.state(), contacts)
        wp.synchronize_device(wp.get_device(device))
    finally:
        output = capture.end()
    attempted = int(contacts.soft_contact_count.numpy()[0])
    test.assertGreater(attempted, 2)
    test.assertIn(f"Soft contact buffer overflowed {attempted} > 2", output)


def test_bvh_graph_capture(test, device):
    """A graph-captured collide() with the BVH back-end matches uncaptured results."""
    model, _box = _build_cloth_over_mesh_box(device)
    pipeline = newton.CollisionPipeline(
        model, soft_contact_gap=0.01, enable_rigid_soft_full_surface_contact=True, full_surface_mesh_backend="bvh"
    )
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.refit_soft_contact_bvh(state)

    pipeline.collide(state, contacts)  # warm-up (module load) + uncaptured reference
    wp.synchronize_device(wp.get_device(device))
    n_ref = int(contacts.soft_contact_count.numpy()[0])
    idx_ref = contacts.soft_contact_indices.numpy()[:n_ref]
    idx_ref = idx_ref[np.lexsort(idx_ref.T[::-1])]  # row-wise order (per-column sort loses rows)

    with wp.ScopedCapture(device) as capture:
        pipeline.collide(state, contacts)
    wp.capture_launch(capture.graph)
    wp.synchronize_device(wp.get_device(device))
    n_cap = int(contacts.soft_contact_count.numpy()[0])
    test.assertEqual(n_cap, n_ref)
    idx_cap = contacts.soft_contact_indices.numpy()[:n_cap]
    idx_cap = idx_cap[np.lexsort(idx_cap.T[::-1])]
    test.assertTrue(np.array_equal(idx_cap, idx_ref))


add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_overflow",
    test_bvh_overflow,
    devices=soft_devices,
    check_output=False,  # the test captures fd 1 itself; CheckOutput would shadow it
)
add_function_test(
    TestBvhFullSurfaceSoftContact,
    "test_bvh_graph_capture",
    test_bvh_graph_capture,
    devices=get_cuda_test_devices(),
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
