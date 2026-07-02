# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the water-tight rigid-soft contact path (SDF back-end).

Covers the ``Contacts`` schema change (standalone length-3 ``soft_contact_count``;
``soft_contact_particle`` renamed to ``soft_contact_primitive``; new
``soft_contact_barycentric`` field), and — as later tasks land — the SDF optimizers, the
edge/face passes, backward-compat, and water-tight regressions.
"""

import unittest
from collections import Counter

import numpy as np
import warp as wp

import newton
from newton import GeoType
from newton._src.geometry.flags import ShapeFlags
from newton._src.geometry.sdf_texture import TextureSDFData
from newton._src.geometry.soft_contacts_sdf import (
    SDF_EDGE_ITERS,
    SDF_FACE_ITERS,
    SDF_LS_ITERS,
    _is_analytic,
    _shape_frames,
    eval_shape_sdf,
    launch_soft_ef_contacts,
    optimize_edge_sdf,
    optimize_face_sdf,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _build_cloth_over_plane(device, particle_z: float = 0.05):
    """A 5x5 cloth grid hovering just above a ground plane (particles overlap within margin)."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5, -0.5, particle_z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=5,
        dim_y=5,
        cell_x=0.2,
        cell_y=0.2,
        mass=0.1,
    )
    return builder.finalize(device=device)


def test_soft_contact_schema(test, device):
    """soft_contact_count is a standalone int32[3]; primitive renamed; kind/bary added."""
    model = _build_cloth_over_plane(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=0.1)
    contacts = pipeline.contacts()

    # Standalone length-3 soft counter: [particle, edge, face].
    test.assertEqual(tuple(contacts.soft_contact_count.shape), (3,))

    # New / renamed fields sized to soft_contact_max.
    test.assertEqual(contacts.soft_contact_primitive.shape[0], contacts.soft_contact_max)
    test.assertEqual(contacts.soft_contact_barycentric.shape[0], contacts.soft_contact_max)

    # soft_contact_particle is a deprecated alias for soft_contact_primitive.
    with test.assertWarns(DeprecationWarning):
        aliased = contacts.soft_contact_particle
    test.assertIs(aliased, contacts.soft_contact_primitive)

    # Rigid counter untouched and independent of the soft counter.
    test.assertEqual(tuple(contacts.rigid_contact_count.shape), (1,))

    # Flag-off collide: the legacy particle pass fills slot 0; edge/face slots stay zero.
    state = model.state()
    pipeline.collide(state, contacts)
    counts = contacts.soft_contact_count.numpy()
    test.assertGreater(int(counts[0]), 0)
    test.assertEqual(int(counts[1]), 0)
    test.assertEqual(int(counts[2]), 0)


devices = get_test_devices()


class TestWaterTightSoftContact(unittest.TestCase):
    pass


add_function_test(
    TestWaterTightSoftContact,
    "test_soft_contact_schema",
    test_soft_contact_schema,
    devices=devices,
)


# ---------------------------------------------------------------------------
# T2: SDF optimizers (Macklin 2020) validated against a brute-force grid min.
# The brute-force reference samples phi on a fine grid and takes the argmin,
# so these isolate "does the optimizer find the minimum of phi".
# ---------------------------------------------------------------------------


def _box_sdf_np(point, half):
    """Reference box SDF (matches geometry.kernels.sdf_box) for brute-force comparison."""
    q = np.abs(point) - half
    return float(np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1], q[2]), 0.0))


@wp.kernel
def _edge_opt_kernel(
    geo: wp.int32,
    scale: wp.vec3,
    p: wp.vec3,
    q: wp.vec3,
    shape_sdf_index: wp.int32,
    table: wp.array[TextureSDFData],
    n_iter: wp.int32,
    out_u: wp.array[float],
    out_phi: wp.array[float],
    out_x: wp.array[wp.vec3],
):
    u, x, phi, _grad = optimize_edge_sdf(geo, scale, p, q, shape_sdf_index, table, n_iter)
    out_u[0] = u
    out_phi[0] = phi
    out_x[0] = x


@wp.kernel
def _face_opt_kernel(
    geo: wp.int32,
    scale: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    shape_sdf_index: wp.int32,
    table: wp.array[TextureSDFData],
    n_iter: wp.int32,
    ls_iter: wp.int32,
    out_bary: wp.array[wp.vec3],
    out_phi: wp.array[float],
    out_x: wp.array[wp.vec3],
):
    bary, x, phi, _grad = optimize_face_sdf(geo, scale, a, b, c, shape_sdf_index, table, n_iter, ls_iter)
    out_bary[0] = bary
    out_phi[0] = phi
    out_x[0] = x


def _empty_sdf_table(device):
    return wp.zeros(0, dtype=TextureSDFData, device=device)


def test_optimize_edge_sdf_box(test, device):
    """Golden-section edge optimizer finds the deepest point of phi along the segment."""
    half = (0.5, 0.5, 0.5)
    p = (0.8, 0.0, 0.0)
    q = (0.0, 0.8, 0.0)
    out_u = wp.zeros(1, dtype=float, device=device)
    out_phi = wp.zeros(1, dtype=float, device=device)
    out_x = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _edge_opt_kernel,
        dim=1,
        inputs=[
            int(GeoType.BOX),
            wp.vec3(*half),
            wp.vec3(*p),
            wp.vec3(*q),
            -1,
            _empty_sdf_table(device),
            SDF_EDGE_ITERS,
        ],
        outputs=[out_u, out_phi, out_x],
        device=device,
    )
    phi_opt = float(out_phi.numpy()[0])
    pa, qa, ha = np.array(p), np.array(q), np.array(half)
    phi_brute = min(_box_sdf_np((1.0 - u) * pa + u * qa, ha) for u in np.linspace(0.0, 1.0, 20001))
    test.assertLess(abs(phi_opt - phi_brute), 1.0e-4)


def test_optimize_face_sdf_box(test, device):
    """Frank-Wolfe face optimizer finds the deepest point of phi over the triangle."""
    half = (0.5, 0.5, 0.5)
    a = (0.9, 0.0, 0.0)
    b = (0.0, 0.9, 0.0)
    c = (0.0, 0.0, 0.9)
    out_bary = wp.zeros(1, dtype=wp.vec3, device=device)
    out_phi = wp.zeros(1, dtype=float, device=device)
    out_x = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _face_opt_kernel,
        dim=1,
        inputs=[
            int(GeoType.BOX),
            wp.vec3(*half),
            wp.vec3(*a),
            wp.vec3(*b),
            wp.vec3(*c),
            -1,
            _empty_sdf_table(device),
            SDF_FACE_ITERS,
            SDF_LS_ITERS,
        ],
        outputs=[out_bary, out_phi, out_x],
        device=device,
    )
    phi_opt = float(out_phi.numpy()[0])
    aa, ba, ca, ha = np.array(a), np.array(b), np.array(c), np.array(half)
    n = 200
    best = min(
        _box_sdf_np((i / n) * aa + (j / n) * ba + (1.0 - i / n - j / n) * ca, ha)
        for i in range(n + 1)
        for j in range(n + 1 - i)
    )
    test.assertLess(abs(phi_opt - best), 2.0e-3)


def _sphere_sdf_np(point, radius):
    """Reference sphere SDF (matches geometry.kernels.sdf_sphere)."""
    return float(np.linalg.norm(point) - radius)


def test_optimize_edge_sdf_sphere(test, device):
    """Golden-section on a smooth field finds the segment's closest approach to the sphere."""
    r = 0.5
    p = (1.0, 0.5, 0.0)
    q = (0.5, 1.0, 0.3)
    out_u = wp.zeros(1, dtype=float, device=device)
    out_phi = wp.zeros(1, dtype=float, device=device)
    out_x = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _edge_opt_kernel,
        dim=1,
        inputs=[
            int(GeoType.SPHERE),
            wp.vec3(r, r, r),
            wp.vec3(*p),
            wp.vec3(*q),
            -1,
            _empty_sdf_table(device),
            SDF_EDGE_ITERS,
        ],
        outputs=[out_u, out_phi, out_x],
        device=device,
    )
    phi_opt = float(out_phi.numpy()[0])
    pa, qa = np.array(p), np.array(q)
    phi_brute = min(_sphere_sdf_np((1.0 - u) * pa + u * qa, r) for u in np.linspace(0.0, 1.0, 20001))
    test.assertLess(abs(phi_opt - phi_brute), 1.0e-4)


def test_optimize_face_sdf_sphere(test, device):
    """Frank-Wolfe on a smooth field moves to a non-centroid optimum (asymmetric triangle)."""
    r = 0.5
    a = (1.0, 0.0, 0.2)
    b = (0.0, 1.0, 0.2)
    c = (0.3, 0.3, 1.2)
    out_bary = wp.zeros(1, dtype=wp.vec3, device=device)
    out_phi = wp.zeros(1, dtype=float, device=device)
    out_x = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _face_opt_kernel,
        dim=1,
        inputs=[
            int(GeoType.SPHERE),
            wp.vec3(r, r, r),
            wp.vec3(*a),
            wp.vec3(*b),
            wp.vec3(*c),
            -1,
            _empty_sdf_table(device),
            SDF_FACE_ITERS,
            SDF_LS_ITERS,
        ],
        outputs=[out_bary, out_phi, out_x],
        device=device,
    )
    phi_opt = float(out_phi.numpy()[0])
    aa, ba, ca = np.array(a), np.array(b), np.array(c)
    n = 200
    best = min(
        _sphere_sdf_np((i / n) * aa + (j / n) * ba + (1.0 - i / n - j / n) * ca, r)
        for i in range(n + 1)
        for j in range(n + 1 - i)
    )
    # Face Frank-Wolfe tail at SDF_FACE_ITERS on a smooth field (~3e-3); ample for contact-within-margin.
    test.assertLess(abs(phi_opt - best), 6.0e-3)


# ---------------------------------------------------------------------------
# T3: edge + face pass kernels (record emission).
# ---------------------------------------------------------------------------


def test_edge_face_passes_box(test, device):
    """A cloth sheet inside a box: every unique soft edge and triangle emits exactly one record."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.5
    )
    builder.add_cloth_grid(
        pos=wp.vec3(-0.4, -0.4, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        cell_x=0.2,
        cell_y=0.2,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    # Large buffer: flag-aware sizing lands in T4; this isolates the kernels.
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=0.1, soft_contact_max=4096)
    contacts = pipeline.contacts()
    state = model.state()
    contacts.soft_contact_count.zero_()
    launch_soft_ef_contacts(model=model, state=state, contacts=contacts, margin=0.1, device=device)

    counts = contacts.soft_contact_count.numpy()
    c0, n_edge, n_face = int(counts[0]), int(counts[1]), int(counts[2])
    n_edges = model.soft_mesh_adjacency.edge_indices.shape[0]
    # Structural dedup: the sheet is entirely inside the box, so every unique edge / triangle
    # emits exactly once (one thread per unique edge / triangle).
    test.assertEqual(n_edge, n_edges)
    test.assertEqual(n_face, model.tri_count)

    prims = contacts.soft_contact_primitive.numpy()
    barys = contacts.soft_contact_barycentric.numpy()
    normals = contacts.soft_contact_normal.numpy()
    body_pos = contacts.soft_contact_body_pos.numpy()
    half = np.array([0.5, 0.5, 0.5])

    # Records pack contiguously and the range IS the kind (no per-record flag): edge range
    # [c0, c0+n_edge), face range [c0+n_edge, c0+n_edge+n_face). n_edge/n_face (asserted above)
    # confirm each range holds exactly the expected feature count.
    for i in range(c0, c0 + n_edge):
        test.assertTrue(0 <= int(prims[i]) < model.tri_count)
        test.assertAlmostEqual(float(barys[i].sum()), 1.0, places=4)
        test.assertGreater(float(normals[i][2]), 0.99)  # +z face of the box
        test.assertLess(abs(_box_sdf_np(body_pos[i], half)), 1.0e-2)  # closest point on the box surface
    for i in range(c0 + n_edge, c0 + n_edge + n_face):
        test.assertTrue(0 <= int(prims[i]) < model.tri_count)
        test.assertAlmostEqual(float(barys[i].sum()), 1.0, places=4)
        test.assertGreater(float(normals[i][2]), 0.99)
        test.assertLess(abs(_box_sdf_np(body_pos[i], half)), 1.0e-2)


# ---------------------------------------------------------------------------
# T4: dispatch flag — backward-compat (bit-for-bit) and water-tight regression.
# ---------------------------------------------------------------------------


def _sorted_particle_records(contacts, c0):
    """Particle-range records sorted by particle id (emission order is non-deterministic on GPU)."""
    prim = contacts.soft_contact_primitive.numpy()[:c0]
    order = np.argsort(prim, kind="stable")
    return (
        prim[order],
        contacts.soft_contact_shape.numpy()[:c0][order],
        contacts.soft_contact_body_pos.numpy()[:c0][order],
        contacts.soft_contact_normal.numpy()[:c0][order],
    )


def test_backward_compat_bit_for_bit(test, device):
    """Flag on vs off (same buffer): the particle range is bit-identical; on only adds E/F records."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.5
    )
    builder.add_cloth_grid(
        pos=wp.vec3(-0.4, -0.4, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        cell_x=0.2,
        cell_y=0.2,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    state = model.state()

    # The flag is fixed at construction, so off vs on are two separately-sized pipelines.
    pipeline_off = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_water_tight_rigid_soft_contact=False
    )
    contacts_off = pipeline_off.contacts()
    pipeline_off.collide(state, contacts_off)
    counts_off = contacts_off.soft_contact_count.numpy().copy()
    c0 = int(counts_off[0])
    test.assertGreater(c0, 0)
    test.assertEqual(int(counts_off[1]), 0)
    test.assertEqual(int(counts_off[2]), 0)
    prim_off, shape_off, pos_off, nrm_off = _sorted_particle_records(contacts_off, c0)

    pipeline_on = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_water_tight_rigid_soft_contact=True
    )
    contacts_on = pipeline_on.contacts()
    pipeline_on.collide(state, contacts_on)
    counts_on = contacts_on.soft_contact_count.numpy()
    test.assertEqual(int(counts_on[0]), c0)  # legacy particle count unchanged
    prim_on, shape_on, pos_on, nrm_on = _sorted_particle_records(contacts_on, c0)

    # Bit-identical particle range (same legacy kernel, same inputs).
    test.assertTrue(np.array_equal(prim_on, prim_off))
    test.assertTrue(np.array_equal(shape_on, shape_off))
    test.assertTrue(np.array_equal(pos_on, pos_off))
    test.assertTrue(np.array_equal(nrm_on, nrm_off))
    # Flag on only ADDS E/F records.
    test.assertGreater(int(counts_on[1]) + int(counts_on[2]), 0)


def test_water_tight_catches_what_particles_miss(test, device):
    """A soft quad spanning a box with all corners outside margin: per-particle misses, E/F catches."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.5
    )
    # 1x1 cloth = one quad (2 tris); corners at (+-1, +-1, 0.45) are far outside the box margin,
    # but the quad's interior/diagonal cross the box's +z face within margin.
    builder.add_cloth_grid(
        pos=wp.vec3(-1.0, -1.0, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        cell_x=2.0,
        cell_y=2.0,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    state = model.state()

    # Per-particle path alone (flag off at construction): every corner is outside margin -> no contact.
    pipeline_off = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_water_tight_rigid_soft_contact=False
    )
    contacts_off = pipeline_off.contacts()
    pipeline_off.collide(state, contacts_off)
    test.assertEqual(int(contacts_off.soft_contact_count.numpy()[0]), 0)

    # Water-tight path (flag on): the edge/face passes detect the crossing the particles miss.
    pipeline_on = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_water_tight_rigid_soft_contact=True
    )
    contacts_on = pipeline_on.contacts()
    pipeline_on.collide(state, contacts_on)
    counts = contacts_on.soft_contact_count.numpy()
    test.assertEqual(int(counts[0]), 0)  # still no per-particle contact
    test.assertGreater(int(counts[1]) + int(counts[2]), 0)  # caught by edge/face


for _name, _fn in (
    ("test_optimize_edge_sdf_box", test_optimize_edge_sdf_box),
    ("test_optimize_face_sdf_box", test_optimize_face_sdf_box),
    ("test_optimize_edge_sdf_sphere", test_optimize_edge_sdf_sphere),
    ("test_optimize_face_sdf_sphere", test_optimize_face_sdf_sphere),
    ("test_edge_face_passes_box", test_edge_face_passes_box),
    ("test_backward_compat_bit_for_bit", test_backward_compat_bit_for_bit),
    ("test_water_tight_catches_what_particles_miss", test_water_tight_catches_what_particles_miss),
):
    add_function_test(TestWaterTightSoftContact, _name, _fn, devices=devices)


# ---------------------------------------------------------------------------
# T5: mesh volume-SDF provisioning at finalize (B2). Texture SDFs are CUDA-only.
# ---------------------------------------------------------------------------


def test_mesh_sdf_provisioned_and_emits(test, device):
    """A participating MESH shape gets a volume SDF baked at finalize and emits EDGE/FACE records."""
    box_mesh = newton.Mesh.create_box(0.5, 0.5, 0.5)
    builder = newton.ModelBuilder()
    mesh_shape = builder.add_shape_mesh(body=-1, mesh=box_mesh)
    builder.add_cloth_grid(
        pos=wp.vec3(-0.4, -0.4, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        cell_x=0.2,
        cell_y=0.2,
        mass=0.1,
    )
    builder.enable_rigid_mesh_sdfs()
    model = builder.finalize(device=device)
    # B2: the participating mesh now carries a provisioned volume SDF.
    test.assertGreaterEqual(int(model._shape_sdf_index.numpy()[mesh_shape]), 0)

    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_water_tight_rigid_soft_contact=True
    )
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.collide(state, contacts)
    counts = contacts.soft_contact_count.numpy()
    # The mesh's volume SDF feeds the edge/face passes -> records emitted.
    test.assertGreater(int(counts[1]) + int(counts[2]), 0)


def test_optimize_against_mesh_texture_sdf(test, device):
    """optimize_edge/face_sdf against a MESH's provisioned texture SDF match the box it represents.

    Validates the volume-SDF branch of eval_shape_sdf (texture sampling + query-time scaling) end to
    end through the optimizers, to within the texture grid's resolution.
    """
    box_mesh = newton.Mesh.create_box(0.5, 0.5, 0.5)
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=box_mesh)
    builder.enable_rigid_mesh_sdfs()
    model = builder.finalize(device=device)
    sdf_idx = int(model._shape_sdf_index.numpy()[0])
    test.assertGreaterEqual(sdf_idx, 0)
    table = model._texture_sdf_data
    scale = wp.vec3(*(float(s) for s in model.shape_scale.numpy()[0]))
    half = np.array([0.5, 0.5, 0.5])
    tol = 3.0e-2  # texture SDF grid resolution (default 64^3 over a unit box) + optimizer tail

    # Edge: from just inside the +z face to outside; the minimum is the inside endpoint.
    p, q = (0.0, 0.0, 0.45), (0.0, 0.0, 0.65)
    out_u = wp.zeros(1, dtype=float, device=device)
    out_phi = wp.zeros(1, dtype=float, device=device)
    out_x = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _edge_opt_kernel,
        dim=1,
        inputs=[int(GeoType.MESH), scale, wp.vec3(*p), wp.vec3(*q), sdf_idx, table, SDF_EDGE_ITERS],
        outputs=[out_u, out_phi, out_x],
        device=device,
    )
    pa, qa = np.array(p), np.array(q)
    phi_ref_edge = min(_box_sdf_np((1.0 - u) * pa + u * qa, half) for u in np.linspace(0.0, 1.0, 4001))
    test.assertLess(abs(float(out_phi.numpy()[0]) - phi_ref_edge), tol)

    # Face: a small triangle grazing the +z face.
    a, b, c = (0.2, 0.0, 0.45), (-0.2, 0.1, 0.45), (0.0, -0.2, 0.45)
    out_bary = wp.zeros(1, dtype=wp.vec3, device=device)
    out_phi2 = wp.zeros(1, dtype=float, device=device)
    out_x2 = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _face_opt_kernel,
        dim=1,
        inputs=[
            int(GeoType.MESH),
            scale,
            wp.vec3(*a),
            wp.vec3(*b),
            wp.vec3(*c),
            sdf_idx,
            table,
            SDF_FACE_ITERS,
            SDF_LS_ITERS,
        ],
        outputs=[out_bary, out_phi2, out_x2],
        device=device,
    )
    aa, ba, ca = np.array(a), np.array(b), np.array(c)
    n = 80
    phi_ref_face = min(
        _box_sdf_np((i / n) * aa + (j / n) * ba + (1.0 - i / n - j / n) * ca, half)
        for i in range(n + 1)
        for j in range(n + 1 - i)
    )
    test.assertLess(abs(float(out_phi2.numpy()[0]) - phi_ref_face), tol)


for _name, _fn in (
    ("test_mesh_sdf_provisioned_and_emits", test_mesh_sdf_provisioned_and_emits),
    ("test_optimize_against_mesh_texture_sdf", test_optimize_against_mesh_texture_sdf),
):
    add_function_test(TestWaterTightSoftContact, _name, _fn, devices=get_cuda_test_devices())


def test_unprovisioned_mesh_raises(test, device):
    """A participating mesh with no SDF makes CollisionPipeline raise when the flag is enabled.

    Mirrors SolverVBD raising on an uncolored model: enable_rigid_mesh_sdfs() is a required build
    step, and skipping it is an error rather than a silent degrade to the per-particle path.
    """
    box_mesh = newton.Mesh.create_box(0.5, 0.5, 0.5)
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=box_mesh)
    builder.add_cloth_grid(
        pos=wp.vec3(-0.4, -0.4, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        cell_x=0.2,
        cell_y=0.2,
        mass=0.1,
    )
    # enable_rigid_mesh_sdfs() intentionally NOT called -> the mesh carries no SDF.
    model = builder.finalize(device=device)
    with test.assertRaises(ValueError):
        newton.CollisionPipeline(
            model, broad_phase="nxn", soft_contact_margin=0.1, enable_water_tight_rigid_soft_contact=True
        )


add_function_test(
    TestWaterTightSoftContact,
    "test_unprovisioned_mesh_raises",
    test_unprovisioned_mesh_raises,
    devices=devices,
)


# ---------------------------------------------------------------------------
# End-to-end: all shape types + random soft triangles, water-tight on, no
# false positives / false negatives vs a brute-force grid min of the same
# eval_shape_sdf. (For analytic shapes the optimizer evaluates phi on the
# feature, so phi* >= true min => false positives are structurally impossible;
# this guards false negatives and the dispatch/record matching too.)
# ---------------------------------------------------------------------------


@wp.kernel
def _brute_face_min_kernel(
    n_tris: wp.int32,
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    n_grid: wp.int32,
    out_min: wp.array[float],
):
    tid = wp.tid()
    shape_index = tid // n_tris
    t = tid % n_tris
    out_min[tid] = 1.0e10
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return
    geo = shape_type[shape_index]
    sdf_idx = shape_sdf_index[shape_index]
    if (not _is_analytic(geo)) and sdf_idx < 0:
        return
    _X_bs, _X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    a = wp.transform_point(X_sw, particle_q[tri_indices[t, 0]])
    b = wp.transform_point(X_sw, particle_q[tri_indices[t, 1]])
    c = wp.transform_point(X_sw, particle_q[tri_indices[t, 2]])
    scale = shape_scale[shape_index]
    m = float(1.0e10)
    for k in range((n_grid + 1) * (n_grid + 1)):
        i = k // (n_grid + 1)
        j = k % (n_grid + 1)
        if i + j <= n_grid:
            u = float(i) / float(n_grid)
            v = float(j) / float(n_grid)
            phi, _g = eval_shape_sdf(geo, scale, u * a + v * b + (1.0 - u - v) * c, sdf_idx, texture_sdf_table)
            m = wp.min(m, phi)
    out_min[tid] = m


@wp.kernel
def _brute_edge_min_kernel(
    n_edges: wp.int32,
    particle_q: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    n_grid: wp.int32,
    out_min: wp.array[float],
):
    tid = wp.tid()
    shape_index = tid // n_edges
    e = tid % n_edges
    out_min[tid] = 1.0e10
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return
    geo = shape_type[shape_index]
    sdf_idx = shape_sdf_index[shape_index]
    if (not _is_analytic(geo)) and sdf_idx < 0:
        return
    _X_bs, _X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    p = wp.transform_point(X_sw, particle_q[edge_indices[e, 2]])
    q = wp.transform_point(X_sw, particle_q[edge_indices[e, 3]])
    scale = shape_scale[shape_index]
    m = float(1.0e10)
    for i in range(n_grid + 1):
        u = float(i) / float(n_grid)
        phi, _g = eval_shape_sdf(geo, scale, (1.0 - u) * p + u * q, sdf_idx, texture_sdf_table)
        m = wp.min(m, phi)
    out_min[tid] = m


def _build_all_shapes_scene(device, rng):
    """Ground plane + the six analytic primitives, with random soft triangles seeded near each."""
    z = 1.0
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    box_mesh = newton.Mesh.create_box(0.5, 0.5, 0.5)
    primitives = [
        (
            lambda: builder.add_shape_sphere(
                body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()), radius=0.5
            ),
            (0.0, 0.0, z),
            0.5,
        ),
        (
            lambda: builder.add_shape_box(
                body=-1, xform=wp.transform(wp.vec3(2.0, 0.0, z), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.5
            ),
            (2.0, 0.0, z),
            0.6,
        ),
        (
            lambda: builder.add_shape_capsule(
                body=-1, xform=wp.transform(wp.vec3(4.0, 0.0, z), wp.quat_identity()), radius=0.4, half_height=0.4
            ),
            (4.0, 0.0, z),
            0.55,
        ),
        (
            lambda: builder.add_shape_cylinder(
                body=-1, xform=wp.transform(wp.vec3(6.0, 0.0, z), wp.quat_identity()), radius=0.5, half_height=0.4
            ),
            (6.0, 0.0, z),
            0.6,
        ),
        (
            lambda: builder.add_shape_cone(
                body=-1, xform=wp.transform(wp.vec3(8.0, 0.0, z), wp.quat_identity()), radius=0.5, half_height=0.5
            ),
            (8.0, 0.0, z),
            0.6,
        ),
        (
            lambda: builder.add_shape_ellipsoid(
                body=-1, xform=wp.transform(wp.vec3(10.0, 0.0, z), wp.quat_identity()), rx=0.5, ry=0.4, rz=0.6
            ),
            (10.0, 0.0, z),
            0.6,
        ),
        # MESH (a box-shaped triangle mesh): on CUDA its texture SDF is provisioned at finalize and
        # validated; on CPU texture SDFs are unavailable so the passes and the brute-force reference
        # both gate it out identically.
        (
            lambda: builder.add_shape_mesh(
                body=-1, xform=wp.transform(wp.vec3(12.0, 0.0, z), wp.quat_identity()), mesh=box_mesh
            ),
            (12.0, 0.0, z),
            0.6,
        ),
    ]
    centers, sizes = [], []
    for add, center, size in primitives:
        add()
        centers.append(np.array(center))
        sizes.append(size)

    verts, indices = [], []

    def add_tri(centroid):
        base = len(verts)
        for _ in range(3):
            off = rng.normal(0.0, 0.12, 3)
            verts.append(wp.vec3(float(centroid[0] + off[0]), float(centroid[1] + off[1]), float(centroid[2] + off[2])))
        indices.extend([base, base + 1, base + 2])

    for center, size in zip(centers, sizes, strict=True):
        for _ in range(5):
            d = rng.normal(0.0, 1.0, 3)
            d = d / np.linalg.norm(d)
            add_tri(center + d * (size + rng.uniform(-0.12, 0.18)))
    for _ in range(6):  # near the ground plane (z ~ 0)
        add_tri(np.array([rng.uniform(0.0, 10.0), rng.uniform(-1.0, 1.0), rng.uniform(-0.05, 0.22)]))

    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=verts,
        indices=indices,
        density=0.1,
        particle_radius=0.0,  # so the pass threshold is exactly `margin` (matches the brute-force check)
    )
    builder.enable_rigid_mesh_sdfs()
    return builder.finalize(device=device)


def test_end_to_end_no_false_pos_neg(test, device):
    """All shapes + random triangles: water-tight emissions match a brute-force grid min (no FP/FN)."""
    margin = 0.1
    model = _build_all_shapes_scene(device, np.random.default_rng(0))
    n_tris = model.tri_count
    n_edges = model.soft_mesh_adjacency.edge_indices.shape[0]
    n_shapes = model.shape_count

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        soft_contact_margin=margin,
        soft_contact_max=n_shapes * (n_tris + n_edges) + 16,
        enable_water_tight_rigid_soft_contact=True,
    )
    contacts = pipeline.contacts()
    state = model.state()
    contacts.soft_contact_count.zero_()
    launch_soft_ef_contacts(model=model, state=state, contacts=contacts, margin=margin, device=device)

    counts = contacts.soft_contact_count.numpy()
    n_edge_rec, n_face_rec = int(counts[1]), int(counts[2])
    test.assertGreater(n_edge_rec + n_face_rec, 0)  # the scene actually generates contacts

    # Brute-force ground truth: min phi per (shape, feature) using the same eval_shape_sdf.
    face_min = wp.empty(n_shapes * n_tris, dtype=float, device=device)
    edge_min = wp.empty(n_shapes * n_edges, dtype=float, device=device)
    shape_args = [
        model.shape_body,
        model.shape_type,
        model.shape_flags,
        model.shape_transform,
        model.shape_scale,
        state.body_q,
        model._shape_sdf_index,
        model._texture_sdf_data,
    ]
    # MeshAdjacency.edge_indices is host numpy; upload for the brute-force kernel.
    edge_indices_dev = wp.array(
        np.ascontiguousarray(model.soft_mesh_adjacency.edge_indices, dtype=np.int32), dtype=wp.int32, device=device
    )
    wp.launch(
        _brute_face_min_kernel,
        dim=n_shapes * n_tris,
        inputs=[n_tris, state.particle_q, model.tri_indices, *shape_args, 40],
        outputs=[face_min],
        device=device,
    )
    wp.launch(
        _brute_edge_min_kernel,
        dim=n_shapes * n_edges,
        inputs=[n_edges, state.particle_q, edge_indices_dev, *shape_args, 200],
        outputs=[edge_min],
        device=device,
    )
    face_min = face_min.numpy().reshape(n_shapes, n_tris)
    edge_min = edge_min.numpy().reshape(n_shapes, n_edges)

    # Emitted records (counter zeroed -> edge range [0, n_edge), face [n_edge, n_edge+n_face)).
    rec_shape = contacts.soft_contact_shape.numpy()
    rec_prim = contacts.soft_contact_primitive.numpy()
    edge_owner = np.asarray(model.soft_mesh_adjacency.edge_tri_indices)[:, 0]

    emitted_faces = {(int(rec_shape[i]), int(rec_prim[i])) for i in range(n_edge_rec, n_edge_rec + n_face_rec)}
    emitted_edge_owner = Counter((int(rec_shape[i]), int(rec_prim[i])) for i in range(n_edge_rec))

    delta = 0.03  # margin band: optimizer tail + brute grid step; borderline cases are not asserted

    # Faces match exactly on (shape, tri).
    for s in range(n_shapes):
        for t in range(n_tris):
            if face_min[s, t] < margin - delta:
                test.assertIn(
                    (s, t), emitted_faces, f"false negative: face (shape {s}, tri {t}) phi={face_min[s, t]:.4f}"
                )
    for s, t in emitted_faces:
        test.assertLess(face_min[s, t], margin + delta, f"false positive: face (shape {s}, tri {t})")

    # Edges: one record per near owned edge, but the bary degenerates to a vertex when the contact is
    # at an endpoint, so match by owner triangle + count. For each (shape, owner-tri) the number of
    # emitted edge records must lie within [#edges clearly inside, #edges possibly inside].
    for s in range(n_shapes):
        for t in range(n_tris):
            near_lo = sum(1 for e in range(n_edges) if edge_owner[e] == t and edge_min[s, e] < margin - delta)
            near_hi = sum(1 for e in range(n_edges) if edge_owner[e] == t and edge_min[s, e] < margin + delta)
            got = emitted_edge_owner[(s, t)]
            test.assertGreaterEqual(got, near_lo, f"false negative: edges of (shape {s}, tri {t}): {got} < {near_lo}")
            test.assertLessEqual(got, near_hi, f"false positive: edges of (shape {s}, tri {t}): {got} > {near_hi}")


add_function_test(
    TestWaterTightSoftContact,
    "test_end_to_end_no_false_pos_neg",
    test_end_to_end_no_false_pos_neg,
    devices=devices,
    check_output=False,  # CPU emits a benign warning when the mesh's texture SDF cannot be provisioned
)


def test_graph_capture_stable(test, device):
    """A flag-on collide is CUDA-graph-capturable and replays to identical soft-contact counts."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.5
    )
    builder.add_cloth_grid(
        pos=wp.vec3(-0.4, -0.4, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        cell_x=0.2,
        cell_y=0.2,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_water_tight_rigid_soft_contact=True
    )
    contacts = pipeline.contacts()
    state = model.state()

    # Warm up so all kernels are compiled before capture.
    pipeline.collide(state, contacts)
    counts0 = contacts.soft_contact_count.numpy().copy()
    test.assertGreater(int(counts0[1]) + int(counts0[2]), 0)

    # Capture the flag-on collide and replay it; counts must be stable across replays.
    with wp.ScopedCapture(device) as capture:
        pipeline.collide(state, contacts)
    for _ in range(3):
        wp.capture_launch(capture.graph)
        test.assertTrue(np.array_equal(contacts.soft_contact_count.numpy(), counts0))


add_function_test(
    TestWaterTightSoftContact,
    "test_graph_capture_stable",
    test_graph_capture_stable,
    devices=get_cuda_test_devices(),
)


def test_face_cull_uses_max_vertex_reach(test, device):
    """Regression (B6): the FACE cull reach must be the max centroid-to-vertex distance, not circumradius.

    A deliberately non-equilateral triangle whose near vertex is also the one *farthest* from the
    centroid, so circumradius (~0.124) is smaller than the true reach (~0.163). The near vertex sits
    inside the sphere's contact margin (phi ~= 0.005 < 0.01), so a real FACE contact exists -- but the
    centroid SDF (~0.168) exceeds ``margin + circumradius`` (~0.134), so the old circumradius cull
    dropped the whole triangle. The correct reach keeps it (``margin + reach`` ~= 0.173 > 0.168).
    A sphere gives an unambiguous radial SDF, so the culled point is genuinely within margin.
    """
    builder = newton.ModelBuilder()
    builder.add_shape_sphere(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.1)
    # Near vertex b0 (just outside the sphere along +x) is farthest from the centroid; the a0/c0
    # cluster sits far out along +x, making the triangle strongly non-equilateral.
    b0 = builder.add_particle(wp.vec3(0.105, 0.0, 0.0), wp.vec3(0.0), 0.0, radius=0.0)
    a0 = builder.add_particle(wp.vec3(0.35, 0.03, 0.0), wp.vec3(0.0), 0.0, radius=0.0)
    c0 = builder.add_particle(wp.vec3(0.35, -0.03, 0.0), wp.vec3(0.0), 0.0, radius=0.0)
    builder.add_triangle(b0, a0, c0)

    builder.color()
    builder.enable_rigid_mesh_sdfs()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.01, enable_water_tight_rigid_soft_contact=True
    )
    contacts = pipeline.contacts()
    state = model.state()

    pipeline.collide(state, contacts)
    counts = contacts.soft_contact_count.numpy()
    # counts = [particle, edge, face]. The FACE pass must emit the contact circumradius used to drop.
    test.assertGreater(
        int(counts[2]), 0, "FACE contact wrongly culled: the cull reach must be the max centroid-to-vertex distance"
    )


add_function_test(
    TestWaterTightSoftContact,
    "test_face_cull_uses_max_vertex_reach",
    test_face_cull_uses_max_vertex_reach,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
