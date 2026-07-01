# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for VBD's consumption of water-tight EDGE/FACE soft contacts.

The detection back-end (``newton/tests/test_water_tight_soft_contact.py``) verifies the
``Contacts`` records. These tests verify the *solver* side: ``SolverVBD`` must turn an
edge/face soft contact into per-vertex forces (barycentric distribution) so a soft body
whose vertices are all outside the legacy particle margin still cannot tunnel through a
rigid shape.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.particle_vbd_kernels import accumulate_particle_body_contact_force_and_hessian
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_edge_over_post(device):
    """One soft triangle whose v0-v1 edge spans across a narrow tall box ("post").

    All three vertices sit well outside the box's contact margin (so the legacy
    particle-vs-shape pass emits *nothing*: ``soft_contact_count[0] == 0``), while the
    edge interior and the face centroid dip ~0.03 below the box's top (+y) face. Only the
    water-tight EDGE/FACE passes can detect this, and only the new VBD section 2 can act
    on it. Gravity is disabled so the contact push-out is the only force.
    """
    builder = newton.ModelBuilder()
    builder.gravity = 0.0

    # Narrow tall post centered at the origin: x,z in [-0.1, 0.1], top face at y = +0.5.
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=0.1, hy=0.5, hz=0.1
    )

    # Triangle at y = 0.47 (0.03 below the top face). v0/v1 span the post in x; v2 reaches
    # out in +z. Every vertex is >= 0.3 outside the post in x or z -> outside any margin.
    v0 = builder.add_particle(wp.vec3(-0.4, 0.47, 0.0), wp.vec3(0.0), 0.1)
    v1 = builder.add_particle(wp.vec3(0.4, 0.47, 0.0), wp.vec3(0.0), 0.1)
    v2 = builder.add_particle(wp.vec3(0.0, 0.47, 0.4), wp.vec3(0.0), 0.1)
    builder.add_triangle(v0, v1, v2)

    builder.color()
    model = builder.finalize(device=device, enable_water_tight_rigid_soft_contact=True)
    return model, (v0, v1, v2)


def test_edge_face_pushes_vertices_out(test, device):
    """A soft edge/face penetrating a rigid box pushes its triangle's vertices out (+y).

    With section 2 absent the particle force stays zero (legacy count is 0, gravity off),
    so the vertices never move. With section 2 present the barycentric distribution drives
    v0 and v1 (the spanning edge) up out of the box.
    """
    model, (v0, v1, _v2) = _build_edge_over_post(device)

    margin = 0.1
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=margin, enable_water_tight_rigid_soft_contact=True
    )
    contacts = pipeline.contacts()
    state_in = model.state()
    state_out = model.state()

    pipeline.collide(state_in, contacts)

    counts = contacts.soft_contact_count.numpy()
    # Precondition: legacy particle pass found nothing; the edge/face passes did.
    test.assertEqual(int(counts[0]), 0, "vertices should be outside the legacy particle margin")
    test.assertGreater(int(counts[1]) + int(counts[2]), 0, "edge/face contacts must be detected")

    solver = newton.solvers.SolverVBD(model)

    y0_before = state_in.particle_q.numpy()[:, 1].copy()
    solver.step(state_in, state_out, None, contacts, dt=1.0 / 60.0)
    y0_after = state_out.particle_q.numpy()[:, 1]

    # The two vertices of the spanning edge are pushed up out of the +y face.
    test.assertGreater(y0_after[v0] - y0_before[v0], 1.0e-3, "v0 should be pushed +y")
    test.assertGreater(y0_after[v1] - y0_before[v1], 1.0e-3, "v1 should be pushed +y")


def _build_sphere_on_fixed_soft_triangle(device):
    """A dynamic sphere resting on a FIXED soft triangle via a soft FACE contact.

    The triangle's three vertices have mass 0 (kinematic -> VBD never moves them) and lie in
    the z=0 plane, spanning wider than the sphere. The sphere bottom starts just below z=0 so
    the triangle face penetrates immediately, and gravity (-z) pulls the sphere down. Every
    triangle vertex is well outside the sphere, so the legacy particle pass finds nothing:
    only the *body-side* reaction from the soft FACE contact can keep the sphere from falling
    through. A sphere (convex SDF, unambiguous radial normal) keeps the contact normal stable
    as the body moves, isolating the body-side reaction under test.
    """
    builder = newton.ModelBuilder()  # up_axis = Z, gravity = -9.81 along -Z

    v0 = builder.add_particle(wp.vec3(-0.3, -0.3, 0.0), wp.vec3(0.0), 0.0, radius=0.0)
    v1 = builder.add_particle(wp.vec3(0.3, -0.3, 0.0), wp.vec3(0.0), 0.0, radius=0.0)
    v2 = builder.add_particle(wp.vec3(0.0, 0.3, 0.0), wp.vec3(0.0), 0.0, radius=0.0)
    builder.add_triangle(v0, v1, v2)

    # Sphere bottom (z = center - radius) starts slightly below z=0 -> immediate penetration.
    inertia = wp.mat33(2.0e-3, 0.0, 0.0, 0.0, 2.0e-3, 0.0, 0.0, 0.0, 2.0e-3)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.095), wp.quat_identity()),
        mass=0.5,
        inertia=inertia,
        lock_inertia=True,
    )
    builder.add_shape_sphere(body=body, radius=0.1)

    builder.color()
    model = builder.finalize(device=device, enable_water_tight_rigid_soft_contact=True)
    return model, body


def test_edge_face_reacts_on_rigid_body(test, device):
    """The body-side reaction from a soft FACE contact supports a falling rigid box (S-a).

    Without the body-side section the body gets no reaction and free-falls through the fixed
    triangle (~4.9 m over 1 s); with it, the body is held up near its initial height.
    """
    model, body = _build_sphere_on_fixed_soft_triangle(device)

    margin = 0.1
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=margin, enable_water_tight_rigid_soft_contact=True
    )
    contacts = pipeline.contacts()
    state_in = model.state()
    state_out = model.state()

    pipeline.collide(state_in, contacts)
    counts = contacts.soft_contact_count.numpy()
    test.assertEqual(int(counts[0]), 0, "triangle vertices should be outside the legacy particle margin")
    test.assertGreater(int(counts[1]) + int(counts[2]), 0, "a soft edge/face contact must be detected")

    solver = newton.solvers.SolverVBD(model)
    dt = 1.0 / 60.0
    z_before = float(state_in.body_q.numpy()[body, 2])

    for _ in range(60):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in

    z_after = float(state_in.body_q.numpy()[body, 2])
    test.assertGreater(z_after, z_before - 0.05, "box should be supported by the soft contact, not free-fall")


def _set_slot(arr, idx, value):
    a = arr.numpy()
    a[idx] = value
    arr.assign(a)


def _run_face_section2(device, shape_margin):
    """Build a single soft-FACE contact and launch the particle-side kernel once with the given
    ``shape_margin`` array. The geometry gives a 0.05 penetration along +z; returns
    ``(forces, hessians, ke, bary, (p0, p1, p2))``. All vertices share color 0 so one launch
    processes the whole triangle."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0), wp.quat_identity()), hx=1.0, hy=1.0, hz=1.0)
    p0 = builder.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0), 0.1, radius=0.0)
    p1 = builder.add_particle(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0), 0.1, radius=0.0)
    p2 = builder.add_particle(wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0), 0.1, radius=0.0)
    builder.add_triangle(p0, p1, p2)
    model = builder.finalize(device=device, enable_water_tight_rigid_soft_contact=True)

    smax = 8
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=0.1, soft_contact_max=smax)
    contacts = pipeline.contacts()
    state = model.state()

    # One FACE record. Contact point x = 0.6 v0 + 0.3 v1 + 0.1 v2 = (0.3, 0.1, 0); put the
    # rigid point 0.05 above it along +z so penetration = -(dot(n, x - bx)) = 0.05 > 0.
    bary = [0.6, 0.3, 0.1]
    contacts.soft_contact_count.assign([0, 0, 1])
    _set_slot(contacts.soft_contact_primitive, 0, 0)  # soft triangle 0
    _set_slot(contacts.soft_contact_barycentric, 0, bary)
    _set_slot(contacts.soft_contact_shape, 0, 0)
    _set_slot(contacts.soft_contact_body_pos, 0, [0.3, 0.1, 0.05])
    _set_slot(contacts.soft_contact_body_vel, 0, [0.0, 0.0, 0.0])
    _set_slot(contacts.soft_contact_normal, 0, [0.0, 0.0, 1.0])
    model.particle_colors.assign([0, 0, 0])

    # Dummy single-entry body arrays (the record's shape is on the world, body = -1, so these
    # are never indexed) to avoid passing empty/None body state.
    body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
    body_qd = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    body_com = wp.zeros(1, dtype=wp.vec3, device=device)
    forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
    hessians = wp.zeros(model.particle_count, dtype=wp.mat33, device=device)
    unused = wp.zeros(smax, dtype=float, device=device)  # AVBD params (section 1 never runs, c0 == 0)

    wp.launch(
        accumulate_particle_body_contact_force_and_hessian,
        dim=smax,
        inputs=[
            0.01,  # dt
            0,  # current_color
            state.particle_q,  # pos_anchor == pos -> no damping / friction
            state.particle_q,
            model.particle_colors,
            1.0,  # friction_epsilon
            model.particle_radius,
            contacts.soft_contact_primitive,
            contacts.soft_contact_count,
            smax,
            unused,  # body_particle_contact_penalty_k
            unused,  # body_particle_contact_material_ke
            unused,  # body_particle_contact_material_kd
            unused,  # body_particle_contact_material_mu
            model.shape_material_mu,
            model.shape_body,
            body_q,
            body_q,
            body_qd,
            body_com,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
            shape_margin,
            model.tri_indices,
            contacts.soft_contact_barycentric,
            model.soft_contact_ke,
            model.soft_contact_kd,
            model.soft_contact_mu,
        ],
        outputs=[forces, hessians],
        device=device,
    )
    return forces.numpy(), hessians.numpy(), float(model.soft_contact_ke), bary, (p0, p1, p2)


def test_barycentric_force_distribution(test, device):
    """Section 2 distributes a contact at x = sum_i bary_i*v_i as bary_i*F and bary_i^2*H.

    A single FACE record with an asymmetric barycentric weight isolates the distribution math:
    the per-vertex force must scale with bary_i and the per-vertex Hessian block with bary_i^2.
    """
    f, h, ke, bary, (p0, p1, p2) = _run_face_section2(device, wp.zeros(0, dtype=float, device=device))
    single_force = np.array([0.0, 0.0, 0.05 * ke])  # F = n * penetration * ke

    for i, vi in enumerate([p0, p1, p2]):
        np.testing.assert_allclose(f[vi], bary[i] * single_force, rtol=2e-4, atol=1e-4)
        # Hessian block = bary_i^2 * ke * outer(n, n); only the zz entry is non-zero.
        np.testing.assert_allclose(h[vi][2, 2], bary[i] ** 2 * ke, rtol=2e-4, atol=1e-4)
    # The distributed force sums back to the single-point force (sum of bary == 1).
    np.testing.assert_allclose(f[p0] + f[p1] + f[p2], single_force, rtol=2e-4, atol=1e-4)


def test_edge_face_uses_shape_margin(test, device):
    """A per-shape contact margin (#2994) widens the edge/face penetration by ``margin``.

    Same single-FACE scene; the geometric penetration is 0.05. With ``shape_margin = 0`` the
    total force is ke*0.05; with ``shape_margin = m`` for the contacted shape it is ke*(0.05+m).
    """
    m = 0.02
    f0, _, ke, _, verts = _run_face_section2(device, wp.zeros(0, dtype=float, device=device))
    fm, _, _, _, _ = _run_face_section2(device, wp.array([m], dtype=float, device=device))  # shape 0 margin
    verts = list(verts)
    np.testing.assert_allclose(f0[verts].sum(axis=0), [0.0, 0.0, 0.05 * ke], rtol=2e-4, atol=1e-4)
    np.testing.assert_allclose(fm[verts].sum(axis=0), [0.0, 0.0, (0.05 + m) * ke], rtol=2e-4, atol=1e-4)


def test_flag_off_is_inert(test, device):
    """With the flag off the edge/face passes produce nothing and section 2 is a pure no-op.

    Reuses the edge-over-post scene (gravity disabled, every vertex outside the legacy
    margin). Flag on pushes the vertices out (test_edge_face_pushes_vertices_out); flag off
    must leave them exactly where they started -- the new path is inert and the legacy path
    is untouched, so flag-off behavior is unchanged.
    """
    model, _verts = _build_edge_over_post(device)
    # Flag OFF at construction: the buffer has no edge/face headroom and the passes never run.
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_water_tight_rigid_soft_contact=False
    )
    contacts = pipeline.contacts()
    state_in = model.state()
    state_out = model.state()

    pipeline.collide(state_in, contacts)
    counts = contacts.soft_contact_count.numpy()
    test.assertEqual(int(counts[0]) + int(counts[1]) + int(counts[2]), 0, "flag off => no soft contacts")

    q_before = state_in.particle_q.numpy().copy()
    solver = newton.solvers.SolverVBD(model)
    solver.step(state_in, state_out, None, contacts, dt=1.0 / 60.0)
    q_after = state_out.particle_q.numpy()

    np.testing.assert_allclose(q_after, q_before, atol=1.0e-6, err_msg="flag off must not move the soft body")


devices = get_test_devices()


class TestVBDWaterTightContact(unittest.TestCase):
    pass


add_function_test(
    TestVBDWaterTightContact,
    "test_edge_face_pushes_vertices_out",
    test_edge_face_pushes_vertices_out,
    devices=devices,
)
add_function_test(
    TestVBDWaterTightContact,
    "test_edge_face_reacts_on_rigid_body",
    test_edge_face_reacts_on_rigid_body,
    devices=devices,
)
add_function_test(
    TestVBDWaterTightContact,
    "test_barycentric_force_distribution",
    test_barycentric_force_distribution,
    devices=devices,
)
add_function_test(
    TestVBDWaterTightContact,
    "test_edge_face_uses_shape_margin",
    test_edge_face_uses_shape_margin,
    devices=devices,
)
add_function_test(
    TestVBDWaterTightContact,
    "test_flag_off_is_inert",
    test_flag_off_is_inert,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
