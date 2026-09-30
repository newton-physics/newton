# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contact force export of :class:`newton.solvers.SolverVBD`.

``SolverVBD.update_contacts`` writes ``Contacts.force``: the rigid-contact rows with the body-body wrenches
when VBD integrates the rigid bodies, and the soft-contact rows with the rigid-soft wrenches (particle, edge,
and face records against rigid shapes). Every row holds the force on body 0 (the contacted shape's body for
soft records) and its torque about that body's center of mass (world origin for static shapes), evaluated at
the final configuration of the preceding step with the solver's own contact law.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorContact
from newton.tests.unittest_utils import (
    add_function_test,
    configure_sdf_for_collision_shapes,
    get_cuda_test_devices,
    get_test_devices,
)

devices = get_test_devices()
cuda_devices = get_cuda_test_devices()

GRAVITY = 9.81


# -----------------------------------------------------------------------------
# NumPy reference of the documented body-particle contact law
# -----------------------------------------------------------------------------


def _quat_rotate_np(q, v):
    q_vec = np.asarray(q[:3], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(q_vec, v)
    return v + float(q[3]) * t + np.cross(q_vec, t)


def _transform_point_np(xform, point):
    return np.asarray(xform[:3], dtype=np.float64) + _quat_rotate_np(xform[3:], point)


def _active_soft_rows(contacts):
    """Return ``(count, rows)`` where ``rows`` are the active soft-contact rows of ``contacts.force``."""
    count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
    start = contacts.rigid_contact_max
    return count, contacts.force.numpy()[start : start + count].astype(np.float64)


def _active_rigid_rows(contacts):
    """Return ``(count, rows)`` where ``rows`` are the active rigid-contact rows of ``contacts.force``."""
    count = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
    return count, contacts.force.numpy()[:count].astype(np.float64)


def _expected_soft_contact_wrenches(
    model,
    contacts,
    *,
    particle_q,
    particle_q_prev,
    body_q,
    body_q_prev,
    body_qd,
    dt,
    friction_epsilon,
):
    """Evaluate the documented body-particle contact law in NumPy for every active soft contact.

    Reproduces the law from model data alone: the mixed contact material (arithmetic-mean ke/kd,
    geometric-mean mu of the global soft material and the shape material) at full stiffness, absolute
    damping while the contact point approaches the surface, and regularized isotropic Coulomb friction
    on the slip between the barycentric soft point and the shape surface over the step. Returns the
    force on the shape's body and its torque about the body COM (world origin for static shapes) as an
    array of shape ``(count, 6)``.
    """
    count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
    indices = contacts.soft_contact_indices.numpy()[:count]
    barycentric = contacts.soft_contact_barycentric.numpy()[:count].astype(np.float64)
    shapes = contacts.soft_contact_shape.numpy()[:count]
    body_pos = contacts.soft_contact_body_pos.numpy()[:count].astype(np.float64)
    body_vel = contacts.soft_contact_body_vel.numpy()[:count].astype(np.float64)
    normals = contacts.soft_contact_normal.numpy()[:count].astype(np.float64)

    shape_body = model.shape_body.numpy()
    shape_ke = model.shape_material_ke.numpy()
    shape_kd = model.shape_material_kd.numpy()
    shape_mu = model.shape_material_mu.numpy()
    shape_margin = model.shape_margin.numpy()
    particle_radius = model.particle_radius.numpy()
    body_com = model.body_com.numpy() if model.body_count > 0 else None

    particle_q = np.asarray(particle_q, dtype=np.float64)
    particle_q_prev = np.asarray(particle_q_prev, dtype=np.float64)

    expected = np.zeros((count, 6))
    for i in range(count):
        corners = [int(c) for c in indices[i] if c >= 0]
        weights = barycentric[i][: len(corners)]
        x = sum(w * particle_q[c] for w, c in zip(weights, corners, strict=True))
        x_prev = sum(w * particle_q_prev[c] for w, c in zip(weights, corners, strict=True))
        radius = max(float(particle_radius[c]) for c in corners)

        shape = int(shapes[i])
        body = int(shape_body[shape])
        n = normals[i]
        if body >= 0:
            xform = np.asarray(body_q[body], dtype=np.float64)
            com_world = _transform_point_np(xform, body_com[body])
            bx = _transform_point_np(xform, body_pos[i])
            surface_vel = _quat_rotate_np(xform[3:], body_vel[i])
            if body_q_prev is not None:
                bx_prev = _transform_point_np(np.asarray(body_q_prev[body], dtype=np.float64), body_pos[i])
                bv = (bx - bx_prev) / dt + surface_vel
            else:
                twist = np.asarray(body_qd[body], dtype=np.float64)
                bv = twist[:3] + np.cross(twist[3:], bx - com_world) + surface_vel
        else:
            com_world = np.zeros(3)
            bx = body_pos[i]
            bv = body_vel[i]

        margin = float(shape_margin[shape]) if shape_margin.shape[0] > 0 else 0.0
        penetration = radius + margin - float(np.dot(n, x - bx))
        if penetration <= 0.0:
            continue

        ke = 0.5 * (float(model.soft_contact_ke) + float(shape_ke[shape]))
        kd = 0.5 * (float(model.soft_contact_kd) + float(shape_kd[shape]))
        mu = np.sqrt(float(model.soft_contact_mu) * float(shape_mu[shape]))

        f_n = ke * penetration
        force = n * f_n
        u = (x - x_prev) - bv * dt
        u_n = float(np.dot(n, u))
        if u_n < 0.0:
            force = force - (kd / dt) * n * u_n
        u_t = u - n * u_n
        slip = float(np.linalg.norm(u_t))
        eps_u = friction_epsilon * dt
        if slip > 0.0:
            scale = 1.0 / slip if slip > eps_u else (-slip / eps_u + 2.0) / eps_u
            force = force - mu * f_n * scale * u_t

        force_on_body = -force
        expected[i, :3] = force_on_body
        expected[i, 3:] = np.cross(bx - com_world, force_on_body)
    return expected


# -----------------------------------------------------------------------------
# Scenes
# -----------------------------------------------------------------------------


def _build_particle_on_ground(device, *, pos, radius, mass=1.0, gravity=True):
    """A single particle over a static ground plane (Z up); no rigid bodies."""
    builder = newton.ModelBuilder()
    if not gravity:
        builder.gravity = (0.0, 0.0, 0.0)
    builder.add_ground_plane()
    builder.add_particle(pos=wp.vec3(*pos), vel=wp.vec3(0.0), mass=mass, radius=radius)
    builder.color()
    return builder.finalize(device=device)


def _build_sphere_on_ground(device, *, with_particle):
    """A dynamic sphere resting on the ground plane, optionally with a particle resting nearby.

    Stiff shape material keeps the sphere's penetration small. The sphere sits away from the world
    origin, so a rigid row whose shape 0 is the static ground carries a torque about the origin.
    Returns ``(model, body, sphere_shape)``.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e2
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()
    radius = 0.1
    body = builder.add_body(xform=wp.transform(wp.vec3(0.3, -0.2, radius - 0.002), wp.quat_identity()))
    sphere_shape = builder.add_shape_sphere(body, radius=radius)
    if with_particle:
        # 0.2 mm penetration is close to the particle's equilibrium against the stiff shape material,
        # so it stays loaded instead of bouncing clear of the surface.
        builder.add_particle(pos=wp.vec3(-0.5, 0.4, 0.0498), vel=wp.vec3(0.0), mass=1.0, radius=0.05)
    builder.color()
    return builder.finalize(device=device), body, sphere_shape


def _build_box_under_particles(device, *, penetration, particle_radius=0.02, box_mass=1.0e4):
    """A heavy dynamic box (top face at z = 0.1) with two light particles pressed into its top face.

    Particle 0 is kinematic (zero mass) so a moving box slides under it; particle 1 is dynamic and is
    dragged along by friction. Gravity is disabled so the box keeps a prescribed velocity through the
    step, and both particles sit off-center so the contact forces produce torques about the box COM.
    Returns ``(model, body)``.
    """
    builder = newton.ModelBuilder()
    builder.gravity = (0.0, 0.0, 0.0)
    inertia = wp.mat33(1.0e3, 0.0, 0.0, 0.0, 1.0e3, 0.0, 0.0, 0.0, 1.0e3)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=box_mass,
        inertia=inertia,
        lock_inertia=True,
    )
    builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.1)
    z = 0.1 + particle_radius - penetration
    builder.add_particle(pos=wp.vec3(0.05, -0.02, z), vel=wp.vec3(0.0), mass=0.0, radius=particle_radius)
    builder.add_particle(pos=wp.vec3(-0.15, 0.1, z), vel=wp.vec3(0.0), mass=1.0, radius=particle_radius)
    builder.color()
    return builder.finalize(device=device), body


def _build_sphere_on_fixed_triangle(device):
    """A dynamic sphere resting on a fixed soft triangle through a soft FACE contact.

    The triangle vertices are massless (never moved by VBD) and lie far outside the sphere, so only the
    full-surface face pass detects the contact and only its body-side reaction supports the sphere.
    """
    builder = newton.ModelBuilder()
    v0 = builder.add_particle(wp.vec3(-0.3, -0.3, 0.0), wp.vec3(0.0), 0.0, radius=0.0)
    v1 = builder.add_particle(wp.vec3(0.3, -0.3, 0.0), wp.vec3(0.0), 0.0, radius=0.0)
    v2 = builder.add_particle(wp.vec3(0.0, 0.3, 0.0), wp.vec3(0.0), 0.0, radius=0.0)
    builder.add_triangle(v0, v1, v2)
    inertia = wp.mat33(2.0e-3, 0.0, 0.0, 0.0, 2.0e-3, 0.0, 0.0, 0.0, 2.0e-3)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.095), wp.quat_identity()),
        mass=0.5,
        inertia=inertia,
        lock_inertia=True,
    )
    builder.add_shape_sphere(body=body, radius=0.1)
    builder.color()
    configure_sdf_for_collision_shapes(builder)
    return builder.finalize(device=device), body


def _build_edge_over_post(device):
    """A two-triangle soft strip whose shared edge spans a narrow static post; gravity disabled.

    Every vertex is outside the post's contact margin, so only the full-surface EDGE/FACE passes detect
    the ~0.03 dip of the shared edge and of both faces below the post's top (+y) face. The shared edge
    is registered with :meth:`ModelBuilder.add_edge` so the EDGE pass emits an edge record for it.
    """
    builder = newton.ModelBuilder()
    builder.gravity = (0.0, 0.0, 0.0)
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=0.1, hy=0.5, hz=0.1
    )
    v0 = builder.add_particle(wp.vec3(-0.4, 0.47, 0.0), wp.vec3(0.0), 0.1)
    v1 = builder.add_particle(wp.vec3(0.4, 0.47, 0.0), wp.vec3(0.0), 0.1)
    o0 = builder.add_particle(wp.vec3(0.0, 0.47, 0.4), wp.vec3(0.0), 0.1)
    o1 = builder.add_particle(wp.vec3(0.0, 0.47, -0.4), wp.vec3(0.0), 0.1)
    builder.add_triangle(v0, v1, o0)
    builder.add_triangle(v1, v0, o1)
    builder.add_edge(o0, o1, v0, v1)
    builder.color()
    configure_sdf_for_collision_shapes(builder)
    return builder.finalize(device=device)


def _advance(pipeline, solver, contacts, state_in, state_out, dt, *, export=True):
    """Collide, step, optionally export, and copy the result back into ``state_in``.

    The copy-back (instead of a Python-level swap) keeps the same arrays bound across calls, so the
    sequence can be captured into a CUDA graph and replayed to advance the simulation.
    """
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, None, contacts, dt)
    if export:
        solver.update_contacts(contacts, state_out)
    wp.copy(state_in.particle_q, state_out.particle_q)
    wp.copy(state_in.particle_qd, state_out.particle_qd)
    if state_in.body_q is not None:
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_vbd_soft_contact_force_static_equilibrium(test, device):
    """Report the weight of a resting particle as a downward force on the static ground with torque about the origin.

    A particle settling on the ground plane under gravity must carry a contact force equal to its
    weight. The force is reported on the shape's body (the static ground, body -1), so it points down,
    and the torque is the moment of that force about the world origin applied at the ground contact
    point below the particle.
    """
    mass = 1.0
    radius = 0.05
    pos = (0.3, -0.2, radius - 0.004)
    model = _build_particle_on_ground(device, pos=pos, radius=radius, mass=mass)
    model.request_contact_attributes("force")

    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01)
    contacts = pipeline.contacts()
    test.assertIsNotNone(contacts.force)
    solver = newton.solvers.SolverVBD(model, iterations=10)
    state_in, state_out = model.state(), model.state()

    dt = 1.0 / 60.0
    for _ in range(300):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in
    solver.update_contacts(contacts, state_in)

    count, rows = _active_soft_rows(contacts)
    test.assertEqual(count, 1)
    weight = mass * GRAVITY
    force = rows[0, :3]
    torque = rows[0, 3:]
    np.testing.assert_allclose(force, [0.0, 0.0, -weight], rtol=1.0e-3, atol=1.0e-3 * weight)
    # Contact point on the ground is directly below the particle: torque = r x F about the world origin.
    q = state_in.particle_q.numpy()[0].astype(np.float64)
    expected_torque = np.cross([q[0], q[1], 0.0], [0.0, 0.0, -weight])
    np.testing.assert_allclose(torque, expected_torque, rtol=1.0e-3, atol=1.0e-3 * weight)
    test.assertGreater(abs(expected_torque[0]) + abs(expected_torque[1]), 0.1 * weight)

    # The particle has come to rest and its weight is balanced by the reported force.
    np.testing.assert_allclose(state_in.particle_qd.numpy()[0], 0.0, atol=1.0e-4)


def test_vbd_soft_contact_force_zero_when_separated(test, device):
    """Report exactly zero force for a soft contact record whose particle does not touch the shape."""
    radius = 0.05
    gap = 0.02
    model = _build_particle_on_ground(device, pos=(0.0, 0.0, radius + 0.5 * gap), radius=radius, gravity=False)
    model.request_contact_attributes("force")

    pipeline = newton.CollisionPipeline(model, soft_contact_gap=gap)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(model, iterations=2)
    state_in, state_out = model.state(), model.state()

    pipeline.collide(state_in, contacts)
    test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0, "gap should produce a soft contact record")
    solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)
    solver.update_contacts(contacts, state_out)

    count, rows = _active_soft_rows(contacts)
    test.assertEqual(count, 1)
    np.testing.assert_array_equal(rows, 0.0)
    np.testing.assert_array_equal(state_out.particle_q.numpy(), state_in.particle_q.numpy())


def _check_moving_box_contact(test, device, *, external_rigid, tangential_speed):
    """Compare exported wrenches against the contact law for a box sliding and rising under two particles.

    ``tangential_speed`` sets the slip the kinematic particle sees over one step: above the friction
    regularization band the drag sits on the Coulomb limit, inside the band it follows the regularized ramp.
    """
    dt = 1.0 / 60.0
    friction_epsilon = 1.0e-2
    velocity = np.array([tangential_speed, 0.0, 0.3])
    penetration = 0.005

    model, body = _build_box_under_particles(device, penetration=penetration)
    model.request_contact_attributes("force")
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=10,
        friction_epsilon=friction_epsilon,
        rigid_compliant_alm=True,
        integrate_with_external_rigid_solver=external_rigid,
    )
    state_in, state_out = model.state(), model.state()

    twist = np.zeros((1, 6), dtype=np.float32)
    twist[0, :3] = velocity
    state_in.body_qd.assign(twist)
    if external_rigid:
        # The caller integrates the body: the pose at the end of the step goes to state_out.
        moved = state_in.body_q.numpy().copy()
        moved[0, :3] += (velocity * dt).astype(np.float32)
        state_out.body_q.assign(moved)
        state_out.body_qd.assign(twist)

    pipeline.collide(state_in, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 2)
    particle_q_prev = state_in.particle_q.numpy().copy()
    body_q_prev = state_in.body_q.numpy().copy()

    solver.step(state_in, state_out, None, contacts, dt)
    solver.update_contacts(contacts, state_out)

    body_q = state_out.body_q.numpy()
    # Sanity: the heavy box followed its prescribed velocity through the step.
    np.testing.assert_allclose(body_q[0, :3] - body_q_prev[0, :3], velocity * dt, rtol=1.0e-3, atol=1.0e-5)

    particle_q = state_out.particle_q.numpy()
    expected = _expected_soft_contact_wrenches(
        model,
        contacts,
        particle_q=particle_q,
        particle_q_prev=particle_q_prev,
        body_q=body_q,
        body_q_prev=body_q_prev,
        body_qd=state_out.body_qd.numpy(),
        dt=dt,
        friction_epsilon=friction_epsilon,
    )
    count, rows = _active_soft_rows(contacts)
    test.assertEqual(count, 2)
    scale = float(np.max(np.abs(expected[:, :3])))
    test.assertGreater(scale, 1.0)
    np.testing.assert_allclose(rows, expected, rtol=2.0e-4, atol=2.0e-4 * scale)

    normals = contacts.soft_contact_normal.numpy()[:count].astype(np.float64)
    np.testing.assert_allclose(normals, [[0.0, 0.0, 1.0]] * count, atol=1.0e-6)
    particles = contacts.soft_contact_indices.numpy()[:count, 0]
    kinematic = int(np.flatnonzero(particles == 0)[0])
    dynamic = int(np.flatnonzero(particles == 1)[0])

    shape_ke = float(model.shape_material_ke.numpy()[0])
    ke = 0.5 * (float(model.soft_contact_ke) + shape_ke)
    mu = np.sqrt(float(model.soft_contact_mu) * float(model.shape_material_mu.numpy()[0]))
    com_world = _transform_point_np(body_q[0].astype(np.float64), model.body_com.numpy()[body])
    body_pos = contacts.soft_contact_body_pos.numpy()[:count]

    particle_radius = model.particle_radius.numpy()
    penetrations = {}
    for i in (kinematic, dynamic):
        n = normals[i]
        particle = int(particles[i])
        bx = _transform_point_np(body_q[0].astype(np.float64), body_pos[i])
        gap = float(np.dot(n, particle_q[particle].astype(np.float64) - bx))
        penetrations[i] = float(particle_radius[particle]) - gap
        test.assertGreater(penetrations[i], 0.0)
        force_on_box = rows[i, :3]
        f_n = -float(np.dot(force_on_box, n))
        f_t = force_on_box - np.dot(force_on_box, n) * n
        # Normal: the box presses into the particle, so the force on the box points along -n.
        test.assertGreater(f_n, 0.0)
        # Friction: the box slides in +x under a particle that lags behind it, so the box feels a drag
        # opposing its motion, bounded by mu times the elastic normal load (the solver's Coulomb bound
        # excludes the damping part of the normal force).
        test.assertLess(f_t[0], -1.0e-3 * f_n)
        test.assertLessEqual(np.linalg.norm(f_t), mu * ke * penetrations[i] * (1.0 + 1.0e-3))
        # Torque reference: moment of the force about the box COM, applied at the shape-side contact point.
        np.testing.assert_allclose(
            rows[i, 3:], np.cross(bx - com_world, force_on_box), rtol=2.0e-4, atol=2.0e-4 * scale
        )
        test.assertGreater(float(np.linalg.norm(rows[i, 3:])), 1.0e-2 * f_n)

    # Kinematic particle: the box displacement is the whole slip, so the drag follows the regularized
    # Coulomb law exactly -- the limit ``mu * f_n`` beyond the regularization band, the smooth ramp
    # inside it. The box also rises into the fixed particle, so damping adds to the elastic normal force.
    n = normals[kinematic]
    force_on_box = rows[kinematic, :3]
    f_n = -float(np.dot(force_on_box, n))
    f_t = force_on_box - np.dot(force_on_box, n) * n
    # The slip is the tangential displacement of the shape-side contact point over the step (the box
    # is heavy but not infinitely so, so use its actual motion rather than the prescribed velocity).
    bx_prev = _transform_point_np(body_q_prev[0].astype(np.float64), body_pos[kinematic])
    bx = _transform_point_np(body_q[0].astype(np.float64), body_pos[kinematic])
    slip_vec = bx - bx_prev
    slip_vec = slip_vec - n * np.dot(n, slip_vec)
    slip = float(np.linalg.norm(slip_vec))
    eps_u = friction_epsilon * dt
    np.testing.assert_allclose(slip, tangential_speed * dt, rtol=1.0e-2)
    if tangential_speed * dt > eps_u:
        test.assertGreater(slip, eps_u)
        ramp = 1.0
    else:
        test.assertLess(slip, eps_u)
        ramp = slip * (-slip / eps_u + 2.0) / eps_u
    np.testing.assert_allclose(np.linalg.norm(f_t), mu * ke * penetrations[kinematic] * ramp, rtol=1.0e-3)
    test.assertGreater(f_n, ke * penetrations[kinematic] * 1.05)
    np.testing.assert_array_equal(particle_q[0], particle_q_prev[0])

    # Dynamic particle: friction drags it along the box's motion.
    test.assertGreater(float(particle_q[1, 0] - particle_q_prev[1, 0]), 0.0)


def test_vbd_soft_contact_force_friction_damping_moving_body(test, device):
    """Match the contact law at the Coulomb limit, with damping, for a body integrated by VBD."""
    _check_moving_box_contact(test, device, external_rigid=False, tangential_speed=1.0)


def test_vbd_soft_contact_force_friction_damping_external_body(test, device):
    """Match the contact law at the Coulomb limit, with damping, for an externally integrated body."""
    _check_moving_box_contact(test, device, external_rigid=True, tangential_speed=1.0)


def test_vbd_soft_contact_force_regularized_friction(test, device):
    """Match the regularized friction ramp when the slip over one step lies inside the regularization band."""
    # slip = 0.5 * friction_epsilon * dt -> drag = 0.75 * mu * elastic normal load.
    _check_moving_box_contact(test, device, external_rigid=False, tangential_speed=0.005)


def test_vbd_soft_contact_force_face_records_support_body(test, device):
    """Sum face-contact forces on a sphere resting on a fixed soft triangle to its weight.

    The sphere is supported only by the body-side reaction of soft FACE records, so the exported rows
    (force on the sphere) must add up to the sphere's weight pointing up, each row must push the sphere
    away from the triangle along the record normal, and each torque must be the moment of the row's
    force about the sphere COM applied at the shape-side contact point.
    """
    model, body = _build_sphere_on_fixed_triangle(device)
    model.request_contact_attributes("force")
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_gap=0.1, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(model, iterations=10, rigid_compliant_alm=True)
    state_in, state_out = model.state(), model.state()

    dt = 1.0 / 60.0
    for _ in range(240):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in
    solver.update_contacts(contacts, state_in)

    count, rows = _active_soft_rows(contacts)
    test.assertGreater(count, 0)
    indices = contacts.soft_contact_indices.numpy()[:count]
    test.assertTrue(np.all(indices >= 0), "only face records are expected in this scene")

    mass = float(model.body_mass.numpy()[body])
    weight = mass * GRAVITY
    np.testing.assert_allclose(rows[:, :3].sum(axis=0), [0.0, 0.0, weight], rtol=2.0e-2, atol=2.0e-2 * weight)
    # Vertical equilibrium: the vertical velocity has settled (a slow creep along the face is allowed).
    test.assertLess(abs(float(state_in.body_qd.numpy()[body, 2])), 1.0e-2)

    normals = contacts.soft_contact_normal.numpy()[:count].astype(np.float64)
    body_q = state_in.body_q.numpy()[body].astype(np.float64)
    com_world = _transform_point_np(body_q, model.body_com.numpy()[body])
    body_pos = contacts.soft_contact_body_pos.numpy()[:count]
    for i in range(count):
        test.assertLess(float(np.dot(rows[i, :3], normals[i])), 0.0)
        bx = _transform_point_np(body_q, body_pos[i])
        np.testing.assert_allclose(
            rows[i, 3:], np.cross(bx - com_world, rows[i, :3]), rtol=1.0e-4, atol=1.0e-4 * weight
        )


def test_vbd_soft_contact_force_edge_records_match_law(test, device):
    """Match the contact law for edge and face records pushing a soft triangle off a static post."""
    dt = 1.0 / 60.0
    friction_epsilon = 1.0e-2
    model = _build_edge_over_post(device)
    model.request_contact_attributes("force")
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_gap=0.1, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(model, iterations=10, friction_epsilon=friction_epsilon)
    state_in, state_out = model.state(), model.state()

    pipeline.collide(state_in, contacts)
    count = int(contacts.soft_contact_count.numpy()[0])
    indices = contacts.soft_contact_indices.numpy()[:count]
    test.assertGreater(count, 0)
    test.assertTrue(np.all(indices[:, 1] >= 0), "vertices are outside the particle margin")
    test.assertTrue(np.any(indices[:, 2] < 0), "an edge record is expected")

    particle_q_prev = state_in.particle_q.numpy().copy()
    solver.step(state_in, state_out, None, contacts, dt)
    solver.update_contacts(contacts, state_out)

    expected = _expected_soft_contact_wrenches(
        model,
        contacts,
        particle_q=state_out.particle_q.numpy(),
        particle_q_prev=particle_q_prev,
        body_q=None,
        body_q_prev=None,
        body_qd=None,
        dt=dt,
        friction_epsilon=friction_epsilon,
    )
    count, rows = _active_soft_rows(contacts)
    scale = float(np.max(np.abs(expected)))
    test.assertGreater(scale, 1.0)
    np.testing.assert_allclose(rows, expected, rtol=2.0e-4, atol=2.0e-4 * scale)

    # Every record with penetration pushes the post along -n (force on the static post's world body).
    normals = contacts.soft_contact_normal.numpy()[:count].astype(np.float64)
    active = np.linalg.norm(rows[:, :3], axis=1) > 0.0
    test.assertTrue(np.any(active))
    test.assertTrue(np.all(np.einsum("ij,ij->i", rows[active, :3], normals[active]) < 0.0))


def _settle_sphere_on_ground(device, *, with_particle, sensor=False, steps=300):
    """Settle the sphere scene and return the objects needed to inspect the last step's export."""
    model, body, sphere_shape = _build_sphere_on_ground(device, with_particle=with_particle)
    contact_sensor = None
    if sensor:
        contact_sensor = SensorContact(model, sensing_bodies=[body], verbose=False)
    else:
        model.request_contact_attributes("force")
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(model, iterations=10, rigid_compliant_alm=True)
    state_in, state_out = model.state(), model.state()

    dt = 1.0 / 60.0
    body_q_prev = None
    for step in range(steps):
        pipeline.collide(state_in, contacts)
        if step == steps - 1:
            body_q_prev = wp.clone(solver.body_q_prev)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in
    solver.update_contacts(contacts, state_in)
    return model, body, sphere_shape, solver, contacts, state_in, body_q_prev, dt, contact_sensor


def test_vbd_rigid_contact_force_static_equilibrium(test, device):
    """Report a resting sphere's weight in the rigid rows, matching the collector and the torque reference.

    Each rigid row is the force on body 0 by body 1 with its torque about body 0's center of mass
    (world origin when shape 0 is the static ground). The rows must sum to the sphere's weight once
    normalized to the sphere side, equal the negated force the public collector reports on body 1,
    and carry the moment of the force about body 0's reference taken at the geometric contact point.
    """
    model, body, sphere_shape, solver, contacts, state, body_q_prev, dt, _ = _settle_sphere_on_ground(
        device, with_particle=True
    )
    count, rows = _active_rigid_rows(contacts)
    test.assertGreater(count, 0)
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    test.assertTrue(np.all((shape0 == sphere_shape) | (shape1 == sphere_shape)))

    weight = float(model.body_mass.numpy()[body]) * GRAVITY
    sign = np.where(shape0 == sphere_shape, 1.0, -1.0)
    force_on_sphere = (sign[:, None] * rows[:, :3]).sum(axis=0)
    np.testing.assert_allclose(force_on_sphere, [0.0, 0.0, weight], rtol=2.0e-2, atol=2.0e-2 * weight)
    test.assertLess(abs(float(state.body_qd.numpy()[body, 2])), 1.0e-2)

    # The public collector evaluates the same contact law; it reports the force on body 1.
    body0, _body1, _p0, _p1, force_on_body1, collected = solver.collect_rigid_contact_forces(
        state.body_q, body_q_prev, contacts, dt
    )
    test.assertEqual(int(collected.numpy()[0]), count)
    np.testing.assert_allclose(rows[:, :3], -force_on_body1.numpy()[:count], rtol=1.0e-5, atol=1.0e-5 * weight)

    # Torque reference: body 0's COM, or the world origin for the static ground.
    body_q = state.body_q.numpy()
    body_com = model.body_com.numpy()
    point0 = contacts.rigid_contact_point0.numpy()[:count]
    body0 = body0.numpy()[:count]
    ground_rows = 0
    for i in range(count):
        if body0[i] >= 0:
            xform = body_q[body0[i]].astype(np.float64)
            lever = _transform_point_np(xform, point0[i]) - _transform_point_np(xform, body_com[body0[i]])
        else:
            lever = point0[i].astype(np.float64)
            ground_rows += 1
        np.testing.assert_allclose(rows[i, 3:], np.cross(lever, rows[i, :3]), atol=5.0e-3 * weight)
    if ground_rows:
        # The sphere sits 0.36 m from the origin, so ground-side rows carry a clear moment.
        test.assertGreater(float(np.max(np.linalg.norm(rows[:, 3:], axis=1))), 0.2 * weight)


def test_vbd_rigid_contact_force_feeds_sensor_contact(test, device):
    """Report a resting sphere's weight through SensorContact after update_contacts."""
    model, body, _shape, _solver, contacts, state, _prev, _dt, sensor = _settle_sphere_on_ground(
        device, with_particle=False, sensor=True
    )
    test.assertIsNotNone(contacts.force)
    sensor.update(state, contacts)
    weight = float(model.body_mass.numpy()[body]) * GRAVITY
    total = sensor.total_force.numpy()[0]
    np.testing.assert_allclose(total, [0.0, 0.0, weight], rtol=2.0e-2, atol=2.0e-2 * weight)


def test_vbd_contact_force_external_rigid_leaves_rigid_rows(test, device):
    """Leave the rigid rows to the external rigid solver and still write the soft rows."""
    model, _body, _shape = _build_sphere_on_ground(device, with_particle=True)
    model.request_contact_attributes("force")
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model, iterations=4, rigid_compliant_alm=True, integrate_with_external_rigid_solver=True
    )
    state_in, state_out = model.state(), model.state()
    # The external solver stands still: the body keeps its pose through the step.
    wp.copy(state_out.body_q, state_in.body_q)
    wp.copy(state_out.body_qd, state_in.body_qd)

    pipeline.collide(state_in, contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 1)
    solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)

    contacts.force.fill_(wp.spatial_vector(7.0, 7.0, 7.0, 7.0, 7.0, 7.0))
    solver.update_contacts(contacts, state_out)
    force = contacts.force.numpy()
    rigid_max = contacts.rigid_contact_max
    np.testing.assert_array_equal(force[:rigid_max], 7.0)
    test.assertLess(force[rigid_max, 2], 0.0)
    np.testing.assert_array_equal(force[rigid_max + 1 :], 0.0)


def test_vbd_soft_contact_force_export_leaves_simulation_unchanged(test, device):
    """Produce bitwise identical trajectories with and without the contact force export enabled."""
    dt = 1.0 / 60.0

    def run(build, *, request_force, steps, **pipeline_kwargs):
        model = build(device)
        if isinstance(model, tuple):
            model = model[0]
        if request_force:
            model.request_contact_attributes("force")
        pipeline = newton.CollisionPipeline(model, **pipeline_kwargs)
        contacts = pipeline.contacts()
        test.assertEqual(contacts.force is not None, request_force)
        solver_kwargs = {"rigid_compliant_alm": True} if model.body_count > 0 else {}
        solver = newton.solvers.SolverVBD(model, iterations=4, **solver_kwargs)
        state_in, state_out = model.state(), model.state()
        for _ in range(steps):
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, None, contacts, dt)
            if request_force:
                solver.update_contacts(contacts, state_out)
            state_in, state_out = state_out, state_in
        result = {"particle_q": state_in.particle_q.numpy().copy(), "particle_qd": state_in.particle_qd.numpy().copy()}
        if model.body_count > 0:
            result["body_q"] = state_in.body_q.numpy().copy()
            result["body_qd"] = state_in.body_qd.numpy().copy()
        return result

    scenes = {
        "particle_on_ground": (
            lambda d: _build_particle_on_ground(d, pos=(0.3, -0.2, 0.046), radius=0.05),
            {"soft_contact_gap": 0.01},
        ),
        "sphere_on_fixed_triangle": (
            _build_sphere_on_fixed_triangle,
            {"broad_phase": "nxn", "soft_contact_gap": 0.1, "enable_rigid_soft_full_surface_contact": True},
        ),
        "sphere_on_ground_with_particle": (
            lambda d: _build_sphere_on_ground(d, with_particle=True),
            {},
        ),
    }
    for name, (build, pipeline_kwargs) in scenes.items():
        with test.subTest(scene=name):
            baseline = run(build, request_force=False, steps=20, **pipeline_kwargs)
            exported = run(build, request_force=True, steps=20, **pipeline_kwargs)
            for key, value in baseline.items():
                np.testing.assert_array_equal(exported[key], value, err_msg=f"{name}: {key} differs with export on")


def test_vbd_contact_force_layout(test, device):
    """Write the active rigid and soft rows of ``Contacts.force`` and zero the inactive rows of both segments."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    box = builder.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 0.249), wp.quat_identity()))
    builder.add_shape_box(body=box, hx=0.25, hy=0.25, hz=0.25)
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.046), vel=wp.vec3(0.0), mass=1.0, radius=0.05)
    builder.color()
    model = builder.finalize(device=device)
    model.request_contact_attributes("force")

    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01, soft_contact_max=4)
    contacts = pipeline.contacts()
    test.assertGreater(contacts.rigid_contact_max, 0)
    test.assertEqual(contacts.soft_contact_max, 4)
    solver = newton.solvers.SolverVBD(model, iterations=4, rigid_compliant_alm=True)
    state_in, state_out = model.state(), model.state()

    pipeline.collide(state_in, contacts)
    n_rigid = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(n_rigid, 0)
    test.assertLess(n_rigid, contacts.rigid_contact_max)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 1)
    solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)

    contacts.force.fill_(wp.spatial_vector(7.0, 7.0, 7.0, 7.0, 7.0, 7.0))
    solver.update_contacts(contacts, state_out)

    force = contacts.force.numpy()
    rigid_max = contacts.rigid_contact_max
    # Active rigid rows carry the box-ground contact; the rest of the rigid segment is zero.
    test.assertGreater(float(np.max(np.abs(force[:n_rigid, :3]))), 0.0)
    np.testing.assert_array_equal(force[n_rigid:rigid_max], 0.0)
    # The active soft row carries the particle's contact force; the remaining soft rows are zero.
    test.assertLess(force[rigid_max, 2], 0.0)
    np.testing.assert_array_equal(force[rigid_max + 1 :], 0.0)


def test_vbd_update_contacts_requires_force_attribute(test, device):
    """Raise ValueError from update_contacts when ``contacts.force`` is not allocated."""
    model = _build_particle_on_ground(device, pos=(0.0, 0.0, 0.046), radius=0.05)
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(model, iterations=2)
    state_in, state_out = model.state(), model.state()

    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)

    test.assertIsNone(contacts.force)
    with test.assertRaises(ValueError):
        solver.update_contacts(contacts, state_out)


def test_vbd_update_contacts_requires_exporting_step(test, device):
    """Raise ValueError from update_contacts before a step has run with force-enabled contacts."""
    model = _build_particle_on_ground(device, pos=(0.0, 0.0, 0.046), radius=0.05)
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01)
    plain_contacts = pipeline.contacts()
    model.request_contact_attributes("force")
    contacts = pipeline.contacts()
    test.assertIsNotNone(contacts.force)
    solver = newton.solvers.SolverVBD(model, iterations=2)
    state_in, state_out = model.state(), model.state()

    with test.assertRaises(ValueError):
        solver.update_contacts(contacts, state_out)

    # A step without the force attribute does not produce export data either.
    pipeline.collide(state_in, plain_contacts)
    solver.step(state_in, state_out, None, plain_contacts, 1.0 / 60.0)
    with test.assertRaises(ValueError):
        solver.update_contacts(contacts, state_out)


def test_vbd_update_contacts_rejects_capacity_mismatch(test, device):
    """Raise ValueError from update_contacts when the Contacts capacity differs from the stepped buffer."""
    model = _build_particle_on_ground(device, pos=(0.0, 0.0, 0.046), radius=0.05)
    model.request_contact_attributes("force")
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01, soft_contact_max=4)
    contacts = pipeline.contacts()
    other_pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01, soft_contact_max=8)
    other_contacts = other_pipeline.contacts()
    solver = newton.solvers.SolverVBD(model, iterations=2)
    state_in, state_out = model.state(), model.state()

    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)
    solver.update_contacts(contacts, state_out)
    with test.assertRaises(ValueError):
        solver.update_contacts(other_contacts, state_out)


def test_vbd_contact_force_graph_capture(test, device):
    """Replay a captured collide/step/update_contacts sequence and match an uncaptured run."""
    dt = 1.0 / 120.0
    box_twist = np.zeros((1, 6), dtype=np.float32)
    box_twist[0, :3] = [0.5, 0.0, 0.0]
    scenes = {
        "particle_on_ground": (
            lambda d: (_build_particle_on_ground(d, pos=(0.3, -0.2, 0.046), radius=0.05), None),
            {"soft_contact_gap": 0.01},
            {},
            None,
            False,
        ),
        "box_under_particles": (
            lambda d: _build_box_under_particles(d, penetration=0.004),
            {"soft_contact_gap": 0.01},
            {"rigid_compliant_alm": True},
            box_twist,
            False,
        ),
        "sphere_on_ground_with_particle": (
            lambda d: _build_sphere_on_ground(d, with_particle=True)[:2],
            {},
            {"rigid_compliant_alm": True},
            None,
            True,
        ),
    }

    def make(build, pipeline_kwargs, solver_kwargs, twist):
        model, _body = build(device)
        model.request_contact_attributes("force")
        pipeline = newton.CollisionPipeline(model, **pipeline_kwargs)
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverVBD(model, iterations=4, **solver_kwargs)
        state_in, state_out = model.state(), model.state()
        if twist is not None:
            state_in.body_qd.assign(twist)
        return pipeline, solver, contacts, state_in, state_out

    with wp.ScopedDevice(device):
        for name, (build, pipeline_kwargs, solver_kwargs, twist, expect_rigid) in scenes.items():
            with test.subTest(scene=name):
                captured = make(build, pipeline_kwargs, solver_kwargs, twist)
                reference = make(build, pipeline_kwargs, solver_kwargs, twist)

                # One uncaptured step sizes the export buffers before capture.
                _advance(*captured, dt)
                _advance(*reference, dt)
                soft_buffer = captured[1]._body_particle_contact_force
                rigid_buffer = captured[1]._body_body_contact_force

                with wp.ScopedCapture(device=device) as capture:
                    _advance(*captured, dt)
                test.assertIsNotNone(capture.graph)
                test.assertIs(captured[1]._body_particle_contact_force, soft_buffer)
                test.assertIs(captured[1]._body_body_contact_force, rigid_buffer)

                replays = 3
                for _ in range(replays):
                    wp.capture_launch(capture.graph)
                    _advance(*reference, dt)

                count, rows = _active_soft_rows(captured[2])
                ref_count, ref_rows = _active_soft_rows(reference[2])
                test.assertEqual(count, ref_count)
                test.assertGreater(count, 0)
                test.assertGreater(float(np.max(np.abs(rows))), 0.0)
                np.testing.assert_allclose(rows, ref_rows, rtol=1.0e-5, atol=1.0e-5 * float(np.max(np.abs(ref_rows))))
                rigid_count, rigid_rows = _active_rigid_rows(captured[2])
                test.assertEqual(rigid_count, _active_rigid_rows(reference[2])[0])
                if expect_rigid:
                    test.assertGreater(rigid_count, 0)
                    test.assertGreater(float(np.max(np.abs(rigid_rows))), 0.0)
                force = captured[2].force.numpy()
                ref_force = reference[2].force.numpy()
                np.testing.assert_allclose(
                    force, ref_force, rtol=1.0e-5, atol=1.0e-5 * float(np.max(np.abs(ref_force)))
                )
                np.testing.assert_allclose(
                    captured[3].particle_q.numpy(), reference[3].particle_q.numpy(), rtol=1.0e-6, atol=1.0e-7
                )


class TestSolverVBDContactForce(unittest.TestCase):
    pass


add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_soft_contact_force_static_equilibrium",
    test_vbd_soft_contact_force_static_equilibrium,
    devices=devices,
)
add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_soft_contact_force_zero_when_separated",
    test_vbd_soft_contact_force_zero_when_separated,
    devices=devices,
)
add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_soft_contact_force_friction_damping_moving_body",
    test_vbd_soft_contact_force_friction_damping_moving_body,
    devices=devices,
)
add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_soft_contact_force_friction_damping_external_body",
    test_vbd_soft_contact_force_friction_damping_external_body,
    devices=devices,
)
add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_soft_contact_force_regularized_friction",
    test_vbd_soft_contact_force_regularized_friction,
    devices=devices,
)
add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_soft_contact_force_face_records_support_body",
    test_vbd_soft_contact_force_face_records_support_body,
    devices=devices,
)
add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_soft_contact_force_edge_records_match_law",
    test_vbd_soft_contact_force_edge_records_match_law,
    devices=devices,
)
add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_soft_contact_force_export_leaves_simulation_unchanged",
    test_vbd_soft_contact_force_export_leaves_simulation_unchanged,
    devices=devices,
)
add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_rigid_contact_force_static_equilibrium",
    test_vbd_rigid_contact_force_static_equilibrium,
    devices=devices,
)
add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_rigid_contact_force_feeds_sensor_contact",
    test_vbd_rigid_contact_force_feeds_sensor_contact,
    devices=devices,
)
add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_contact_force_external_rigid_leaves_rigid_rows",
    test_vbd_contact_force_external_rigid_leaves_rigid_rows,
    devices=devices,
)
add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_contact_force_layout",
    test_vbd_contact_force_layout,
    devices=devices,
)
add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_update_contacts_requires_force_attribute",
    test_vbd_update_contacts_requires_force_attribute,
    devices=devices,
)
add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_update_contacts_requires_exporting_step",
    test_vbd_update_contacts_requires_exporting_step,
    devices=devices,
)
add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_update_contacts_rejects_capacity_mismatch",
    test_vbd_update_contacts_rejects_capacity_mismatch,
    devices=devices,
)
add_function_test(
    TestSolverVBDContactForce,
    "test_vbd_contact_force_graph_capture",
    test_vbd_contact_force_graph_capture,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
