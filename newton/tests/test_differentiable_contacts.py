# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def test_no_overhead_when_disabled(test, device):
    """Differentiable arrays are None when requires_grad=False."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
        builder.add_shape_sphere(body=body, radius=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=False)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()

        test.assertIsNone(contacts.rigid_contact_diff_distance)
        test.assertIsNone(contacts.rigid_contact_diff_normal)
        test.assertIsNone(contacts.rigid_contact_diff_point0_world)
        test.assertIsNone(contacts.rigid_contact_diff_point1_world)


def test_arrays_allocated_when_enabled(test, device):
    """Differentiable arrays are allocated when requires_grad=True."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
        builder.add_shape_sphere(body=body, radius=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()

        test.assertIsNotNone(contacts.rigid_contact_diff_distance)
        test.assertIsNotNone(contacts.rigid_contact_diff_normal)
        test.assertIsNotNone(contacts.rigid_contact_diff_point0_world)
        test.assertIsNotNone(contacts.rigid_contact_diff_point1_world)
        test.assertTrue(contacts.rigid_contact_diff_distance.requires_grad)


def test_sphere_on_plane_distance(test, device):
    """Sphere penetrating ground plane produces correct differentiable distance."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        sphere_height = 0.3
        sphere_radius = 0.5
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, sphere_height)))
        builder.add_shape_sphere(body=body, radius=sphere_radius)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state()

        pipeline.collide(state, contacts)

        count = contacts.rigid_contact_count.numpy()[0]
        test.assertGreater(count, 0, "Expected at least one contact")

        diff_dist = contacts.rigid_contact_diff_distance.numpy()[:count]
        test.assertTrue(
            np.any(diff_dist < 0.0),
            f"Expected negative (penetrating) distance, got {diff_dist}",
        )


def test_gradient_flow_through_body_q(test, device):
    """Verify gradients flow from diff distance through body_q via wp.Tape."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3)))
        builder.add_shape_sphere(body=body, radius=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state(requires_grad=True)

        with wp.Tape() as tape:
            pipeline.collide(state, contacts)

        tape.backward(
            grads={
                contacts.rigid_contact_diff_distance: wp.ones(contacts.rigid_contact_max, dtype=float, device=device)
            }
        )

        grad_q = tape.gradients.get(state.body_q)
        test.assertIsNotNone(grad_q, "body_q gradient should be recorded on tape")

        grad_np = grad_q.numpy()
        test.assertFalse(
            np.allclose(grad_np, 0.0),
            "body_q gradient should be non-zero for penetrating sphere",
        )


def test_gradient_direction(test, device):
    """Moving the sphere upward should increase (make less negative) the contact distance."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3)))
        builder.add_shape_sphere(body=body, radius=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state(requires_grad=True)

        with wp.Tape() as tape:
            pipeline.collide(state, contacts)

        tape.backward(
            grads={
                contacts.rigid_contact_diff_distance: wp.ones(contacts.rigid_contact_max, dtype=float, device=device)
            }
        )

        grad_q = tape.gradients.get(state.body_q)
        test.assertIsNotNone(grad_q)

        grad_np = grad_q.numpy()
        # wp.transform stores (px, py, pz, qw, qx, qy, qz)
        dz = grad_np[0, 2]  # body 0, z-translation component
        test.assertGreater(
            dz,
            0.0,
            f"Expected positive z-gradient (moving up increases distance), got dz={dz}",
        )


def test_collide_outside_tape(test, device):
    """collide() works correctly outside a tape (no gradients, no crash)."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3)))
        builder.add_shape_sphere(body=body, radius=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state()

        pipeline.collide(state, contacts)

        count = contacts.rigid_contact_count.numpy()[0]
        test.assertGreater(count, 0)
        diff_dist = contacts.rigid_contact_diff_distance.numpy()[:count]
        test.assertTrue(np.any(diff_dist < 0.0))


def test_two_body_contact(test, device):
    """Two dynamic bodies in contact both receive non-zero gradients."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
        builder.add_shape_box(body=body_a, hx=0.5, hy=0.5, hz=0.5)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)))
        builder.add_shape_box(body=body_b, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state(requires_grad=True)

        with wp.Tape() as tape:
            pipeline.collide(state, contacts)

        count = contacts.rigid_contact_count.numpy()[0]
        test.assertGreater(count, 0, "Expected contacts between two overlapping boxes")

        tape.backward(
            grads={
                contacts.rigid_contact_diff_distance: wp.ones(contacts.rigid_contact_max, dtype=float, device=device)
            }
        )

        grad_q = tape.gradients.get(state.body_q)
        test.assertIsNotNone(grad_q)
        grad_np = grad_q.numpy()

        grad_a = grad_np[0]
        grad_b = grad_np[1]
        test.assertFalse(np.allclose(grad_a, 0.0), f"Body A gradient should be non-zero, got {grad_a}")
        test.assertFalse(np.allclose(grad_b, 0.0), f"Body B gradient should be non-zero, got {grad_b}")


def test_world_points_correctness(test, device):
    """Differentiable world-space points and distance are geometrically consistent."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        sphere_height = 0.3
        sphere_radius = 0.5
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, sphere_height)))
        builder.add_shape_sphere(body=body, radius=sphere_radius)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state()
        pipeline.collide(state, contacts)

        count = contacts.rigid_contact_count.numpy()[0]
        test.assertGreater(count, 0)

        p0 = contacts.rigid_contact_diff_point0_world.numpy()[:count]
        p1 = contacts.rigid_contact_diff_point1_world.numpy()[:count]
        normals = contacts.rigid_contact_diff_normal.numpy()[:count]
        distances = contacts.rigid_contact_diff_distance.numpy()[:count]
        margins0 = contacts.rigid_contact_margin0.numpy()[:count]
        margins1 = contacts.rigid_contact_margin1.numpy()[:count]

        for i in range(count):
            # Verify distance identity: d = dot(n, p1 - p0) - thickness
            gap = np.dot(normals[i], p1[i] - p0[i])
            thickness = margins0[i] + margins1[i]
            expected_d = gap - thickness
            test.assertAlmostEqual(
                float(distances[i]),
                float(expected_d),
                places=4,
                msg=f"Contact {i}: distance {distances[i]} != dot(n, p1-p0) - thickness = {expected_d}",
            )

            # Normal should be approximately unit length
            n_len = np.linalg.norm(normals[i])
            test.assertAlmostEqual(
                n_len,
                1.0,
                places=3,
                msg=f"Contact {i}: normal length {n_len} != 1.0",
            )


def test_finite_difference_distance_gradient(test, device):
    """Tape gradient of distance w.r.t. z-translation matches finite differences."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        h0 = 0.3
        r = 0.5
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, h0)))
        builder.add_shape_sphere(body=body, radius=r)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()

        # Analytical gradient via tape
        state = model.state(requires_grad=True)
        with wp.Tape() as tape:
            pipeline.collide(state, contacts)

        count = contacts.rigid_contact_count.numpy()[0]
        test.assertGreater(count, 0)
        grad_seed = wp.zeros(contacts.rigid_contact_max, dtype=float, device=device)
        grad_seed_np = grad_seed.numpy()
        grad_seed_np[:count] = 1.0
        grad_seed = wp.array(grad_seed_np, dtype=float, device=device)

        tape.backward(grads={contacts.rigid_contact_diff_distance: grad_seed})
        grad_q = tape.gradients.get(state.body_q)
        analytic_dz = grad_q.numpy()[0, 2]

        # Finite difference: perturb z by eps
        eps = 1e-4
        dist_vals = []
        for sign in [-1.0, 1.0]:
            state_fd = model.state()
            q_np = state_fd.body_q.numpy()
            q_np[0, 2] += sign * eps
            state_fd.body_q = wp.array(q_np, dtype=wp.transform, device=device)
            pipeline.collide(state_fd, contacts)
            c = contacts.rigid_contact_count.numpy()[0]
            d = contacts.rigid_contact_diff_distance.numpy()[:c].sum() if c > 0 else 0.0
            dist_vals.append(d)

        fd_dz = (dist_vals[1] - dist_vals[0]) / (2.0 * eps)

        test.assertAlmostEqual(
            analytic_dz,
            fd_dz,
            places=2,
            msg=f"Analytic dz={analytic_dz:.6f} vs FD dz={fd_dz:.6f}",
        )


def test_repeated_collide_independent_gradients(test, device):
    """Calling collide() twice in separate tapes gives independent gradients."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3)))
        builder.add_shape_sphere(body=body, radius=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()

        # First tape
        state1 = model.state(requires_grad=True)
        with wp.Tape() as tape1:
            pipeline.collide(state1, contacts)
        tape1.backward(
            grads={
                contacts.rigid_contact_diff_distance: wp.ones(contacts.rigid_contact_max, dtype=float, device=device)
            }
        )
        grad1 = tape1.gradients.get(state1.body_q).numpy().copy()

        # Second tape with same state values
        state2 = model.state(requires_grad=True)
        with wp.Tape() as tape2:
            pipeline.collide(state2, contacts)
        tape2.backward(
            grads={
                contacts.rigid_contact_diff_distance: wp.ones(contacts.rigid_contact_max, dtype=float, device=device)
            }
        )
        grad2 = tape2.gradients.get(state2.body_q).numpy().copy()

        np.testing.assert_allclose(
            grad1,
            grad2,
            atol=1e-6,
            err_msg="Repeated collide() should produce identical gradients",
        )


def test_finite_difference_two_body_gradient(test, device):
    """Tape gradients match central finite differences for two overlapping boxes across all translation DOFs."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
        builder.add_shape_box(body=body_a, hx=0.5, hy=0.5, hz=0.5)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)))
        builder.add_shape_box(body=body_b, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()

        # Analytical gradient via tape
        state = model.state(requires_grad=True)
        with wp.Tape() as tape:
            pipeline.collide(state, contacts)

        count = contacts.rigid_contact_count.numpy()[0]
        test.assertGreater(count, 0, "Expected contacts between two overlapping boxes")

        grad_seed = wp.zeros(contacts.rigid_contact_max, dtype=float, device=device)
        grad_seed_np = grad_seed.numpy()
        grad_seed_np[:count] = 1.0
        grad_seed = wp.array(grad_seed_np, dtype=float, device=device)

        tape.backward(grads={contacts.rigid_contact_diff_distance: grad_seed})
        grad_q = tape.gradients.get(state.body_q)
        test.assertIsNotNone(grad_q)
        analytic_grad = grad_q.numpy()

        eps = 1e-4
        for body_idx in range(2):
            for axis in range(3):
                dist_vals = []
                for sign in [-1.0, 1.0]:
                    state_fd = model.state()
                    q_np = state_fd.body_q.numpy()
                    q_np[body_idx, axis] += sign * eps
                    state_fd.body_q = wp.array(q_np, dtype=wp.transform, device=device)
                    pipeline.collide(state_fd, contacts)
                    c = contacts.rigid_contact_count.numpy()[0]
                    d = contacts.rigid_contact_diff_distance.numpy()[:c].sum() if c > 0 else 0.0
                    dist_vals.append(d)

                fd_grad = (dist_vals[1] - dist_vals[0]) / (2.0 * eps)
                analytic_val = float(analytic_grad[body_idx, axis])

                test.assertAlmostEqual(
                    analytic_val,
                    fd_grad,
                    places=2,
                    msg=f"Body {body_idx} axis {axis}: analytic={analytic_val:.6f} vs FD={fd_grad:.6f}",
                )


@wp.kernel
def _body_position_loss_kernel(
    body_q: wp.array[wp.transform],
    target: wp.vec3,
    loss: wp.array[float],
):
    pos = wp.transform_get_translation(body_q[0])
    delta = pos - target
    loss[0] = wp.dot(delta, delta)


def test_multistep_gradient_flow(test, device):
    """Multi-step tape gradient of position loss w.r.t. initial z matches finite differences."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=-9.81)
        sphere_height = 2.0
        sphere_radius = 0.5
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, sphere_height)))
        builder.add_shape_sphere(body=body, radius=sphere_radius)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)
        model.soft_contact_ke = 1000.0
        model.soft_contact_kd = 10.0
        model.soft_contact_kf = 100.0
        model.soft_contact_mu = 0.5

        solver = newton.solvers.SolverSemiImplicit(model)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            soft_contact_margin=10.0,
            requires_grad=True,
        )

        sim_substeps = 4
        sim_dt = 1.0 / 60.0 / float(sim_substeps)
        target = wp.vec3(0.0, 0.0, 5.0)

        # --- Analytical gradient via tape ---
        control = model.control()
        states = [model.state(requires_grad=True) for _ in range(sim_substeps + 1)]
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

        with wp.Tape() as tape:
            for t in range(sim_substeps):
                states[t].clear_forces()
                contacts = pipeline.contacts()
                pipeline.collide(states[t], contacts)
                solver.step(states[t], states[t + 1], control, contacts, sim_dt)

            wp.launch(
                _body_position_loss_kernel,
                dim=1,
                inputs=[states[-1].body_q, target],
                outputs=[loss],
                device=device,
            )

        tape.backward(loss)

        grad_q0 = tape.gradients.get(states[0].body_q)
        test.assertIsNotNone(grad_q0, "Initial body_q gradient should exist after multi-step backward")
        analytic_dz = float(grad_q0.numpy()[0, 2])

        # --- Finite-difference reference ---
        eps = 1e-4
        loss_vals = []
        for sign in [-1.0, 1.0]:
            q_np = states[0].body_q.numpy()
            q_np[0, 2] = sphere_height + sign * eps
            model_fd = builder.finalize(device=device, requires_grad=False)
            model_fd.soft_contact_ke = 1000.0
            model_fd.soft_contact_kd = 10.0
            model_fd.soft_contact_kf = 100.0
            model_fd.soft_contact_mu = 0.5
            solver_fd = newton.solvers.SolverSemiImplicit(model_fd)
            pipeline_fd = newton.CollisionPipeline(
                model_fd,
                broad_phase="explicit",
                soft_contact_margin=10.0,
                requires_grad=False,
            )
            # Override initial body_q with perturbed value
            fd_states_0 = model_fd.state()
            fd_q = fd_states_0.body_q.numpy()
            fd_q[0, 2] = sphere_height + sign * eps
            fd_states_0.body_q = wp.array(fd_q, dtype=wp.transform, device=device)

            fd_control = model_fd.control()
            fd_states = [fd_states_0] + [model_fd.state() for _ in range(sim_substeps)]
            fd_loss = wp.zeros(1, dtype=float, device=device)
            for t in range(sim_substeps):
                fd_states[t].clear_forces()
                fd_contacts = pipeline_fd.contacts()
                pipeline_fd.collide(fd_states[t], fd_contacts)
                solver_fd.step(fd_states[t], fd_states[t + 1], fd_control, fd_contacts, sim_dt)
            wp.launch(
                _body_position_loss_kernel,
                dim=1,
                inputs=[fd_states[-1].body_q, target],
                outputs=[fd_loss],
                device=device,
            )
            loss_vals.append(fd_loss.numpy()[0])

        fd_dz = (loss_vals[1] - loss_vals[0]) / (2.0 * eps)

        # Verify sign: target above sphere, so moving up reduces loss
        test.assertLess(
            analytic_dz,
            0.0,
            f"Expected negative z-gradient (moving up reduces loss toward target above), got dz={analytic_dz}",
        )

        # Verify magnitude matches finite differences
        test.assertAlmostEqual(
            analytic_dz,
            fd_dz,
            places=1,
            msg=f"Multi-step analytic dz={analytic_dz:.6f} vs FD dz={fd_dz:.6f}",
        )


class TestDifferentiableContacts(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
add_function_test(
    TestDifferentiableContacts, "test_no_overhead_when_disabled", test_no_overhead_when_disabled, devices=devices
)
add_function_test(
    TestDifferentiableContacts,
    "test_arrays_allocated_when_enabled",
    test_arrays_allocated_when_enabled,
    devices=devices,
)
add_function_test(
    TestDifferentiableContacts, "test_sphere_on_plane_distance", test_sphere_on_plane_distance, devices=devices
)
add_function_test(
    TestDifferentiableContacts, "test_gradient_flow_through_body_q", test_gradient_flow_through_body_q, devices=devices
)
add_function_test(TestDifferentiableContacts, "test_gradient_direction", test_gradient_direction, devices=devices)
add_function_test(TestDifferentiableContacts, "test_collide_outside_tape", test_collide_outside_tape, devices=devices)
add_function_test(TestDifferentiableContacts, "test_two_body_contact", test_two_body_contact, devices=devices)
add_function_test(
    TestDifferentiableContacts, "test_world_points_correctness", test_world_points_correctness, devices=devices
)
add_function_test(
    TestDifferentiableContacts,
    "test_finite_difference_distance_gradient",
    test_finite_difference_distance_gradient,
    devices=devices,
)
add_function_test(
    TestDifferentiableContacts,
    "test_repeated_collide_independent_gradients",
    test_repeated_collide_independent_gradients,
    devices=devices,
)
add_function_test(
    TestDifferentiableContacts,
    "test_finite_difference_two_body_gradient",
    test_finite_difference_two_body_gradient,
    devices=devices,
)
add_function_test(
    TestDifferentiableContacts,
    "test_multistep_gradient_flow",
    test_multistep_gradient_flow,
    devices=devices,
)


@wp.kernel
def _soft_contact_gap_kernel(
    count: wp.array[int],
    corners: wp.array[wp.vec3i],
    bary: wp.array[wp.vec3],
    shape: wp.array[int],
    body_pos: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    loss: wp.array[float],
):
    """Sum the soft-contact gap ``dot(n, x - bx)``, ``x = sum_i bary[i] * particle_q[corners[i]]``.

    Replicates the differentiable core of the VBD penetration formula
    (``rigid_vbd_kernels._eval_soft_ef_contact``): the contact geometry (``bary``, ``body_pos``,
    ``normal``) is frozen and only the live particle positions enter, so the gradient of this loss
    w.r.t. ``particle_q`` is the gap derivative the solver relies on -- ``bary_i * n`` per corner.
    """
    i = wp.tid()
    if i >= count[0]:
        return
    c = corners[i]
    b = bary[i]
    x = b[0] * particle_q[c[0]]
    if c[1] >= 0:
        x = x + b[1] * particle_q[c[1]]
    if c[2] >= 0:
        x = x + b[2] * particle_q[c[2]]
    X_wb = wp.transform_identity()
    body_idx = shape_body[shape[i]]
    if body_idx >= 0:
        X_wb = body_q[body_idx]
    bx = wp.transform_point(X_wb, body_pos[i])
    wp.atomic_add(loss, 0, wp.dot(normal[i], x - bx))


def _assert_soft_contact_gap_differentiable(test, device, builder, full_surface):
    """collide once, freeze the contact geometry, then check the gap is differentiable w.r.t. the
    live particle positions -- analytic tape gradient vs central finite differences. The barycentric
    contact point is intentionally frozen (fixed-contact-point model); only the gap must be
    differentiable, which is what the VBD contact force consumes."""
    model = builder.finalize(device=device, requires_grad=True)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        soft_contact_margin=0.25,
        enable_rigid_soft_full_surface_contact=full_surface,
        requires_grad=True,
    )
    contacts = pipeline.contacts()
    state = model.state(requires_grad=True)
    pipeline.collide(state, contacts)

    count = int(contacts.soft_contact_count.numpy()[0])
    test.assertGreater(count, 0, "expected at least one soft contact")
    idx = contacts.soft_contact_indices.numpy()[:count]

    # Freeze the contact geometry as detached constants; only particle_q stays differentiable.
    frozen_cnt = wp.array([count], dtype=int, device=device)
    frozen_corners = wp.array(idx, dtype=wp.vec3i, device=device)
    frozen_bary = wp.array(contacts.soft_contact_barycentric.numpy()[:count], dtype=wp.vec3, device=device)
    frozen_shape = wp.array(contacts.soft_contact_shape.numpy()[:count], dtype=int, device=device)
    frozen_body_pos = wp.array(contacts.soft_contact_body_pos.numpy()[:count], dtype=wp.vec3, device=device)
    frozen_normal = wp.array(contacts.soft_contact_normal.numpy()[:count], dtype=wp.vec3, device=device)

    def _launch(pq, loss):
        wp.launch(
            _soft_contact_gap_kernel,
            dim=count,
            inputs=[
                frozen_cnt,
                frozen_corners,
                frozen_bary,
                frozen_shape,
                frozen_body_pos,
                frozen_normal,
                pq,
                model.shape_body,
                state.body_q,
            ],
            outputs=[loss],
            device=device,
        )

    q0 = state.particle_q.numpy()
    pq = wp.array(q0, dtype=wp.vec3, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        _launch(pq, loss)
    tape.backward(loss)
    grad = pq.grad.numpy()

    def _loss_at(qnp):
        p2 = wp.array(qnp, dtype=wp.vec3, device=device)
        loss_fd = wp.zeros(1, dtype=float, device=device)
        _launch(p2, loss_fd)
        return float(loss_fd.numpy()[0])

    test.assertGreater(
        np.count_nonzero(np.abs(grad) > 1e-9), 0, "gap gradient must be nonzero (path is differentiable)"
    )
    eps = 1e-3
    participating = sorted({int(v) for row in idx for v in row if v >= 0})
    for vi in participating:
        for comp in range(3):
            qp = q0.copy()
            qp[vi, comp] += eps
            qm = q0.copy()
            qm[vi, comp] -= eps
            fd = (_loss_at(qp) - _loss_at(qm)) / (2.0 * eps)
            test.assertAlmostEqual(
                float(grad[vi, comp]),
                fd,
                places=3,
                msg=f"gap gradient v{vi}.{'xyz'[comp]}: analytic={grad[vi, comp]:.5f} vs FD={fd:.5f}",
            )


def test_particle_soft_contact_gap_differentiable(test, device):
    """The particle soft-contact gap is differentiable w.r.t. the particle position (analytic tape
    gradient matches finite differences; d(gap)/d(particle) = contact normal)."""
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)
    builder.add_particle(wp.vec3(0.6, 0.1, 0.05), wp.vec3(0.0), 1.0, radius=0.0)
    _assert_soft_contact_gap_differentiable(test, device, builder, full_surface=False)


def test_full_surface_gap_differentiable(test, device):
    """The full-surface edge/face gap is differentiable w.r.t. the soft vertices with the barycentric
    contact point frozen: gap = dot(n, sum_i bary_i * pos_i - bx), so d(gap)/d(pos_i) = bary_i * n.
    Analytic tape gradient matches finite differences (confirms EF autodiff through the frozen point)."""
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)
    # A triangle whose v0-v1 edge grazes the +x face; the third vertex is far, and the endpoints are
    # far enough in y that they are NOT within the per-particle margin -> an edge/face record only.
    v0 = builder.add_particle(wp.vec3(0.6, -1.0, 0.0), wp.vec3(0.0), 1.0, radius=0.0)
    v1 = builder.add_particle(wp.vec3(0.6, 1.0, 0.0), wp.vec3(0.0), 1.0, radius=0.0)
    v2 = builder.add_particle(wp.vec3(2.0, 0.0, 0.0), wp.vec3(0.0), 1.0, radius=0.0)
    builder.add_triangle(v0, v1, v2)
    _assert_soft_contact_gap_differentiable(test, device, builder, full_surface=True)


for _name, _fn in (
    ("test_particle_soft_contact_gap_differentiable", test_particle_soft_contact_gap_differentiable),
    ("test_full_surface_gap_differentiable", test_full_surface_gap_differentiable),
):
    add_function_test(TestDifferentiableContacts, _name, _fn, devices=devices)


@wp.kernel
def _sum_soft_contact_bodypos_y(count: wp.array[int], body_pos: wp.array[wp.vec3], loss: wp.array[float]):
    i = wp.tid()
    if i < count[0]:
        wp.atomic_add(loss, 0, body_pos[i][1])


def test_soft_contact_detection_differentiable_through_collide(test, device):
    """Differentiating THROUGH ``collide()`` (the detection recorded on the tape) propagates a correct
    gradient to ``particle_q`` -- this is the path that actually exercises ``create_soft_contacts`` ->
    :func:`~newton._src.geometry.kernels.counter_increment` -> its ``@wp.func_replay`` and the
    per-thread ``tids`` replay array. A single particle-vs-box contact has a differentiable contact
    point, so the analytic tape gradient of ``sum(body_pos.y)`` matches finite differences.

    This measures the *contact-point location* (``body_pos``) route. For an edge/face record the SDF
    argmin freezes that location, so *this* route is zero -- but that does NOT mean the EF path is
    non-differentiable: the sim-relevant gradient flows through the gap ``dot(n, sum_i bary_i * pos_i
    - bx)`` w.r.t. the live positions (``bary_i * n``), which is nonzero and covered by
    ``test_full_surface_gap_differentiable`` above.
    """
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)
    builder.add_particle(wp.vec3(0.6, 0.1, 0.05), wp.vec3(0.0), 1.0, radius=0.0)
    model = builder.finalize(device=device, requires_grad=True)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=0.25, requires_grad=True)
    q0 = model.state().particle_q.numpy()

    def _loss(qnp, on_tape):
        state = model.state(requires_grad=on_tape)
        state.particle_q.assign(qnp)
        contacts = pipeline.contacts()
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        if on_tape:
            tape = wp.Tape()
            with tape:
                pipeline.collide(state, contacts)
                wp.launch(
                    _sum_soft_contact_bodypos_y,
                    dim=pipeline.soft_contact_max,
                    inputs=[contacts.soft_contact_count, contacts.soft_contact_body_pos],
                    outputs=[loss],
                    device=device,
                )
            tape.backward(loss)
            return state.particle_q.grad.numpy(), int(contacts.soft_contact_count.numpy()[0])
        pipeline.collide(state, contacts)
        wp.launch(
            _sum_soft_contact_bodypos_y,
            dim=pipeline.soft_contact_max,
            inputs=[contacts.soft_contact_count, contacts.soft_contact_body_pos],
            outputs=[loss],
            device=device,
        )
        return float(loss.numpy()[0]), int(contacts.soft_contact_count.numpy()[0])

    grad, count = _loss(q0, on_tape=True)
    test.assertEqual(count, 1, "expected exactly one particle-vs-box contact")
    test.assertGreater(
        np.count_nonzero(np.abs(grad) > 1e-9),
        0,
        "detection must be differentiable through collide() (counter_increment replay path)",
    )

    eps = 1e-3
    for comp in range(3):
        if abs(grad[0, comp]) < 1e-9:
            continue
        qp = q0.copy()
        qp[0, comp] += eps
        qm = q0.copy()
        qm[0, comp] -= eps
        lp, cp = _loss(qp, on_tape=False)
        lm, cm = _loss(qm, on_tape=False)
        test.assertEqual(cp, count)
        test.assertEqual(cm, count)
        fd = (lp - lm) / (2.0 * eps)
        test.assertAlmostEqual(
            float(grad[0, comp]),
            fd,
            places=3,
            msg=f"through-collide gradient p.{'xyz'[comp]}: analytic={grad[0, comp]:.5f} vs FD={fd:.5f}",
        )


add_function_test(
    TestDifferentiableContacts,
    "test_soft_contact_detection_differentiable_through_collide",
    test_soft_contact_detection_differentiable_through_collide,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
