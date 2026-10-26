# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise current-position mesh selection across VBD contact consumers."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.soft_contacts_mesh import _make_soft_contact_filter_data
from newton._src.solvers.vbd.particle_vbd_kernels import (
    gather_particle_body_contact_force_and_hessian,
    scatter_particle_body_contact_force_and_hessian,
)
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    accumulate_body_particle_contacts_per_body,
    update_duals_body_particle_contacts,
)
from newton.solvers import SolverSemiImplicit, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupled
from newton.tests.unittest_utils import NewtonTestCase, add_function_test, get_test_devices
from newton.tests.utils.soft_contacts import evaluate_soft_contact_force_mask

_RECORD_FIELDS = (
    "soft_contact_particle",
    "soft_contact_indices",
    "soft_contact_shape",
    "soft_contact_barycentric",
    "soft_contact_body_pos",
    "soft_contact_body_vel",
    "soft_contact_normal",
)


def _scene(device, two_bodies=False):
    """Create a perturbed cloth patch contacting a mesh body with a nonzero global ID."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    # The contacted body has global ID 1; a coupled view must map it to local ID 0.
    builder.add_body(xform=wp.transform(wp.vec3(10.0), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3)))
    body = builder.add_body()
    builder.add_shape_mesh(body, mesh=newton.Mesh.create_box(0.2, 0.2, 0.05))
    builder.add_cloth_grid(
        pos=wp.vec3(-0.24, -0.24, 0.055),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=8,
        dim_y=8,
        cell_x=0.06,
        cell_y=0.06,
        mass=0.1,
        particle_radius=0.03,
    )
    if two_bodies:
        builder.add_shape_mesh(0, mesh=newton.Mesh.create_box(0.2, 0.2, 0.05))
        builder.add_cloth_grid(
            pos=wp.vec3(9.76, 9.76, 10.055),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=8,
            dim_y=8,
            cell_x=0.06,
            cell_y=0.06,
            mass=0.1,
            particle_radius=0.03,
        )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    q = state.particle_q.numpy()
    state.particle_q.assign(q + np.random.default_rng(7).normal(scale=0.006, size=q.shape).astype(np.float32))
    pipeline = newton.CollisionPipeline(
        model,
        enable_rigid_soft_full_surface_contact=True,
        soft_contact_gap=0.03,
        soft_contact_max=4096,
    )
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return model, state, pipeline, contacts


def _solver(model, contacts):
    """Prepare both gather and scatter consumers for the supplied raw contacts."""
    solver = SolverVBD(model, iterations=2)
    # Build gather adjacency even when a subsequent evaluation chooses scatter.
    solver._particle_contact_use_gather = True
    solver._refresh_body_particle_contact_state(contacts, True)
    return solver


def _evaluate(solver, state, contacts, gather, buffers):
    """Evaluate production contact kernels at a fixed state, without integrating positions."""
    model, device = solver.model, solver.device
    particle_f, particle_h, body_f, body_t, body_ll, body_al, body_aa, exported, feedback, mapping, bodies = buffers
    for array in buffers[:9]:
        array.zero_()
    solver.body_particle_contact_penalty_k.fill_(100.0)
    solver.body_particle_contact_material_ke.fill_(1000.0)
    solver.body_particle_contact_material_kd.zero_()
    solver.body_particle_contact_material_mu.zero_()
    wp.copy(solver.particle_q_prev, state.particle_q)
    wp.copy(solver.body_q_prev, state.body_q)
    wp.copy(solver._coupling_body_q_prev_snapshot, state.body_q)
    data = _make_soft_contact_filter_data(model, contacts, state.particle_q, state.body_q)
    material_and_body = [
        solver.body_particle_contact_penalty_k,
        solver.body_particle_contact_material_kd,
        solver.body_particle_contact_material_mu,
        model.shape_body,
        state.body_q,
        solver.body_q_prev,
        state.body_qd,
        model.body_com,
        contacts.soft_contact_shape,
        contacts.soft_contact_body_pos,
        contacts.soft_contact_body_vel,
        contacts.soft_contact_normal,
        model.shape_margin,
        contacts.soft_contact_barycentric,
    ]
    for color, group in enumerate(model.particle_color_groups):
        if gather:
            wp.launch(
                gather_particle_body_contact_force_and_hessian,
                dim=group.size,
                inputs=[
                    0.01,
                    group,
                    solver.particle_q_prev,
                    state.particle_q,
                    solver.friction_epsilon,
                    False,
                    model.particle_radius,
                    contacts.soft_contact_indices,
                    solver._particle_contact_head,
                    solver._particle_contact_next,
                    *material_and_body,
                    data,
                ],
                outputs=[particle_f, particle_h],
                device=device,
            )
        else:
            wp.launch(
                scatter_particle_body_contact_force_and_hessian,
                dim=solver._particle_contact_worker_count,
                inputs=[
                    0.01,
                    color,
                    solver.particle_q_prev,
                    state.particle_q,
                    model.particle_colors,
                    solver.friction_epsilon,
                    False,
                    model.particle_radius,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_count,
                    data,
                    contacts.soft_contact_max,
                    solver._particle_contact_worker_count,
                    *material_and_body,
                ],
                outputs=[particle_f, particle_h],
                device=device,
            )
    wp.launch(
        accumulate_body_particle_contacts_per_body,
        dim=model.body_count * 4,
        inputs=[
            0.01,
            bodies,
            4,
            state.particle_q,
            solver.particle_q_prev,
            model.particle_radius,
            solver.body_q_prev,
            state.body_q,
            state.body_qd,
            model.body_com,
            model.body_inv_mass,
            model.shape_body,
            solver.friction_epsilon,
            False,
            solver.body_particle_contact_penalty_k,
            solver.body_particle_contact_material_ke,
            solver.body_particle_contact_material_kd,
            solver.body_particle_contact_material_mu,
            contacts.soft_contact_count,
            contacts.soft_contact_indices,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
            contacts.soft_contact_barycentric,
            model.shape_margin,
            solver.body_particle_contact_buffer_pre_alloc,
            solver.body_particle_contact_counts,
            solver.body_particle_contact_indices,
            data,
        ],
        outputs=[body_f, body_t, body_ll, body_al, body_aa],
        device=device,
    )
    solver._export_body_particle_contact_forces(state, state, contacts, 0.01, exported)
    solver.coupling_harvest_proxy_wrenches(
        mapping,
        feedback,
        body_qd_before=state.body_qd,
        state=state,
        state_out=state,
        contacts=contacts,
        dt=0.01,
    )
    wp.launch(
        update_duals_body_particle_contacts,
        dim=contacts.soft_contact_max,
        inputs=[
            contacts.soft_contact_count,
            data,
            contacts.soft_contact_indices,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_normal,
            contacts.soft_contact_barycentric,
            state.particle_q,
            model.particle_radius,
            model.shape_body,
            model.shape_margin,
            state.body_q,
            solver.body_particle_contact_material_ke,
            100.0,
            solver.body_particle_contact_penalty_k,
        ],
        device=device,
    )


def _buffers(model, contacts):
    """Allocate force, Hessian, export, and feedback outputs for the production consumers."""
    device = model.device
    return [
        wp.zeros(model.particle_count, dtype=wp.vec3, device=device),
        wp.zeros(model.particle_count, dtype=wp.mat33, device=device),
        *[wp.zeros(model.body_count, dtype=wp.vec3, device=device) for _ in range(2)],
        *[wp.zeros(model.body_count, dtype=wp.mat33, device=device) for _ in range(3)],
        wp.zeros(contacts.soft_contact_max, dtype=wp.spatial_vector, device=device),
        wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device),
        wp.array(np.arange(model.body_count), dtype=int, device=device),
        wp.array(np.arange(model.body_count), dtype=int, device=device),
    ]


def test_current_mesh_filter_consumers(test, device):
    """Refresh geometry and activity across force consumers, including CUDA graph replay."""
    # Warp CPU printf capture is unreliable on Windows; verify overflow through its counter.
    test.allowOutputRegex(
        r"^Warning: Per-body particle contact buffer overflowed \d+ > 256; scanning all stored contacts\.$",
        stream="stdout",
    )
    model, state, _pipeline, contacts = _scene(device)
    count = int(contacts.soft_contact_count.numpy()[0])
    test.assertLess(count, contacts.soft_contact_max)
    original = {
        name: getattr(contacts, name).numpy().copy() for name in (*_RECORD_FIELDS, "_soft_contact_mesh_features")
    }
    q0, b0 = state.particle_q.numpy().copy(), state.body_q.numpy().copy()
    # Candidate lists retain every detected row, including currently ineligible rows.
    test.assertFalse(hasattr(contacts, "soft_contact_force_mask"))
    solver = _solver(model, contacts)
    test.assertEqual(int(solver.body_particle_contact_counts.numpy()[1]), count)
    test.assertGreater(int(solver.body_particle_contact_overflow_max.numpy()[0]), 256)
    full_buffers = _buffers(model, contacts)
    compact = newton.Contacts(contacts.rigid_contact_max, contacts.soft_contact_max, device=device)
    reference = _solver(model, compact)
    reference_buffers = _buffers(model, compact)
    masks, forces = [], []
    for gather in (False, True):
        _evaluate(solver, state, contacts, gather, full_buffers)
        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                _evaluate(solver, state, contacts, gather, full_buffers)
        for particle_shift, body_shift in ((0.0, 0.0), (-0.01, 0.0), (0.0, -0.01), (0.0, 0.0)):
            state.particle_q.assign(q0 + np.float32(particle_shift))
            poses = b0.copy()
            poses[1, :3] += body_shift
            state.body_q.assign(poses)
            mask = evaluate_soft_contact_force_mask(model, state, contacts).numpy()
            rows = np.flatnonzero(mask[:count])
            masks.append(mask.copy())
            # Independent reference: remove rejected rows and evaluate without mesh filtering.
            compact.soft_contact_count.assign([len(rows)])
            for name in _RECORD_FIELDS:
                values = getattr(compact, name).numpy()
                values[: len(rows)] = original[name][rows]
                getattr(compact, name).assign(values)
            reference._refresh_body_particle_contact_state(compact, True)
            _evaluate(reference, state, compact, gather, reference_buffers)
            if device.is_cuda:
                wp.capture_launch(capture.graph)
            else:
                _evaluate(solver, state, contacts, gather, full_buffers)
            for i in (*range(7), 8):
                np.testing.assert_allclose(full_buffers[i].numpy(), reference_buffers[i].numpy(), rtol=3e-5, atol=3e-4)
            actual = full_buffers[7].numpy()
            forces.append(actual.copy())
            np.testing.assert_allclose(actual[rows], reference_buffers[7].numpy()[: len(rows)], rtol=2e-5, atol=1e-5)
            np.testing.assert_array_equal(actual[~mask], 0.0)
            penalty = solver.body_particle_contact_penalty_k.numpy()
            np.testing.assert_allclose(
                penalty[rows], reference.body_particle_contact_penalty_k.numpy()[: len(rows)], atol=1e-6
            )
            np.testing.assert_array_equal(penalty[~mask], 100.0)
            np.testing.assert_allclose(full_buffers[8].numpy()[:, :3], full_buffers[2].numpy(), rtol=3e-5, atol=3e-4)
            np.testing.assert_allclose(full_buffers[8].numpy()[:, 3:], full_buffers[3].numpy(), rtol=3e-5, atol=3e-4)
            for name, value in original.items():
                np.testing.assert_array_equal(getattr(contacts, name).numpy(), value)
            test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), count)
        # Independent activity oracles: no force from an entirely inactive mesh,
        # and unchanged forces from ACTIVE particles with zero inverse mass.
        # The graph was captured before these in-place changes to particle flags.
        baseline = [array.numpy().copy() for array in full_buffers[:9]]
        baseline_penalty_np = solver.body_particle_contact_penalty_k.numpy().copy()
        particle_flags_np = model.particle_flags.numpy().copy()
        particle_inv_mass_np = model.particle_inv_mass.numpy().copy()
        soft_contact_indices_np = original["soft_contact_indices"][:count]
        families_np = original["_soft_contact_mesh_features"][:count, 0] & 7
        interior_tv_np = np.all(original["soft_contact_barycentric"][:count] > 0.0, axis=1)
        stable_owner_np = (families_np != 1) | interior_tv_np
        for activity in ("inactive", "mixed", "active_zero_mass", "restored"):
            with test.subTest(gather=gather, activity=activity):
                current_flags_np = particle_flags_np.copy()
                if activity == "inactive":
                    current_flags_np &= ~int(newton.ParticleFlags.ACTIVE)
                elif activity == "mixed":
                    current_flags_np[q0[:, 0] <= 0.0] &= ~int(newton.ParticleFlags.ACTIVE)
                model.particle_flags.assign(current_flags_np)
                model.particle_inv_mass.assign(
                    np.zeros_like(particle_inv_mass_np) if activity == "active_zero_mass" else particle_inv_mass_np
                )
                if device.is_cuda:
                    wp.capture_launch(capture.graph)
                else:
                    _evaluate(solver, state, contacts, gather, full_buffers)
                penalty_np = solver.body_particle_contact_penalty_k.numpy()
                if activity == "inactive":
                    for array in full_buffers[:9]:
                        np.testing.assert_array_equal(array.numpy(), 0.0)
                    np.testing.assert_array_equal(penalty_np, 100.0)
                elif activity == "mixed":
                    active_np = np.any(
                        (soft_contact_indices_np >= 0)
                        & (
                            (
                                current_flags_np[np.maximum(soft_contact_indices_np, 0)]
                                & int(newton.ParticleFlags.ACTIVE)
                            )
                            != 0
                        ),
                        axis=1,
                    )
                    test.assertTrue(np.any(~active_np))
                    exported_np = full_buffers[7].numpy()[:count]
                    np.testing.assert_array_equal(exported_np[~active_np], 0.0)
                    np.testing.assert_array_equal(penalty_np[:count][~active_np], 100.0)
                    retained_np = active_np & stable_owner_np
                    test.assertTrue(np.any(np.linalg.norm(baseline[7][:count][retained_np], axis=1) > 0.0))
                    np.testing.assert_allclose(
                        exported_np[retained_np], baseline[7][:count][retained_np], rtol=2e-5, atol=1e-5
                    )
                else:
                    test.assertGreater(np.linalg.norm(baseline[2]), 0.0)
                    for array, expected in zip(full_buffers[:9], baseline, strict=True):
                        np.testing.assert_allclose(array.numpy(), expected, rtol=3e-5, atol=3e-4)
                    np.testing.assert_allclose(penalty_np, baseline_penalty_np, atol=1e-6)
                for name, value in original.items():
                    np.testing.assert_array_equal(getattr(contacts, name).numpy(), value)
                test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), count)
    families = original["_soft_contact_mesh_features"][:count, 0] & 7
    changed = (masks[0] != masks[1])[:count] | (masks[0] != masks[2])[:count]
    test.assertTrue({0, 1, 2} <= set(families[changed]))
    became_active = masks[1] & ~masks[0]
    became_inactive = masks[0] & ~masks[1]
    test.assertTrue(np.any(np.linalg.norm(forces[1][became_active], axis=1) > 0.0))
    test.assertTrue(np.any(np.linalg.norm(forces[0][became_inactive], axis=1) > 0.0))


def test_coupled_mesh_filter_compact_body_mapping(test, device):
    """Preserve row provenance and evaluate against compact entry poses without another detection."""
    model, state, pipeline, contacts = _scene(device)
    coupled = SolverCoupled(
        model,
        [
            SolverCoupled.Entry(
                "vbd",
                lambda view: SolverVBD(view, iterations=2),
                bodies=[1],
                particles=range(model.particle_count),
            )
        ],
    )
    entry = coupled._entries["vbd"]
    test.assertEqual(entry.view.body_count, 1)
    test.assertEqual(int(entry.view.shape_body.numpy()[0]), 0)
    # Empty preallocated Contacts are legal before the first collision detection.
    empty = pipeline.contacts()
    test.assertFalse(np.any(evaluate_soft_contact_force_mask(model, state, empty).numpy()))
    count = int(contacts.soft_contact_count.numpy()[0])
    q0, b0 = state.particle_q.numpy().copy(), state.body_q.numpy().copy()
    initial_mask = None
    for shift in (0.0, 0.01, -0.01, 0.0):
        state.particle_q.assign(q0)
        poses = b0.copy()
        poses[1, :3] += shift
        state.body_q.assign(poses)
        coupled._distribute_state(state, dt=0.01)
        copied = coupled._contacts_for_entry(entry, contacts)
        mapping = coupled._entry_soft_contact_src_to_dst["vbd"].numpy()[:count]
        test.assertTrue(np.all(mapping >= 0))
        test.assertEqual(int(copied.soft_contact_count.numpy()[0]), count)
        np.testing.assert_array_equal(
            copied._soft_contact_mesh_features.numpy()[mapping],
            contacts._soft_contact_mesh_features.numpy()[:count],
        )
        test.assertIs(copied._soft_contact_mesh_data, contacts._soft_contact_mesh_data)
        test.assertIsNone(copied._soft_contact_mesh_params)
        test.assertIsNone(copied._soft_contact_mesh_state)
        mask = evaluate_soft_contact_force_mask(model, state, contacts).numpy()[:count]
        copied_mask = evaluate_soft_contact_force_mask(entry.view, entry.state_0, copied).numpy()
        np.testing.assert_array_equal(copied_mask[mapping], mask)
        if initial_mask is None:
            initial_mask = mask.copy()
        elif shift:
            test.assertTrue(np.any(mask != initial_mask))


def test_body_candidate_overflow_preserves_reactions(test, device):
    """Retain each body's reactions and feedback when candidate lists overflow."""
    # The counter assertions below verify overflow even when native printf is not captured.
    test.allowOutputRegex(
        r"^Warning: Per-body particle contact buffer overflowed \d+ > 4; scanning all stored contacts\.$",
        stream="stdout",
    )
    model, state, _pipeline, contacts = _scene(device, two_bodies=True)
    count = int(contacts.soft_contact_count.numpy()[0])
    test.assertLess(count, contacts.soft_contact_max)
    limited = SolverVBD(model, iterations=2, rigid_body_particle_contact_buffer_size=4)
    full = SolverVBD(model, iterations=2, rigid_body_particle_contact_buffer_size=contacts.soft_contact_max)
    for solver in (limited, full):
        solver._particle_contact_use_gather = True
        solver._refresh_body_particle_contact_state(contacts, True)
    test.assertTrue(np.all(limited.body_particle_contact_counts.numpy() > 4))
    test.assertGreater(int(limited.body_particle_contact_overflow_max.numpy()[0]), 4)
    full_buffers, limited_buffers = _buffers(model, contacts), _buffers(model, contacts)
    _evaluate(limited, state, contacts, True, limited_buffers)
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as captured:
            _evaluate(limited, state, contacts, True, limited_buffers)
    q0 = state.particle_q.numpy().copy()
    for shift in (0.0, -0.01, 0.0):
        state.particle_q.assign(q0 + np.float32(shift))
        _evaluate(full, state, contacts, True, full_buffers)
        if device.is_cuda:
            wp.capture_launch(captured.graph)
        else:
            _evaluate(limited, state, contacts, True, limited_buffers)
        for actual, expected in zip(limited_buffers[:9], full_buffers[:9], strict=True):
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=5e-5, atol=5e-4)
        test.assertTrue(np.all(np.linalg.norm(full_buffers[2].numpy(), axis=1) > 0.0))
        np.testing.assert_allclose(
            limited_buffers[0].numpy().sum(axis=0) + limited_buffers[2].numpy().sum(axis=0),
            0.0,
            atol=5e-4,
        )


def test_mesh_contacts_reused_by_particle_pipeline(test, device):
    """Invalidate mesh tags before a reused buffer receives analytic particle contacts."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_shape_mesh(
        -1,
        mesh=newton.Mesh(
            vertices=np.array([(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
        ),
    )
    builder.add_shape_sphere(-1, xform=wp.transform(wp.vec3(10.0, 0.0, 0.0), wp.quat_identity()), radius=1.0)
    builder.add_particle((0.0, 0.0, 0.01), wp.vec3(), 1.0, radius=0.02)
    builder.color()
    model = builder.finalize(device=device)
    full = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True, soft_contact_max=8)
    particle_only = newton.CollisionPipeline(model, soft_contact_max=8)
    contacts = full.contacts()
    features = contacts._soft_contact_mesh_features
    state, output = model.state(), model.state()
    solver = SolverVBD(model, iterations=1)
    full.collide(state, contacts)
    solver.step(state, output, None, contacts, 0.001)
    particle_only.collide(state, contacts)
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as captured:
            particle_only.collide(state, contacts)

    for _ in range(2):
        state.particle_q.assign([[0.0, 0.0, 0.01]])
        full.collide(state, contacts)
        test.assertTrue(np.any(features.numpy()[:, 0] >= 0))
        solver.step(state, output, None, contacts, 0.001)
        state.particle_q.assign([[10.0, 0.0, 1.01]])
        if device.is_cuda:
            wp.capture_launch(captured.graph)
        else:
            particle_only.collide(state, contacts)
        test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 1)
        test.assertEqual(int(contacts.soft_contact_shape.numpy()[0]), 1)
        # Check before solving: a stale tag would query the sphere's null mesh pointer.
        np.testing.assert_array_equal(features.numpy(), -1)
        solver.step(state, output, None, contacts, 0.001)
        test.assertTrue(np.isfinite(output.particle_q.numpy()).all())
        test.assertIs(contacts._soft_contact_mesh_features, features)


def test_coupled_full_surface_rigid_only_fallback(test, device):
    """Allow a rigid-only fallback view even when it retains the parent's particle allocation."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    parent = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    child = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = builder.add_joint_revolute(parent=parent, child=child, axis=(0.0, 0.0, 1.0))
    builder.add_shape_box(child, hx=0.1, hy=0.1, hz=0.1)
    builder.add_particle((0.0, 0.0, 0.11), wp.vec3(), 1.0, radius=0.02)
    builder.color()
    model = builder.finalize(device=device)
    coupled = SolverCoupled(
        model,
        [
            SolverCoupled.Entry("rigid", SolverSemiImplicit, bodies=[child], joints=[joint]),
            SolverCoupled.Entry("soft", lambda view: SolverVBD(view, iterations=1), particles=[0]),
        ],
    )
    rigid = coupled._entries["rigid"]
    test.assertEqual(rigid.view.particle_count, 1)
    test.assertEqual(rigid.particle_indices.size, 0)
    test.assertEqual(rigid.proxy_particle_local_indices.size, 0)
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
    state, output, contacts = model.state(), model.state(), pipeline.contacts()
    pipeline.collide(state, contacts)
    test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
    coupled.step(state, output, None, contacts, 0.001)
    copied = coupled._entry_contact_buffers["rigid"]
    test.assertFalse(copied._enable_rigid_soft_full_surface_contact)
    test.assertEqual(int(copied.soft_contact_count.numpy()[0]), 0)
    test.assertTrue(np.isfinite(output.particle_q.numpy()).all())
    test.assertTrue(np.isfinite(output.body_q.numpy()).all())


def test_coupled_mesh_source_overflow(test, device):
    """Preserve overflow force policy through packing, cached copies, and new detections."""
    test.allowOutputRegex(
        r"^Mesh soft-contact capacity exceeded: soft_contact_max=2, required=\d+\. "
        r"Increase soft_contact_max to at least \d+ for this collision pass; "
        r"this collision result is incomplete\.$",
        stream="stdout",
    )
    model, state, _pipeline, _contacts = _scene(device)
    q0 = state.particle_q.numpy().copy()
    q0[0] = (0.0, 0.0, 0.055)
    state.particle_q.assign(q0)
    pipeline = newton.CollisionPipeline(
        model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.03, soft_contact_max=2
    )
    contacts = pipeline.contacts()
    coupled = SolverCoupled(
        model,
        [
            SolverCoupled.Entry(
                "vbd", lambda view: SolverVBD(view, iterations=1), bodies=[1], particles=range(model.particle_count)
            )
        ],
    )
    entry = coupled._entries["vbd"]
    downstream = SolverCoupled(
        model,
        [
            SolverCoupled.Entry(
                "vbd", lambda view: SolverVBD(view, iterations=1), bodies=[1], particles=range(model.particle_count)
            )
        ],
    )

    def detect_and_copy():
        pipeline.collide(state, contacts)
        coupled._distribute_state(state, dt=0.01)
        return coupled._contacts_for_entry(entry, contacts)

    copied = detect_and_copy()
    source_solver, entry_solver = _solver(model, contacts), _solver(entry.view, copied)
    source_buffers, entry_buffers = _buffers(model, contacts), _buffers(entry.view, copied)
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as captured:
            detect_and_copy()
    for shift in (0.0, 10.0, 0.0):
        state.particle_q.assign(q0 + np.float32(shift))
        if device.is_cuda:
            wp.capture_launch(captured.graph)
        else:
            detect_and_copy()
        overflow = int(contacts.soft_contact_count.numpy()[0]) > contacts.soft_contact_max
        test.assertEqual(overflow, shift == 0.0)
        test.assertEqual(int(copied._soft_contact_source_overflow.numpy()[0]), int(overflow))
        # Repacking an already packed buffer must propagate its loss even though
        # the intermediate count contains no overflow sentinel.
        forwarded = downstream._contacts_for_entry(downstream._entries["vbd"], copied)
        test.assertEqual(int(forwarded._soft_contact_source_overflow.numpy()[0]), int(overflow))
        # A cached copy must keep the same provenance even though its packed count fits.
        coupled._contacts_for_entry(entry, contacts)
        count = int(copied.soft_contact_count.numpy()[0])
        test.assertLessEqual(count, copied.soft_contact_max)
        test.assertEqual(int(copied._soft_contact_source_overflow.numpy()[0]), int(overflow))
        for solver, buffer in ((source_solver, contacts), (entry_solver, copied)):
            solver._refresh_body_particle_contact_state(buffer, True)
        _evaluate(source_solver, state, contacts, True, source_buffers)
        _evaluate(entry_solver, entry.state_0, copied, True, entry_buffers)
        mapping = coupled._entry_soft_contact_src_to_dst["vbd"].numpy()[:count]
        test.assertTrue(np.all(mapping >= 0))
        np.testing.assert_allclose(
            entry_buffers[7].numpy()[mapping], source_buffers[7].numpy()[:count], rtol=2e-5, atol=1e-5
        )
        mask = evaluate_soft_contact_force_mask(entry.view, entry.state_0, copied).numpy()
        np.testing.assert_array_equal(mask[:count], True)
        np.testing.assert_array_equal(mask[count:], False)
    copied.clear()
    test.assertEqual(int(copied._soft_contact_source_overflow.numpy()[0]), 0)
    contacts._soft_contact_source_overflow.fill_(1)
    state.particle_q.assign(q0 + np.float32(10.0))
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0)
    test.assertEqual(int(contacts._soft_contact_source_overflow.numpy()[0]), 0)


class TestVBDCurrentMeshFilter(NewtonTestCase):
    def setUp(self):
        self.allowOutputRegex(r"^Module .* load on device .*\n?", stream="stdout")


for fn in (
    test_current_mesh_filter_consumers,
    test_coupled_mesh_filter_compact_body_mapping,
    test_body_candidate_overflow_preserves_reactions,
    test_mesh_contacts_reused_by_particle_pipeline,
    test_coupled_full_surface_rigid_only_fallback,
    test_coupled_mesh_source_overflow,
):
    add_function_test(TestVBDCurrentMeshFilter, fn.__name__, fn, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
