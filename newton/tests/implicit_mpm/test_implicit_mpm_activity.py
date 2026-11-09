# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Protect captured sleeping, compact particle identity, and fail-closed capacity handling."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.implicit_mpm.implicit_mpm_model import ImplicitMPMModel
from newton._src.solvers.implicit_mpm.particle_activity import ParticleActivity
from newton._src.solvers.implicit_mpm.solver_implicit_mpm import (
    _require_conditional_allocations,
    _UnsupportedConditionalAllocations,
)
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices

HISTORY_FIELDS = ("particle_qd_grad", "particle_Jp", "particle_stress", "particle_elastic_strain", "particle_transform")


def _make_scene(device, *, worlds=1, collider=False, particle_bases=False, colored_contacts=False):
    """Build a small heterogeneous bed whose logical and spatial particle orders differ."""
    local = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(local)
    locations = [(x, y, z) for x in range(24) for y in range(2) for z in range(2)]
    order = np.random.default_rng(17).permutation(len(locations))
    if colored_contacts:
        # Keep a forced contact at the last eligible ID, beyond a compact pool's last slot.
        contact_slot = int(np.flatnonzero(order == locations.index((12, 0, 0)))[0])
        order[-1], order[contact_slot] = order[contact_slot], order[-1]
    for index in order:
        x, y, z = locations[index]
        local.add_particle(
            pos=(0.0125 + 0.025 * x, 0.0125 + 0.025 * y, 0.0125 + 0.025 * z),
            vel=(0.0, 0.0, 0.0),
            mass=0.0075 + 0.0025 * (index % 4),
            radius=0.01 + 0.001 * (index % 3),
            custom_attributes={
                "mpm:young_modulus": 8000.0 + 1500.0 * (index % 5),
                "mpm:poisson_ratio": 0.15 + 0.02 * (index % 4),
                "mpm:yield_pressure": 1.0e7,
                "mpm:yield_stress": 1.0e7,
                "mpm:tensile_yield_ratio": 1.0,
            },
        )
    local.add_particle(pos=(0.2625, 0.0125, 0.0125), vel=(0.0, 0.0, 0.0), mass=1000.0, radius=0.0125, flags=0)
    local.add_ground_plane(height=-0.05)
    if collider:
        body = local.add_body(xform=wp.transform(wp.vec3(2.0, 0.025, 0.025), wp.quat_identity()), is_kinematic=True)
        local.add_shape_sphere(body, radius=0.025, cfg=local.ShapeConfig(density=0.0))
    if worlds > 1:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverImplicitMPM.register_custom_attributes(builder)
        for _ in range(worlds):
            builder.add_world(local)
    else:
        builder = local
    model = builder.finalize(device=device)

    def make_solver(*, collider_velocity_mode="forward"):
        solver = SolverImplicitMPM(
            model,
            SolverImplicitMPM.Config(
                grid_type="sparse",
                voxel_size=0.05,
                max_active_cell_count=512,
                velocity_basis="B2" if particle_bases else "Q1",
                strain_basis="pic8" if particle_bases else "P0",
                collider_basis="pic8" if particle_bases or colored_contacts else "Q1",
                max_iterations=5,
                tolerance=0.0,
                warmstart_mode="particles",
                separate_worlds=worlds > 1,
                reuse_grid_topology=True,
                collider_velocity_mode=collider_velocity_mode,
            ),
            enable_timers=False,
        )
        solver.set_particle_sleeping(
            sleep_time=0.02,
            collider_margin=0.025,
            velocity_threshold=0.1,
            strain_threshold=1.0e6,
            stress_threshold=1.0e6,
            collider_lookahead=0.0,
        )
        return solver

    return model, make_solver


def _snapshot(state):
    return {
        "particle_q": state.particle_q.numpy().copy(),
        "particle_qd": state.particle_qd.numpy().copy(),
        **{name: getattr(state.mpm, name).numpy().copy() for name in HISTORY_FIELDS},
    }


def _assert_state(state, expected, *, mask=slice(None), exact=True):
    for name, values in expected.items():
        array = getattr(state, name) if name in ("particle_q", "particle_qd") else getattr(state.mpm, name)
        if exact:
            np.testing.assert_array_equal(array.numpy()[mask], values[mask])
        else:
            np.testing.assert_allclose(array.numpy()[mask], values[mask], rtol=2.0e-4, atol=2.0e-5)


def _require_capture(test, device):
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture requires the Warp memory pool")
    if not wp.is_conditional_graph_supported():
        test.skipTest("Guarded MPM requires CUDA conditional graphs")
    try:
        _require_conditional_allocations(wp.get_device(device).alias)
    except _UnsupportedConditionalAllocations as error:
        test.skipTest(str(error))


def test_activity_request_safety(test, device):
    """Exercise classification edge cases without requiring CUDA capture support."""
    local = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(local)
    for x in (0.0975, 0.1125, 0.2025, 0.4025):
        local.add_particle(pos=(x, 0.0125, 0.0125), vel=(0.0, 0.0, 0.0), mass=0.01, radius=0.01)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for _ in range(2):
        builder.add_world(local)
    model = builder.finalize(device=device)
    mpm = ImplicitMPMModel(model, SolverImplicitMPM.Config(voxel_size=0.05, separate_worlds=True))
    base_material_flags = wp.clone(mpm.material_particle_flags)
    base_density = wp.clone(mpm.particle_density)

    def make_activity(*, prescribed=False, **parameters):
        mpm.particle_flags.assign(model.particle_flags)
        mpm.material_particle_flags.assign(base_material_flags)
        mpm.particle_density.assign(base_density)
        if prescribed:
            material_flags = base_material_flags.numpy()
            material_flags[0] = 0
            mpm.material_particle_flags.assign(material_flags)
        activity = ParticleActivity(
            model,
            mpm,
            voxel_size=0.05,
            patch_size=0.1,
            velocity_threshold=0.01,
            strain_threshold=0.1,
            stress_threshold=0.05,
            sleep_time=0.0,
            collider_margin=0.0,
            collider_lookahead=0.0,
            **parameters,
        )
        return activity, model.state()

    def update(activity, state, dt=0.005):
        activity.update(state, dt, mpm.collider, None, True)

    # A prescribed particle remains classified awake even at rest. Its actual
    # translation or affine motion, rather than that mode, must retain support.
    for affine in (False, True):
        activity, state = make_activity(prescribed=True, planning_interval=4)
        if affine:
            gradients = np.zeros((model.particle_count, 3, 3), dtype=np.float32)
            gradients[0, 0, 0] = 0.2
            state.mpm.particle_qd_grad.assign(gradients)
        else:
            velocities = np.zeros((model.particle_count, 3), dtype=np.float32)
            velocities[0, 0] = -0.02  # Sweep away from the stationary neighbor.
            state.particle_qd.assign(velocities)
        for _ in range(4):
            update(activity, state)
            np.testing.assert_array_equal(activity.activity.numpy(), [2, 1, 0, 0, 0, 0, 0, 0])
        state.particle_qd.zero_()
        state.mpm.particle_qd_grad.zero_()
        update(activity, state)
        np.testing.assert_array_equal(activity.activity.numpy(), [2, 0, 0, 0, 0, 0, 0, 0])

    # Both planning and urgent per-tick sweeps must fail conservatively within
    # the particle's world when a finite velocity's displacement overflows.
    for interval, min_count in ((1, 1), (4, 1), (1, 2)):
        activity, state = make_activity(planning_interval=interval, min_moving_particles=min_count)
        update(activity, state)
        velocities = np.zeros((model.particle_count, 3), dtype=np.float32)
        velocities[0, 0] = 1.0e19
        dt = 20.0
        if min_count > 1:
            # Exercise count-gated transport separately from urgent spill motion.
            positions = state.particle_q.numpy()
            positions[1] = positions[0]
            state.particle_q.assign(positions)
            velocities[:2, 0] = 0.02
            dt = 1.0e22
        state.particle_qd.assign(velocities)
        update(activity, state, dt=dt)
        np.testing.assert_array_equal(activity.activity.numpy(), [2, 2, 2, 2, 0, 0, 0, 0])

    # At two-voxel patch size, B2 and B3 particles at x=1.95h and 4.05h
    # share node 3h despite being two patches apart; the latter is support.
    activity, state = make_activity(support_radius=2)
    forces = np.zeros((model.particle_count, 3), dtype=np.float32)
    forces[0, 0] = 0.001
    state.particle_f.assign(forces)
    update(activity, state)
    np.testing.assert_array_equal(activity.activity.numpy(), [2, 1, 1, 0, 0, 0, 0, 0])


def test_captured_sleep_support_and_dirty_wake(test, device):
    """Retain evolved dormant history while a moving collider changes the cached active region."""
    _require_capture(test, device)
    model, make_solver = _make_scene(device, collider=True)
    solver, state = make_solver(), model.state()
    positions = model.particle_q.numpy()
    flags, masses = model.particle_flags.numpy().copy(), model.particle_mass.numpy().copy()
    materials = {name: getattr(model.mpm, name).numpy().copy() for name in ("young_modulus", "poisson_ratio")}
    eligible = (flags & int(newton.ParticleFlags.ACTIVE)) != 0
    force = np.zeros((model.particle_count, 3), dtype=np.float32)
    force[np.flatnonzero(eligible & (positions[:, 0] < 0.1))[0], 0] = 0.003
    state.particle_f.assign(force)
    for _ in range(3):
        solver.step(state, state, None, None, 0.005)
    test.assertGreater(np.linalg.norm(state.mpm.particle_stress.numpy()), 1.0e-6)
    state.clear_forces()
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state, state, None, None, 0.005)
    graph = capture.graph
    pointers = (state.particle_q.ptr, state.particle_qd.ptr, state.mpm.particle_Jp.ptr, solver.step_status.ptr)
    for _ in range(10):
        wp.capture_launch(graph)
    np.testing.assert_array_equal(solver.particle_activity.numpy()[eligible], 0)
    test.assertEqual(solver._scratchpad.grid.cell_grid.get_active_stats().voxel_count, 0)
    frozen = _snapshot(state)
    grains = solver.sample_render_grains(state, grains_per_particle=2)
    frozen_grains = grains.numpy().copy()
    solver.update_render_grains(state, state, grains, dt=0.005)
    np.testing.assert_array_equal(grains.numpy(), frozen_grains)
    for _ in range(3):
        wp.capture_launch(graph)
    _assert_state(state, frozen)
    # Out-of-place output starts with fresh buffers and must inherit dormant history.
    copied = model.state()
    solver.step(state, copied, None, None, 0.005)
    _assert_state(copied, frozen)

    # With zero optional lookahead, the current step still sweeps to x=0.4.
    # Neither the current sphere at x=0.8 nor its previous pose touches the bed.
    state.body_q.assign([wp.transform(wp.vec3(0.8, 0.025, 0.025), wp.quat_identity())])
    state.body_qd.assign([wp.spatial_vector(-80.0, 0.0, 0.0, 0.0, 0.0, 0.0)])
    wp.capture_launch(graph)
    test.assertTrue(np.any(solver.particle_activity.numpy()[eligible & (positions[:, 0] > 0.45)] == 2))
    # Enter the SDF query band to exercise the resulting contact response.
    state.body_q.assign([wp.transform(wp.vec3(0.55, 0.025, 0.025), wp.quat_identity())])
    state.body_qd.assign([wp.spatial_vector(-0.5, 0.0, 0.0, 0.0, 0.0, 0.0)])
    for _ in range(3):
        wp.capture_launch(graph)
    modes = solver.particle_activity.numpy()
    test.assertTrue(np.any(modes[eligible & (positions[:, 0] > 0.45)] == 2))
    test.assertTrue(np.any(modes == 1))
    remote = eligible & (positions[:, 0] < 0.1)
    np.testing.assert_array_equal(modes[remote], 0)
    _assert_state(state, frozen, mask=remote)
    test.assertGreater(np.linalg.norm(state.particle_qd.numpy()[positions[:, 0] > 0.3]), 1.0e-6)
    test.assertGreater(solver._scratchpad.grid.cell_grid.get_active_stats().voxel_count, 0)
    test.assertEqual(
        pointers, (state.particle_q.ptr, state.particle_qd.ptr, state.mpm.particle_Jp.ptr, solver.step_status.ptr)
    )
    np.testing.assert_array_equal(model.particle_flags.numpy(), flags)
    np.testing.assert_array_equal(model.particle_mass.numpy(), masses)
    for name, values in materials.items():
        np.testing.assert_array_equal(getattr(model.mpm, name).numpy(), values)
    solver.check_sparse_grid_rebuild_status()
    with wp.ScopedCapture(device=device) as reset_capture:
        solver.reset(state, flags=0)
    wp.capture_launch(reset_capture.graph)
    wp.capture_launch(graph)
    np.testing.assert_array_equal(solver.particle_activity.numpy()[eligible], 2)
    test.assertEqual(int(solver.step_status.numpy()[0]), 0)

    # Backward mode must ignore stale prescribed velocity and follow pose motion.
    backward, backward_state = make_solver(collider_velocity_mode="backward"), model.state()
    backward_state.body_q.assign([wp.transform(wp.vec3(1.2, 0.025, 0.025), wp.quat_identity())])
    for _ in range(10):
        backward.step(backward_state, backward_state, None, None, 0.005)
    np.testing.assert_array_equal(backward.particle_activity.numpy()[eligible], 0)
    backward_state.body_q.assign([wp.transform(wp.vec3(0.8, 0.025, 0.025), wp.quat_identity())])
    backward_state.body_qd.assign([wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)])
    with wp.ScopedCapture(device=device) as backward_capture:
        backward.step(backward_state, backward_state, None, None, 0.005)
    wp.capture_launch(backward_capture.graph)
    test.assertTrue(np.any(backward.particle_activity.numpy()[eligible & (positions[:, 0] > 0.45)] == 2))
    backward.check_sparse_grid_rebuild_status()


def test_working_set_equivalence(test, device, *, particle_bases=False, page_size=0, colored_contacts=False):
    """Match full sleeping storage when compact evaluation changes IDs across overlapping worlds."""
    _require_capture(test, device)
    model, make_solver = _make_scene(device, worlds=2, particle_bases=particle_bases, colored_contacts=colored_contacts)
    variants = 3 if page_size else 2
    solvers, states = [make_solver() for _ in range(variants)], [model.state() for _ in range(variants)]
    positions, worlds = model.particle_q.numpy(), model.particle_world.numpy()
    flags, masses = model.particle_flags.numpy().copy(), model.particle_mass.numpy().copy()
    eligible = (flags & int(newton.ParticleFlags.ACTIVE)) != 0
    targets = [
        np.flatnonzero(eligible & (worlds == world) & (positions[:, 0] > x) & (positions[:, 0] < x + 0.05))[0]
        for world, x in ((0, 0.25), (1, 0.5), (1, 0.1))
    ]
    if page_size:
        # Logical pages intentionally contain spatially distant particles. One
        # lane must sleep while its distant page neighbor remains awake.
        for first in np.flatnonzero(eligible & (worlds == 1) & (positions[:, 0] > 0.45)):
            same_page = np.arange(model.particle_count) // page_size == first // page_size
            neighbors = np.flatnonzero(same_page & eligible & (worlds == 1) & (positions[:, 0] < 0.2))
            if len(neighbors):
                targets[1], targets[2] = first, neighbors[0]
                break
        else:
            test.fail("The scene must contain distant active IDs in one logical page")
    force = np.zeros((model.particle_count, 3), dtype=np.float32)
    force[targets[:2], 0] = 0.0003
    if particle_bases or colored_contacts:
        mesh = wp.Mesh(
            points=wp.array(
                [(-2.0, -2.0, 0.02), (2.0, -2.0, 0.02), (2.0, 2.0, 0.02), (-2.0, 2.0, 0.02)],
                dtype=wp.vec3,
                device=device,
            ),
            indices=wp.array([0, 1, 2, 0, 2, 3], dtype=int, device=device),
            velocities=wp.zeros(4, dtype=wp.vec3, device=device),
        )
        contact = eligible & (positions[:, 0] > 0.2) & (positions[:, 0] < 0.4) & (positions[:, 2] < 0.02)
        force[contact, 0], force[contact, 2] = 0.002, -0.01
        for solver in solvers:
            solver.setup_collider(collider_meshes=[mesh], collider_friction=[0.3])
    for solver, state in zip(solvers, states, strict=True):
        state.mpm.particle_Jp.assign(np.linspace(0.73, 0.97, model.particle_count, dtype=np.float32))
        state.particle_f.assign(force)
        for _ in range(10):
            solver.step(state, state, None, None, 0.005)
    reference, compact = solvers[0], solvers[-1]
    capacity = 192 if particle_bases or colored_contacts else 160
    for index, (solver, state) in enumerate(zip(solvers[1:], states[1:], strict=True), start=1):
        variant_page_size = page_size if index == len(solvers) - 1 else 0
        variant_capacity = (
            ((model.particle_count + page_size - 1) // page_size) * page_size if variant_page_size else capacity
        )
        solver.set_particle_working_set(
            state, particle_capacity=variant_capacity, page_size=variant_page_size, max_active_cell_count=128
        )
    if not page_size:
        test.assertLess(compact.particle_working_set_capacity, model.particle_count)
    if particle_bases or colored_contacts:
        for solver in solvers:
            solver._last_step_data.ws_impulse_field.dof_values.zero_()
    graphs = []
    for solver, state in zip(solvers, states, strict=True):
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state, state, None, None, 0.005)
        graphs.append(capture.graph)
    for step in range(10 if page_size else 6):
        if step == 2:
            force[targets[1], 0], force[targets[2], 1] = 0.0, -0.0003
            for state in states:
                state.particle_f.assign(force)
        frozen = _snapshot(states[-1])
        for graph in graphs:
            wp.capture_launch(graph)
        if colored_contacts and step == 0:
            scratch = compact._scratchpad
            rows = np.flatnonzero(np.diff(scratch.collider_matrix.offsets.numpy()) > 0)
            nodes = scratch._collision_space_restriction.space_partition.space_node_indices().numpy()[rows]
            contact_ids = scratch.collision_pic.cell_particle_indices.numpy()[nodes]
            test.assertTrue(np.any(contact_ids >= compact.particle_working_set_capacity))
        for state in states[1:]:
            _assert_state(state, _snapshot(states[0]), exact=False)
        modes = compact.particle_activity.numpy()
        np.testing.assert_array_equal(modes, reference.particle_activity.numpy())
        _assert_state(states[-1], frozen, mask=modes != 2)
    if page_size:
        test.assertEqual(targets[1] // page_size, targets[2] // page_size)
        test.assertEqual(modes[targets[1]], 0)
        test.assertEqual(modes[targets[2]], 2)
    test.assertTrue(np.any(compact.particle_activity.numpy() == 1))
    for solver in solvers:
        solver.check_sparse_grid_rebuild_status()
    if particle_bases or colored_contacts:
        impulses = [solver._last_step_data.ws_impulse_field.dof_values.numpy() for solver in solvers]
        for values in impulses:
            test.assertEqual(values.shape, (model.particle_count, 3))
            test.assertGreater(np.linalg.norm(values, axis=1).max(), 1.0e-8)
            np.testing.assert_array_equal(values[~eligible], 0.0)
        np.testing.assert_allclose(impulses[1], impulses[0], rtol=2.0e-4, atol=2.0e-5)
    np.testing.assert_array_equal(model.particle_flags.numpy(), flags)
    np.testing.assert_array_equal(model.particle_mass.numpy(), masses)


def test_captured_overflow_reset(test, device, *, grid_overflow=False):
    """Freeze the full state on either capacity fault until an explicit captured reset."""
    _require_capture(test, device)
    model, make_solver = _make_scene(device, worlds=1 if grid_overflow else 2)
    solver, state = make_solver(), model.state()
    for _ in range(10):
        solver.step(state, state, None, None, 0.005)
    eligible = (model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0
    np.testing.assert_array_equal(solver.particle_activity.numpy()[eligible], 0)
    state.mpm.particle_Jp.fill_(0.81)
    state.mpm.particle_qd_grad.fill_(wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1))
    page_size = 0 if grid_overflow else 8
    solver.set_particle_working_set(
        state,
        particle_capacity=model.particle_count if grid_overflow else page_size,
        page_size=page_size,
        max_active_cell_count=1 if grid_overflow else 512,
    )
    solver.step(state, state, None, None, 0.005)
    test.assertEqual(int(solver.step_status.numpy()[0]), 0)
    previous = _snapshot(state)
    grains = solver.sample_render_grains(state, grains_per_particle=2)
    previous_grains = grains.numpy().copy()
    with wp.ScopedCapture(device=device) as step_capture:
        solver.step(state, state, None, None, 0.005)
        solver.update_particle_frames(state, state, 0.005)
        solver.update_render_grains(state, state, grains, 0.005)
    graph = step_capture.graph
    force = np.zeros((model.particle_count, 3), dtype=np.float32)
    force[eligible, 0] = 0.0001
    state.particle_f.assign(force)
    wp.capture_launch(graph)
    test.assertNotEqual(int(solver.step_status.numpy()[0]), 0)
    with test.assertRaises(RuntimeError):
        solver.check_sparse_grid_rebuild_status()
    _assert_state(state, previous)
    np.testing.assert_array_equal(grains.numpy(), previous_grains)
    copied = model.state()
    solver.update_particle_frames(state, copied, 0.005)
    np.testing.assert_array_equal(copied.mpm.particle_transform.numpy(), previous["particle_transform"])
    state.clear_forces()
    wp.capture_launch(graph)
    test.assertNotEqual(int(solver.step_status.numpy()[0]), 0)
    _assert_state(state, previous)
    np.testing.assert_array_equal(grains.numpy(), previous_grains)
    reset_mask = None if grid_overflow else wp.zeros(model.world_count + 1, dtype=wp.bool, device=device)
    with wp.ScopedCapture(device=device) as reset_capture:
        solver.reset(state, world_mask=reset_mask, flags=0)
    if reset_mask is not None:
        wp.capture_launch(reset_capture.graph)
        test.assertNotEqual(int(solver.step_status.numpy()[0]), 0)
        _assert_state(state, previous)
        selected_worlds = np.zeros(model.world_count + 1, dtype=bool)
        selected_worlds[0] = True
        reset_mask.assign(selected_worlds)
    wp.capture_launch(reset_capture.graph)
    test.assertEqual(int(solver.step_status.numpy()[0]), 0)
    np.testing.assert_array_equal(solver.particle_activity.numpy()[eligible], 2)
    if reset_mask is not None:
        # The reset changes no active ID: the same oversized population must
        # be rejected again, rather than reusing an empty pool as valid.
        wp.capture_launch(graph)
        test.assertNotEqual(int(solver.step_status.numpy()[0]), 0)
        with test.assertRaises(RuntimeError):
            solver.check_particle_working_set_status()
        _assert_state(state, previous)
    del graph, step_capture, reset_capture
    capacity = ((model.particle_count + page_size - 1) // page_size) * page_size if page_size else model.particle_count
    solver.set_particle_working_set(state, particle_capacity=capacity, page_size=page_size, max_active_cell_count=512)
    state.particle_f.assign(force)
    solver.step(state, state, None, None, 0.005)
    with wp.ScopedCapture(device=device) as recovered:
        solver.step(state, state, None, None, 0.005)
    wp.capture_launch(recovered.graph)
    solver.check_sparse_grid_rebuild_status()
    test.assertEqual(int(solver.step_status.numpy()[0]), 0)
    test.assertGreater(np.linalg.norm(state.particle_qd.numpy()[eligible]), 0.0)


def test_material_change_retains_velocity_partition(test, device):
    """Allocate elasticity when rigid material becomes compliant on a retained partition."""
    _require_capture(test, device)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for x in (0.0125, 0.0375):
        builder.add_particle(
            pos=(x, 0.0125, 0.0125),
            vel=(0.0, 0.0, 0.0),
            mass=0.01,
            radius=0.01,
            custom_attributes={"mpm:young_modulus": 1.0e15},
        )
    model = builder.finalize(device=device)
    solver = SolverImplicitMPM(
        model,
        SolverImplicitMPM.Config(
            voxel_size=0.05,
            max_active_cell_count=128,
            velocity_basis="Q1",
            strain_basis="P0",
            collider_basis="Q1",
            max_iterations=5,
            warmstart_mode="none",
            reuse_grid_topology=True,
        ),
    )
    state = model.state()
    solver.step(state, state, None, None, 0.005)
    model.mpm.young_modulus.fill_(1.0e4)
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    state.particle_f.assign(np.array([[0.001, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    solver.step(state, state, None, None, 0.005)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state, state, None, None, 0.005)
    wp.capture_launch(capture.graph)
    solver.check_sparse_grid_rebuild_status()
    test.assertTrue(np.isfinite(state.mpm.particle_stress.numpy()).all())
    test.assertGreater(np.linalg.norm(state.mpm.particle_stress.numpy()), 1.0e-6)


class TestImplicitMPMActivity(unittest.TestCase):
    """Exercise graph replay through public sleeping, working-set, and reset APIs."""


add_function_test(
    TestImplicitMPMActivity,
    "test_activity_request_safety",
    test_activity_request_safety,
    devices=["cpu"],
    check_output=False,
)

for function in (
    test_captured_sleep_support_and_dirty_wake,
    test_working_set_equivalence,
    test_captured_overflow_reset,
    test_material_change_retains_velocity_partition,
):
    add_function_test(
        TestImplicitMPMActivity,
        function.__name__,
        function,
        devices=get_selected_cuda_test_devices(mode="basic"),
        check_output=False,
    )
add_function_test(
    TestImplicitMPMActivity,
    "test_working_set_pages",
    test_working_set_equivalence,
    devices=get_selected_cuda_test_devices(mode="basic"),
    page_size=8,
    check_output=False,
)
add_function_test(
    TestImplicitMPMActivity,
    "test_working_set_point_bases",
    test_working_set_equivalence,
    devices=get_selected_cuda_test_devices(mode="basic"),
    particle_bases=True,
    check_output=False,
)
add_function_test(
    TestImplicitMPMActivity,
    "test_grid_overflow_reset",
    test_captured_overflow_reset,
    devices=get_selected_cuda_test_devices(mode="basic"),
    grid_overflow=True,
    check_output=False,
)

add_function_test(
    TestImplicitMPMActivity,
    "test_working_set_colored_contacts",
    test_working_set_equivalence,
    devices=get_selected_cuda_test_devices(mode="basic"),
    colored_contacts=True,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
