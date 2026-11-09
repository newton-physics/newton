# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp
import warp.fem as fem

import newton
from newton._src.solvers.implicit_mpm.solver_implicit_mpm import ImplicitMPMScratchpad
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices


def _make_particle_model(device, positions, inactive_indices=()):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverImplicitMPM.register_custom_attributes(builder)
    for position in positions:
        builder.add_particle(wp.vec3(*position), wp.vec3(0.0), mass=1.0)

    model = builder.finalize(device=device)
    if inactive_indices:
        flags = model.particle_flags.numpy()
        for particle_index in inactive_indices:
            flags[particle_index] &= ~int(newton.ParticleFlags.ACTIVE)
        model.particle_flags.assign(flags)
    return model


def _make_sparse_solver(
    model,
    max_active_cell_count,
    collider_basis="Q1",
    voxel_size=0.1,
    warmstart_mode="none",
    **config_kwargs,
):
    config = SolverImplicitMPM.Config(
        grid_type="sparse",
        voxel_size=voxel_size,
        max_active_cell_count=max_active_cell_count,
        velocity_basis="Q1",
        strain_basis="P0",
        collider_basis=collider_basis,
        max_iterations=2,
        warmstart_mode=warmstart_mode,
        **config_kwargs,
    )
    return SolverImplicitMPM(model, config, verbose=False)


def test_rebuildable_sparse_s2_is_enabled(test, device):
    """Verify S2 collider bases enable rebuildable sparse grids."""
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    solver = _make_sparse_solver(model, max_active_cell_count=4, collider_basis="S2")
    test.assertTrue(solver._sparse_rebuildable)


def test_rebuildable_sparse_rejects_grid_warmstart(test, device):
    """Verify rebuildable sparse grids reject grid-backed warm starts."""
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    for warmstart_mode in ("grid", "smoothed"):
        with (
            test.subTest(warmstart_mode=warmstart_mode),
            test.assertRaisesRegex(ValueError, f"warmstart_mode={warmstart_mode!r}"),
        ):
            _make_sparse_solver(model, max_active_cell_count=4, warmstart_mode=warmstart_mode)


def test_rebuildable_sparse_auto_uses_particle_warmstart(test, device):
    """Verify automatic warm starts use particles for rebuildable sparse grids."""
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])

    rebuildable = _make_sparse_solver(model, max_active_cell_count=4, warmstart_mode="auto")
    test.assertTrue(rebuildable._sparse_rebuildable)
    test.assertEqual(rebuildable._stress_warmstart, "particles")

    allocating = _make_sparse_solver(model, max_active_cell_count=-1, warmstart_mode="auto")
    test.assertFalse(allocating._sparse_rebuildable)
    test.assertEqual(allocating._stress_warmstart, "grid")


def test_rebuildable_sparse_refreshes_retained_topologies(test, device):
    """Verify retained function-space topologies rebuild in place."""
    del test, device
    geometry = object()
    topologies = [SimpleNamespace(rebuild=mock.Mock()) for _ in range(3)]
    scratch = ImplicitMPMScratchpad.__new__(ImplicitMPMScratchpad)
    scratch.grid = geometry
    scratch._velocity_basis = SimpleNamespace(topology=topologies[0])
    scratch._strain_basis = SimpleNamespace(topology=topologies[1])
    scratch._collision_basis = SimpleNamespace(topology=topologies[2])

    with (
        mock.patch.object(scratch, "_create_velocity_function_space"),
        mock.patch.object(scratch, "_create_collider_function_space"),
        mock.patch.object(scratch, "_create_strain_function_space"),
    ):
        scratch.rebuild_function_spaces(
            SimpleNamespace(domain=SimpleNamespace(geometry=geometry)),
            velocity_basis_str="Q1",
            strain_basis_str="P0",
            collider_basis_str="S2",
            max_cell_count=8,
            environment_first=False,
            temporary_store=None,
        )

    for topology in topologies:
        topology.rebuild.assert_called_once()


def test_rebuildable_sparse_node_capacity_validation(test, device):
    """Verify rebuildable sparse-grid node capacities reject invalid values."""
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    for name in ("max_leaf_node_count", "max_lower_node_count", "max_upper_node_count"):
        for value in (0, -2, True, 1.5):
            with test.subTest(name=name, value=value), test.assertRaisesRegex(ValueError, name):
                _make_sparse_solver(model, max_active_cell_count=64, **{name: value})


def test_rebuildable_sparse_grid_reserves_explicit_hierarchy_capacity(test, device):
    """Verify explicit sparse-grid hierarchy capacities are reserved."""
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    solver = _make_sparse_solver(
        model,
        max_active_cell_count=64,
        max_leaf_node_count=48,
        max_lower_node_count=24,
        max_upper_node_count=12,
    )

    rebuild_info = solver._scratchpad.grid.cell_grid.get_rebuild_info()
    test.assertEqual(rebuild_info.max_voxel_count, 64)
    test.assertEqual(rebuild_info.max_leaf_node_count, 48)
    test.assertEqual(rebuild_info.max_lower_node_count, 24)
    test.assertEqual(rebuild_info.max_upper_node_count, 12)


def test_rebuildable_sparse_automatic_hierarchy_respects_explicit_leaf(test, device):
    """Verify automatic internal-node capacities respect an explicit leaf capacity."""
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    solver = _make_sparse_solver(model, max_active_cell_count=64, max_leaf_node_count=1)

    rebuild_info = solver._scratchpad.grid.cell_grid.get_rebuild_info()
    test.assertEqual(rebuild_info.max_leaf_node_count, 1)
    test.assertEqual(rebuild_info.max_lower_node_count, 1)
    test.assertEqual(rebuild_info.max_upper_node_count, 1)


def test_rebuildable_sparse_automatic_upper_capacity_respects_explicit_lower(test, device):
    """Verify automatic upper-node capacity respects an explicit lower capacity."""
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    solver = _make_sparse_solver(model, max_active_cell_count=64, max_lower_node_count=1)

    rebuild_info = solver._scratchpad.grid.cell_grid.get_rebuild_info()
    test.assertEqual(rebuild_info.max_leaf_node_count, 64)
    test.assertEqual(rebuild_info.max_lower_node_count, 1)
    test.assertEqual(rebuild_info.max_upper_node_count, 1)


def test_rebuildable_sparse_rejects_inconsistent_hierarchy_capacity(test, device):
    """Verify inconsistent sparse-grid hierarchy capacities are rejected."""
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    with test.assertRaisesRegex(ValueError, "capacity hierarchy"):
        _make_sparse_solver(
            model,
            max_active_cell_count=256,
            max_leaf_node_count=256,
            max_lower_node_count=16,
            max_upper_node_count=32,
        )


def test_rebuildable_sparse_grid_excludes_inactive_particles(test, device):
    """Verify inactive particles do not contribute to sparse-grid construction."""
    model = _make_particle_model(device, [(0.01, 0.01, 0.01), (1000.01, 1000.01, 1000.01)], (1,))
    solver = _make_sparse_solver(model, max_active_cell_count=2)

    test.assertTrue(solver._sparse_rebuildable)
    test.assertEqual(solver._scratchpad.grid.cell_grid.get_active_stats().voxel_count, 1)
    solver.check_sparse_grid_rebuild_status()


def test_rebuildable_sparse_grid_excludes_deformable_collider_particles(test, device):
    """Verify deformable collider particles are excluded from sparse-grid rebuilds."""
    positions = (
        (0.01, 0.01, 0.01),
        (0.02, 0.01, 0.01),
        (0.01, 0.02, 0.01),
        (0.02, 0.02, 0.01),
    )
    model = _make_particle_model(device, positions)
    solver = _make_sparse_solver(model, max_active_cell_count=1)
    collider_points = wp.array(positions[:3], dtype=wp.vec3, device=device)
    collider_mesh = wp.Mesh(
        points=collider_points,
        indices=wp.array((0, 1, 2), dtype=wp.int32, device=device),
        velocities=wp.zeros_like(collider_points),
    )
    solver.setup_collider(collider_meshes=[collider_mesh], collider_particle_ids=[[1, 2, 3]])

    active = int(newton.ParticleFlags.ACTIVE)
    test.assertTrue(np.all(model.particle_flags.numpy() & active))
    np.testing.assert_array_equal(solver._mpm_model.particle_flags.numpy() & active, [active, 0, 0, 0])

    moved_positions = np.asarray(positions, dtype=np.float32)
    moved_positions[1:] = ((1000.01, 0.01, 0.01), (0.01, 1000.01, 0.01), (0.01, 0.01, 1000.01))
    moved_positions = wp.array(moved_positions, dtype=wp.vec3, device=device)
    observed_point_masks = []

    class _StopAfterMask(RuntimeError):
        pass

    def observe_rebuild(*args, **kwargs):
        observed_point_masks.append(kwargs["point_mask"].numpy().copy())
        raise _StopAfterMask

    with (
        mock.patch.object(fem.Nanogrid, "rebuild", autospec=True, side_effect=observe_rebuild),
        test.assertRaises(_StopAfterMask),
    ):
        solver._particles_to_cells(moved_positions)

    test.assertEqual(len(observed_point_masks), 1)
    np.testing.assert_array_equal(observed_point_masks[0], [1, 0, 0, 0])


def test_rebuildable_sparse_grid_excludes_nonfinite_particles_before_rebuild(test, device):
    """Verify non-finite particles are excluded before sparse-grid rebuilds."""
    model = _make_particle_model(
        device,
        [(0.01, 0.01, 0.01), (1.01, 1.01, 1.01), (2.01, 2.01, 2.01), (3.01, 3.01, 3.01)],
        (3,),
    )
    solver = _make_sparse_solver(model, max_active_cell_count=8)
    positions = model.particle_q.numpy()
    positions[1] = (np.nan, 1.01, 1.01)
    positions[2] = (2.01, np.inf, -np.inf)
    poisoned_positions = wp.array(positions, dtype=wp.vec3, device=device)
    observed_point_masks = []

    class _StopAfterMask(RuntimeError):
        pass

    def observe_rebuild(*args, **kwargs):
        observed_point_masks.append(kwargs["point_mask"].numpy().copy())
        raise _StopAfterMask

    with (
        mock.patch.object(fem.Nanogrid, "rebuild", autospec=True, side_effect=observe_rebuild),
        test.assertRaises(_StopAfterMask),
    ):
        solver._particles_to_cells(poisoned_positions)

    test.assertEqual(len(observed_point_masks), 1)
    np.testing.assert_array_equal(observed_point_masks[0], [1, 0, 0, 0])


def test_rebuildable_sparse_preserves_inactive_particle_strain(test, device):
    """Verify inactive particles keep their plastic, elastic strain and stress across a step.

    Covers inactive particles next to active ones and inactive particles outside any active cell,
    which get no quadrature point on a rebuildable grid.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverImplicitMPM.register_custom_attributes(builder)
    # Active block, inactive block in the neighboring cell, inactive block far from any active cell
    for x0 in (0.025, 0.125, 1.125):
        builder.add_particle_grid(
            pos=wp.vec3(x0, 0.025, 0.025),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2,
            dim_y=2,
            dim_z=2,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            mass=0.01,
            jitter=0.0,
            radius_mean=0.025,
            custom_attributes={"mpm:young_modulus": 1.0e4, "mpm:poisson_ratio": 0.2, "mpm:hardening": 10.0},
        )
    model = builder.finalize(device=device)

    x = model.particle_q.numpy()[:, 0]
    adjacent = (x > 0.1) & (x < 1.0)
    far = x > 1.0
    flags = model.particle_flags.numpy()
    flags[adjacent | far] &= ~int(newton.ParticleFlags.ACTIVE)
    model.particle_flags.assign(flags)

    solver = _make_sparse_solver(model, max_active_cell_count=64, warmstart_mode="particles")
    state_0 = model.state()
    state_1 = model.state()

    particle_count = model.particle_count
    jp = np.full(particle_count, 0.9, dtype=np.float32)
    strain = np.tile(np.diag([1.01, 0.99, 1.0]).astype(np.float32), (particle_count, 1, 1))
    stress = np.tile(np.diag([-10.0, -20.0, -30.0]).astype(np.float32), (particle_count, 1, 1))
    state_0.mpm.particle_Jp.assign(jp)
    state_0.mpm.particle_elastic_strain.assign(strain)
    state_0.mpm.particle_stress.assign(stress)

    solver.step(state_0, state_1, None, None, 0.001)

    for name, inactive in (("adjacent", adjacent), ("far", far)):
        with test.subTest(inactive=name):
            np.testing.assert_array_equal(state_1.mpm.particle_Jp.numpy()[inactive], jp[inactive])
            np.testing.assert_array_equal(state_1.mpm.particle_elastic_strain.numpy()[inactive], strain[inactive])
            np.testing.assert_array_equal(state_1.mpm.particle_stress.numpy()[inactive], stress[inactive])


def test_rebuildable_sparse_rebuild_uses_mpm_transfer_flags(test, device):
    """Verify sparse-grid rebuilds use MPM transfer flags."""
    model = _make_particle_model(device, [(0.01, 0.01, 0.01), (1000.01, 1000.01, 1000.01)], (1,))
    solver = _make_sparse_solver(model, max_active_cell_count=2)
    state_in = model.state()
    state_out = model.state()

    # Coupled views may expose model flags with a different shape; the solver's
    # transfer flags remain particle-aligned and are the source used elsewhere.
    model.particle_flags = wp.ones(model.particle_count + 1, dtype=wp.int32, device=device)
    solver.step(state_in, state_out, None, None, 0.001)

    test.assertEqual(solver._scratchpad.grid.cell_grid.get_active_stats().voxel_count, 1)
    solver.check_sparse_grid_rebuild_status()


def test_rebuildable_sparse_grid_reserves_empty_capacity(test, device):
    """Verify empty sparse grids retain their configured capacity."""
    model = _make_particle_model(device, [(0.01, 0.01, 0.01), (1.01, 1.01, 1.01)], (0, 1))
    solver = _make_sparse_solver(model, max_active_cell_count=4)

    rebuild_info = solver._scratchpad.grid.cell_grid.get_rebuild_info()
    test.assertEqual(rebuild_info.max_voxel_count, 4)
    test.assertEqual(rebuild_info.max_leaf_node_count, 4)


def test_rebuildable_sparse_grid_reports_initial_overflow(test, device):
    """Verify initial sparse-grid capacity overflow raises."""
    model = _make_particle_model(device, [(0.01, 0.01, 0.01), (1.01, 1.01, 1.01)])

    with test.assertRaisesRegex(RuntimeError, "sparse grid rebuild capacity"):
        _make_sparse_solver(model, max_active_cell_count=1)


def _check_rebuildable_sparse_auto_gs_cuda_graph(test, device, collider_basis):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle_grid(
        pos=wp.vec3(0.05, 0.2, 0.05),
        rot=wp.quat_identity(),
        vel=wp.vec3(25.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        mass=1.0,
        jitter=0.0,
    )
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    eager_state_0 = model.state()
    eager_state_1 = model.state()
    eager_solver = _make_sparse_solver(
        model,
        max_active_cell_count=64,
        collider_basis=collider_basis,
        solver="auto",
    )
    test.assertEqual(eager_solver.solver, ("gs",))
    for _ in range(5):
        eager_solver.step(eager_state_0, eager_state_1, None, None, 0.005)
        eager_state_0, eager_state_1 = eager_state_1, eager_state_0
    eager_positions = eager_state_0.particle_q.numpy()
    eager_velocities = eager_state_0.particle_qd.numpy()

    state_0 = model.state()
    state_1 = model.state()
    solver = _make_sparse_solver(
        model,
        max_active_cell_count=64,
        collider_basis=collider_basis,
        solver="auto",
    )
    test.assertEqual(solver.solver, ("gs",))
    grid = solver._scratchpad.grid
    if collider_basis == "S2":
        test.assertIsNotNone(grid._edge_grid)

    # Build the inner GS solve graph before recording the outer graph.
    solver.step(state_0, state_1, None, None, 0.005)
    state_0, state_1 = state_1, state_0
    cell_grid_id = grid.cell_grid.id
    initial_cell_count = grid.cell_grid.get_active_stats().voxel_count
    initial_cells = {tuple(ijk) for ijk in grid.cell_grid.get_voxels().numpy()[:initial_cell_count]}
    if collider_basis == "S2":
        test.assertIsNotNone(grid._edge_grid)
        edge_grid_id = grid.edge_grid.id
        initial_edge_count = grid.edge_grid.get_active_stats().voxel_count
        initial_edges = {tuple(ijk) for ijk in grid.edge_grid.get_voxels().numpy()[:initial_edge_count]}

    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_0, state_1, None, None, 0.005)
        solver.step(state_1, state_0, None, None, 0.005)

    for _ in range(2):
        wp.capture_launch(capture.graph)

    solver.check_sparse_grid_rebuild_status()
    test.assertEqual(solver._scratchpad.grid.cell_grid.id, cell_grid_id)
    final_cell_count = grid.cell_grid.get_active_stats().voxel_count
    final_cells = {tuple(ijk) for ijk in grid.cell_grid.get_voxels().numpy()[:final_cell_count]}
    test.assertNotEqual(final_cells, initial_cells)
    if collider_basis == "S2":
        test.assertEqual(solver._scratchpad.grid.edge_grid.id, edge_grid_id)
        final_edge_count = grid.edge_grid.get_active_stats().voxel_count
        final_edges = {tuple(ijk) for ijk in grid.edge_grid.get_voxels().numpy()[:final_edge_count]}
        test.assertNotEqual(final_edges, initial_edges)
    test.assertTrue(np.isfinite(state_0.particle_q.numpy()).all())
    test.assertTrue(np.isfinite(state_0.particle_qd.numpy()).all())
    np.testing.assert_allclose(state_0.particle_q.numpy(), eager_positions, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(state_0.particle_qd.numpy(), eager_velocities, rtol=1.0e-5, atol=1.0e-5)


def test_rebuildable_sparse_auto_gs_cuda_graph(test, device):
    """Verify automatic Gauss-Seidel matches eager execution under graph replay."""
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture requires the Warp memory pool")

    for collider_basis in ("Q1", "S2"):
        with test.subTest(collider_basis=collider_basis):
            _check_rebuildable_sparse_auto_gs_cuda_graph(test, device, collider_basis)


def test_rebuildable_sparse_cuda_graph_reports_overflow(test, device):
    """Verify CUDA graph replay reports sparse-grid capacity overflow."""
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture requires the Warp memory pool")

    model = _make_particle_model(device, [(0.01, 0.01, 0.01), (0.02, 0.02, 0.02)])
    solver = _make_sparse_solver(model, max_active_cell_count=1)
    state_in = model.state()
    state_out = model.state()

    positions = state_in.particle_q.numpy()
    positions[1] = (1.01, 1.01, 1.01)
    state_in.particle_q.assign(positions)

    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_in, state_out, None, None, 0.001)
    wp.capture_launch(capture.graph)

    with test.assertRaisesRegex(RuntimeError, "sparse grid rebuild capacity"):
        solver.check_sparse_grid_rebuild_status()
    status = int(solver._grid_accumulated_status.numpy()[0])
    test.assertTrue(status & wp.Volume.REBUILD_VOXEL_CAPACITY_EXCEEDED)
    solver._clear_sparse_grid_rebuild_status()
    solver.check_sparse_grid_rebuild_status()


def test_rebuildable_sparse_warmstarts_collider_impulses(test, device):
    """Carry grid collider impulses across in-place rebuilds, matching a sparse grid allocated each step.

    Two separate worlds with different gravities keep their own impulses, and a masked reset clears only one.
    """
    spacing = 0.025
    local = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverImplicitMPM.register_custom_attributes(local)
    local.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, 0.5 * spacing),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=8,
        dim_y=8,
        dim_z=12,
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=spacing**3 * 1000.0,
        jitter=0.0,
        radius_mean=0.5 * spacing,
        custom_attributes={"mpm:young_modulus": 1.0e5, "mpm:friction": 0.5},
    )
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(local)
    builder.add_world(local)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -20.0), world=1)
    reset_mask = wp.array([False, True, False], dtype=wp.bool, device=device)

    def settle(max_active_cell_count, capture=False):
        # Few iterations cannot rebuild the ground's support from cold impulses.
        config = SolverImplicitMPM.Config(
            grid_type="sparse",
            voxel_size=2.0 * spacing,
            max_active_cell_count=max_active_cell_count,
            collider_basis="S2",
            warmstart_mode="particles",
            max_iterations=10,
            tolerance=0.0,
            separate_worlds=True,
        )
        solver = SolverImplicitMPM(model, config, verbose=False)
        state = model.state()
        graph = None
        if capture:
            with wp.ScopedCapture(device=device) as capture_scope:
                solver.step(state, state, None, None, 1.0 / 120.0)
            graph = capture_scope.graph
        for step in range(40):
            if step == 20:
                solver.reset(state, world_mask=reset_mask, flags=0)
            if graph is None:
                solver.step(state, state, None, None, 1.0 / 120.0)
            else:
                wp.capture_launch(graph)
        return state.particle_q.numpy()

    reference = settle(-1)
    test.assertGreater(reference[:, 2].min(), -0.5 * spacing)
    np.testing.assert_allclose(settle(8192), reference, atol=1.0e-6)
    # Graphs captured before the first step keep the warm start across replays.
    np.testing.assert_allclose(settle(8192, capture=True), reference, atol=1.0e-6)


def test_particle_properties_notify_refreshes_masks_in_place(test, device):
    """Verify PARTICLE_PROPERTIES refreshes the particle masks in place, with flag-independent extrema."""
    active = int(newton.ParticleFlags.ACTIVE)
    proxy = int(newton.ParticleFlags.PROXY)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverImplicitMPM.register_custom_attributes(builder)
    for i, flags in enumerate((active, active, 0, active | proxy)):
        builder.add_particle(
            wp.vec3(0.01 + 0.05 * i, 0.01, 0.01),
            wp.vec3(0.0),
            mass=1.0,
            radius=0.01 * (i + 1),
            flags=flags,
            custom_attributes={"mpm:viscosity": 10.0 if i == 2 else 0.0, "mpm:dilatancy": 0.5 if i == 3 else 0.0},
        )
    model = builder.finalize(device=device)
    solver = _make_sparse_solver(model, max_active_cell_count=8)
    mpm_model = solver._mpm_model
    masks = (mpm_model.particle_flags, mpm_model.material_particle_flags, mpm_model.material_particle_volume)
    pointers = [array.ptr for array in masks]
    volume = mpm_model.particle_volume.numpy()

    # The inactive viscous particle and the dilatant proxy particle select their code paths up front.
    test.assertTrue(mpm_model.has_viscosity)
    test.assertTrue(mpm_model.has_dilatancy)

    def check_masks(flags):
        material = np.where(flags & proxy, flags & ~active, flags)
        np.testing.assert_array_equal(mpm_model.particle_flags.numpy(), flags)
        np.testing.assert_array_equal(mpm_model.material_particle_flags.numpy(), material)
        np.testing.assert_array_equal(
            mpm_model.material_particle_volume.numpy(), np.where(material & active, volume, 0.0)
        )
        test.assertEqual([array.ptr for array in masks], pointers)
        test.assertTrue(mpm_model.has_viscosity)
        test.assertTrue(mpm_model.has_dilatancy)

    check_masks(model.particle_flags.numpy())

    flags = model.particle_flags.numpy()
    flags[0] &= ~active
    flags[2] |= active
    model.particle_flags.assign(flags)
    solver.notify_model_changed(newton.ModelFlags.PARTICLE_PROPERTIES)
    check_masks(flags)

    flags[3] &= ~proxy
    model.particle_flags.assign(flags)
    solver.notify_model_changed(newton.ModelFlags.PARTICLE_PROPERTIES)
    check_masks(flags)

    flags[1] &= ~active
    model.particle_flags.assign(flags)
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    check_masks(flags)


def test_notify_rejects_particle_count_change(test, device):
    """Verify MODEL_PROPERTIES and PARTICLE_PROPERTIES notifications reject a changed particle count."""
    model = _make_particle_model(device, [(0.01, 0.01, 0.01), (0.03, 0.01, 0.01)])
    solver = _make_sparse_solver(model, max_active_cell_count=8)
    masks = solver._mpm_model.particle_flags
    model.particle_count += 1
    for flag in (newton.ModelFlags.MODEL_PROPERTIES, newton.ModelFlags.PARTICLE_PROPERTIES):
        with test.subTest(flag=flag.name):
            with test.assertRaisesRegex(RuntimeError, "particle_count changed after construction"):
                solver.notify_model_changed(flag)
    test.assertIs(solver._mpm_model.particle_flags, masks)
    model.particle_count -= 1
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)


def test_particle_properties_notify_in_cuda_graph(test, device):
    """Verify particle activity notified inside a CUDA graph matches eager MODEL_PROPERTIES notifications."""
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture requires the Warp memory pool")

    active = int(newton.ParticleFlags.ACTIVE)
    cluster = [(0.01 + 0.02 * i, 0.5 + 0.02 * j, 0.01 + 0.02 * k) for i in range(2) for j in range(2) for k in range(2)]
    positions = cluster + [(x + 1.0, y, z) for x, y, z in cluster]
    model = _make_particle_model(device, positions)
    all_active = model.particle_flags.numpy()
    second_frozen = all_active.copy()
    second_frozen[len(cluster) :] &= ~active
    phases = (second_frozen, all_active, second_frozen)

    def run(capture):
        model.particle_flags.assign(all_active)
        solver = _make_sparse_solver(model, max_active_cell_count=64, solver="auto")
        state_0, state_1 = model.state(), model.state()
        solver.step(state_0, state_1, None, None, 0.005)  # builds the inner solve graph before capture
        state_0, state_1 = state_1, state_0

        def two_steps():
            solver.step(state_0, state_1, None, None, 0.005)
            solver.step(state_1, state_0, None, None, 0.005)

        graph = None
        if capture:
            with wp.ScopedCapture(device=device) as scope:
                solver.notify_model_changed(newton.ModelFlags.PARTICLE_PROPERTIES)
                two_steps()
            graph = scope.graph

        results = []
        for flags in phases:
            before = state_0.particle_q.numpy()
            model.particle_flags.assign(flags)
            if capture:
                wp.capture_launch(graph)
            else:
                solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
                two_steps()
            results.append((before, state_0.particle_q.numpy()))
        solver.check_sparse_grid_rebuild_status()
        return results

    eager = run(capture=False)
    graph = run(capture=True)
    for (eager_before, eager_after), (graph_before, graph_after), flags in zip(eager, graph, phases, strict=True):
        frozen = (flags & active) == 0
        np.testing.assert_array_equal(eager_after[frozen], eager_before[frozen])
        np.testing.assert_array_equal(graph_after[frozen], graph_before[frozen])
        test.assertTrue(np.all(graph_after[~frozen, 1] < graph_before[~frozen, 1]))
        np.testing.assert_allclose(graph_after, eager_after, rtol=1.0e-5, atol=1.0e-6)


def _make_material_block_model(device, **custom_attributes):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverImplicitMPM.register_custom_attributes(builder)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                builder.add_particle(
                    wp.vec3(0.01 + 0.02 * i, 0.5 + 0.02 * j, 0.01 + 0.02 * k),
                    wp.vec3(0.0),
                    mass=1.0,
                    radius=0.01,
                    custom_attributes=custom_attributes,
                )
    return builder.finalize(device=device)


def test_particle_properties_notify_refreshes_material_extrema(test, device):
    """Verify PARTICLE_PROPERTIES refreshes volumes and material extrema in place."""
    model = _make_material_block_model(device)
    solver = _make_sparse_solver(model, max_active_cell_count=8)
    mpm_model = solver._mpm_model
    arrays = (mpm_model._material_extrema, mpm_model.particle_volume, mpm_model.particle_density)
    pointers = [array.ptr for array in arrays]
    test.assertEqual(mpm_model.code_path_switches(), (False, False, False, False))

    model.mpm.young_modulus.fill_(1.0e4)
    model.mpm.hardening.fill_(5.0)
    model.mpm.viscosity.fill_(2.0)
    model.mpm.dilatancy.fill_(0.5)
    model.particle_radius.fill_(0.02)
    solver.notify_model_changed(newton.ModelFlags.PARTICLE_PROPERTIES)

    test.assertAlmostEqual(mpm_model.min_young_modulus, 1.0e4, places=2)
    test.assertAlmostEqual(mpm_model.max_hardening, 5.0, places=5)
    test.assertEqual(mpm_model.code_path_switches(), (True, True, True, True))
    np.testing.assert_allclose(mpm_model.particle_volume.numpy(), 8.0 * 0.02**3, rtol=1.0e-5)
    test.assertEqual([array.ptr for array in arrays], pointers)


def test_material_notify_in_cuda_graph_syncs_on_eager_step(test, device):
    """Verify a material change notified inside a CUDA graph reaches the host on the next eager step."""
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture requires the Warp memory pool")

    model = _make_material_block_model(device)
    solver = _make_sparse_solver(model, max_active_cell_count=64)
    mpm_model = solver._mpm_model
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, None, None, 0.005)

    with wp.ScopedCapture(device=device) as scope:
        solver.notify_model_changed(newton.ModelFlags.PARTICLE_PROPERTIES)
    graph = scope.graph

    model.mpm.viscosity.fill_(2.0)
    wp.capture_launch(graph)
    # The host has not read the graph's extrema back yet.
    test.assertFalse(mpm_model.has_viscosity)
    solver.step(state_1, state_0, None, None, 0.005)
    test.assertTrue(mpm_model.has_viscosity)

    # Every eager step reads back once a notification was captured.
    model.mpm.viscosity.zero_()
    wp.capture_launch(graph)
    solver.step(state_0, state_1, None, None, 0.005)
    test.assertFalse(mpm_model.has_viscosity)


def test_material_change_after_step_capture_warns(test, device):
    """Verify switching a code path after capturing step() warns, and other material changes do not."""
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture requires the Warp memory pool")

    model = _make_material_block_model(device, **{"mpm:young_modulus": 1.0e4})
    solver = _make_sparse_solver(model, max_active_cell_count=64)
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, None, None, 0.005)
    with wp.ScopedCapture(device=device):
        solver.step(state_1, state_0, None, None, 0.005)

    # Values within the captured code paths are read at run time: no warning.
    model.mpm.young_modulus.fill_(1.0e3)
    model.mpm.friction.fill_(0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solver.notify_model_changed(newton.ModelFlags.PARTICLE_PROPERTIES)

    model.mpm.viscosity.fill_(2.0)
    with test.assertWarnsRegex(UserWarning, "viscosity on.*capture them again"):
        solver.notify_model_changed(newton.ModelFlags.PARTICLE_PROPERTIES)

    # Warned once per capture.
    model.mpm.dilatancy.fill_(0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solver.notify_model_changed(newton.ModelFlags.PARTICLE_PROPERTIES)


class TestImplicitMPMRebuildableSparse(unittest.TestCase):
    pass


devices = get_test_devices(mode="basic")
cuda_devices = get_selected_cuda_test_devices(mode="basic")

add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_s2_is_enabled",
    test_rebuildable_sparse_s2_is_enabled,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_refreshes_retained_topologies",
    test_rebuildable_sparse_refreshes_retained_topologies,
    devices=None,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_rejects_grid_warmstart",
    test_rebuildable_sparse_rejects_grid_warmstart,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_auto_uses_particle_warmstart",
    test_rebuildable_sparse_auto_uses_particle_warmstart,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_node_capacity_validation",
    test_rebuildable_sparse_node_capacity_validation,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_grid_reserves_explicit_hierarchy_capacity",
    test_rebuildable_sparse_grid_reserves_explicit_hierarchy_capacity,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_automatic_hierarchy_respects_explicit_leaf",
    test_rebuildable_sparse_automatic_hierarchy_respects_explicit_leaf,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_automatic_upper_capacity_respects_explicit_lower",
    test_rebuildable_sparse_automatic_upper_capacity_respects_explicit_lower,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_rejects_inconsistent_hierarchy_capacity",
    test_rebuildable_sparse_rejects_inconsistent_hierarchy_capacity,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_grid_excludes_inactive_particles",
    test_rebuildable_sparse_grid_excludes_inactive_particles,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_grid_excludes_deformable_collider_particles",
    test_rebuildable_sparse_grid_excludes_deformable_collider_particles,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_grid_excludes_nonfinite_particles_before_rebuild",
    test_rebuildable_sparse_grid_excludes_nonfinite_particles_before_rebuild,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_preserves_inactive_particle_strain",
    test_rebuildable_sparse_preserves_inactive_particle_strain,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_rebuild_uses_mpm_transfer_flags",
    test_rebuildable_sparse_rebuild_uses_mpm_transfer_flags,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_grid_reserves_empty_capacity",
    test_rebuildable_sparse_grid_reserves_empty_capacity,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_grid_reports_initial_overflow",
    test_rebuildable_sparse_grid_reports_initial_overflow,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_auto_gs_cuda_graph",
    test_rebuildable_sparse_auto_gs_cuda_graph,
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_cuda_graph_reports_overflow",
    test_rebuildable_sparse_cuda_graph_reports_overflow,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_warmstarts_collider_impulses",
    test_rebuildable_sparse_warmstarts_collider_impulses,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_particle_properties_notify_refreshes_masks_in_place",
    test_particle_properties_notify_refreshes_masks_in_place,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_notify_rejects_particle_count_change",
    test_notify_rejects_particle_count_change,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_particle_properties_notify_in_cuda_graph",
    test_particle_properties_notify_in_cuda_graph,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_particle_properties_notify_refreshes_material_extrema",
    test_particle_properties_notify_refreshes_material_extrema,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_material_notify_in_cuda_graph_syncs_on_eager_step",
    test_material_notify_in_cuda_graph_syncs_on_eager_step,
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_material_change_after_step_capture_warns",
    test_material_change_after_step_capture_warns,
    devices=cuda_devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
