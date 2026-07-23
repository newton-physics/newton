# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
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
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    solver = _make_sparse_solver(model, max_active_cell_count=4, collider_basis="S2")
    test.assertTrue(solver._sparse_rebuildable)


def test_rebuildable_sparse_rejects_grid_warmstart(test, device):
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    for warmstart_mode in ("grid", "smoothed"):
        with (
            test.subTest(warmstart_mode=warmstart_mode),
            test.assertRaisesRegex(ValueError, f"warmstart_mode={warmstart_mode!r}"),
        ):
            _make_sparse_solver(model, max_active_cell_count=4, warmstart_mode=warmstart_mode)


def test_rebuildable_sparse_auto_uses_particle_warmstart(test, device):
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])

    rebuildable = _make_sparse_solver(model, max_active_cell_count=4, warmstart_mode="auto")
    test.assertTrue(rebuildable._sparse_rebuildable)
    test.assertEqual(rebuildable._stress_warmstart, "particles")

    allocating = _make_sparse_solver(model, max_active_cell_count=-1, warmstart_mode="auto")
    test.assertFalse(allocating._sparse_rebuildable)
    test.assertEqual(allocating._stress_warmstart, "grid")


def test_rebuildable_sparse_refreshes_retained_topologies(test, device):
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
            temporary_store=None,
        )

    for topology in topologies:
        topology.rebuild.assert_called_once()


def test_rebuildable_sparse_node_capacity_validation(test, device):
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    for name in ("max_leaf_node_count", "max_lower_node_count", "max_upper_node_count"):
        for value in (0, -2, True, 1.5):
            with test.subTest(name=name, value=value), test.assertRaisesRegex(ValueError, name):
                _make_sparse_solver(model, max_active_cell_count=64, **{name: value})


def test_rebuildable_sparse_grid_reserves_explicit_hierarchy_capacity(test, device):
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
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    solver = _make_sparse_solver(model, max_active_cell_count=64, max_leaf_node_count=1)

    rebuild_info = solver._scratchpad.grid.cell_grid.get_rebuild_info()
    test.assertEqual(rebuild_info.max_leaf_node_count, 1)
    test.assertEqual(rebuild_info.max_lower_node_count, 1)
    test.assertEqual(rebuild_info.max_upper_node_count, 1)


def test_rebuildable_sparse_automatic_upper_capacity_respects_explicit_lower(test, device):
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    solver = _make_sparse_solver(model, max_active_cell_count=64, max_lower_node_count=1)

    rebuild_info = solver._scratchpad.grid.cell_grid.get_rebuild_info()
    test.assertEqual(rebuild_info.max_leaf_node_count, 64)
    test.assertEqual(rebuild_info.max_lower_node_count, 1)
    test.assertEqual(rebuild_info.max_upper_node_count, 1)


def test_rebuildable_sparse_rejects_inconsistent_hierarchy_capacity(test, device):
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
    model = _make_particle_model(device, [(0.01, 0.01, 0.01), (1000.01, 1000.01, 1000.01)], (1,))
    solver = _make_sparse_solver(model, max_active_cell_count=2)

    test.assertTrue(solver._sparse_rebuildable)
    test.assertEqual(solver._scratchpad.grid.cell_grid.get_active_stats().voxel_count, 1)
    solver.check_status()


def test_rebuildable_sparse_grid_excludes_deformable_collider_particles(test, device):
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


def test_rebuildable_sparse_rebuild_uses_mpm_transfer_flags(test, device):
    model = _make_particle_model(device, [(0.01, 0.01, 0.01), (1000.01, 1000.01, 1000.01)], (1,))
    solver = _make_sparse_solver(model, max_active_cell_count=2)
    state_in = model.state()
    state_out = model.state()

    # Coupled views may expose model flags with a different shape; the solver's
    # transfer flags remain particle-aligned and are the source used elsewhere.
    model.particle_flags = wp.ones(model.particle_count + 1, dtype=wp.int32, device=device)
    solver.step(state_in, state_out, None, None, 0.001)

    test.assertEqual(solver._scratchpad.grid.cell_grid.get_active_stats().voxel_count, 1)
    solver.check_status()


def test_rebuildable_sparse_grid_reserves_empty_capacity(test, device):
    model = _make_particle_model(device, [(0.01, 0.01, 0.01), (1.01, 1.01, 1.01)], (0, 1))
    solver = _make_sparse_solver(model, max_active_cell_count=4)

    rebuild_info = solver._scratchpad.grid.cell_grid.get_rebuild_info()
    test.assertEqual(rebuild_info.max_voxel_count, 4)
    test.assertEqual(rebuild_info.max_leaf_node_count, 4)


def test_rebuildable_sparse_grid_reports_initial_overflow(test, device):
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
    test.assertTrue(solver.supports_graph_capture)
    solver.prepare_graph_capture()

    # Materialize persistent topology and the GS solve graph before outer capture.
    solver.step(state_0, state_1, None, None, 0.005)
    state_0, state_1 = state_1, state_0
    grid = solver._scratchpad.grid
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

    solver.check_status()
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
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture requires the Warp memory pool")

    for collider_basis in ("Q1", "S2"):
        with test.subTest(collider_basis=collider_basis):
            _check_rebuildable_sparse_auto_gs_cuda_graph(test, device, collider_basis)


def test_rebuildable_sparse_cuda_graph_reports_overflow(test, device):
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
        solver.check_status()
    status = int(solver._grid_accumulated_status.numpy()[0])
    test.assertTrue(status & wp.Volume.REBUILD_VOXEL_CAPACITY_EXCEEDED)
    solver._clear_sparse_grid_rebuild_status()
    solver.check_status()


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


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
