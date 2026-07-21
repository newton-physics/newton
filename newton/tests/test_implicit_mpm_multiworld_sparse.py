# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for coupled multi-world implicit MPM."""

import unittest
from unittest import mock

import numpy as np
import warp as wp
import warp.fem as fem

import newton
from newton.solvers import SolverImplicitMPM, SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _make_triangle_mesh(device) -> wp.Mesh:
    points = wp.array(
        ((-0.05, 0.0, -0.05), (0.05, 0.0, -0.05), (0.0, 0.0, 0.05)),
        dtype=wp.vec3,
        device=device,
    )
    indices = wp.array((0, 1, 2), dtype=wp.int32, device=device)
    return wp.Mesh(points=points, indices=indices, velocities=wp.zeros_like(points))


def _make_mpm_config() -> SolverImplicitMPM.Config:
    config = SolverImplicitMPM.Config()
    config.separate_worlds = True
    config.grid_type = "fixed"
    config.grid_padding = 1
    config.max_iterations = 1
    config.solver = "jacobi"
    config.transfer_scheme = "pic"
    config.warmstart_mode = "none"
    return config


def _make_two_world_particle_model(device) -> newton.Model:
    world_builder = newton.ModelBuilder(gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(world_builder)
    for pos in ((-0.05, 0.0, -0.05), (0.05, 0.0, -0.05), (0.0, 0.0, 0.05)):
        world_builder.add_particle(pos=pos, vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.025)

    builder = newton.ModelBuilder(gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(world_builder)
    builder.add_world(world_builder)
    return builder.finalize(device=device)


def _assert_collider_unchanged(test, solver, expected_worlds, expected_particle_ids):
    collider = solver._mpm_model.collider
    np.testing.assert_array_equal(collider.collider_world.numpy(), expected_worlds)
    np.testing.assert_array_equal(collider.collider_particle_ids.numpy(), expected_particle_ids)


def test_mismatched_deformable_collider_particle_world_rejected(test, device):
    """Verify mismatched deformable collider particle world rejected."""
    model = _make_two_world_particle_model(device)
    solver = SolverImplicitMPM(model, config=_make_mpm_config())
    collider = solver._mpm_model.collider
    initial_worlds = collider.collider_world.numpy().copy()
    initial_particle_ids = collider.collider_particle_ids.numpy().copy()
    world_starts = model.particle_world_start.numpy()
    world_0_particle_ids = list(range(world_starts[0], world_starts[1]))

    with test.assertRaisesRegex(
        ValueError,
        r"collider_particle_ids\[0\].*collider world 1.*particle world IDs \[0\]",
    ):
        solver.setup_collider(
            collider_meshes=[_make_triangle_mesh(device)],
            collider_particle_ids=[world_0_particle_ids],
            collider_world_ids=[1],
        )

    _assert_collider_unchanged(test, solver, initial_worlds, initial_particle_ids)


def test_global_deformable_collider_rejected(test, device):
    """Verify global deformable collider rejected."""
    model = _make_two_world_particle_model(device)
    solver = SolverImplicitMPM(model, config=_make_mpm_config())
    collider = solver._mpm_model.collider
    initial_worlds = collider.collider_world.numpy().copy()
    initial_particle_ids = collider.collider_particle_ids.numpy().copy()
    world_starts = model.particle_world_start.numpy()
    world_0_particle_ids = list(range(world_starts[0], world_starts[1]))

    with test.assertRaisesRegex(
        ValueError,
        r"collider_particle_ids\[0\].*global deformable collider.*isolated worlds",
    ):
        solver.setup_collider(
            collider_meshes=[_make_triangle_mesh(device)],
            collider_particle_ids=[world_0_particle_ids],
            collider_world_ids=[-1],
        )

    _assert_collider_unchanged(test, solver, initial_worlds, initial_particle_ids)


def test_external_deformable_collider_particle_mapping_rejected(test, device):
    """Verify external deformable collider particle mapping rejected."""
    model = _make_two_world_particle_model(device)
    external_model = _make_two_world_particle_model(device)
    solver = SolverImplicitMPM(model, config=_make_mpm_config())
    collider = solver._mpm_model.collider
    initial_worlds = collider.collider_world.numpy().copy()
    initial_particle_ids = collider.collider_particle_ids.numpy().copy()
    world_starts = model.particle_world_start.numpy()
    world_0_particle_ids = list(range(world_starts[0], world_starts[1]))

    with test.assertRaisesRegex(ValueError, r"collider_particle_ids.*solver model"):
        solver.setup_collider(
            collider_meshes=[_make_triangle_mesh(device)],
            collider_particle_ids=[world_0_particle_ids],
            collider_world_ids=[0],
            model=external_model,
        )

    _assert_collider_unchanged(test, solver, initial_worlds, initial_particle_ids)


def test_coupled_multiworld_isolation(test, device):
    """Verify coupled multi-world isolation."""
    config = _make_mpm_config()
    test.assertTrue(config.separate_worlds)

    world_builder = newton.ModelBuilder(gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(world_builder)

    # Three deformable-collider proxies, one transfer-only proxy, and one MPM
    # material particle. Replication keeps the two worlds spatially colocated.
    for pos in (
        (-0.05, 0.0, -0.05),
        (0.05, 0.0, -0.05),
        (0.0, 0.0, 0.05),
        (0.0, 0.1, 0.0),
        (0.0, 0.2, 0.0),
    ):
        world_builder.add_particle(pos=pos, vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.025)

    dynamic_body = world_builder.add_body(
        xform=wp.transform((0.0, -0.1, 0.0), wp.quat_identity()),
        inertia=wp.mat33(np.eye(3)),
        mass=1.0,
        lock_inertia=True,
    )
    world_builder.add_shape_box(dynamic_body, hx=0.2, hy=0.05, hz=0.2)

    builder = newton.ModelBuilder(gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(world_builder)
    builder.add_world(world_builder)
    model = builder.finalize(device=device)

    particle_starts = model.particle_world_start.numpy()
    body_starts = model.body_world_start.numpy()
    shape_starts = model.shape_world_start.numpy()
    collider_proxy_ids = [
        *range(particle_starts[0], particle_starts[0] + 3),
        *range(particle_starts[1], particle_starts[1] + 3),
    ]
    transfer_proxy_ids = [particle_starts[0] + 3, particle_starts[1] + 3]
    material_particle_ids = [particle_starts[0] + 4, particle_starts[1] + 4]
    proxy_particle_ids = collider_proxy_ids + transfer_proxy_ids
    collider_body_ids = [int(body_starts[0]), int(body_starts[1])]
    collider_shape_ids = [int(shape_starts[0]), int(shape_starts[1])]

    coupled = SolverCoupledProxy(
        model=model,
        entries=(
            SolverCoupled.Entry(name="xpbd", solver=SolverXPBD, particles=proxy_particle_ids),
            SolverCoupled.Entry(
                name="mpm",
                solver=lambda view: SolverImplicitMPM(view, config=config),
                bodies=collider_body_ids,
                particles=material_particle_ids,
                shapes=collider_shape_ids,
            ),
        ),
        coupling=SolverCoupledProxy.Config(
            proxies=(
                SolverCoupledProxy.Proxy(
                    source="xpbd",
                    destination="mpm",
                    particles=proxy_particle_ids,
                ),
            )
        ),
    )

    mpm_solver = coupled.solver("mpm")
    mpm_model = mpm_solver._mpm_model
    collider = mpm_model.collider
    expected_worlds = np.array([0, 1], dtype=np.int32)
    expected_body_ids = np.array(collider_body_ids, dtype=np.int32)

    test.assertEqual(mpm_solver._environment_count, 2)
    np.testing.assert_array_equal(collider.collider_world.numpy(), expected_worlds)
    np.testing.assert_array_equal(collider.collider_body_index.numpy(), expected_body_ids)
    test.assertTrue(np.all(mpm_model.collider_body_mass.numpy()[expected_body_ids] > 0.0))
    test.assertGreater(mpm_model.min_collider_mass, 0.0)

    triangle_meshes = [_make_triangle_mesh(device), _make_triangle_mesh(device)]
    deformable_ids_by_world = [collider_proxy_ids[:3], collider_proxy_ids[3:]]
    mpm_solver.setup_collider(
        collider_meshes=triangle_meshes,
        collider_particle_ids=deformable_ids_by_world,
        collider_world_ids=[0, 1],
        model=coupled.view("mpm"),
    )

    active = int(newton.ParticleFlags.ACTIVE)

    np.testing.assert_array_equal(collider.collider_world.numpy(), np.array([0, 1], dtype=np.int32))
    test.assertEqual(collider.world_collider_offsets.shape[0], model.world_count + 1)
    np.testing.assert_array_equal(collider.collider_particle_offsets.numpy(), np.array([0, 3, 6], dtype=np.int32))
    np.testing.assert_array_equal(collider.collider_particle_ids.numpy(), np.array(collider_proxy_ids, dtype=np.int32))

    transfer_flags = mpm_model.particle_flags.numpy()
    material_flags = mpm_model.material_particle_flags.numpy()
    for particle_id in collider_proxy_ids:
        test.assertEqual(transfer_flags[particle_id] & active, 0)
        test.assertEqual(material_flags[particle_id] & active, 0)
    for particle_id in transfer_proxy_ids:
        test.assertNotEqual(transfer_flags[particle_id] & active, 0)
        test.assertEqual(material_flags[particle_id] & active, 0)
    for particle_id in material_particle_ids:
        test.assertNotEqual(transfer_flags[particle_id] & active, 0)
        test.assertNotEqual(material_flags[particle_id] & active, 0)


def _make_sparse_capture_config() -> SolverImplicitMPM.Config:
    config = SolverImplicitMPM.Config()
    config.separate_worlds = True
    config.grid_type = "sparse"
    config.voxel_size = 0.1
    config.grid_padding = 0
    config.max_active_cell_count = 128
    config.max_iterations = 5
    config.tolerance = 0.0
    config.solver = "jacobi"
    config.warmstart_mode = "none"
    config.transfer_scheme = "pic"
    config.integration_scheme = "pic"
    config.strain_basis = "P0"
    config.velocity_basis = "Q1"
    config.collider_basis = "Q1"
    return config


def _make_sparse_capture_case(device):
    world_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(world_builder)
    world_builder.add_particle_grid(
        pos=wp.vec3(0.025, 0.025, 0.025),
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
        custom_attributes={"mpm:young_modulus": 1.0e4, "mpm:poisson_ratio": 0.2},
    )

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(world_builder)
    builder.add_world(world_builder)
    model = builder.finalize(device=device)

    starts = model.particle_world_start.numpy()
    velocities = np.zeros((model.particle_count, 3), dtype=np.float32)
    velocities[starts[0] : starts[1], 0] = 3.0
    velocities[starts[1] : starts[2], 0] = -2.0
    model.particle_qd.assign(velocities)

    solver = SolverImplicitMPM(model, config=_make_sparse_capture_config(), enable_timers=False)
    return model, solver, model.state(), model.state()


def _make_sparse_reset_case(device):
    world_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(world_builder)
    world_builder.add_particle(pos=(0.025, 0.025, 0.025), vel=(0.0, 0.0, 0.0), mass=0.01, radius=0.025)
    world_builder.add_body(
        xform=wp.transform((0.0, -0.1, 0.0), wp.quat_identity()),
        inertia=wp.mat33(np.eye(3)),
        mass=1.0,
    )

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(world_builder)
    builder.add_world(world_builder)
    model = builder.finalize(device=device)
    solver = SolverImplicitMPM(model, config=_make_sparse_capture_config(), enable_timers=False)
    return model, solver, model.state()


def _make_point_warmstart_reset_case(device):
    model = _make_two_world_particle_model(device)
    config = _make_mpm_config()
    config.max_active_cell_count = 64
    config.collider_basis = "pic"
    config.strain_basis = "pic"
    config.warmstart_mode = "grid"
    solver = SolverImplicitMPM(model, config=config, enable_timers=False)
    return model, solver, model.state()


def _require_sparse_capture_prerequisites(test, device):
    if not device.is_cuda:
        test.skipTest("Sparse implicit MPM outer capture requires CUDA.")
    if not device.is_mempool_supported or not wp.is_mempool_enabled(device):
        test.skipTest("Sparse implicit MPM outer capture requires a CUDA memory pool.")
    if not wp.is_conditional_graph_supported():
        test.skipTest("Sparse implicit MPM outer capture requires conditional CUDA graphs.")


def _warm_sparse_solver(model, solver, dt):
    warm_state_0 = model.state()
    warm_state_1 = model.state()
    solver.step(warm_state_0, warm_state_1, control=None, contacts=None, dt=dt)


def _sparse_grid_snapshot(grid):
    active_cell_count = grid.cell_grid.get_active_stats().voxel_count
    cell_ijks = wp.empty(grid.cell_count(), dtype=wp.vec3i, device=grid.cell_env.device)
    grid.cell_grid.get_voxels(out=cell_ijks)
    cell_env = grid.cell_env.numpy()[:active_cell_count]
    env_offsets = grid.env_offsets.numpy().copy()
    packed_cell_ijks = cell_ijks.numpy()[:active_cell_count]
    local_cell_ijks = packed_cell_ijks - env_offsets[cell_env]
    return {
        "cell_env": cell_env,
        "env_offsets": env_offsets,
        "packed_cell_ijks": packed_cell_ijks,
        "local_cell_ijks": local_cell_ijks,
    }


def _sparse_case_state_arrays(state):
    return {
        "particle_q": state.particle_q,
        "particle_qd": state.particle_qd,
        "particle_qd_grad": state.mpm.particle_qd_grad,
        "particle_elastic_strain": state.mpm.particle_elastic_strain,
        "particle_Jp": state.mpm.particle_Jp,
        "particle_stress": state.mpm.particle_stress,
        "particle_transform": state.mpm.particle_transform,
    }


def test_sparse_multiworld_constructs_environment_grid(test, device):
    """Verify sparse multi-world constructs environment grid."""
    model, solver, _state_0, _state_1 = _make_sparse_capture_case(device)
    grid = solver._scratchpad.grid

    test.assertEqual(model.world_count, 2)
    test.assertTrue(solver._separate_worlds)
    test.assertTrue(solver._sparse_rebuildable)
    test.assertEqual(grid.environment_count(), 2)
    test.assertTrue(solver.supports_graph_capture)
    test.assertEqual(solver.max_active_cell_count, 128)


def test_sparse_multiworld_node_capacities_are_total_reserves(test, device):
    """Verify sparse multi-world node capacities are total reserves."""
    _require_sparse_capture_prerequisites(test, device)
    model = _make_two_world_particle_model(device)
    positions = model.particle_q.numpy()
    positions[::2, 1] = -0.06
    positions[1::2, 1] = 0.06
    model.particle_q.assign(positions)
    config = _make_sparse_capture_config()
    config.max_active_cell_count = 256
    config.max_upper_node_count = 32
    config.collider_basis = "Q1"
    solver = SolverImplicitMPM(model, config=config, enable_timers=False)

    capacity = solver._scratchpad.grid.cell_grid.get_rebuild_info()
    test.assertEqual(solver._environment_count, 2)
    test.assertEqual(capacity.max_voxel_count, 256)
    test.assertEqual(capacity.max_leaf_node_count, 256)
    test.assertEqual(capacity.max_lower_node_count, 32)
    test.assertEqual(capacity.max_upper_node_count, 32)


def test_sparse_multiworld_pic_cache_distinguishes_partition_types(test, device):
    """Verify sparse multi-world PIC caches distinguish partition types."""
    plain_model = _make_two_world_particle_model(device)
    plain_config = _make_sparse_capture_config()
    plain_config.max_active_cell_count = -1
    plain_solver = SolverImplicitMPM(plain_model, config=plain_config, enable_timers=False)
    _warm_sparse_solver(plain_model, plain_solver, dt=0.05)

    rebuildable_model, rebuildable_solver, _state_0, _state_1 = _make_sparse_capture_case(device)
    _warm_sparse_solver(rebuildable_model, rebuildable_solver, dt=0.05)

    test.assertEqual(int(rebuildable_solver._grid_status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)


def test_graph_capture_capability_and_preparation(test, device):
    """Verify graph capture capability and preparation."""
    model, solver, state_0, state_1 = _make_sparse_capture_case(device)
    test.assertTrue(solver.supports_graph_capture)

    for solver_name in ("auto", "cg", "cr", "gmres"):
        config = _make_sparse_capture_config()
        config.solver = solver_name
        compatible_solver = SolverImplicitMPM(model, config=config, enable_timers=False)
        test.assertEqual(compatible_solver._environment_count, 2)
        test.assertTrue(compatible_solver._sparse_rebuildable)
        test.assertTrue(compatible_solver.supports_graph_capture, solver_name)
        compatible_solver.prepare_graph_capture()

    fixed_config = _make_mpm_config()
    fixed_config.max_active_cell_count = 128
    fixed_solver = SolverImplicitMPM(model, config=fixed_config, enable_timers=False)
    test.assertTrue(fixed_solver.supports_graph_capture)

    uncapped_fixed_config = _make_mpm_config()
    uncapped_fixed_config.max_active_cell_count = -1
    uncapped_fixed_solver = SolverImplicitMPM(model, config=uncapped_fixed_config, enable_timers=False)
    test.assertFalse(uncapped_fixed_solver.supports_graph_capture)
    with test.assertRaisesRegex(RuntimeError, "max_active_cell_count > 0"):
        uncapped_fixed_solver.prepare_graph_capture()

    incompatible_sparse_configs = []
    plain_sparse_config = _make_sparse_capture_config()
    plain_sparse_config.max_active_cell_count = -1
    incompatible_sparse_configs.append(("plain", plain_sparse_config))
    padded_sparse_config = _make_sparse_capture_config()
    padded_sparse_config.grid_padding = 1
    incompatible_sparse_configs.append(("padding", padded_sparse_config))
    unsupported_basis_config = _make_sparse_capture_config()
    unsupported_basis_config.velocity_basis = "B2"
    incompatible_sparse_configs.append(("basis", unsupported_basis_config))
    for reason, config in incompatible_sparse_configs:
        incompatible_solver = SolverImplicitMPM(model, config=config, enable_timers=False)
        test.assertFalse(incompatible_solver._sparse_rebuildable, reason)
        test.assertFalse(incompatible_solver.supports_graph_capture, reason)
        with test.assertRaisesRegex(RuntimeError, "rebuildable sparse grid"):
            incompatible_solver.prepare_graph_capture()

    state_before = {
        name: array.numpy().copy()
        for name, array in {
            **_sparse_case_state_arrays(state_0),
            "particle_q_out": state_1.particle_q,
            "particle_qd_out": state_1.particle_qd,
        }.items()
    }
    grid = solver._scratchpad.grid
    scratchpad = solver._scratchpad
    vertex_grid = grid.vertex_grid

    solver.prepare_graph_capture()

    test.assertIs(solver._scratchpad, scratchpad)
    test.assertIs(grid.vertex_grid, vertex_grid)
    for name, expected in state_before.items():
        if name == "particle_q_out":
            actual = state_1.particle_q.numpy()
        elif name == "particle_qd_out":
            actual = state_1.particle_qd.numpy()
        else:
            actual = _sparse_case_state_arrays(state_0)[name].numpy()
        np.testing.assert_array_equal(actual, expected)

    solver.enable_timers = True
    test.assertFalse(solver.supports_graph_capture)
    with test.assertRaisesRegex(RuntimeError, "enable_timers=False"):
        solver.prepare_graph_capture()
    solver.enable_timers = False

    with mock.patch.object(wp, "is_mempool_enabled", return_value=False):
        test.assertFalse(solver.supports_graph_capture)
        with test.assertRaisesRegex(RuntimeError, "memory pool"):
            solver.prepare_graph_capture()

    with mock.patch.object(wp, "is_conditional_graph_supported", return_value=False):
        test.assertFalse(solver.supports_graph_capture)
        with test.assertRaisesRegex(RuntimeError, "conditional graph"):
            solver.prepare_graph_capture()

    solver.grid_type = "dense"
    test.assertFalse(solver.supports_graph_capture)
    with test.assertRaisesRegex(RuntimeError, "grid_type='dense'"):
        solver.prepare_graph_capture()
    solver.grid_type = "sparse"

    cpu_model = _make_two_world_particle_model("cpu")
    cpu_config = _make_mpm_config()
    cpu_config.max_active_cell_count = 128
    cpu_solver = SolverImplicitMPM(cpu_model, config=cpu_config, enable_timers=False)
    test.assertFalse(cpu_solver.supports_graph_capture)
    with test.assertRaisesRegex(RuntimeError, "CPU graph capture is not currently supported"):
        cpu_solver.prepare_graph_capture()


def test_sparse_status_is_sticky_until_explicitly_cleared(test, device):
    """Verify sparse status is sticky until explicitly cleared."""
    _require_sparse_capture_prerequisites(test, device)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle((0.01, 0.01, 0.01), (0.0, 0.0, 0.0), mass=1.0)
    builder.add_particle((0.02, 0.02, 0.02), (0.0, 0.0, 0.0), mass=1.0)
    model = builder.finalize(device=device)
    config = _make_sparse_capture_config()
    config.max_active_cell_count = 1
    config.collider_basis = "Q1"
    solver = SolverImplicitMPM(model, config=config, enable_timers=False)
    state_in = model.state()
    state_out = model.state()

    overflow_positions = state_in.particle_q.numpy()
    overflow_positions[1] = (1.01, 1.01, 1.01)
    state_in.particle_q.assign(overflow_positions)

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        solver.step(state_in, state_out, control=None, contacts=None, dt=0.001)

    wp.capture_launch(capture.graph)
    success_positions = state_in.particle_q.numpy()
    success_positions[1] = (0.02, 0.02, 0.02)
    state_in.particle_q.assign(success_positions)
    wp.capture_launch(capture.graph)

    test.assertEqual(int(solver._grid_status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)
    test.assertTrue(int(solver._grid_accumulated_status.numpy()[0]) & wp.Volume.REBUILD_VOXEL_CAPACITY_EXCEEDED)
    with test.assertRaisesRegex(RuntimeError, "sparse grid rebuild capacity"):
        solver.check_status()

    solver._clear_sparse_grid_rebuild_status()
    test.assertEqual(int(solver._grid_status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)
    test.assertEqual(int(solver._grid_accumulated_status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)
    solver.check_status()

    with wp.ScopedCapture(device=device, force_module_load=False):
        with test.assertRaisesRegex(RuntimeError, "clear sparse grid rebuild status.*graph capture"):
            solver._clear_sparse_grid_rebuild_status()


def _assign_nondefault_mpm_history(state):
    particle_count = state.particle_q.shape[0]
    matrices = np.arange(1, particle_count * 9 + 1, dtype=np.float32).reshape(particle_count, 3, 3)
    state.mpm.particle_elastic_strain.assign(matrices)
    state.mpm.particle_transform.assign(matrices + 100.0)
    state.mpm.particle_qd_grad.assign(matrices + 200.0)
    state.mpm.particle_stress.assign(matrices + 300.0)
    state.mpm.particle_Jp.assign(np.arange(2, particle_count + 2, dtype=np.float32))


def _mpm_history_snapshot(state):
    return {
        "particle_elastic_strain": state.mpm.particle_elastic_strain.numpy().copy(),
        "particle_transform": state.mpm.particle_transform.numpy().copy(),
        "particle_qd_grad": state.mpm.particle_qd_grad.numpy().copy(),
        "particle_stress": state.mpm.particle_stress.numpy().copy(),
        "particle_Jp": state.mpm.particle_Jp.numpy().copy(),
    }


def _expected_grid_warmstart_after_mask(field, scratch_field, values, environment):
    partition = scratch_field.space_partition
    if field.space.topology != partition.space_topology:
        raise AssertionError("Test helper requires matching warm-start and scratch topologies.")
    offsets = partition.env_offsets.numpy()
    node_indices = partition.space_node_indices().numpy()
    expected = values.copy()
    expected[node_indices[offsets[environment] : offsets[environment + 1]]] = 0.0
    return expected


def test_masked_reset_restores_only_selected_world_history(test, device):
    """Verify masked reset restores only selected world history."""
    model, solver, state = _make_sparse_reset_case(device)
    _assign_nondefault_mpm_history(state)
    grid_warmstarts = (
        solver._last_step_data.ws_impulse_field,
        solver._last_step_data.ws_stress_field,
    )
    for index, field in enumerate(grid_warmstarts, start=1):
        test.assertNotIsInstance(field.space.basis, fem.PointBasisSpace)
        field.dof_values.assign(np.full_like(field.dof_values.numpy(), float(index)))
    warmstarts_before = tuple(field.dof_values.numpy().copy() for field in grid_warmstarts)
    starts = model.particle_world_start.numpy()
    selected = slice(starts[0], starts[1])
    unselected = slice(starts[1], starts[2])
    before = _mpm_history_snapshot(state)

    body_q = state.body_q.numpy()
    body_q[:, :3] = np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=np.float32)
    state.body_q.assign(body_q)
    solver._last_step_data.body_q_prev.zero_()
    solver._grid_status.fill_(wp.Volume.REBUILD_VOXEL_CAPACITY_EXCEEDED)
    solver._grid_accumulated_status.fill_(wp.Volume.REBUILD_VOXEL_CAPACITY_EXCEEDED)

    world_mask = wp.array((True, False), dtype=wp.bool, device=device)
    solver.reset(state, world_mask=world_mask)

    after = _mpm_history_snapshot(state)
    identity = np.eye(3, dtype=np.float32)[None, ...]
    np.testing.assert_array_equal(after["particle_elastic_strain"][selected], identity)
    np.testing.assert_array_equal(after["particle_transform"][selected], identity)
    np.testing.assert_array_equal(after["particle_qd_grad"][selected], np.zeros((1, 3, 3), dtype=np.float32))
    np.testing.assert_array_equal(after["particle_stress"][selected], np.zeros((1, 3, 3), dtype=np.float32))
    np.testing.assert_array_equal(after["particle_Jp"][selected], np.ones(1, dtype=np.float32))
    for name in after:
        np.testing.assert_array_equal(after[name][unselected], before[name][unselected])
    scratch_warmstarts = (solver._scratchpad.impulse_field, solver._scratchpad.stress_field)
    for field, scratch_field, values in zip(grid_warmstarts, scratch_warmstarts, warmstarts_before, strict=True):
        expected = _expected_grid_warmstart_after_mask(field, scratch_field, values, environment=0)
        np.testing.assert_array_equal(field.dof_values.numpy(), expected)
    expected_body_q_prev = np.zeros_like(body_q)
    selected_bodies = model.body_world.numpy() == 0
    expected_body_q_prev[selected_bodies] = body_q[selected_bodies]
    np.testing.assert_array_equal(solver._last_step_data.body_q_prev.numpy(), expected_body_q_prev)
    test.assertEqual(int(solver._grid_status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)
    test.assertEqual(int(solver._grid_accumulated_status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)

    _assign_nondefault_mpm_history(state)
    before_body_only = _mpm_history_snapshot(state)
    solver.reset(state, world_mask=world_mask, flags=newton.StateFlags.BODY_Q)
    for name, expected in before_body_only.items():
        np.testing.assert_array_equal(_mpm_history_snapshot(state)[name], expected)

    _assign_nondefault_mpm_history(state)
    before_combined = _mpm_history_snapshot(state)
    solver.reset(
        state,
        world_mask=world_mask,
        flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
    )
    after_combined = _mpm_history_snapshot(state)
    np.testing.assert_array_equal(after_combined["particle_elastic_strain"][selected], identity)
    np.testing.assert_array_equal(after_combined["particle_transform"][selected], identity)
    np.testing.assert_array_equal(after_combined["particle_Jp"][selected], np.ones(1, dtype=np.float32))
    for name in after_combined:
        np.testing.assert_array_equal(after_combined[name][unselected], before_combined[name][unselected])

    solver.reset(state, world_mask=None)
    after_full = _mpm_history_snapshot(state)
    expected_identity = np.repeat(identity, model.particle_count, axis=0)
    np.testing.assert_array_equal(after_full["particle_elastic_strain"], expected_identity)
    np.testing.assert_array_equal(after_full["particle_transform"], expected_identity)
    np.testing.assert_array_equal(after_full["particle_qd_grad"], np.zeros_like(expected_identity))
    np.testing.assert_array_equal(after_full["particle_stress"], np.zeros_like(expected_identity))
    np.testing.assert_array_equal(after_full["particle_Jp"], np.ones(model.particle_count, dtype=np.float32))
    for field, expected in zip(grid_warmstarts, warmstarts_before, strict=True):
        np.testing.assert_array_equal(field.dof_values.numpy(), np.zeros_like(expected))


def test_masked_reset_clears_selected_fixed_dense_and_allocating_sparse_warmstarts(test, device):
    """Verify masked reset clears selected fixed dense and allocating sparse warm starts."""
    for grid_type in ("fixed", "dense", "sparse"):
        with test.subTest(grid_type=grid_type):
            model = _make_two_world_particle_model(device)
            config = _make_mpm_config()
            config.grid_type = grid_type
            config.grid_padding = 1
            config.max_active_cell_count = 64 if grid_type == "fixed" else -1
            config.collider_basis = "Q1"
            config.strain_basis = "P0"
            config.warmstart_mode = "grid"
            solver = SolverImplicitMPM(model, config=config, enable_timers=False)
            state = model.state()
            fields = (solver._last_step_data.ws_impulse_field, solver._last_step_data.ws_stress_field)
            scratch_fields = (solver._scratchpad.impulse_field, solver._scratchpad.stress_field)
            before = []
            for index, field in enumerate(fields, start=1):
                values = np.full_like(field.dof_values.numpy(), float(index))
                field.dof_values.assign(values)
                before.append(values)

            solver.reset(state, world_mask=wp.array((True, False), dtype=wp.bool, device=device))

            for field, scratch_field, values in zip(fields, scratch_fields, before, strict=True):
                expected = _expected_grid_warmstart_after_mask(field, scratch_field, values, environment=0)
                np.testing.assert_array_equal(field.dof_values.numpy(), expected)


def test_masked_reset_rejects_shared_grid_warmstarts_before_mutation(test, device):
    """Verify masked reset rejects shared grid warm starts before mutation."""
    model = _make_two_world_particle_model(device)
    config = _make_mpm_config()
    config.separate_worlds = SolverImplicitMPM.Config().separate_worlds
    test.assertFalse(config.separate_worlds)
    solver = SolverImplicitMPM(model, config=config, enable_timers=False)
    state = model.state()
    _assign_nondefault_mpm_history(state)
    history_before = _mpm_history_snapshot(state)
    fields = (solver._last_step_data.ws_impulse_field, solver._last_step_data.ws_stress_field)
    field_values = []
    for index, field in enumerate(fields, start=1):
        values = np.full_like(field.dof_values.numpy(), float(index))
        field.dof_values.assign(values)
        field_values.append(values)

    with test.assertRaisesRegex(RuntimeError, "cannot selectively clear grid-backed warm starts"):
        solver.reset(state, world_mask=wp.array((True, False), dtype=wp.bool, device=device))

    for name, expected in history_before.items():
        np.testing.assert_array_equal(_mpm_history_snapshot(state)[name], expected)
    for field, expected in zip(fields, field_values, strict=True):
        np.testing.assert_array_equal(field.dof_values.numpy(), expected)


def test_masked_reset_clears_only_selected_point_warmstarts(test, device):
    """Verify masked reset clears only selected point warm starts."""
    model, solver, state = _make_point_warmstart_reset_case(device)
    impulse = solver._last_step_data.ws_impulse_field
    stress = solver._last_step_data.ws_stress_field
    test.assertIsInstance(impulse.space.basis, fem.PointBasisSpace)
    test.assertIsInstance(stress.space.basis, fem.PointBasisSpace)
    test.assertEqual(impulse.dof_values.shape, (model.particle_count,))
    test.assertEqual(stress.dof_values.shape, (model.particle_count,))

    impulse_values = np.arange(1, model.particle_count * 3 + 1, dtype=np.float32).reshape(-1, 3)
    stress_values = np.arange(101, 101 + model.particle_count * 6, dtype=np.float32).reshape(-1, 6)
    impulse.dof_values.assign(impulse_values)
    stress.dof_values.assign(stress_values)
    impulse_before = impulse.dof_values.numpy().copy()
    stress_before = stress.dof_values.numpy().copy()
    starts = model.particle_world_start.numpy()
    selected = slice(starts[0], starts[1])
    unselected = slice(starts[1], starts[2])

    solver.reset(state, world_mask=wp.array((True, False), dtype=wp.bool, device=device))

    impulse_after = impulse.dof_values.numpy()
    stress_after = stress.dof_values.numpy()
    np.testing.assert_array_equal(impulse_after[selected], np.zeros_like(impulse_before[selected]))
    np.testing.assert_array_equal(stress_after[selected], np.zeros_like(stress_before[selected]))
    np.testing.assert_array_equal(impulse_after[unselected], impulse_before[unselected])
    np.testing.assert_array_equal(stress_after[unselected], stress_before[unselected])

    impulse.dof_values.assign(impulse_values + 1000.0)
    stress.dof_values.assign(stress_values + 1000.0)
    solver.reset(state, world_mask=None)
    np.testing.assert_array_equal(impulse.dof_values.numpy(), np.zeros_like(impulse_values))
    np.testing.assert_array_equal(stress.dof_values.numpy(), np.zeros_like(stress_values))

    valid_impulse_values = impulse.dof_values
    impulse.dof_values = wp.zeros(model.particle_count + 1, dtype=valid_impulse_values.dtype, device=device)
    stress.dof_values.assign(stress_values)
    with test.assertRaisesRegex(ValueError, "ws_impulse_field.*shape"):
        solver.reset(state, world_mask=None)
    np.testing.assert_array_equal(stress.dof_values.numpy(), stress_values)
    impulse.dof_values = valid_impulse_values


def test_coupled_mpm_non_in_place_capture_replays_after_reset(test, device):
    """Verify coupled MPM non-in-place capture replays after reset."""
    _require_sparse_capture_prerequisites(test, device)
    model = _make_two_world_particle_model(device)
    config = _make_mpm_config()
    config.max_active_cell_count = 64
    coupled = SolverCoupled(
        model=model,
        entries=(
            SolverCoupled.Entry(
                name="mpm",
                solver=lambda view: SolverImplicitMPM(view, config=config, enable_timers=False),
                particles=range(model.particle_count),
                substeps=2,
            ),
        ),
    )
    test.assertTrue(coupled.supports_graph_capture)
    coupled.prepare_graph_capture()

    state_0 = model.state()
    state_1 = model.state()
    coupled.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)
    coupled.reset(state_1)

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        coupled.step(state_1, state_0, control=None, contacts=None, dt=1.0e-4)
        coupled.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    wp.capture_launch(capture.graph)
    wp.capture_launch(capture.graph)
    coupled.reset(state_1, world_mask=wp.array((True, False), dtype=wp.bool, device=device))
    wp.capture_launch(capture.graph)

    test.assertTrue(np.isfinite(state_0.particle_q.numpy()).all())
    test.assertTrue(np.isfinite(state_1.particle_q.numpy()).all())
    entry = coupled._entries["mpm"]
    for values in _mpm_history_snapshot(entry.state_1).values():
        test.assertTrue(np.isfinite(values).all())


def test_reset_validates_state_and_world_mask_before_mutation(test, device):
    """Verify reset validates state and world mask before mutation."""
    model, solver, state = _make_sparse_reset_case(device)
    _assign_nondefault_mpm_history(state)
    expected = _mpm_history_snapshot(state)
    solver._grid_status.fill_(wp.Volume.REBUILD_VOXEL_CAPACITY_EXCEEDED)
    solver._grid_accumulated_status.fill_(wp.Volume.REBUILD_VOXEL_CAPACITY_EXCEEDED)

    invalid_masks = (
        wp.array((True,), dtype=wp.bool, device=device),
        wp.array((1, 0), dtype=wp.int32, device=device),
        wp.array((True, False), dtype=wp.bool, device="cpu"),
    )
    for world_mask in invalid_masks:
        with test.subTest(shape=world_mask.shape, dtype=world_mask.dtype, device=str(world_mask.device)):
            with test.assertRaises((TypeError, ValueError)):
                solver.reset(state, world_mask=world_mask)
            for name, values in expected.items():
                np.testing.assert_array_equal(_mpm_history_snapshot(state)[name], values)
            test.assertNotEqual(int(solver._grid_accumulated_status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)

    valid_mask = wp.array((True, False), dtype=wp.bool, device=device)
    original_jp = state.mpm.particle_Jp
    state.mpm.particle_Jp = wp.zeros(model.particle_count + 1, dtype=float, device=device)
    with test.assertRaisesRegex(ValueError, "particle_Jp.*shape"):
        solver.reset(state, world_mask=valid_mask)
    state.mpm.particle_Jp = original_jp
    for name, values in expected.items():
        np.testing.assert_array_equal(_mpm_history_snapshot(state)[name], values)
    test.assertNotEqual(int(solver._grid_accumulated_status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)

    def assert_reset_inputs_unchanged():
        for name, values in expected.items():
            np.testing.assert_array_equal(_mpm_history_snapshot(state)[name], values)
        test.assertNotEqual(int(solver._grid_status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)
        test.assertNotEqual(int(solver._grid_accumulated_status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)

    with test.subTest(count="world_count"):
        initial_world_count = model.world_count
        model.world_count = 1
        try:
            with test.assertRaisesRegex(
                RuntimeError,
                r"model\.world_count changed after construction: expected 2, got 1",
            ):
                solver.reset(state, world_mask=wp.array((True,), dtype=wp.bool, device=device))
        finally:
            model.world_count = initial_world_count
        assert_reset_inputs_unchanged()

    # Restore the fixture even when the preceding subtest exposed a mutation,
    # so particle-count drift is tested independently.
    _assign_nondefault_mpm_history(state)
    expected = _mpm_history_snapshot(state)
    solver._grid_status.fill_(wp.Volume.REBUILD_VOXEL_CAPACITY_EXCEEDED)
    solver._grid_accumulated_status.fill_(wp.Volume.REBUILD_VOXEL_CAPACITY_EXCEEDED)
    with test.subTest(count="particle_count"):
        initial_particle_count = model.particle_count
        model.particle_count = 1
        try:
            with test.assertRaisesRegex(
                RuntimeError,
                r"model\.particle_count changed after construction: expected 2, got 1",
            ):
                solver.reset(state, world_mask=valid_mask)
        finally:
            model.particle_count = initial_particle_count
        assert_reset_inputs_unchanged()


def test_sparse_multiworld_capture_rebuilds_isolated_topology(test, device):
    """Verify sparse multi-world capture rebuilds isolated topology."""
    _require_sparse_capture_prerequisites(test, device)
    model, solver, state_0, state_1 = _make_sparse_capture_case(device)
    dt = 0.05
    _warm_sparse_solver(model, solver, dt)

    grid = solver._scratchpad.grid
    initial = _sparse_grid_snapshot(grid)
    cell_grid_id = grid.cell_grid.id
    vertex_grid = grid.vertex_grid
    vertex_grid_id = vertex_grid.id
    test.assertEqual(solver._scratchpad._collision_space.topology._vertex_grid, vertex_grid_id)

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        solver.step(state_1, state_0, control=None, contacts=None, dt=dt)

    wp.capture_launch(capture.graph)
    solver.check_status()
    rebuilt = _sparse_grid_snapshot(grid)

    test.assertEqual(int(solver._grid_status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)
    test.assertEqual(grid.cell_grid.id, cell_grid_id)
    test.assertIs(grid.vertex_grid, vertex_grid)
    test.assertEqual(grid.vertex_grid.id, vertex_grid_id)
    test.assertEqual(solver._scratchpad._collision_space.topology._vertex_grid, vertex_grid_id)
    test.assertEqual(grid.environment_count(), 2)
    test.assertEqual(set(rebuilt["cell_env"].tolist()), {0, 1})
    test.assertFalse(np.array_equal(rebuilt["env_offsets"], initial["env_offsets"]))

    packed_by_environment = []
    for environment in range(2):
        initial_local = initial["local_cell_ijks"][initial["cell_env"] == environment]
        rebuilt_local = rebuilt["local_cell_ijks"][rebuilt["cell_env"] == environment]
        test.assertGreater(initial_local.shape[0], 0)
        test.assertGreater(rebuilt_local.shape[0], 0)
        packed_by_environment.append(
            {tuple(cell) for cell in rebuilt["packed_cell_ijks"][rebuilt["cell_env"] == environment].tolist()}
        )

    test.assertGreater(
        np.mean(rebuilt["local_cell_ijks"][rebuilt["cell_env"] == 0, 0]),
        np.mean(initial["local_cell_ijks"][initial["cell_env"] == 0, 0]) + 0.5,
    )
    test.assertLess(
        np.mean(rebuilt["local_cell_ijks"][rebuilt["cell_env"] == 1, 0]),
        np.mean(initial["local_cell_ijks"][initial["cell_env"] == 1, 0]) - 0.5,
    )
    test.assertTrue(packed_by_environment[0].isdisjoint(packed_by_environment[1]))


def test_sparse_multiworld_outer_capture_matches_eager(test, device):
    """Verify sparse multi-world outer capture matches eager."""
    _require_sparse_capture_prerequisites(test, device)
    eager_model, eager_solver, eager_state_0, eager_state_1 = _make_sparse_capture_case(device)
    captured_model, captured_solver, captured_state_0, captured_state_1 = _make_sparse_capture_case(device)
    dt = 0.02
    _warm_sparse_solver(eager_model, eager_solver, dt)
    _warm_sparse_solver(captured_model, captured_solver, dt)

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        captured_solver.step(captured_state_0, captured_state_1, control=None, contacts=None, dt=dt)
        captured_solver.step(captured_state_1, captured_state_0, control=None, contacts=None, dt=dt)

    for cycle in range(3):
        eager_solver.step(eager_state_0, eager_state_1, control=None, contacts=None, dt=dt)
        eager_solver.step(eager_state_1, eager_state_0, control=None, contacts=None, dt=dt)
        wp.capture_launch(capture.graph)
        captured_solver.check_status()

        test.assertEqual(int(captured_solver._grid_status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)
        eager_arrays = _sparse_case_state_arrays(eager_state_0)
        captured_arrays = _sparse_case_state_arrays(captured_state_0)
        for name, eager_array in eager_arrays.items():
            eager_values = eager_array.numpy()
            captured_values = captured_arrays[name].numpy()
            test.assertTrue(np.isfinite(eager_values).all(), f"{name} is non-finite after eager cycle {cycle}")
            test.assertTrue(np.isfinite(captured_values).all(), f"{name} is non-finite after capture cycle {cycle}")
            np.testing.assert_allclose(
                captured_values,
                eager_values,
                rtol=1.0e-5,
                atol=1.0e-6,
                equal_nan=False,
                err_msg=f"{name} differs after capture replay cycle {cycle}",
            )


class TestImplicitMPMMultiworldSparse(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_mismatched_deformable_collider_particle_world_rejected",
    test_mismatched_deformable_collider_particle_world_rejected,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_global_deformable_collider_rejected",
    test_global_deformable_collider_rejected,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_external_deformable_collider_particle_mapping_rejected",
    test_external_deformable_collider_particle_mapping_rejected,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_coupled_multiworld_isolation",
    test_coupled_multiworld_isolation,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_sparse_multiworld_constructs_environment_grid",
    test_sparse_multiworld_constructs_environment_grid,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_sparse_multiworld_node_capacities_are_total_reserves",
    test_sparse_multiworld_node_capacities_are_total_reserves,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_sparse_multiworld_pic_cache_distinguishes_partition_types",
    test_sparse_multiworld_pic_cache_distinguishes_partition_types,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_graph_capture_capability_and_preparation",
    test_graph_capture_capability_and_preparation,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_sparse_status_is_sticky_until_explicitly_cleared",
    test_sparse_status_is_sticky_until_explicitly_cleared,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_masked_reset_restores_only_selected_world_history",
    test_masked_reset_restores_only_selected_world_history,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_masked_reset_clears_selected_fixed_dense_and_allocating_sparse_warmstarts",
    test_masked_reset_clears_selected_fixed_dense_and_allocating_sparse_warmstarts,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_masked_reset_rejects_shared_grid_warmstarts_before_mutation",
    test_masked_reset_rejects_shared_grid_warmstarts_before_mutation,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_masked_reset_clears_only_selected_point_warmstarts",
    test_masked_reset_clears_only_selected_point_warmstarts,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_coupled_mpm_non_in_place_capture_replays_after_reset",
    test_coupled_mpm_non_in_place_capture_replays_after_reset,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_reset_validates_state_and_world_mask_before_mutation",
    test_reset_validates_state_and_world_mask_before_mutation,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_sparse_multiworld_capture_rebuilds_isolated_topology",
    test_sparse_multiworld_capture_rebuilds_isolated_topology,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_sparse_multiworld_outer_capture_matches_eager",
    test_sparse_multiworld_outer_capture_matches_eager,
    devices=devices,
)
