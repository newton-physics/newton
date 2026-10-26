# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test live MPM resolution changes and swept particle projection."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices


def _make_solver(
    device,
    *,
    moving_wall=False,
    wall_positions=(0.0,),
    particle_positions=((0.5, 0.0, 0.0),),
    grid_type=None,
    integration_scheme="pic",
):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for position in particle_positions:
        builder.add_particle(position, (0.0, 0.0, 0.0), mass=0.01, radius=0.025)
    for x in wall_positions:
        body = builder.add_body(is_kinematic=True) if moving_wall else -1
        builder.add_shape_box(
            body,
            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            hx=0.01,
            hy=1.0,
            hz=1.0,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, margin=0.0),
        )
    model = builder.finalize(device=device)
    config = SolverImplicitMPM.Config(
        voxel_size=0.1,
        grid_type=grid_type or ("sparse" if device.is_cuda else "dense"),
        grid_padding=0 if device.is_cuda and grid_type != "fixed" else 1,
        max_active_cell_count=64 if device.is_cuda else -1,
        max_iterations=1,
        warmstart_mode="particles",
        collider_basis="Q1",
        integration_scheme=integration_scheme,
    )
    return model, SolverImplicitMPM(model, config, verbose=False)


def _particle_snapshot(model, state):
    arrays = {"particle_q": state.particle_q, "particle_qd": state.particle_qd, "particle_mass": model.particle_mass}
    arrays.update({f"mpm.{name}": value for name, value in vars(state.mpm).items() if isinstance(value, wp.array)})
    return {name: array.numpy().copy() for name, array in arrays.items()}


def _assert_particle_snapshot(model, state, expected):
    actual = _particle_snapshot(model, state)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values, err_msg=name)


def test_swept_projection_thin_wall(test, device):
    """Stop a particle whose complete trajectory crosses a thin wall."""
    model, solver = _make_solver(device)
    previous, current = model.state(), model.state()
    previous.particle_q.fill_(wp.vec3(0.5, 0.0, 0.0))
    current.particle_q.fill_(wp.vec3(-0.5, 0.0, 0.0))
    current.particle_qd.fill_(wp.vec3(-10.0, 0.0, 0.0))

    # Endpoint-only projection cannot see the wall after the particle crosses it.
    solver.project_outside(current, current, 0.1)
    test.assertLess(current.particle_q.numpy()[0, 0], -0.4)

    solver.project_outside(current, current, 0.1, state_prev=previous)
    test.assertGreaterEqual(current.particle_q.numpy()[0, 0], 0.009)
    test.assertAlmostEqual(float(current.particle_qd.numpy()[0, 0]), 0.0, places=5)


def test_swept_projection_first_wall(test, device):
    """Select the earliest collision independently of collider ordering."""
    model, solver = _make_solver(device, wall_positions=(0.0, 0.2))
    previous, current = model.state(), model.state()
    current.particle_q.fill_(wp.vec3(-0.5, 0.0, 0.0))
    current.particle_qd.fill_(wp.vec3(-10.0, 0.0, 0.0))
    solver.project_outside(current, current, 0.1, state_prev=previous)
    test.assertGreaterEqual(current.particle_q.numpy()[0, 0], 0.209)


def test_swept_projection_corner(test, device):
    """Stop the remaining slide from crossing a second thin wall at a corner."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle((0.5, 0.5, 0.0), (0.0, 0.0, 0.0), mass=0.01, radius=0.025)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, margin=0.0)
    builder.add_shape_box(-1, hx=0.01, hy=1.0, hz=1.0, cfg=cfg)
    builder.add_shape_box(-1, hx=1.0, hy=0.01, hz=1.0, cfg=cfg)
    model = builder.finalize(device=device)
    solver = SolverImplicitMPM(
        model,
        SolverImplicitMPM.Config(
            voxel_size=0.1,
            grid_type="sparse" if device.is_cuda else "dense",
            grid_padding=0 if device.is_cuda else 1,
            max_active_cell_count=64 if device.is_cuda else -1,
            collider_basis="Q1",
            max_iterations=1,
        ),
    )
    previous, current = model.state(), model.state()
    current.particle_q.fill_(wp.vec3(-0.5, -0.8, 0.0))
    current.particle_qd.fill_(wp.vec3(-10.0, -13.0, 0.0))
    solver.project_outside(current, current, 0.1, state_prev=previous)
    position = current.particle_q.numpy()[0]
    test.assertGreaterEqual(position[0], 0.009)
    test.assertGreaterEqual(position[1], 0.009)


def test_swept_projection_moving_wall(test, device):
    """Catch a translating wall that passes completely through a stationary particle."""
    model, solver = _make_solver(device, moving_wall=True)
    previous, current = model.state(), model.state()
    previous.particle_q.zero_()
    current.particle_q.zero_()
    previous.body_q.fill_(wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()))
    current.body_q.fill_(wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()))
    solver.project_outside(current, current, 0.1, state_prev=previous)
    test.assertGreaterEqual(current.particle_q.numpy()[0, 0], 0.259)
    test.assertAlmostEqual(float(current.particle_qd.numpy()[0, 0]), 5.0, places=4)


def test_swept_projection_inactive(test, device):
    """Preserve inactive and zero-mass particles during swept projection."""
    model, solver = _make_solver(device)
    previous, current = model.state(), model.state()
    current.particle_q.fill_(wp.vec3(-0.5, 0.0, 0.0))
    current.particle_qd.fill_(wp.vec3(-10.0, 0.0, 0.0))
    for inactive in (True, False):
        if inactive:
            model.particle_flags.zero_()
        else:
            model.particle_flags.fill_(int(newton.ParticleFlags.ACTIVE))
            model.particle_mass.zero_()
        solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        solver.project_outside(current, current, 0.1, state_prev=previous)
        test.assertAlmostEqual(float(current.particle_q.numpy()[0, 0]), -0.5)
        test.assertAlmostEqual(float(current.particle_qd.numpy()[0, 0]), -10.0)


def test_resize_preserves_particle_state(test, device):
    """Rebuild PIC and GIMP grids without resetting particle history."""
    # GIMP also initializes overlap quadrature and its contact/history buffers.
    for integration_scheme in ("pic", "gimp"):
        with test.subTest(integration_scheme=integration_scheme):
            model, solver = _make_solver(device, integration_scheme=integration_scheme)
            current, next_state = model.state(), model.state()
            current.particle_q.fill_(wp.vec3(0.55, 0.02, 0.03))
            current.particle_qd.fill_(wp.vec3(0.1, 0.2, 0.3))
            current.mpm.particle_stress.fill_(wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1))
            current.mpm.particle_Jp.fill_(0.9)
            arrays = [current.particle_q, current.particle_qd]
            arrays.extend(value for value in vars(current.mpm).values() if isinstance(value, wp.array))
            snapshots = [array.numpy().copy() for array in arrays]
            old_grid = solver._scratchpad.grid

            solver.set_voxel_size(0.05, state=current)

            test.assertEqual(solver.voxel_size, 0.05)
            test.assertIsNot(solver._scratchpad.grid, old_grid)
            for array, snapshot in zip(arrays, snapshots, strict=True):
                np.testing.assert_array_equal(array.numpy(), snapshot)
            solver.step(current, next_state, None, None, 0.001)
            solver.check_sparse_grid_rebuild_status()
            test.assertTrue(np.isfinite(next_state.particle_q.numpy()).all())
            test.assertGreater(next_state.particle_q.numpy()[0, 0], 0.5)


def test_resize_validation(test, device):
    """Reject invalid resolution changes before changing solver resources."""
    model, solver = _make_solver(device)
    current = model.state()
    grid = solver._scratchpad.grid
    for value in (0.0, -0.1, np.inf, np.nan):
        with test.subTest(value=value), test.assertRaises(ValueError):
            solver.set_voxel_size(value, state=current)
        test.assertEqual(solver.voxel_size, 0.1)
        test.assertIs(solver._scratchpad.grid, grid)
    solver.set_voxel_size(0.1, state=current)
    test.assertIs(solver._scratchpad.grid, grid)


def test_radius_preserves_particle_state(test, device):
    """Update every MPM support radius and density without resetting particle history."""
    model, solver = _make_solver(device, particle_positions=((0.5, 0.0, 0.0), (0.65, 0.0, 0.0)))
    current, next_state = model.state(), model.state()
    current.particle_qd.fill_(wp.vec3(0.1, 0.2, 0.3))
    current.mpm.particle_stress.fill_(wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1))
    current.mpm.particle_Jp.fill_(0.9)
    before = _particle_snapshot(model, current)
    grid = solver._scratchpad.grid
    original_radii = model.particle_radius.numpy().copy()
    for radius in (0.0, -0.01, np.inf, np.nan):
        with test.subTest(radius=radius), test.assertRaises(ValueError):
            solver.set_particle_radius(radius)
        np.testing.assert_array_equal(model.particle_radius.numpy(), original_radii)
        test.assertAlmostEqual(model.particle_max_radius, 0.025)
        test.assertIs(solver._scratchpad.grid, grid)
        _assert_particle_snapshot(model, current, before)

    solver.set_particle_radius(0.02)

    _assert_particle_snapshot(model, current, before)
    np.testing.assert_allclose(model.particle_radius.numpy(), 0.02)
    test.assertAlmostEqual(model.particle_max_radius, 0.02)
    np.testing.assert_allclose(solver._mpm_model.particle_volume.numpy(), 8.0 * 0.02**3)
    np.testing.assert_allclose(solver._mpm_model.particle_density.numpy(), before["particle_mass"] / (8.0 * 0.02**3))
    solver.step(current, next_state, None, None, 0.001)
    solver.check_sparse_grid_rebuild_status()
    test.assertTrue(np.isfinite(next_state.particle_q.numpy()).all())
    test.assertTrue(np.isfinite(next_state.particle_qd.numpy()).all())


def test_resize_fixed_grid(test, device):
    """Reject fixed-grid resizing with a clear error."""
    model, solver = _make_solver(device, grid_type="fixed")
    with test.assertRaisesRegex(ValueError, "fixed"):
        solver.set_voxel_size(0.05, state=model.state())


def test_swept_projection_world_isolation(test, device):
    """Apply swept projection only to colliders belonging to the particle's world."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for world in range(2):
        builder.begin_world()
        builder.add_particle((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), mass=0.01, radius=0.025)
        if world == 0:
            builder.add_shape_box(-1, hx=0.01, hy=1.0, hz=1.0)
        builder.end_world()
    model = builder.finalize(device=device)
    solver = SolverImplicitMPM(
        model,
        SolverImplicitMPM.Config(
            voxel_size=0.1,
            grid_type="sparse" if device.is_cuda else "dense",
            grid_padding=0 if device.is_cuda else 1,
            max_active_cell_count=64 if device.is_cuda else -1,
            collider_basis="Q1",
            max_iterations=1,
            separate_worlds=True,
        ),
    )
    previous, current = model.state(), model.state()
    current.particle_q.fill_(wp.vec3(-0.5, 0.0, 0.0))
    solver.project_outside(current, current, 0.1, state_prev=previous)
    positions = current.particle_q.numpy()
    test.assertGreater(positions[0, 0], 0.0)
    test.assertAlmostEqual(float(positions[1, 0]), -0.5)


def test_resize_capacity_rollback(test, device):
    """Keep the original grid and particle state when a finer grid exceeds capacity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for i in range(8):
        builder.add_particle((0.01 + i * 0.1, 0.01, 0.01), (0.0, 0.0, 0.0), mass=0.01, radius=0.025)
    model = builder.finalize(device=device)
    solver = SolverImplicitMPM(
        model,
        SolverImplicitMPM.Config(voxel_size=1.0, max_active_cell_count=4, collider_basis="Q1", max_iterations=1),
    )
    state = model.state()
    positions = state.particle_q.numpy().copy()
    grid = solver._scratchpad.grid
    with test.assertRaisesRegex(RuntimeError, "capacity"):
        solver.set_voxel_size(0.01, state=state)
    test.assertEqual(solver.voxel_size, 1.0)
    test.assertIs(solver._scratchpad.grid, grid)
    solver.check_sparse_grid_rebuild_status()
    np.testing.assert_array_equal(state.particle_q.numpy(), positions)
    solver.step(state, state, None, None, 0.001)
    test.assertTrue(np.isfinite(state.particle_q.numpy()).all())


def test_resize_capture_rejection(test, device):
    """Reject voxel and radius changes during capture before altering live state."""
    model, solver = _make_solver(device)
    state = model.state()
    grid = solver._scratchpad.grid
    before = _particle_snapshot(model, state)
    radii = model.particle_radius.numpy().copy()
    with wp.ScopedCapture(device=device):
        with test.assertRaisesRegex(RuntimeError, "capture"):
            solver.set_voxel_size(0.05, state=state)
        with test.assertRaisesRegex(RuntimeError, "capture"):
            solver.set_particle_radius(0.02)
    test.assertEqual(solver.voxel_size, 0.1)
    test.assertIs(solver._scratchpad.grid, grid)
    test.assertAlmostEqual(model.particle_max_radius, 0.025)
    np.testing.assert_array_equal(model.particle_radius.numpy(), radii)
    _assert_particle_snapshot(model, state, before)


def test_bowl_graph_and_runtime_changes(test, device):
    """Preserve the MPM bowl state across voxel and radius changes with graph recapture."""
    from newton.examples.mpm.example_mpm_bowl import Example  # noqa: PLC0415
    from newton.viewer import ViewerNull  # noqa: PLC0415

    if not wp.is_mempool_enabled(device) or not wp.is_conditional_graph_supported():
        test.skipTest("MPM graph capture requires a memory pool and conditional graph support")

    with wp.ScopedDevice(device):
        args = Example.create_parser().parse_args(
            ["--particle-radius", "0.04", "--voxel-size", "0.1", "--iterations", "3"]
        )
        example = Example(ViewerNull(), args)
        test.assertIsNotNone(example.graph)
        example.step()
        before = _particle_snapshot(example.model, example.state_0)
        sim_time = example.sim_time

        for update in (
            lambda: example.set_voxel_size(0.08),
            lambda: example.set_particle_radius(0.036),
            example.capture,
        ):
            old_graph = example.graph
            update()
            _assert_particle_snapshot(example.model, example.state_0, before)
            test.assertEqual(example.sim_time, sim_time)
            test.assertAlmostEqual(float(example.sim_clock.numpy()[0]), sim_time, places=6)
            test.assertIsNotNone(example.graph)
            test.assertIsNot(example.graph, old_graph)

        test.assertAlmostEqual(example.mpm_solver.voxel_size, 0.08)
        test.assertAlmostEqual(example.model.particle_max_radius, 0.036)
        np.testing.assert_allclose(example.model.particle_radius.numpy(), 0.036)
        example.step()
        test.assertGreater(example.sim_time, sim_time)
        example.test_post_step()
        example.mpm_solver.check_sparse_grid_rebuild_status()


class TestImplicitMPMAdaptive(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_swept_projection_thin_wall,
        test_swept_projection_first_wall,
        test_swept_projection_corner,
        test_swept_projection_moving_wall,
        test_swept_projection_inactive,
        test_resize_preserves_particle_state,
        test_resize_validation,
        test_radius_preserves_particle_state,
        test_resize_fixed_grid,
        test_swept_projection_world_isolation,
    ):
        add_function_test(TestImplicitMPMAdaptive, _test.__name__, _test, devices=[_device])

for _test in (test_resize_capacity_rollback, test_resize_capture_rejection, test_bowl_graph_and_runtime_changes):
    add_function_test(TestImplicitMPMAdaptive, _test.__name__, _test, devices=get_selected_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
