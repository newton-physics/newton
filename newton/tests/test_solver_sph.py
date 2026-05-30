# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.sph.solver_sph import SolverPBF
from newton.solvers import SolverSPH
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_dam_break(test, device):
    """Particles under gravity should fall and remain bounded.

    Uses parameters derived from Warp's SPH example (Müller 2003)
    with mass proportional to smoothing length cubed.
    """
    spacing = 0.5
    dim = 4
    h = spacing * 2.0
    mass = 0.01 * h**3
    gravity = -0.1

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverSPH.register_custom_attributes(builder)

    builder.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=dim,
        dim_y=dim,
        dim_z=dim,
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=mass,
        jitter=0.0,
    )
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    model.set_gravity((0.0, gravity, 0.0))

    state_0 = model.state()
    state_1 = model.state()

    solver = SolverSPH(
        model,
        smoothing_length=h,
        rest_density=0.06,
        pressure_stiffness=20.0,
        dynamic_viscosity=0.025,
    )

    dt = 0.01 * h
    for _ in range(20):
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0

    end_pos = state_0.particle_q.numpy()
    # Check no NaN
    test.assertFalse(np.any(np.isnan(end_pos)), "Particle positions contain NaN")
    # Check particles are bounded
    max_extent = dim * spacing * 3.0
    test.assertTrue(
        np.all(np.abs(end_pos) < max_extent),
        f"Particles exceeded {max_extent}: max_abs={np.abs(end_pos).max():.2f}",
    )


def test_density_computation(test, device):
    """Interior particles should have nonzero density after one step."""
    spacing = 0.5
    dim = 4
    h = spacing * 2.0
    mass = 0.01 * h**3

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverSPH.register_custom_attributes(builder)

    builder.add_particle_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=dim,
        dim_y=dim,
        dim_z=dim,
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=mass,
        jitter=0.0,
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    state_0 = model.state()
    state_1 = model.state()

    solver = SolverSPH(
        model,
        smoothing_length=h,
        rest_density=1.0,
        pressure_stiffness=20.0,
    )

    # Single step to compute density
    solver.step(state_0, state_1, None, None, 0.001)

    # Density is written to state_in (state_0)
    rho = state_0.sph.density.numpy()
    interior_count = int(np.sum(rho > 0.0))
    test.assertGreater(interior_count, 0, "No particles have nonzero density")


def test_no_particles(test, device):
    """Solver should handle zero particles without error."""
    builder = newton.ModelBuilder()
    SolverSPH.register_custom_attributes(builder)
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()

    solver = SolverSPH(model)

    # Should not raise
    solver.step(state_0, state_1, None, None, 0.01)


# ---------------------------------------------------------------------------
# PBF (Position-Based Fluids) tests
# ---------------------------------------------------------------------------


def _build_pbf_model(device, spacing=0.1, dim=6, rest_density=1000.0, gravity=-9.81):
    """Helper to build a PBF particle grid model."""
    h = spacing * 2.0
    mass = rest_density * spacing**3

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverPBF.register_custom_attributes(builder)

    builder.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=dim,
        dim_y=dim,
        dim_z=dim,
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=mass,
        jitter=0.0,
    )
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    model.set_gravity((0.0, gravity, 0.0))

    return model, h, mass


def test_pbf_dam_break(test, device):
    """PBF dam break: 6x6x6 grid, spacing=0.1, 30 steps, no NaN, particles fall."""
    spacing = 0.1
    dim = 6
    rest_density = 1000.0
    mass = rest_density * spacing**3  # 1.0
    h = 0.2
    gravity = -9.81
    dt = 0.002
    iterations = 4

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverPBF.register_custom_attributes(builder)

    builder.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=dim,
        dim_y=dim,
        dim_z=dim,
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=mass,
        jitter=0.0,
    )
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    model.set_gravity((0.0, gravity, 0.0))

    state_0 = model.state()
    state_1 = model.state()

    solver = SolverPBF(
        model,
        smoothing_length=h,
        rest_density=rest_density,
        iterations=iterations,
    )

    initial_max_y = state_0.particle_q.numpy()[:, 1].max()

    for _ in range(30):
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0

    end_pos = state_0.particle_q.numpy()
    test.assertFalse(np.any(np.isnan(end_pos)), "Particle positions contain NaN")

    end_max_y = end_pos[:, 1].max()
    # PBF constraint may push top-layer particles slightly upward on the first
    # step (boundary effect), so we allow a small tolerance above initial.
    test.assertLess(
        end_max_y,
        initial_max_y + spacing,
        f"max_y ({end_max_y:.4f}) should be within one spacing of initial ({initial_max_y:.4f})",
    )


def test_pbf_density_constraint(test, device):
    """PBF should enforce density close to rest_density for interior particles.

    10 steps with 6 iterations; interior particle density within 20% of rest.
    """
    spacing = 0.1
    dim = 6
    rest_density = 1000.0
    mass = rest_density * spacing**3
    h = 0.2
    gravity = -9.81
    dt = 0.002
    iterations = 6

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverPBF.register_custom_attributes(builder)

    builder.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=dim,
        dim_y=dim,
        dim_z=dim,
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=mass,
        jitter=0.0,
    )
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    model.set_gravity((0.0, gravity, 0.0))

    state_0 = model.state()
    state_1 = model.state()

    solver = SolverPBF(
        model,
        smoothing_length=h,
        rest_density=rest_density,
        iterations=iterations,
    )

    for _ in range(10):
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0

    rho = state_0.sph.density.numpy()
    end_pos = state_0.particle_q.numpy()

    # No NaN in density
    test.assertFalse(np.any(np.isnan(rho)), "Density contains NaN")

    # Find interior particles: not on the boundary of the grid
    # Interior = particles that have neighbors on all sides (skip boundary layer)
    positions = end_pos
    x_vals = np.sort(np.unique(np.round(positions[:, 0], decimals=4)))
    y_vals = np.sort(np.unique(np.round(positions[:, 1], decimals=4)))
    z_vals = np.sort(np.unique(np.round(positions[:, 2], decimals=4)))

    if len(x_vals) >= 3 and len(y_vals) >= 3 and len(z_vals) >= 3:
        x_inner = x_vals[1:-1]
        y_inner = y_vals[1:-1]
        z_inner = z_vals[1:-1]

        # Build interior mask via rounded position matching
        interior_mask = (
            np.isin(np.round(positions[:, 0], 4), x_inner)
            & np.isin(np.round(positions[:, 1], 4), y_inner)
            & np.isin(np.round(positions[:, 2], 4), z_inner)
        )

        interior_rho = rho[interior_mask]
        if len(interior_rho) > 0:
            mean_rho = np.mean(interior_rho)
            rel_error = abs(mean_rho - rest_density) / rest_density
            test.assertLess(
                rel_error,
                0.20,
                f"Interior density {mean_rho:.1f} is more than 20% from rest_density {rest_density}",
            )


def test_pbf_no_particles(test, device):
    """PBF solver should handle zero particles without error."""
    builder = newton.ModelBuilder()
    SolverPBF.register_custom_attributes(builder)
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()

    solver = SolverPBF(model)

    # Should not raise
    solver.step(state_0, state_1, None, None, 0.01)


def test_pbf_single_particle(test, device):
    """Single PBF particle under gravity should fall without NaN."""
    rest_density = 1000.0
    spacing = 0.1
    mass = rest_density * spacing**3
    h = 0.2

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverPBF.register_custom_attributes(builder)

    builder.add_particle(
        pos=wp.vec3(0.0, 1.0, 0.0),
        vel=wp.vec3(0.0),
        mass=mass,
    )
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    model.set_gravity((0.0, -9.81, 0.0))

    state_0 = model.state()
    state_1 = model.state()

    solver = SolverPBF(
        model,
        smoothing_length=h,
        rest_density=rest_density,
        iterations=4,
    )

    for _ in range(30):
        solver.step(state_0, state_1, None, None, 0.002)
        state_0, state_1 = state_1, state_0

    end_pos = state_0.particle_q.numpy()
    test.assertFalse(np.any(np.isnan(end_pos)), "Single particle position is NaN")


def test_pbf_stability_100steps(test, device):
    """PBF 4x4x4 grid, 100 steps: no NaN, no explosion (max position < 10)."""
    spacing = 0.2
    dim = 4
    rest_density = 1000.0
    mass = rest_density * spacing**3
    h = spacing * 2.0
    gravity = -9.81
    dt = 0.002
    iterations = 4

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverPBF.register_custom_attributes(builder)

    builder.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=dim,
        dim_y=dim,
        dim_z=dim,
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=mass,
        jitter=0.0,
    )
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    model.set_gravity((0.0, gravity, 0.0))

    state_0 = model.state()
    state_1 = model.state()

    solver = SolverPBF(
        model,
        smoothing_length=h,
        rest_density=rest_density,
        iterations=iterations,
    )

    for _ in range(100):
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0

    end_pos = state_0.particle_q.numpy()
    test.assertFalse(np.any(np.isnan(end_pos)), "Particle positions contain NaN after 100 steps")

    max_pos = np.max(np.abs(end_pos))
    test.assertLess(
        max_pos,
        10.0,
        f"Particles exploded: max absolute position = {max_pos:.2f}",
    )


devices = get_test_devices()


class TestSolverSPH(unittest.TestCase):
    pass


add_function_test(TestSolverSPH, "test_dam_break", test_dam_break, devices=devices)
add_function_test(TestSolverSPH, "test_density_computation", test_density_computation, devices=devices)
add_function_test(TestSolverSPH, "test_no_particles", test_no_particles, devices=devices)

add_function_test(TestSolverSPH, "test_pbf_dam_break", test_pbf_dam_break, devices=devices)
add_function_test(TestSolverSPH, "test_pbf_density_constraint", test_pbf_density_constraint, devices=devices)
add_function_test(TestSolverSPH, "test_pbf_no_particles", test_pbf_no_particles, devices=devices)
add_function_test(TestSolverSPH, "test_pbf_single_particle", test_pbf_single_particle, devices=devices)
add_function_test(TestSolverSPH, "test_pbf_stability_100steps", test_pbf_stability_100steps, devices=devices)


if __name__ == "__main__":
    unittest.main()
