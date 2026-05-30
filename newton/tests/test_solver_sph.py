# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
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


devices = get_test_devices()


class TestSolverSPH(unittest.TestCase):
    pass


add_function_test(TestSolverSPH, "test_dam_break", test_dam_break, devices=devices)
add_function_test(TestSolverSPH, "test_density_computation", test_density_computation, devices=devices)
add_function_test(TestSolverSPH, "test_no_particles", test_no_particles, devices=devices)


if __name__ == "__main__":
    unittest.main()
