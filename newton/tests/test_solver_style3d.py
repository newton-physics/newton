# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_constructor_precomputes_fixed_pd_matrix(test, device):
    builder = newton.ModelBuilder()
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_grid(
        builder,
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
        tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
        edge_aniso_ke=wp.vec3(2.0e-4, 1.0e-4, 5.0e-5),
    )
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverStyle3D(model, iterations=1, linear_iterations=1)

    test.assertGreater(float(solver.pd_diags.numpy().sum()), 0.0)
    test.assertGreater(int(solver.pd_non_diags.num_nz.numpy().sum()), 0)


def test_zero_mass_isolated_particle_remains_finite(test, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_mesh(
        builder,
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(1.0, 2.0, 3.0),
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 2.0, 0.0)),
        indices=(0, 1, 2),
        density=1.0,
    )
    model = builder.finalize(device=device)
    test.assertEqual(float(model.particle_mass.numpy()[3]), 0.0)
    test.assertTrue(int(model.particle_flags.numpy()[3]) & int(newton.ParticleFlags.ACTIVE))

    solver = newton.solvers.SolverStyle3D(model, iterations=1, linear_iterations=1)
    state_0 = model.state()
    state_1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    initial_position = state_0.particle_q.numpy()[3].copy()
    solver.step(state_0, state_1, model.control(), contacts, 0.01)

    positions = state_1.particle_q.numpy()
    velocities = state_1.particle_qd.numpy()
    test.assertTrue(np.isfinite(positions).all())
    test.assertTrue(np.isfinite(velocities).all())
    np.testing.assert_allclose(positions[3], initial_position)
    np.testing.assert_allclose(velocities[3], 0.0)


def test_solver_flags_deactivate_zero_mass_without_mutating_model(test, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_mesh(
        builder,
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 2.0, 0.0)),
        indices=(0, 1, 2),
        density=1.0,
    )
    builder.particle_flags[3] |= newton.ParticleFlags.PROXY
    model = builder.finalize(device=device)
    model_flags = model.particle_flags.numpy().copy()

    solver = newton.solvers.SolverStyle3D(model, iterations=1, linear_iterations=1)

    np.testing.assert_array_equal(model.particle_flags.numpy(), model_flags)
    solver_flags = solver._particle_flags.numpy()
    active = int(newton.ParticleFlags.ACTIVE)
    proxy = int(newton.ParticleFlags.PROXY)
    test.assertTrue(np.all((solver_flags[:3] & active) != 0))
    test.assertEqual(int(solver_flags[3]) & active, 0)
    test.assertNotEqual(int(solver_flags[3]) & proxy, 0)


def test_solver_flags_track_runtime_model_changes(test, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_grid(
        builder,
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverStyle3D(model, iterations=1, linear_iterations=1)

    masses = model.particle_mass.numpy().copy()
    masses[0] = 0.0
    model.particle_mass.assign(masses)
    model_flags = model.particle_flags.numpy().copy()
    model_flags[1] = 0
    model.particle_flags.assign(model_flags)

    state_0 = model.state()
    state_1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    solver.step(state_0, state_1, model.control(), contacts, 0.01)

    solver_flags = solver._particle_flags.numpy()
    active = int(newton.ParticleFlags.ACTIVE)
    test.assertEqual(int(solver_flags[0]), 0)
    test.assertEqual(int(solver_flags[1]), 0)
    test.assertNotEqual(int(solver_flags[2]) & active, 0)
    np.testing.assert_array_equal(model.particle_flags.numpy(), model_flags)


devices = get_test_devices()


class TestSolverStyle3D(unittest.TestCase):
    pass


add_function_test(
    TestSolverStyle3D,
    "test_constructor_precomputes_fixed_pd_matrix",
    test_constructor_precomputes_fixed_pd_matrix,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_zero_mass_isolated_particle_remains_finite",
    test_zero_mass_isolated_particle_remains_finite,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_solver_flags_deactivate_zero_mass_without_mutating_model",
    test_solver_flags_deactivate_zero_mass_without_mutating_model,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_solver_flags_track_runtime_model_changes",
    test_solver_flags_track_runtime_model_changes,
    devices=devices,
    check_output=False,
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
