# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.style3d.kernels import (
    accumulate_dragging_pd_diag_kernel,
    prepare_jacobi_preconditioner_kernel,
)
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
    builder = newton.ModelBuilder(gravity=0.0)
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
    initial_position = state_0.particle_q.numpy()[3].copy()
    solver.step(state_0, state_1, model.control(), model.contacts(), 0.01)

    positions = state_1.particle_q.numpy()
    velocities = state_1.particle_qd.numpy()
    test.assertTrue(np.isfinite(positions).all())
    test.assertTrue(np.isfinite(velocities).all())
    np.testing.assert_allclose(positions[3], initial_position)
    np.testing.assert_allclose(velocities[3], 0.0)


def test_zero_mass_particle_excluded_from_optional_diagonals(test, device):
    active = int(newton.ParticleFlags.ACTIVE)
    masses = wp.array([0.0, 1.0, 1.0], dtype=float, device=device)
    flags = wp.array([active, active, active], dtype=wp.int32, device=device)

    dragging_diags = wp.zeros(3, dtype=float, device=device)
    drag_face = wp.array([0], dtype=int, device=device)
    drag_coords = wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3, device=device)
    faces = wp.array([[0, 1, 2]], dtype=wp.int32, device=device)
    wp.launch(
        accumulate_dragging_pd_diag_kernel,
        dim=1,
        inputs=[
            3.0,
            drag_face,
            drag_coords,
            faces,
            masses,
            flags,
        ],
        outputs=[dragging_diags],
        device=device,
    )
    np.testing.assert_allclose(dragging_diags.numpy(), [0.0, 3.0, 3.0])

    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    inverse_diags = wp.empty(3, dtype=wp.mat33, device=device)
    static_diags = wp.ones(3, dtype=float, device=device)
    contact_diags = wp.array([identity, identity, identity], dtype=wp.mat33, device=device)
    wp.launch(
        prepare_jacobi_preconditioner_kernel,
        dim=3,
        inputs=[
            static_diags,
            contact_diags,
            masses,
            flags,
        ],
        outputs=[inverse_diags],
        device=device,
    )
    expected = np.stack((np.eye(3), np.eye(3) * 0.5, np.eye(3) * 0.5))
    np.testing.assert_allclose(inverse_diags.numpy(), expected)


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
    "test_zero_mass_particle_excluded_from_optional_diagonals",
    test_zero_mass_particle_excluded_from_optional_diagonals,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
