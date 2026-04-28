# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

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


devices = get_test_devices(mode="basic")


class TestSolverStyle3D(unittest.TestCase):
    pass


add_function_test(
    TestSolverStyle3D,
    "test_constructor_precomputes_fixed_pd_matrix",
    test_constructor_precomputes_fixed_pd_matrix,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
