# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest

import numpy as np
import warp as wp

import newton
from newton.math import quat_between_axes
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestQuatBetweenAxes(unittest.TestCase):
    def test_three_axis_sequence_applies_rotations_in_order(self):
        """Apply each pairwise rotation to a vector in sequence."""
        vector = wp.vec3(0.0, 0.0, 1.0)
        first = quat_between_axes("x", "y")
        second = quat_between_axes("y", "z")
        expected = wp.quat_rotate(second, wp.quat_rotate(first, vector))

        rotation = quat_between_axes("x", "y", "z")
        np.testing.assert_allclose(wp.quat_rotate(rotation, vector), expected, atol=1.0e-6)

    def test_three_axis_sequence_reaches_destination(self):
        """Rotate the first axis onto the last axis in a three-axis sequence."""
        axes = {
            "x": wp.vec3(1.0, 0.0, 0.0),
            "y": wp.vec3(0.0, 1.0, 0.0),
            "z": wp.vec3(0.0, 0.0, 1.0),
        }
        for source, middle, destination in itertools.permutations(axes):
            with self.subTest(axes=(source, middle, destination)):
                rotation = quat_between_axes(source, middle, destination)
                rotated = wp.quat_rotate(rotation, axes[source])
                np.testing.assert_allclose(rotated, axes[destination], atol=1.0e-6)


class TestControlForce(unittest.TestCase):
    pass


def test_gravity(test: TestControlForce, device, solver_fn, up_axis: newton.Axis):
    builder = newton.ModelBuilder(
        up_axis=up_axis, gravity=tuple(component * -9.81 for component in up_axis.to_vector())
    )

    b = builder.add_body()
    # Apply axis rotation to transform
    xform = wp.transform(wp.vec3(), quat_between_axes(newton.Axis.Z, up_axis))
    builder.add_shape_capsule(b, xform=xform)

    model = builder.finalize(device=device)

    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()
    control = model.control()

    sim_dt = 1.0 / 10.0
    solver.step(state_0, state_1, control, None, sim_dt)

    lin_vel = state_1.body_qd.numpy()[0, :3]
    test.assertAlmostEqual(lin_vel[up_axis.value], -0.981, delta=1e-5)


devices = get_test_devices()
solvers = {
    "featherstone": lambda model: newton.solvers.SolverFeatherstone(model, angular_damping=0.0),
    "mujoco_cpu": lambda model: newton.solvers.SolverMuJoCo(
        model, use_mujoco_cpu=True, update_data_interval=0, disable_contacts=True
    ),
    "mujoco_warp": lambda model: newton.solvers.SolverMuJoCo(
        model, use_mujoco_cpu=False, update_data_interval=0, disable_contacts=True
    ),
    "xpbd": lambda model: newton.solvers.SolverXPBD(model, angular_damping=0.0),
    "semi_implicit": lambda model: newton.solvers.SolverSemiImplicit(model, angular_damping=0.0),
    "kamino": newton.solvers.SolverKamino,
}
for device in devices:
    for solver_name, solver_fn in solvers.items():
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue
        add_function_test(
            TestControlForce,
            f"test_gravity_y_up_{solver_name}",
            test_gravity,
            devices=[device],
            solver_fn=solver_fn,
            up_axis=newton.Axis.Y,
        )
        add_function_test(
            TestControlForce,
            f"test_gravity_z_up_{solver_name}",
            test_gravity,
            devices=[device],
            solver_fn=solver_fn,
            up_axis=newton.Axis.Z,
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
