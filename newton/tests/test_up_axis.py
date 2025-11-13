# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import warp as wp

import newton
from newton._src.core import quat_between_axes
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestControlForce(unittest.TestCase):
    pass


def test_gravity(test: TestControlForce, device, solver_fn, up_axis: newton.Axis):
    builder = newton.ModelBuilder(up_axis=up_axis, gravity=-9.81)

    b = builder.add_body()
    # Apply axis rotation to transform
    xform = wp.transform(wp.vec3(), quat_between_axes(newton.Axis.Z, up_axis))
    builder.add_shape_capsule(b, xform=xform)
    builder.add_joint_free(b)

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
