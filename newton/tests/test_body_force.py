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

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestBodyForce(unittest.TestCase):
    pass


def test_floating_body(test: TestBodyForce, device, solver_fn, test_angular=True, up_axis=newton.Axis.Y):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=up_axis)

    # easy case: identity transform, zero center of mass
    pos = wp.vec3(1.0, 2.0, 3.0)
    rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi * 0.0)

    b = builder.add_body(xform=wp.transform(pos, rot))
    builder.add_shape_box(b, hx=0.25, hy=0.5, hz=1.0)
    builder.add_joint_free(b)
    builder.joint_q = [*pos, *rot]

    model = builder.finalize(device=device)

    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    input = np.zeros(model.body_count * 6, dtype=np.float32)

    if test_angular:
        test_index = 5
        test_value = 0.96
    else:
        test_index = 1
        test_value = 0.1

    input[test_index] = 1000.0
    state_0.body_f.assign(input)
    state_1.body_f.assign(input)

    sim_dt = 1.0 / 10.0
    for _ in range(1):
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0

    body_qd = state_0.body_qd.numpy()[0]
    test.assertAlmostEqual(body_qd[test_index], test_value, delta=1e-2)
    for i in range(1):
        if i == test_index:
            continue
        test.assertAlmostEqual(body_qd[i], 0.0, delta=1e-2)


def test_3d_articulation(test: TestBodyForce, device, solver_fn, test_angular, up_axis):
    # test mechanism with 3 orthogonally aligned prismatic joints
    # which allows to test all 3 dimensions of the control force independently
    builder = newton.ModelBuilder(gravity=0.0, up_axis=up_axis)
    builder.default_shape_cfg.density = 1000.0

    b = builder.add_body()
    builder.add_shape_box(b, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=0.25, hy=0.5, hz=1.0)
    builder.add_joint_d6(
        -1,
        b,
        linear_axes=[
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z),
        ],
        angular_axes=[
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z),
        ],
    )

    model = builder.finalize(device=device)
    test.assertEqual(model.joint_dof_count, 6)

    angular_values = [0.24, 0.282353, 0.96]
    for control_dim in range(3):
        solver = solver_fn(model)
        state_0, state_1 = model.state(), model.state()

        if test_angular:
            control_idx = control_dim + 3
            test_value = angular_values[control_dim]
        else:
            control_idx = control_dim
            test_value = 0.1

        input = np.zeros(model.body_count * 6, dtype=np.float32)
        input[control_idx] = 1000.0
        state_0.body_f.assign(input)
        state_1.body_f.assign(input)

        sim_dt = 1.0 / 10.0

        for _ in range(1):
            solver.step(state_0, state_1, None, None, sim_dt)
            state_0, state_1 = state_1, state_0

        if not isinstance(solver, newton.solvers.SolverMuJoCo | newton.solvers.SolverFeatherstone):
            # need to compute joint_qd from body_qd
            newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)

        body_qd = state_0.body_qd.numpy()[0]

        test.assertAlmostEqual(body_qd[control_idx], test_value, delta=1e-4)
        for i in range(6):
            if i == control_idx:
                continue
            test.assertAlmostEqual(body_qd[i], 0.0, delta=1e-2)


devices = get_test_devices()
solvers = {
    "featherstone": lambda model: newton.solvers.SolverFeatherstone(model, angular_damping=0.0),
    "mujoco_cpu": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True),
    "mujoco_warp": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True),
    "xpbd": lambda model: newton.solvers.SolverXPBD(model, angular_damping=0.0),
    "semi_implicit": lambda model: newton.solvers.SolverSemiImplicit(model, angular_damping=0.0),
}
for device in devices:
    for solver_name, solver_fn in solvers.items():
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue
        add_function_test(
            TestBodyForce,
            f"test_floating_body_linear_{solver_name}",
            test_floating_body,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=False,
        )
        add_function_test(
            TestBodyForce,
            f"test_floating_body_angular_up_axis_Y_{solver_name}",
            test_floating_body,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=True,
            up_axis=newton.Axis.Y,
        )
        add_function_test(
            TestBodyForce,
            f"test_floating_body_angular_up_axis_Z_{solver_name}",
            test_floating_body,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=True,
            up_axis=newton.Axis.Z,
        )
        add_function_test(
            TestBodyForce,
            f"test_floating_body_linear_up_axis_Y_{solver_name}",
            test_floating_body,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=False,
            up_axis=newton.Axis.Y,
        )
        add_function_test(
            TestBodyForce,
            f"test_floating_body_linear_up_axis_Z_{solver_name}",
            test_floating_body,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=False,
            up_axis=newton.Axis.Z,
        )

        # test 3d articulation
        add_function_test(
            TestBodyForce,
            f"test_3d_articulation_up_axis_Y_{solver_name}",
            test_3d_articulation,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=True,
            up_axis=newton.Axis.Y,
        )
        add_function_test(
            TestBodyForce,
            f"test_3d_articulation_up_axis_Z_{solver_name}",
            test_3d_articulation,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=True,
            up_axis=newton.Axis.Z,
        )
        add_function_test(
            TestBodyForce,
            f"test_3d_articulation_linear_up_axis_Y_{solver_name}",
            test_3d_articulation,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=False,
            up_axis=newton.Axis.Y,
        )
        add_function_test(
            TestBodyForce,
            f"test_3d_articulation_linear_up_axis_Z_{solver_name}",
            test_3d_articulation,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=False,
            up_axis=newton.Axis.Z,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
