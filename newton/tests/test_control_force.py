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

# TODO:
# - Fix Featherstone solver for floating body
# - Fix linear force application to floating body for SolverMuJoCo

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestControlForce(unittest.TestCase):
    pass


def test_floating_body(test: TestControlForce, device, solver_fn, test_angular=True):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)

    # easy case: identity transform, zero center of mass
    b = builder.add_body()
    builder.add_shape_box(b)
    builder.add_joint_free(b)
    builder.joint_q = [1.0, 2.0, 3.0, *wp.quat_rpy(-1.3, 0.8, 2.4)]

    model = builder.finalize(device=device)

    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    control = model.control()
    if test_angular:
        control.joint_f.assign(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 100.0], dtype=np.float32))
        test_index = 5
    else:
        control.joint_f.assign(np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        test_index = 1

    sim_dt = 1.0 / 10.0

    for _ in range(4):
        solver.step(state_0, state_1, control, None, sim_dt)
        state_0, state_1 = state_1, state_0

    body_qd = state_0.body_qd.numpy()[0]
    test.assertGreater(body_qd[test_index], 0.04)
    test.assertLess(body_qd[test_index], 0.4)
    for i in range(6):
        if i == test_index:
            continue
        test.assertAlmostEqual(body_qd[i], 0.0, delta=1e-6)
    # TODO test joint_qd for MJC, Featherstone solvers


def test_3d_articulation(test: TestControlForce, device, solver_fn):
    # test mechanism with 3 orthogonally aligned prismatic joints
    # which allows to test all 3 dimensions of the control force independently
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_shape_cfg.density = 100.0

    b = builder.add_body()
    builder.add_shape_sphere(b)
    builder.add_joint_d6(
        -1,
        b,
        linear_axes=[
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, armature=0.0),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y, armature=0.0),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z, armature=0.0),
        ],
    )

    model = builder.finalize(device=device)

    test.assertEqual(model.joint_dof_count, 3)

    for control_dim in range(3):
        solver = solver_fn(model)

        state_0, state_1 = model.state(), model.state()

        control = model.control()
        control_input = np.zeros(model.joint_dof_count, dtype=np.float32)
        control_input[control_dim] = 100.0
        control.joint_f.assign(control_input)

        sim_dt = 1.0 / 10.0

        for _ in range(4):
            solver.step(state_0, state_1, control, None, sim_dt)
            state_0, state_1 = state_1, state_0

        if not isinstance(solver, newton.solvers.SolverMuJoCo | newton.solvers.SolverFeatherstone):
            # need to compute joint_qd from body_qd
            newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)

        qd = state_0.joint_qd.numpy()
        test.assertGreater(qd[control_dim], 0.009)
        test.assertLess(qd[control_dim], 0.4)
        for i in range(model.joint_dof_count):
            if i == control_dim:
                continue
            test.assertAlmostEqual(qd[i], 0.0, delta=1e-6)


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
        # add_function_test(TestControlForce, f"test_floating_body_linear_{solver_name}", test_floating_body, devices=[device], solver_fn=solver_fn, test_angular=False)
        add_function_test(
            TestControlForce,
            f"test_floating_body_angular_{solver_name}",
            test_floating_body,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=True,
        )
        add_function_test(
            TestControlForce,
            f"test_3d_articulation_{solver_name}",
            test_3d_articulation,
            devices=[device],
            solver_fn=solver_fn,
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
