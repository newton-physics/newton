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

wp.config.quiet = True


class TestBodyForce(unittest.TestCase):
    pass


def test_floating_body(test: TestBodyForce, device, solver_fn, test_angular=True):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)

    # easy case: identity transform, zero center of mass
    pos = wp.vec3(1.0, 2.0, 3.0) 
    rot = wp.quat_rpy(-1.3, 0.8, 2.4)
    b = builder.add_body(xform= wp.transform(pos, rot))
    builder.add_shape_box(b) # density = 1000.0, mass = 1000.0. Ixx = 1000/6 *
    builder.add_joint_free(b)
    builder.joint_q = [*pos, *rot]

    model = builder.finalize(device=device)
    model.ground = False

    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()

    newton.core.articulation.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    input = np.zeros(model.body_count * 6, dtype=np.float32)
    if test_angular:
        test_index = 2
        test_value = 2.4
    else:
        test_index = 4
        test_value = 0.4

    input[test_index] = 1000.0
    state_0.body_f.assign(input)
    state_1.body_f.assign(input)
    
    sim_dt = 1.0 / 10.0
    # F = m * a, a = 1.0, dt = 0.4 -> V = 0.4
    # T = I * alpha, alpha_ii = 6.0, dt = 0.4 -> W = 2.4
    for _ in range(4):
        solver.step(model, state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0

    body_qd = state_0.body_qd.numpy()[0]
    test.assertAlmostEqual(body_qd[test_index], test_value, delta=1e-2)
    for i in range(6):
        if i == test_index:
            continue
        test.assertAlmostEqual(body_qd[i], 0.0, delta=1e-2)


def test_3d_articulation(test: TestBodyForce, device, solver_fn):
    # test mechanism with 3 orthogonally aligned prismatic joints
    # which allows to test all 3 dimensions of the control force independently
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_shape_cfg.density = 1000.0

    b = builder.add_body()
    builder.add_shape_box(b)
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
    model.ground = False

    test.assertEqual(model.joint_dof_count, 3)

    for control_dim in range(3):
        solver = solver_fn(model)

        state_0, state_1 = model.state(), model.state()

        input = np.zeros(model.body_count * 6, dtype=np.float32)
        input[control_dim+3] = 1000.0
        state_0.body_f.assign(input)
        state_1.body_f.assign(input)

        sim_dt = 1.0 / 10.0

        for _ in range(4):
            solver.step(model, state_0, state_1, None, None, sim_dt)
            state_0, state_1 = state_1, state_0

        if not isinstance(solver, (newton.solvers.MuJoCoSolver, newton.solvers.FeatherstoneSolver)):
            # need to compute joint_qd from body_qd
            newton.core.articulation.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)

        body_qd = state_0.body_qd.numpy()[0]
        test.assertAlmostEqual(body_qd[control_dim+3], 0.4, delta=1e-4)
        for i in range(6):
            if i == control_dim+3:
                continue
            test.assertAlmostEqual(body_qd[i], 0.0, delta=1e-2)


devices = get_test_devices()
solvers = {
    # "featherstone": lambda model: newton.solvers.FeatherstoneSolver(model, angular_damping=0.0),
    "mujoco_c": lambda model: newton.solvers.MuJoCoSolver(model, disable_contacts=True),
    "mujoco_warp": lambda model: newton.solvers.MuJoCoSolver(model, use_mujoco=False, disable_contacts=True),
    "xpbd": lambda model: newton.solvers.XPBDSolver(model, angular_damping=0.0),
    "semi_implicit": lambda model: newton.solvers.SemiImplicitSolver(model, angular_damping=0.0),
}
for device in ["cpu"]:
    for solver_name, solver_fn in solvers.items():
        # add_function_test(TestBodyForce, f"test_floating_body_linear_{solver_name}", test_floating_body, devices=[device], solver_fn=solver_fn, test_angular=False)
        add_function_test(
            TestBodyForce,
            f"test_floating_body_angular_{solver_name}",
            test_floating_body,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=True,
        )
        add_function_test(
            TestBodyForce,
            f"test_floating_body_linear_{solver_name}",
            test_floating_body,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=False,
        )
        # test 3d articulation
        add_function_test(
            TestBodyForce,
            f"test_3d_articulation_{solver_name}",
            test_3d_articulation,
            devices=[device],
            solver_fn=solver_fn,
        )



if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
