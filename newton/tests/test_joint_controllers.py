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

import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestJointController(unittest.TestCase):
    pass


def test_revolute_controller(test: TestJointController, device, solver_fn, joint_mode, target_value):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    box_mass = 1.0
    box_inertia = wp.mat33((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    # easy case: identity transform, zero center of mass
    b = builder.add_body(armature=0.0, I_m=box_inertia, mass=box_mass)
    builder.add_shape_box(body=b, hx=0.2, hy=0.2, hz=0.2, cfg=newton.ModelBuilder.ShapeConfig(density=1))
    # Create a revolute joint
    builder.add_joint_revolute(
        parent=-1,
        child=b,
        parent_xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 0.0, 1.0),
        target=target_value,
        armature=0.0,
        mode=joint_mode,
        # limit_lower=-wp.pi,
        # limit_upper=wp.pi,
        limit_ke=0.0,
        limit_kd=0.0,
        target_ke=2000.0,
        target_kd=500.0,
    )

    model = builder.finalize(device=device)
    model.ground = False

    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    control = model.control()
    control.joint_target = wp.array([target_value], dtype=wp.float32, device=device)

    sim_dt = 1.0 / 60.0
    sim_time = 0.0
    for _ in range(100):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, sim_dt)
        state_0, state_1 = state_1, state_0

        sim_time += sim_dt

    if not isinstance(solver, newton.solvers.SolverMuJoCo | newton.solvers.SolverFeatherstone):
        newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)

    joint_q = state_0.joint_q.numpy()
    joint_qd = state_0.joint_qd.numpy()
    if joint_mode == newton.JointMode.TARGET_POSITION:
        test.assertAlmostEqual(joint_q[0], target_value, delta=1e-2)
        test.assertAlmostEqual(joint_qd[0], 0.0, delta=1e-2)
    elif joint_mode == newton.JointMode.TARGET_VELOCITY:
        test.assertAlmostEqual(joint_qd[0], target_value, delta=1e-2)


devices = get_test_devices()
solvers = {
    # "featherstone": lambda model: newton.solvers.SolverFeatherstone(model, angular_damping=0.0),
    "mujoco_cpu": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True),
    "mujoco_warp": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True),
    "xpbd": lambda model: newton.solvers.SolverXPBD(model, angular_damping=0.0, iterations=5),
    # "semi_implicit": lambda model: newton.solvers.SolverSemiImplicit(model, angular_damping=0.0),
}
for device in devices:
    for solver_name, solver_fn in solvers.items():
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue
        # add_function_test(TestJointController, f"test_floating_body_linear_{solver_name}", test_floating_body, devices=[device], solver_fn=solver_fn, test_angular=False)
        add_function_test(
            TestJointController,
            f"test_revolute_joint_controller_position_target_{solver_name}",
            test_revolute_controller,
            devices=[device],
            solver_fn=solver_fn,
            joint_mode=newton.JointMode.TARGET_POSITION,
            target_value=wp.pi / 2.0,
        )
        # TODO: XPBD velocity control is not working correctly
        if solver_name == "mujoco_warp" or solver_name == "mujoco_cpu":
            add_function_test(
                TestJointController,
                f"test_revolute_joint_controller_velocity_target_{solver_name}",
                test_revolute_controller,
                devices=[device],
                solver_fn=solver_fn,
                joint_mode=newton.JointMode.TARGET_VELOCITY,
                target_value=wp.pi / 2.0,
            )

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
