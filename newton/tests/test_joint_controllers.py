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


class TestJointController(unittest.TestCase):
    pass


def test_revolute_controller(
    test: TestJointController,
    device,
    solver_fn,
    pos_target_val,
    vel_target_val,
    expected_pos,
    expected_vel,
    target_ke,
    target_kd,
):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    box_mass = 1.0
    box_inertia = wp.mat33((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    # easy case: identity transform, zero center of mass
    b = builder.add_link(armature=0.0, I_m=box_inertia, mass=box_mass)
    builder.add_shape_box(body=b, hx=0.2, hy=0.2, hz=0.2, cfg=newton.ModelBuilder.ShapeConfig(density=1))

    # Create a revolute joint
    j = builder.add_joint_revolute(
        parent=-1,
        child=b,
        parent_xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 0.0, 1.0),
        target_pos=pos_target_val,
        target_vel=vel_target_val,
        armature=0.0,
        # limit_lower=-wp.pi,
        # limit_upper=wp.pi,
        limit_ke=0.0,
        limit_kd=0.0,
        target_ke=target_ke,
        target_kd=target_kd,
    )
    builder.add_articulation([j])

    model = builder.finalize(device=device)
    model.ground = False

    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    control = model.control()
    control.joint_target_pos = wp.array([pos_target_val], dtype=wp.float32, device=device)
    control.joint_target_vel = wp.array([vel_target_val], dtype=wp.float32, device=device)

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
    if expected_pos is not None:
        test.assertAlmostEqual(joint_q[0], expected_pos, delta=1e-2)
    if expected_vel is not None:
        test.assertAlmostEqual(joint_qd[0], expected_vel, delta=1e-2)


def test_effort_limit_clamping(
    test: TestJointController,
    device,
    solver_fn,
):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)

    # Simple pendulum with known inertia for analytical calculations
    box_mass = 1.0
    inertia_value = 0.1  # Small inertia for easier calculations
    box_inertia = wp.mat33((inertia_value, 0.0, 0.0), (0.0, inertia_value, 0.0), (0.0, 0.0, inertia_value))
    b = builder.add_link(armature=0.0, I_m=box_inertia, mass=box_mass)
    builder.add_shape_box(body=b, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))

    # Very high PD gains that would generate large forces
    high_kp = 10000.0
    high_kd = 1000.0

    # Low effort limit that should clamp the forces
    effort_limit = 5.0

    # Create a revolute joint with high gains but low effort limit
    j = builder.add_joint_revolute(
        parent=-1,
        child=b,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 0.0, 1.0),
        target_pos=0.0,
        target_vel=0.0,
        armature=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        target_ke=high_kp,
        target_kd=high_kd,
        effort_limit=effort_limit,  # This should clamp the total P+D force
    )
    builder.add_articulation([j])

    model = builder.finalize(device=device)
    model.ground = False

    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()

    # Set initial position far from target to generate large error
    # This should create forces that would exceed the effort limit without clamping
    initial_q = 1.0  # radians
    initial_qd = 0.0
    state_0.joint_q.assign([initial_q])
    state_0.joint_qd.assign([initial_qd])
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    control = model.control()
    control.joint_target_pos = wp.array([0.0], dtype=wp.float32, device=device)
    control.joint_target_vel = wp.array([0.0], dtype=wp.float32, device=device)

    dt = 0.01

    # Analytical calculation of expected motion with clamped force
    F_unclamped = -high_kp * initial_q - high_kd * initial_qd
    F_clamped = np.clip(F_unclamped, -effort_limit, effort_limit)

    alpha = F_clamped / inertia_value
    qd_expected = initial_qd + alpha * dt
    q_expected = initial_q + qd_expected * dt

    # Step the solver
    solver.step(state_0, state_1, control, None, dt=dt)

    if not isinstance(solver, newton.solvers.SolverMuJoCo | newton.solvers.SolverFeatherstone):
        newton.eval_ik(model, state_1, state_1.joint_q, state_1.joint_qd)

    q_actual = state_1.joint_q.numpy()[0]
    qd_actual = state_1.joint_qd.numpy()[0]

    # The actual position should be much closer to q_expected (with clamping)
    # than it would be if unlimited force were applied

    # Calculate what would happen without clamping (for comparison)
    alpha_unclamped = F_unclamped / inertia_value
    qd_unclamped = initial_qd + alpha_unclamped * dt
    q_unclamped = initial_q + qd_unclamped * dt

    # Verify that clamping had a significant effect
    test.assertGreater(abs(q_unclamped - q_expected), 0.5, "Clamping should significantly affect the motion")

    tolerance = 0.05
    test.assertAlmostEqual(
        q_actual,
        q_expected,
        delta=tolerance,
        msg=f"Position with clamped effort limit: expected {q_expected:.4f}, got {q_actual:.4f}",
    )
    test.assertAlmostEqual(
        qd_actual,
        qd_expected,
        delta=tolerance * 10,  # Velocity tolerance can be larger
        msg=f"Velocity with clamped effort limit: expected {qd_expected:.4f}, got {qd_actual:.4f}",
    )


devices = get_test_devices()
solvers = {
    "featherstone": lambda model: newton.solvers.SolverFeatherstone(model, angular_damping=0.0),
    "mujoco_cpu": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True),
    "mujoco_warp": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True),
    "xpbd": lambda model: newton.solvers.SolverXPBD(model, angular_damping=0.0, iterations=5),
    # "semi_implicit": lambda model: newton.solvers.SolverSemiImplicit(model, angular_damping=0.0),
}
for device in devices:
    for solver_name, solver_fn in solvers.items():
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue

        if "mujoco" in solver_name:
            add_function_test(
                TestJointController,
                f"test_effort_limit_clamping_{solver_name}",
                test_effort_limit_clamping,
                devices=[device],
                solver_fn=solver_fn,
            )

        # add_function_test(TestJointController, f"test_floating_body_linear_{solver_name}", test_floating_body, devices=[device], solver_fn=solver_fn, test_angular=False)
        add_function_test(
            TestJointController,
            f"test_revolute_joint_controller_position_target_{solver_name}",
            test_revolute_controller,
            devices=[device],
            solver_fn=solver_fn,
            pos_target_val=wp.pi / 2.0,
            vel_target_val=0.0,
            expected_pos=wp.pi / 2.0,
            expected_vel=0.0,
            target_ke=2000.0,
            target_kd=500.0,
        )
        # TODO: XPBD velocity control is not working correctly
        if solver_name != "xpbd":
            add_function_test(
                TestJointController,
                f"test_revolute_joint_controller_velocity_target_{solver_name}",
                test_revolute_controller,
                devices=[device],
                solver_fn=solver_fn,
                pos_target_val=0.0,
                vel_target_val=wp.pi / 2.0,
                expected_pos=None,
                expected_vel=wp.pi / 2.0,
                target_ke=0.0,
                target_kd=500.0,
            )

if __name__ == "__main__":
    unittest.main(verbosity=2)
