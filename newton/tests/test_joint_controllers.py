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
        actuator_mode=newton.ActuatorMode.POSITION_VELOCITY,
    )
    builder.add_articulation([j])

    model = builder.finalize(device=device)

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


def test_ball_controller(
    test: TestJointController,
    device,
    solver_fn,
    pos_target_vals,
    vel_target_vals,
    expected_quat,
    expected_vel,
    target_ke,
    target_kd,
):
    """Test ball joint controller with position and velocity targets."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    box_mass = 1.0
    box_inertia = wp.mat33((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    # easy case: identity transform, zero center of mass
    b = builder.add_link(armature=0.0, I_m=box_inertia, mass=box_mass)
    builder.add_shape_box(body=b, hx=0.2, hy=0.2, hz=0.2, cfg=newton.ModelBuilder.ShapeConfig(density=1))

    # Create a ball joint with POSITION_VELOCITY actuator mode for PD control
    j = builder.add_joint_ball(
        parent=-1,
        child=b,
        parent_xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
        armature=0.0,
        actuator_mode=newton.ActuatorMode.POSITION_VELOCITY,
    )
    builder.add_articulation([j])

    test.assertEqual(builder.joint_count, 1)
    test.assertEqual(builder.joint_dof_count, 3)
    test.assertEqual(builder.joint_coord_count, 4)
    test.assertEqual(builder.joint_type[0], newton.JointType.BALL)
    test.assertEqual(builder.joint_parent[0], -1)
    test.assertEqual(builder.joint_child[0], b)
    test.assertEqual(builder.joint_armature[0], 0.0)
    test.assertEqual(builder.joint_friction[0], 0.0)

    # Set controller gains for the ball joint axes
    # Ball joints have 3 axes (X, Y, Z) that are added to joint_target_ke/kd arrays
    qd_start = builder.joint_qd_start[j]
    for i in range(3):  # 3 angular axes
        builder.joint_target_ke[qd_start + i] = target_ke
        builder.joint_target_kd[qd_start + i] = target_kd
        builder.joint_target_pos[qd_start + i] = pos_target_vals[i]
        builder.joint_target_vel[qd_start + i] = vel_target_vals[i]

    model = builder.finalize(device=device)

    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    control = model.control()
    control.joint_target_pos = wp.array(pos_target_vals, dtype=wp.float32, device=device)
    control.joint_target_vel = wp.array(vel_target_vals, dtype=wp.float32, device=device)

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

    # Ball joint has 4 position coordinates (quaternion) and 3 velocity coordinates
    if expected_quat is not None:
        # Check quaternion (allowing for sign flip since q and -q represent same rotation)
        # Compute dot product between actual and expected quaternions
        dot = abs(
            joint_q[0] * expected_quat[0]
            + joint_q[1] * expected_quat[1]
            + joint_q[2] * expected_quat[2]
            + joint_q[3] * expected_quat[3]
        )
        test.assertAlmostEqual(dot, 1.0, delta=1e-2)

    if expected_vel is not None:
        for i in range(3):
            test.assertAlmostEqual(joint_qd[i], expected_vel[i], delta=1e-2)


def test_effort_limit_clamping(
    test: TestJointController,
    device,
    solver_fn,
):
    """Test that MuJoCo solver correctly clamps actuator forces based on effort_limit."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)

    box_mass = 1.0
    inertia_value = 0.1
    box_inertia = wp.mat33((inertia_value, 0.0, 0.0), (0.0, inertia_value, 0.0), (0.0, 0.0, inertia_value))
    b = builder.add_link(armature=0.0, I_m=box_inertia, mass=box_mass)
    builder.add_shape_box(body=b, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))

    # High PD gains should be clamped by low effort_limit
    high_kp = 10000.0
    high_kd = 1000.0
    effort_limit = 5.0

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
        effort_limit=effort_limit,
        actuator_mode=newton.ActuatorMode.POSITION_VELOCITY,
    )
    builder.add_articulation([j])

    model = builder.finalize(device=device)
    model.ground = False
    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()

    initial_q = 1.0
    initial_qd = 0.0
    state_0.joint_q.assign([initial_q])
    state_0.joint_qd.assign([initial_qd])

    control = model.control()
    control.joint_target_pos = wp.array([0.0], dtype=wp.float32, device=device)
    control.joint_target_vel = wp.array([0.0], dtype=wp.float32, device=device)

    dt = 0.01

    F_unclamped = -high_kp * initial_q - high_kd * initial_qd
    F_clamped = np.clip(F_unclamped, -effort_limit, effort_limit)
    alpha = F_clamped / inertia_value
    qd_expected = initial_qd + alpha * dt
    q_expected = initial_q + qd_expected * dt

    solver.step(state_0, state_1, control, None, dt=dt)

    q_actual = state_1.joint_q.numpy()[0]
    qd_actual = state_1.joint_qd.numpy()[0]

    alpha_unclamped = F_unclamped / inertia_value
    qd_unclamped = initial_qd + alpha_unclamped * dt
    q_unclamped = initial_q + qd_unclamped * dt

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
        delta=tolerance * 10,
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

        # Revolute joint tests
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

        if solver_name == "mujoco_cpu" or solver_name == "mujoco_warp":
            # Ball joint tests
            # Test 1: Position control - rotation around Z axis
            add_function_test(
                TestJointController,
                f"test_ball_joint_controller_position_target_z_{solver_name}",
                test_ball_controller,
                devices=[device],
                solver_fn=solver_fn,
                pos_target_vals=[0.0, 0.0, wp.pi / 2.0],  # Rotate 90 degrees around Z
                vel_target_vals=[0.0, 0.0, 0.0],
                expected_quat=[0.0, 0.0, 0.7071068, 0.7071068],  # quat for 90 deg around Z
                expected_vel=[0.0, 0.0, 0.0],
                target_ke=2000.0,
                target_kd=500.0,
            )

            # Test 2: Position control - rotation around X axis
            add_function_test(
                TestJointController,
                f"test_ball_joint_controller_position_target_x_{solver_name}",
                test_ball_controller,
                devices=[device],
                solver_fn=solver_fn,
                pos_target_vals=[wp.pi / 2.0, 0.0, 0.0],  # Rotate 90 degrees around X
                vel_target_vals=[0.0, 0.0, 0.0],
                expected_quat=[0.7071068, 0.0, 0.0, 0.7071068],  # quat for 90 deg around X
                expected_vel=[0.0, 0.0, 0.0],
                target_ke=2000.0,
                target_kd=500.0,
            )

            # Test 3: Position control - rotation around Y axis
            add_function_test(
                TestJointController,
                f"test_ball_joint_controller_position_target_y_{solver_name}",
                test_ball_controller,
                devices=[device],
                solver_fn=solver_fn,
                pos_target_vals=[0.0, wp.pi / 2.0, 0.0],  # Rotate 90 degrees around Y
                vel_target_vals=[0.0, 0.0, 0.0],
                expected_quat=[0.0, 0.7071068, 0.0, 0.7071068],  # quat for 90 deg around Y
                expected_vel=[0.0, 0.0, 0.0],
                target_ke=2000.0,
                target_kd=500.0,
            )

            # Test 4: Velocity control - angular velocity around Z axis
            add_function_test(
                TestJointController,
                f"test_ball_joint_controller_velocity_target_z_{solver_name}",
                test_ball_controller,
                devices=[device],
                solver_fn=solver_fn,
                pos_target_vals=[0.0, 0.0, 0.0],
                vel_target_vals=[0.0, 0.0, wp.pi / 2.0],  # Angular velocity around Z
                expected_quat=None,  # Don't check position for velocity control
                expected_vel=[0.0, 0.0, wp.pi / 2.0],
                target_ke=0.0,
                target_kd=500.0,
            )

            # Test 5: Velocity control - angular velocity around X axis
            add_function_test(
                TestJointController,
                f"test_ball_joint_controller_velocity_target_x_{solver_name}",
                test_ball_controller,
                devices=[device],
                solver_fn=solver_fn,
                pos_target_vals=[0.0, 0.0, 0.0],
                vel_target_vals=[wp.pi / 2.0, 0.0, 0.0],  # Angular velocity around X
                expected_quat=None,
                expected_vel=[wp.pi / 2.0, 0.0, 0.0],
                target_ke=0.0,
                target_kd=500.0,
            )

            # Test 6: Velocity control - angular velocity around Y axis
            add_function_test(
                TestJointController,
                f"test_ball_joint_controller_velocity_target_y_{solver_name}",
                test_ball_controller,
                devices=[device],
                solver_fn=solver_fn,
                pos_target_vals=[0.0, 0.0, 0.0],
                vel_target_vals=[0.0, wp.pi / 2.0, 0.0],  # Angular velocity around Y
                expected_quat=None,
                expected_vel=[0.0, wp.pi / 2.0, 0.0],
                target_ke=0.0,
                target_kd=500.0,
            )

if __name__ == "__main__":
    unittest.main(verbosity=2)
