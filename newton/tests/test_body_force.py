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

"""
Tests for body force/torque application.

This module includes tests for:
1. Basic force/torque application on floating bodies and articulations
2. Force/torque behavior with non-zero center of mass (CoM) offsets

For non-zero CoM tests:
- When a force is applied (which acts at the CoM), the body should accelerate
  linearly without rotation.
- When a combined force and torque is applied, the body should accelerate linearly and rotate about its CoM
  according to the applied force and torque.

Note: Featherstone solver is excluded from non-zero CoM tests due to known issues
with CoM offset handling in the algorithm.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.viewer.kernels import compute_com_positions
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestBodyForce(unittest.TestCase):
    pass


def compute_com_world_position(body_q, body_com, body_world, world_offsets=None, body_index: int = 0) -> np.ndarray:
    """Compute the center of mass position in world frame."""
    com_world = wp.zeros(body_q.shape[0], dtype=wp.vec3, device=body_q.device)
    wp.launch(
        kernel=compute_com_positions,
        dim=body_q.shape[0],
        inputs=[body_q, body_com, body_world, world_offsets],
        outputs=[com_world],
        device=body_q.device,
    )
    return com_world.numpy()[body_index]


def test_floating_body(test: TestBodyForce, device, solver_fn, test_angular=True, up_axis=newton.Axis.Y):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=up_axis)

    # easy case: identity transform, zero center of mass
    pos = wp.vec3(1.0, 2.0, 3.0)
    rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi * 0.0)

    body_index = builder.add_body(xform=wp.transform(pos, rot))
    builder.add_shape_box(body_index, hx=0.25, hy=0.5, hz=1.0)
    builder.joint_q = [*pos, *rot]

    model = builder.finalize(device=device)

    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    input = np.zeros(model.body_count * 6, dtype=np.float32)
    
    sim_dt = 1.0 / 10.0
    test_force_torque = 1000.0
    relative_tolerance = 5e-2  # for testing expected velocity
    zero_velocity_tolerance = 1e-3  # for testing zero velocities

    if test_angular:
        test_index = 5  # torque about z-axis
        inertia_zz = model.body_inertia.numpy()[body_index][2, 2]
        expected_velocity = test_force_torque / inertia_zz * sim_dt
    else:
        test_index = 1  # force in y-direction
        mass = model.body_mass.numpy()[body_index]
        expected_velocity = test_force_torque / mass * sim_dt

    input[test_index] = test_force_torque
    state_0.body_f.assign(input)
    state_1.body_f.assign(input)

    for _ in range(1):
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0

    body_qd = state_0.body_qd.numpy()[body_index]
    abs_tol_expected_velocity = relative_tolerance * abs(expected_velocity)
    test.assertAlmostEqual(body_qd[test_index], expected_velocity, delta=abs_tol_expected_velocity)
    for i in range(6):
        if i == test_index:
            continue
        test.assertAlmostEqual(body_qd[i], 0.0, delta=zero_velocity_tolerance)


def test_floating_body_control_joint_f(
    test: TestBodyForce,
    device,
    solver_fn,
    test_angular=True,
    up_axis=newton.Axis.Y,
):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=up_axis)

    # easy case: identity transform, zero center of mass
    pos = wp.vec3(1.0, 2.0, 3.0)
    rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi * 0.0)

    body_index = builder.add_body(xform=wp.transform(pos, rot))
    builder.add_shape_box(body_index, hx=0.25, hy=0.5, hz=1.0)
    builder.joint_q = [*pos, *rot]

    model = builder.finalize(device=device)

    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    control = model.control()
    input = np.zeros(model.joint_dof_count, dtype=np.float32)

    sim_dt = 1.0 / 10.0
    test_force_torque = 1000.0
    relative_tolerance = 5e-2  # for testing expected velocity
    zero_velocity_tolerance = 1e-3  # for testing zero velocities

    if test_angular:
        test_index = 5  # torque about z-axis
        inertia_zz = model.body_inertia.numpy()[body_index][2, 2]
        expected_velocity = test_force_torque / inertia_zz * sim_dt
    else:
        test_index = 1  # force in y-direction
        mass = model.body_mass.numpy()[body_index]
        expected_velocity = test_force_torque / mass * sim_dt

    input[test_index] = test_force_torque
    control.joint_f.assign(input)

    for _ in range(1):
        solver.step(state_0, state_1, control, None, sim_dt)
        state_0, state_1 = state_1, state_0

    body_qd = state_0.body_qd.numpy()[body_index]
    abs_tol_expected_velocity = relative_tolerance * abs(expected_velocity)
    test.assertAlmostEqual(body_qd[test_index], expected_velocity, delta=abs_tol_expected_velocity)
    for i in range(6):
        if i == test_index:
            continue
        test.assertAlmostEqual(body_qd[i], 0.0, delta=zero_velocity_tolerance)


def test_3d_articulation(test: TestBodyForce, device, solver_fn, test_angular, up_axis):
    # test mechanism with 3 orthogonally aligned prismatic joints
    # which allows to test all 3 dimensions of the control force independently
    builder = newton.ModelBuilder(gravity=0.0, up_axis=up_axis)
    builder.default_shape_cfg.density = 1000.0

    b = builder.add_link()
    builder.add_shape_box(b, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=0.25, hy=0.5, hz=1.0)
    j = builder.add_joint_d6(
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
    builder.add_articulation([j])

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
        if solver_name == "featherstone":
            continue
        add_function_test(
            TestBodyForce,
            f"test_floating_body_joint_f_linear_{solver_name}",
            test_floating_body_control_joint_f,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=False,
        )
        add_function_test(
            TestBodyForce,
            f"test_floating_body_joint_f_angular_up_axis_Y_{solver_name}",
            test_floating_body_control_joint_f,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=True,
            up_axis=newton.Axis.Y,
        )
        add_function_test(
            TestBodyForce,
            f"test_floating_body_joint_f_angular_up_axis_Z_{solver_name}",
            test_floating_body_control_joint_f,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=True,
            up_axis=newton.Axis.Z,
        )
        add_function_test(
            TestBodyForce,
            f"test_floating_body_joint_f_linear_up_axis_Y_{solver_name}",
            test_floating_body_control_joint_f,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=False,
            up_axis=newton.Axis.Y,
        )
        add_function_test(
            TestBodyForce,
            f"test_floating_body_joint_f_linear_up_axis_Z_{solver_name}",
            test_floating_body_control_joint_f,
            devices=[device],
            solver_fn=solver_fn,
            test_angular=False,
            up_axis=newton.Axis.Z,
        )


# =============================================================================
# Non-zero Center of Mass Tests
# =============================================================================
#
# These tests verify that forces and torques are correctly applied when the body
# has a non-zero center of mass offset.


def test_force_no_rotation(
    test: TestBodyForce,
    device,
    solver_fn,
    com_offset: tuple[float, float, float],
    force_direction: tuple[float, float, float],
):
    """Test that a force applied at the CoM causes linear acceleration without rotation.

    When a body has a non-zero CoM offset and we apply a pure force (no torque),
    the force acts at the CoM, so the body should accelerate linearly without
    rotating.

    Args:
        test: Test case instance
        device: Compute device
        solver_fn: Function that creates a solver given a model
        com_offset: Center of mass offset in body frame (x, y, z)
        force_direction: Direction of applied force (fx, fy, fz)
        tolerance: Maximum allowed rotation (quaternion dot product threshold)
    """
    builder = newton.ModelBuilder(gravity=0.0)

    initial_pos = wp.vec3(0.0, 0.0, 1.0)
    body_index = builder.add_body(xform=wp.transform(initial_pos, wp.quat_identity()))
    builder.add_shape_box(body_index, hx=0.1, hy=0.1, hz=0.1)
    builder.body_com[body_index] = wp.vec3(*com_offset)

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Apply pure force (no torque)
    force_magnitude = 10.0
    body_f = np.array(
        [
            force_direction[0] * force_magnitude,
            force_direction[1] * force_magnitude,
            force_direction[2] * force_magnitude,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    state_0.body_f.assign(body_f)
    state_1.body_f.assign(body_f)

    # Step simulation
    sim_dt = 0.01
    num_steps = 5

    mass = model.body_mass.numpy()[body_index]
    expected_velocity = force_magnitude / mass * sim_dt * num_steps
    abs_tol_expected_velocity = 5e-2 * abs(expected_velocity)
    abs_tol_zero_velocity = 1e-3 # for testing zero velocities

    for _ in range(num_steps):
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0
        # Re-apply force for next step
        state_0.body_f.assign(body_f)
        state_1.body_f.assign(body_f)

    # Body rotation should NOT have accelerated - expect zero velocity for angular components
    body_qd = state_0.body_qd.numpy()[body_index]
    test.assertAlmostEqual(body_qd[3], 0.0, delta=abs_tol_zero_velocity)
    test.assertAlmostEqual(body_qd[4], 0.0, delta=abs_tol_zero_velocity)
    test.assertAlmostEqual(body_qd[5], 0.0, delta=abs_tol_zero_velocity)

    # project linear velocity onto force direction and test against expected velocity
    force_dir = np.array(force_direction, dtype=np.float32)
    force_dir_norm = np.linalg.norm(force_dir)
    test.assertAlmostEqual(force_dir_norm, 1.0, delta=1e-6)
    linear_velocity = body_qd[:3]
    projected_velocity = float(np.dot(force_dir, linear_velocity))
    test.assertAlmostEqual(projected_velocity, expected_velocity, delta=abs_tol_expected_velocity)


def test_combined_force_torque(
    test: TestBodyForce,
    device,
    solver_fn,
    com_offset: tuple[float, float, float],
    use_control: bool = False,
):
    """Test combined force and torque with non-zero CoM offset.

    When both force and torque are applied, the CoM should translate according
    to the force while the body rotates due to the torque.

    Args:
        test: Test case instance
        device: Compute device
        solver_fn: Function that creates a solver given a model
        com_offset: Center of mass offset in body frame (x, y, z)
        use_control: Apply forces via control.joint_f instead of state.body_f
    """
    builder = newton.ModelBuilder(gravity=0.0)

    initial_pos = wp.vec3(0.0, 0.0, 1.0)
    body_index = builder.add_body(xform=wp.transform(initial_pos, wp.quat_identity()))
    builder.add_shape_box(body_index, hx=0.1, hy=0.1, hz=0.1)
    builder.body_com[body_index] = wp.vec3(*com_offset)

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control() if use_control else None

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Apply both force and torque
    force_magnitude = 10.0
    torque_magnitude = 10.0
    wrench = np.array(
        [force_magnitude, 0.0, 0.0, 0.0, 0.0, torque_magnitude],  # Force in X, torque about Z
        dtype=np.float32,
    )
    if use_control:
        control.joint_f.assign(wrench)
    else:
        state_0.body_f.assign(wrench)
        state_1.body_f.assign(wrench)

    # Step simulation
    sim_dt = 0.01
    num_steps = 10
    mass = model.body_mass.numpy()[body_index]
    expected_velocity = force_magnitude / mass * sim_dt * num_steps
    abs_tol_expected_velocity = 5e-2 * abs(expected_velocity)

    expected_angular_velocity = torque_magnitude / model.body_inertia.numpy()[body_index][2, 2] * sim_dt * num_steps
    abs_tol_expected_angular_velocity = 5e-2 * abs(expected_angular_velocity)

    abs_tol_zero_velocities = 1e-3  # for testing zero velocities

    for _ in range(num_steps):
        solver.step(state_0, state_1, control, None, sim_dt)
        state_0, state_1 = state_1, state_0
        # Re-apply force for next step
        if not use_control:
            state_0.body_f.assign(wrench)
            state_1.body_f.assign(wrench)

    # Get final body twist
    body_qd = state_0.body_qd.numpy()[body_index]

    linear_velocity = body_qd[:3]
    test.assertAlmostEqual(linear_velocity[0], expected_velocity, delta=abs_tol_expected_velocity)
    test.assertAlmostEqual(linear_velocity[1], 0.0, delta=abs_tol_zero_velocities)
    test.assertAlmostEqual(linear_velocity[2], 0.0, delta=abs_tol_zero_velocities)

    # Test angular velocity
    angular_velocity = body_qd[3:6]
    test.assertAlmostEqual(angular_velocity[0], 0.0, delta=abs_tol_zero_velocities)
    test.assertAlmostEqual(angular_velocity[1], 0.0, delta=abs_tol_zero_velocities)
    test.assertAlmostEqual(angular_velocity[2], expected_angular_velocity, delta=abs_tol_expected_angular_velocity)

def test_force_no_rotation_control_joint_f(
    test: TestBodyForce,
    device,
    solver_fn,
    com_offset: tuple[float, float, float],
    force_direction: tuple[float, float, float],
):
    """Control.joint_f version of test_force_no_rotation."""
    builder = newton.ModelBuilder(gravity=0.0)

    initial_pos = wp.vec3(0.0, 0.0, 1.0)
    body_index = builder.add_body(xform=wp.transform(initial_pos, wp.quat_identity()))
    builder.add_shape_box(body_index, hx=0.1, hy=0.1, hz=0.1)
    builder.body_com[body_index] = wp.vec3(*com_offset)

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    force_magnitude = 10.0
    joint_f = np.array(
        [
            force_direction[0] * force_magnitude,
            force_direction[1] * force_magnitude,
            force_direction[2] * force_magnitude,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    control.joint_f.assign(joint_f)

    sim_dt = 0.01
    num_steps = 5

    mass = model.body_mass.numpy()[body_index]
    expected_velocity = force_magnitude / mass * sim_dt * num_steps
    abs_tol_expected_velocity = 5e-2 * abs(expected_velocity)
    abs_tol_zero_velocity = 1e-3  # for testing zero velocities

    for _ in range(num_steps):
        solver.step(state_0, state_1, control, None, sim_dt)
        state_0, state_1 = state_1, state_0

    # Body rotation should NOT have accelerated - expect zero velocity for angular components
    body_qd = state_0.body_qd.numpy()[body_index]
    test.assertAlmostEqual(body_qd[3], 0.0, delta=abs_tol_zero_velocity)
    test.assertAlmostEqual(body_qd[4], 0.0, delta=abs_tol_zero_velocity)
    test.assertAlmostEqual(body_qd[5], 0.0, delta=abs_tol_zero_velocity)

    # project linear velocity onto force direction and test against expected velocity
    force_dir = np.array(force_direction, dtype=np.float32)
    force_dir_norm = np.linalg.norm(force_dir)
    test.assertAlmostEqual(force_dir_norm, 1.0, delta=1e-6)
    linear_velocity = body_qd[:3]
    projected_velocity = float(np.dot(force_dir, linear_velocity))
    test.assertAlmostEqual(projected_velocity, expected_velocity, delta=abs_tol_expected_velocity)


# Solvers for non-zero CoM tests
# Tuple format: (solver_fn, tolerance, supports_torque_com_tests)
com_solvers = {
    "mujoco_cpu": (
        # Use RK4 integrator to reduce numerical drift
        lambda model: newton.solvers.SolverMuJoCo(model, integrator="rk4", use_mujoco_cpu=True, disable_contacts=True),
        1e-3,
        True,
    ),
    "mujoco_warp": (
        # Use RK4 integrator to reduce numerical drift
        lambda model: newton.solvers.SolverMuJoCo(model, integrator="rk4", use_mujoco_cpu=False, disable_contacts=True),
        1e-3,
        True,
    ),
    "xpbd": (
        lambda model: newton.solvers.SolverXPBD(model, angular_damping=0.0),
        1e-3,
        True,
    ),
    "semi_implicit": (
        lambda model: newton.solvers.SolverSemiImplicit(model, angular_damping=0.0),
        1e-3,
        True,
    ),
    "featherstone": (
        lambda model: newton.solvers.SolverFeatherstone(model),
        1e-3,
        False,  # Does NOT support torque-CoM tests - uses body origin coordinates internally
    ),
}

# Test configurations for non-zero CoM tests
com_offsets = [
    (0.5, 0.0, 0.0),  # X offset
    (0.0, 0.3, 0.0),  # Y offset
    (0.0, 0.0, 0.4),  # Z offset
    (0.2, 0.3, 0.1),  # Combined offset
]

force_directions = [
    (1.0, 0.0, 0.0),  # X force
    (0.0, 1.0, 0.0),  # Y force
    (0.0, 0.0, 1.0),  # Z force
]

for device in devices:
    for solver_name, (solver_fn, tolerance, supports_torque_com) in com_solvers.items():
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue

        # Test force with CoM offset (no rotation)
        # This should work for all solvers since forces act at the CoM
        for i, com_offset in enumerate(com_offsets):
            for j, force_dir in enumerate(force_directions):
                add_function_test(
                    TestBodyForce,
                    f"test_force_no_rotation_{solver_name}_com{i}_force{j}",
                    test_force_no_rotation,
                    devices=[device],
                    solver_fn=solver_fn,
                    com_offset=com_offset,
                    force_direction=force_dir,
                )
                if solver_name != "featherstone":
                    add_function_test(
                        TestBodyForce,
                        f"test_force_no_rotation_joint_f_{solver_name}_com{i}_force{j}",
                        test_force_no_rotation_control_joint_f,
                        devices=[device],
                        solver_fn=solver_fn,
                        com_offset=com_offset,
                        force_direction=force_dir,
                    )

        # Test combined force and torque with CoM offset
        # Only for solvers that correctly handle torque with CoM offset
        if supports_torque_com:
            for i, com_offset in enumerate(com_offsets):
                add_function_test(
                    TestBodyForce,
                    f"test_combined_force_torque_{solver_name}_com{i}",
                    test_combined_force_torque,
                    devices=[device],
                    solver_fn=solver_fn,
                    com_offset=com_offset,
                    use_control=False,
                )
                if solver_name != "featherstone":
                    add_function_test(
                        TestBodyForce,
                        f"test_combined_force_torque_joint_f_{solver_name}_com{i}",
                        test_combined_force_torque,
                        devices=[device],
                        solver_fn=solver_fn,
                        com_offset=com_offset,
                        use_control=True,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
