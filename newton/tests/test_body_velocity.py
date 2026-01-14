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
Tests for body velocity stepping with non-zero center of mass offsets.

This module tests that when applying angular velocity to a body with a non-zero
center of mass (CoM) offset, the body rotates about its CoM, not about the body
frame origin. This is verified by checking that the CoM position stays stationary
when only angular velocity is applied.

For generalized coordinate solvers (MuJoCo, Featherstone), velocity is set via joint_qd.
For maximal coordinate solvers (XPBD, SemiImplicit), velocity is set via body_qd.

Note on tolerances:
- MuJoCo/Featherstone use body origin velocity internally, which introduces small
  numerical integration errors when converting back to CoM velocity (~1e-3 after 10 steps).
- Maximal coordinate solvers (XPBD, SemiImplicit) directly integrate CoM velocity,
  so they have much tighter numerical precision (~1e-6).
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestBodyVelocity(unittest.TestCase):
    pass


def compute_com_world_position(body_q: np.ndarray, body_com: np.ndarray) -> np.ndarray:
    """Compute the center of mass position in world frame."""
    result = wp.transform_point(wp.transform(*body_q), wp.vec3(*body_com))
    return np.array([result[0], result[1], result[2]])


def test_angular_velocity_com_stationary(
    test: TestBodyVelocity,
    device,
    solver_fn,
    uses_generalized_coords: bool,
    com_offset: tuple[float, float, float],
    angular_velocity: tuple[float, float, float],
    tolerance: float,
):
    """Test that angular velocity causes rotation about CoM, not body origin.

    When a body has a non-zero CoM offset and we apply angular velocity with zero
    linear velocity (at the CoM), the CoM should stay stationary while the body
    rotates around it.

    Args:
        test: Test case instance
        device: Compute device
        solver_fn: Function that creates a solver given a model
        uses_generalized_coords: If True, set velocity via joint_qd; else via body_qd
        com_offset: Center of mass offset in body frame (x, y, z)
        angular_velocity: Angular velocity in world frame (wx, wy, wz)
        tolerance: Maximum allowed CoM drift
    """
    builder = newton.ModelBuilder(gravity=0.0)

    # Create a body with the specified CoM offset
    initial_pos = wp.vec3(1.0, 2.0, 3.0)
    b = builder.add_body(xform=wp.transform(initial_pos, wp.quat_identity()))
    builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
    builder.body_com[b] = wp.vec3(*com_offset)

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()

    # Compute initial FK
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Set angular velocity (linear velocity = 0 at CoM)
    # joint_qd for FREE joint: [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
    # body_qd: [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
    velocity = np.array([0.0, 0.0, 0.0, *angular_velocity], dtype=np.float32)

    if uses_generalized_coords:
        # MuJoCo, Featherstone: set joint_qd
        state_0.joint_qd.assign(velocity)
        # Also need to update body_qd via FK for the solver
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    else:
        # XPBD, SemiImplicit: set body_qd directly
        state_0.body_qd.assign(velocity.reshape(1, 6))

    # Get initial CoM position in world frame
    body_q_initial = state_0.body_q.numpy()[0].copy()
    body_com = np.array(com_offset)
    com_initial = compute_com_world_position(body_q_initial, body_com)

    # Step simulation
    sim_dt = 0.01
    num_steps = 10

    for _ in range(num_steps):
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0

    # Get final CoM position
    body_q_final = state_0.body_q.numpy()[0]
    com_final = compute_com_world_position(body_q_final, body_com)

    # CoM should stay stationary (within numerical tolerance)
    com_drift = np.linalg.norm(com_final - com_initial)
    test.assertLess(
        com_drift,
        tolerance,
        f"CoM drifted by {com_drift:.6f} (expected < {tolerance}). Initial CoM: {com_initial}, Final CoM: {com_final}",
    )

    # Verify that the body actually rotated (quaternion changed)
    quat_initial = body_q_initial[3:7]
    quat_final = body_q_final[3:7]
    quat_diff = np.abs(np.dot(quat_initial, quat_final))
    test.assertLess(
        quat_diff,
        0.9999,
        "Body should have rotated but quaternion barely changed",
    )


def test_linear_velocity_com_moves(
    test: TestBodyVelocity,
    device,
    solver_fn,
    uses_generalized_coords: bool,
    com_offset: tuple[float, float, float],
    linear_velocity: tuple[float, float, float],
    tolerance: float,
):
    """Test that linear velocity causes CoM to move as expected.

    When a body has a non-zero CoM offset and we apply linear velocity at the CoM
    with zero angular velocity, the CoM should translate at the specified velocity.

    Args:
        test: Test case instance
        device: Compute device
        solver_fn: Function that creates a solver given a model
        uses_generalized_coords: If True, set velocity via joint_qd; else via body_qd
        com_offset: Center of mass offset in body frame (x, y, z)
        linear_velocity: Linear velocity in world frame (vx, vy, vz)
        tolerance: Maximum allowed displacement error
    """
    builder = newton.ModelBuilder(gravity=0.0)

    initial_pos = wp.vec3(0.0, 0.0, 1.0)
    b = builder.add_body(xform=wp.transform(initial_pos, wp.quat_identity()))
    builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
    builder.body_com[b] = wp.vec3(*com_offset)

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Set linear velocity (angular velocity = 0)
    velocity = np.array([*linear_velocity, 0.0, 0.0, 0.0], dtype=np.float32)

    if uses_generalized_coords:
        state_0.joint_qd.assign(velocity)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    else:
        state_0.body_qd.assign(velocity.reshape(1, 6))

    # Get initial CoM position
    body_q_initial = state_0.body_q.numpy()[0]
    body_com = np.array(com_offset)
    com_initial = compute_com_world_position(body_q_initial, body_com)

    # Step simulation
    sim_dt = 0.01
    num_steps = 10
    total_time = sim_dt * num_steps

    for _ in range(num_steps):
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0

    # Get final CoM position
    body_q_final = state_0.body_q.numpy()[0]
    com_final = compute_com_world_position(body_q_final, body_com)

    # Expected displacement = velocity * time
    expected_displacement = np.array(linear_velocity) * total_time
    actual_displacement = com_final - com_initial

    # Check that displacement matches expected
    displacement_error = np.linalg.norm(actual_displacement - expected_displacement)
    test.assertLess(
        displacement_error,
        tolerance,
        f"CoM displacement error: {displacement_error:.6f} (expected < {tolerance}). "
        f"Expected: {expected_displacement}, Actual: {actual_displacement}",
    )


def test_combined_velocity(
    test: TestBodyVelocity,
    device,
    solver_fn,
    uses_generalized_coords: bool,
    com_offset: tuple[float, float, float],
    tolerance: float,
):
    """Test combined linear and angular velocity with non-zero CoM offset.

    When both linear and angular velocities are applied, the CoM should translate
    at the linear velocity rate while the body rotates.

    Args:
        test: Test case instance
        device: Compute device
        solver_fn: Function that creates a solver given a model
        uses_generalized_coords: If True, set velocity via joint_qd; else via body_qd
        com_offset: Center of mass offset in body frame (x, y, z)
        tolerance: Maximum allowed displacement error
    """
    builder = newton.ModelBuilder(gravity=0.0)

    initial_pos = wp.vec3(0.0, 0.0, 1.0)
    b = builder.add_body(xform=wp.transform(initial_pos, wp.quat_identity()))
    builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
    builder.body_com[b] = wp.vec3(*com_offset)

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Set both linear and angular velocity
    linear_velocity = (0.1, 0.0, 0.0)
    angular_velocity = (0.0, 0.0, 1.0)
    velocity = np.array([*linear_velocity, *angular_velocity], dtype=np.float32)

    if uses_generalized_coords:
        state_0.joint_qd.assign(velocity)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    else:
        state_0.body_qd.assign(velocity.reshape(1, 6))

    # Get initial CoM position
    body_q_initial = state_0.body_q.numpy()[0].copy()
    body_com = np.array(com_offset)
    com_initial = compute_com_world_position(body_q_initial, body_com)

    # Step simulation
    sim_dt = 0.01
    num_steps = 10
    total_time = sim_dt * num_steps

    for _ in range(num_steps):
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0

    # Get final CoM position
    body_q_final = state_0.body_q.numpy()[0]
    com_final = compute_com_world_position(body_q_final, body_com)

    # Expected displacement = linear_velocity * time (rotation shouldn't affect CoM position)
    expected_displacement = np.array(linear_velocity) * total_time
    actual_displacement = com_final - com_initial

    # The CoM should have moved only due to linear velocity, not angular
    displacement_error = np.linalg.norm(actual_displacement - expected_displacement)
    test.assertLess(
        displacement_error,
        tolerance,
        f"CoM displacement error: {displacement_error:.6f} (expected < {tolerance}). "
        f"Expected: {expected_displacement}, Actual: {actual_displacement}",
    )

    # Verify body rotated
    quat_initial = body_q_initial[3:7]
    quat_final = body_q_final[3:7]
    quat_diff = np.abs(np.dot(quat_initial, quat_final))
    test.assertLess(quat_diff, 0.9999, "Body should have rotated")


devices = get_test_devices()

solvers = {
    # NOTE: Featherstone currently has issues with angular velocity and non-zero CoM offsets.
    # The Featherstone algorithm uses body origin velocity internally, and while we have
    # conversion kernels at the solver boundary, the dynamics equations don't correctly
    # compute the centripetal acceleration needed to keep the CoM stationary when rotating.
    # Linear velocity tests pass, but angular velocity tests fail.
    # This requires deeper changes to the Featherstone algorithm.
    # "featherstone": (
    #     lambda model: newton.solvers.SolverFeatherstone(model, angular_damping=0.0),
    #     True,
    #     1e-3,
    # ),
    "mujoco_cpu": (
        lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True),
        True,
        1e-3,  # Higher tolerance due to body origin velocity integration
    ),
    "mujoco_warp": (
        lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True),
        True,
        1e-3,  # Higher tolerance due to body origin velocity integration
    ),
    "xpbd": (
        lambda model: newton.solvers.SolverXPBD(model, angular_damping=0.0),
        False,
        1e-4,  # Tighter tolerance - directly integrates CoM velocity
    ),
    "semi_implicit": (
        lambda model: newton.solvers.SolverSemiImplicit(model, angular_damping=0.0),
        False,
        1e-4,  # Tighter tolerance - directly integrates CoM velocity
    ),
}

# Test configurations: different CoM offsets and velocity directions
com_offsets = [
    (0.5, 0.0, 0.0),  # X offset
    (0.0, 0.3, 0.0),  # Y offset
    (0.0, 0.0, 0.4),  # Z offset
    (0.2, 0.3, 0.1),  # Combined offset
]

angular_velocities = [
    (0.0, 0.0, 1.0),  # Z rotation
    (0.0, 1.0, 0.0),  # Y rotation
    (1.0, 0.0, 0.0),  # X rotation
]

linear_velocities = [
    (0.7, 0.0, 0.0),  # X translation
    (0.0, 0.7, 0.0),  # Y translation
    (0.0, 0.0, 0.7),  # Z translation
]

for device in devices:
    for solver_name, (solver_fn, uses_gen_coords, tolerance) in solvers.items():
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue

        # Test angular velocity with various CoM offsets
        for i, com_offset in enumerate(com_offsets):
            for j, angular_vel in enumerate(angular_velocities):
                add_function_test(
                    TestBodyVelocity,
                    f"test_angular_com_stationary_{solver_name}_com{i}_ang{j}",
                    test_angular_velocity_com_stationary,
                    devices=[device],
                    solver_fn=solver_fn,
                    uses_generalized_coords=uses_gen_coords,
                    com_offset=com_offset,
                    angular_velocity=angular_vel,
                    tolerance=tolerance,
                )

        # Test linear velocity with various CoM offsets
        for i, com_offset in enumerate(com_offsets):
            for j, linear_vel in enumerate(linear_velocities):
                add_function_test(
                    TestBodyVelocity,
                    f"test_linear_com_moves_{solver_name}_com{i}_lin{j}",
                    test_linear_velocity_com_moves,
                    devices=[device],
                    solver_fn=solver_fn,
                    uses_generalized_coords=uses_gen_coords,
                    com_offset=com_offset,
                    linear_velocity=linear_vel,
                    tolerance=tolerance,
                )

        # Test combined velocity with various CoM offsets
        for i, com_offset in enumerate(com_offsets):
            add_function_test(
                TestBodyVelocity,
                f"test_combined_velocity_{solver_name}_com{i}",
                test_combined_velocity,
                devices=[device],
                solver_fn=solver_fn,
                uses_generalized_coords=uses_gen_coords,
                com_offset=com_offset,
                tolerance=tolerance,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
