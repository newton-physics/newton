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
import newton.core.articulation
import newton.core.ik as ik

# Import IK solver components (adjust import paths as needed)
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices


def test_ik_convergence(test, device):
    """Test that IK solver converges to target positions"""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        num_envs = 3

        # Build simple 2-link planar robots
        for env_idx in range(num_envs):
            x_offset = env_idx * 3.0

            # Link 1
            link1 = builder.add_body(xform=wp.transform([x_offset + 0.5, 0.0, 0.0], wp.quat_identity()), mass=1.0)

            builder.add_joint_revolute(
                parent=-1,
                child=link1,
                parent_xform=wp.transform([x_offset, 0.0, 0.0], wp.quat_identity()),
                child_xform=wp.transform([-0.5, 0.0, 0.0], wp.quat_identity()),
                axis=[0.0, 0.0, 1.0],
            )

            # Link 2
            link2 = builder.add_body(xform=wp.transform([x_offset + 1.5, 0.0, 0.0], wp.quat_identity()), mass=1.0)

            builder.add_joint_revolute(
                parent=link1,
                child=link2,
                parent_xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
                child_xform=wp.transform([-0.5, 0.0, 0.0], wp.quat_identity()),
                axis=[0.0, 0.0, 1.0],
            )

        model = builder.finalize(device=device, requires_grad=True)
        model.ground = True

        # Setup IK targets
        num_links = 2
        ee_link_index = 1
        ee_offset = wp.vec3(0.5, 0.0, 0.0)

        # Set explicit reachable targets for each environment
        targets = [
            [1.5, 1.0, 0.0],  # Env 0
            [4.5, 1.0, 0.0],  # Env 1
            [7.5, 1.0, 0.0],  # Env 2
        ]

        target_positions = wp.array(targets, dtype=wp.vec3)

        # Create position objective
        position_obj = ik.PositionObjective(
            link_index=ee_link_index,
            link_offset=ee_offset,
            target_positions=target_positions,
            num_links=num_links,
            num_envs=num_envs,
            total_residuals=3,
            residual_offset=0,
        )

        # Create IK solver
        ik_sys = ik.create_ik(
            model=model,
            num_envs=num_envs,
            objectives=[position_obj],
            damping=1e-3,
            jacobian_mode=ik.JacobianMode.AUTODIFF,
        )

        # Get initial positions
        state = model.state()
        newton.core.articulation.eval_fk(model, model.joint_q, model.joint_qd, state, None)

        initial_errors = []
        for env_idx in range(num_envs):
            body_idx = env_idx * num_links + ee_link_index
            body_tf = state.body_q.numpy()[body_idx]

            # Convert numpy to Warp types properly
            pos = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
            rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
            tf = wp.transform(pos, rot)
            ee_pos = wp.transform_point(tf, ee_offset)

            # Convert back to numpy for error calculation
            ee_pos_np = np.array([ee_pos[0], ee_pos[1], ee_pos[2]])
            initial_errors.append(np.linalg.norm(ee_pos_np - targets[env_idx]))

        # Solve
        ik_sys.solve(iterations=50)

        # Check convergence
        newton.core.articulation.eval_fk(model, model.joint_q, model.joint_qd, state, None)

        for env_idx in range(num_envs):
            body_idx = env_idx * num_links + ee_link_index
            body_tf = state.body_q.numpy()[body_idx]

            # Convert numpy to Warp types properly
            pos = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
            rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
            tf = wp.transform(pos, rot)
            final_ee_pos = wp.transform_point(tf, ee_offset)

            # Convert back to numpy for error calculation
            final_ee_pos_np = np.array([final_ee_pos[0], final_ee_pos[1], final_ee_pos[2]])
            final_error = np.linalg.norm(final_ee_pos_np - targets[env_idx])

            # Verify improvement
            test.assertLess(final_error, initial_errors[env_idx], f"IK didn't improve for env {env_idx}")
            test.assertLess(final_error, 0.1, f"IK error too large for env {env_idx}: {final_error}")


def test_position_jacobian(test, device):
    """Test analytic vs autodiff Jacobian for position objective"""
    with wp.ScopedDevice(device):
        num_envs = 3

        # Build multiple 2-joint planar robots
        builder = newton.ModelBuilder()

        # Store info for each environment
        link_indices = []

        for env_idx in range(num_envs):
            # Offset each robot in space
            x_offset = env_idx * 3.0

            # Link 1 - 1m long
            link1 = builder.add_body(
                xform=wp.transform([x_offset, 0.0, 0.0], wp.quat_identity()), mass=1.0, key=f"link1_env{env_idx}"
            )

            # Joint 1 - revolute joint at origin
            builder.add_joint_revolute(
                parent=-1,  # World frame
                child=link1,
                parent_xform=wp.transform([x_offset, 0.0, 0.0], wp.quat_identity()),
                child_xform=wp.transform_identity(),
                axis=[0.0, 0.0, 1.0],  # Z-axis rotation
                key=f"joint1_env{env_idx}",
            )

            # Add visual for link1
            builder.add_shape_box(
                body=link1,
                xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
                hx=0.5,
                hy=0.05,
                hz=0.05,
            )

            # Link 2 - 1m long
            link2 = builder.add_body(
                xform=wp.transform([x_offset + 1.0, 0.0, 0.0], wp.quat_identity()), mass=1.0, key=f"link2_env{env_idx}"
            )

            # Joint 2 - revolute joint at end of link1
            builder.add_joint_revolute(
                parent=link1,
                child=link2,
                parent_xform=wp.transform([1.0, 0.0, 0.0], wp.quat_identity()),
                child_xform=wp.transform_identity(),
                axis=[0.0, 0.0, 1.0],  # Z-axis rotation
                key=f"joint2_env{env_idx}",
            )

            # Add visual for link2
            builder.add_shape_box(
                body=link2,
                xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
                hx=0.5,
                hy=0.05,
                hz=0.05,
            )

            # Store link2 index (end-effector link)
            link_indices.append(link2)

        # Build model
        model = builder.finalize(device=device, requires_grad=True)

        # Set different initial joint angles for each environment
        joint_angles = []
        for env_idx in range(num_envs):
            joint_angles.extend(
                [
                    0.0 + env_idx * 0.1,  # joint1 angle
                    np.pi / 4 + env_idx * 0.2,  # joint2 angle
                ]
            )

        model.joint_q = wp.array(joint_angles, dtype=wp.float32, requires_grad=True)
        model.joint_qd = wp.zeros(len(joint_angles), dtype=wp.float32)

        # Compute forward kinematics
        state = model.state()
        newton.core.articulation.eval_fk(model, model.joint_q, model.joint_qd, state, None)

        # Setup for each environment
        ee_link_index = 1  # link2 within each articulation
        ee_offset = wp.vec3(1.0, 0.0, 0.0)  # 1m from joint to tip
        num_links = 2  # 2 links per articulation
        coords_per_env = 2  # 2 revolute joints per articulation
        total_residuals = 3  # Position only (x, y, z)

        # Create different target positions for each environment
        target_positions = []
        for env_idx in range(num_envs):
            x_offset = env_idx * 3.0
            target_positions.append(
                [
                    x_offset + 1.5 + env_idx * 0.1,  # x
                    0.5 + env_idx * 0.1,  # y
                    0.0,  # z
                ]
            )

        target_position_array = wp.array(target_positions, dtype=wp.vec3)

        # Create position objective
        objective = ik.PositionObjective(
            link_index=ee_link_index,
            link_offset=ee_offset,
            target_positions=target_position_array,
            num_links=num_links,
            num_envs=num_envs,
            total_residuals=total_residuals,
            residual_offset=0,
        )

        # Override supports_analytic to return True for testing
        objective.supports_analytic = lambda: True

        # Allocate arrays
        residuals = wp.zeros((num_envs, total_residuals), dtype=wp.float32, requires_grad=True)
        jacobian_autodiff = wp.zeros((num_envs, total_residuals, coords_per_env), dtype=wp.float32)
        jacobian_analytic = wp.zeros((num_envs, total_residuals, coords_per_env), dtype=wp.float32)

        # Compute residuals
        objective.compute_residuals(state, model, residuals, 0)

        # Compute Jacobian with autodiff
        tape = wp.Tape()
        with tape:
            # Need fresh state computation in tape
            newton.core.articulation.eval_fk(model, model.joint_q, model.joint_qd, state, None)
            residuals_compute = wp.zeros((num_envs, total_residuals), dtype=wp.float32, requires_grad=True)
            objective.compute_residuals(state, model, residuals_compute, 0)
            residuals_flat = residuals_compute.flatten()

        tape.outputs = [residuals_flat]
        objective.compute_jacobian_autodiff(tape, model, jacobian_autodiff, 0)

        # Compute Jacobian analytically
        objective.compute_jacobian_analytic(state, model, jacobian_analytic, 0)

        # Compare
        J_auto = jacobian_autodiff.numpy()
        J_analytic = jacobian_analytic.numpy()

        # Check each environment
        for env_idx in range(num_envs):
            # Get actual EE position for debugging
            body_idx = env_idx * num_links + ee_link_index
            body_tf = state.body_q.numpy()[body_idx]

            # Convert properly to avoid warnings
            pos = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
            rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
            tf = wp.transform(pos, rot)
            wp.transform_point(tf, ee_offset)

            # Check this environment
            assert_np_equal(J_analytic[env_idx], J_auto[env_idx], tol=1e-4)


def test_rotation_jacobian(test, device):
    """Test rotation objective Jacobian computation"""
    with wp.ScopedDevice(device):
        num_envs = 3

        # Build multiple 2-joint planar robots
        builder = newton.ModelBuilder()

        # Store info for each environment
        link_indices = []

        for env_idx in range(num_envs):
            # Offset each robot in space
            x_offset = env_idx * 3.0

            # Link 1 - 1m long
            link1 = builder.add_body(
                xform=wp.transform([x_offset, 0.0, 0.0], wp.quat_identity()), mass=1.0, key=f"link1_env{env_idx}"
            )

            # Joint 1 - revolute joint at origin
            builder.add_joint_revolute(
                parent=-1,  # World frame
                child=link1,
                parent_xform=wp.transform([x_offset, 0.0, 0.0], wp.quat_identity()),
                child_xform=wp.transform_identity(),
                axis=[0.0, 0.0, 1.0],  # Z-axis rotation
                key=f"joint1_env{env_idx}",
            )

            # Link 2 - 1m long
            link2 = builder.add_body(
                xform=wp.transform([x_offset + 1.0, 0.0, 0.0], wp.quat_identity()), mass=1.0, key=f"link2_env{env_idx}"
            )

            # Joint 2 - revolute joint at end of link1
            builder.add_joint_revolute(
                parent=link1,
                child=link2,
                parent_xform=wp.transform([1.0, 0.0, 0.0], wp.quat_identity()),
                child_xform=wp.transform_identity(),
                axis=[0.0, 0.0, 1.0],  # Z-axis rotation
                key=f"joint2_env{env_idx}",
            )

            # Store link2 index (end-effector link)
            link_indices.append(link2)

        # Build model
        model = builder.finalize(device=device, requires_grad=True)

        # Set different initial joint angles for each environment
        joint_angles = []
        for env_idx in range(num_envs):
            joint_angles.extend(
                [
                    0.0 + env_idx * 0.1,  # joint1 angle
                    np.pi / 4 + env_idx * 0.2,  # joint2 angle
                ]
            )

        model.joint_q = wp.array(joint_angles, dtype=wp.float32, requires_grad=True)
        model.joint_qd = wp.zeros(len(joint_angles), dtype=wp.float32)

        # Compute forward kinematics
        state = model.state()
        newton.core.articulation.eval_fk(model, model.joint_q, model.joint_qd, state, None)

        # Setup for each environment
        ee_link_index = 1  # link2 within each articulation
        ee_offset_rotation = wp.quat_identity()  # No rotation offset
        num_links = 2  # 2 links per articulation
        coords_per_env = 2  # 2 revolute joints per articulation
        total_residuals = 3  # Rotation quaternion error (x, y, z components)

        # Create different target rotations for each environment
        # Since we have planar robots rotating around Z, we'll create Z-axis rotations
        target_rotations = []
        for env_idx in range(num_envs):
            # Create rotation around Z-axis
            angle = np.pi / 6 + env_idx * np.pi / 8  # Different target angles
            # Quaternion for Z-axis rotation: [0, 0, sin(θ/2), cos(θ/2)]
            qx = 0.0
            qy = 0.0
            qz = np.sin(angle / 2.0)
            qw = np.cos(angle / 2.0)
            target_rotations.append([qx, qy, qz, qw])

        target_rotation_array = wp.array(target_rotations, dtype=wp.vec4)

        # Create rotation objective
        objective = ik.RotationObjective(
            link_index=ee_link_index,
            link_offset_rotation=ee_offset_rotation,
            target_rotations=target_rotation_array,
            num_links=num_links,
            num_envs=num_envs,
            total_residuals=total_residuals,
            residual_offset=0,
        )

        # Allocate arrays
        residuals = wp.zeros((num_envs, total_residuals), dtype=wp.float32, requires_grad=True)
        jacobian_autodiff = wp.zeros((num_envs, total_residuals, coords_per_env), dtype=wp.float32)

        # Compute residuals
        objective.compute_residuals(state, model, residuals, 0)

        # Compute Jacobian with autodiff
        tape = wp.Tape()
        with tape:
            # Need fresh state computation in tape
            newton.core.articulation.eval_fk(model, model.joint_q, model.joint_qd, state, None)
            residuals_compute = wp.zeros((num_envs, total_residuals), dtype=wp.float32, requires_grad=True)
            objective.compute_residuals(state, model, residuals_compute, 0)
            residuals_flat = residuals_compute.flatten()

        tape.outputs = [residuals_flat]
        objective.compute_jacobian_autodiff(tape, model, jacobian_autodiff, 0)

        # Get the autodiff Jacobian
        jacobian_autodiff.numpy()

        # Verify Jacobian with fini


def test_joint_limit_jacobian(test, device):
    """Test joint limit objective Jacobian computation"""
    with wp.ScopedDevice(device):
        num_envs = 2
        dof_per_env = 3

        # Build robots with joint limits
        builder = newton.ModelBuilder()

        for env_idx in range(num_envs):
            x_offset = env_idx * 2.0

            # Add 3-DOF robot
            prev_body = -1
            for i in range(3):
                body = builder.add_body(
                    xform=wp.transform([x_offset, 0.0, i * 0.5 + 0.5], wp.quat_identity()), mass=1.0
                )

                builder.add_joint_revolute(
                    parent=prev_body,
                    child=body,
                    parent_xform=wp.transform([0.0, 0.0, 0.5], wp.quat_identity())
                    if i > 0
                    else wp.transform([x_offset, 0.0, 0.5], wp.quat_identity()),
                    child_xform=wp.transform_identity(),
                    axis=[0.0, 0.0, 1.0],
                    limit_lower=-1.0,
                    limit_upper=1.0,
                )

                # Set some joints near limits
                if i == 0:
                    builder.joint_q[env_idx * 3 + i] = 0.95  # Near upper limit
                elif i == 1:
                    builder.joint_q[env_idx * 3 + i] = -0.95  # Near lower limit
                else:
                    builder.joint_q[env_idx * 3 + i] = 0.0  # Safe

                prev_body = body

        model = builder.finalize(device=device, requires_grad=True)
        state = model.state()

        # Create joint limit objective
        joint_limit_obj = ik.JointLimitObjective(
            joint_limit_lower=model.joint_limit_lower,
            joint_limit_upper=model.joint_limit_upper,
            weight=0.1,
            num_envs=num_envs,
            total_residuals=dof_per_env,
            residual_offset=0,
        )

        # Test Jacobian
        wp.zeros((num_envs, dof_per_env), dtype=wp.float32, requires_grad=True)
        jacobian_autodiff = wp.zeros((num_envs, dof_per_env, dof_per_env), dtype=wp.float32)
        jacobian_analytic = wp.zeros((num_envs, dof_per_env, dof_per_env), dtype=wp.float32)

        # Compute autodiff Jacobian
        tape = wp.Tape()
        with tape:
            residuals_compute = wp.zeros((num_envs, dof_per_env), dtype=wp.float32, requires_grad=True)
            joint_limit_obj.compute_residuals(state, model, residuals_compute, 0)
            residuals_flat = residuals_compute.flatten()

        tape.outputs = [residuals_flat]
        joint_limit_obj.compute_jacobian_autodiff(tape, model, jacobian_autodiff, 0)

        # Compute analytic Jacobian
        joint_limit_obj.compute_jacobian_analytic(state, model, jacobian_analytic, 0)

        # Compare
        J_auto = jacobian_autodiff.numpy()
        J_anal = jacobian_analytic.numpy()

        for env_idx in range(num_envs):
            assert_np_equal(J_anal[env_idx], J_auto[env_idx], tol=1e-5)


def test_multi_target_jacobian(test, device):
    """Test Jacobian computation with multiple end-effectors"""
    with wp.ScopedDevice(device):
        num_envs = 1

        # Build a 4-link robot with 2 end-effectors (branching structure)
        builder = newton.ModelBuilder()

        # Base (fixed)
        builder.add_body(
            xform=wp.transform_identity(),
            mass=0.0,  # Fixed
            key="base",
        )

        # Link 1
        link1 = builder.add_body(xform=wp.transform([0.0, 0.0, 0.5], wp.quat_identity()), mass=1.0, key="link1")
        builder.add_joint_revolute(
            parent=-1,
            child=link1,
            parent_xform=wp.transform([0.0, 0.0, 0.5], wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=[0.0, 0.0, 1.0],
            key="joint1",
        )

        # Branch A - Link 2a (first end-effector)
        link2a = builder.add_body(xform=wp.transform([0.5, 0.0, 1.0], wp.quat_identity()), mass=0.5, key="link2a")
        builder.add_joint_revolute(
            parent=link1,
            child=link2a,
            parent_xform=wp.transform([0.5, 0.0, 0.5], wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=[0.0, 1.0, 0.0],
            key="joint2a",
        )

        # Branch B - Link 2b (second end-effector)
        link2b = builder.add_body(xform=wp.transform([-0.5, 0.0, 1.0], wp.quat_identity()), mass=0.5, key="link2b")
        builder.add_joint_revolute(
            parent=link1,
            child=link2b,
            parent_xform=wp.transform([-0.5, 0.0, 0.5], wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=[0.0, 1.0, 0.0],
            key="joint2b",
        )

        model = builder.finalize(device=device, requires_grad=True)

        # Set initial joint positions
        model.joint_q = wp.array([0.0, 0.3, -0.3], dtype=wp.float32, requires_grad=True)
        model.joint_qd = wp.zeros(3, dtype=wp.float32)

        # Compute FK
        state = model.state()
        newton.core.articulation.eval_fk(model, model.joint_q, model.joint_qd, state, None)

        # Setup multiple end-effectors
        num_links = 4  # base, link1, link2a, link2b
        ee_info = [
            {"link_index": 2, "offset": wp.vec3(0.5, 0.0, 0.0)},  # Tip of link2a
            {"link_index": 3, "offset": wp.vec3(-0.5, 0.0, 0.0)},  # Tip of link2b
        ]
        num_ees = len(ee_info)
        coords = 3

        # Get initial positions and set targets
        targets = []
        for i, ee in enumerate(ee_info):
            body_idx = ee["link_index"]
            body_tf = state.body_q.numpy()[body_idx]

            # Convert properly to avoid warnings
            pos = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
            rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
            tf = wp.transform(pos, rot)
            ee_pos = wp.transform_point(tf, ee["offset"])

            # Set different targets for each EE
            target_offset = wp.vec3(0.1, 0.05, -0.1) * float(i + 1)
            target = wp.vec3(ee_pos[0] + target_offset[0], ee_pos[1] + target_offset[1], ee_pos[2] + target_offset[2])
            targets.append([target[0], target[1], target[2]])

        # Create objectives
        objectives = []
        total_residuals = num_ees * 3 + coords  # Position residuals + joint limits

        # Add position objectives
        for i, ee in enumerate(ee_info):
            obj = ik.PositionObjective(
                link_index=ee["link_index"],
                link_offset=ee["offset"],
                target_positions=wp.array([targets[i]], dtype=wp.vec3),  # Single env
                num_links=num_links,
                num_envs=num_envs,
                total_residuals=total_residuals,
                residual_offset=i * 3,
            )
            # Force analytic for testing
            obj.supports_analytic = lambda: True
            objectives.append(obj)

        # Add joint limit objective
        joint_limit_obj = ik.JointLimitObjective(
            joint_limit_lower=model.joint_limit_lower,
            joint_limit_upper=model.joint_limit_upper,
            weight=0.1,
            num_envs=num_envs,
            total_residuals=total_residuals,
            residual_offset=num_ees * 3,
        )
        objectives.append(joint_limit_obj)

        # Create solver with autodiff
        ik_sys_autodiff = ik.create_ik(
            model=model, num_envs=num_envs, objectives=objectives, damping=1e-3, jacobian_mode=ik.JacobianMode.AUTODIFF
        )

        # Create solver with analytic
        ik_sys_analytic = ik.create_ik(
            model=model, num_envs=num_envs, objectives=objectives, damping=1e-3, jacobian_mode=ik.JacobianMode.ANALYTIC
        )

        # Compute residuals (should be same for both)
        residuals_auto = ik_sys_autodiff.compute_residuals()
        residuals_anal = ik_sys_analytic.compute_residuals()

        # Should have same residuals
        assert_np_equal(residuals_auto.numpy(), residuals_anal.numpy(), tol=1e-6)

        # Compute Jacobians
        jacobian_auto = ik_sys_autodiff.compute_jacobian()
        jacobian_anal = ik_sys_analytic.compute_jacobian()

        # Compare
        J_auto = jacobian_auto.numpy()
        J_anal = jacobian_anal.numpy()

        # Check shape
        test.assertEqual(J_auto.shape, (1, total_residuals, coords))

        # Check each objective's contribution
        for i in range(num_ees):
            # Position objectives should have non-zero Jacobian
            pos_jacobian_auto = J_auto[0, i * 3 : (i + 1) * 3, :]
            pos_jacobian_anal = J_anal[0, i * 3 : (i + 1) * 3, :]

            # Should be non-zero
            test.assertGreater(np.linalg.norm(pos_jacobian_auto), 1e-6, f"Autodiff Jacobian for EE {i} is zero")
            test.assertGreater(np.linalg.norm(pos_jacobian_anal), 1e-6, f"Analytic Jacobian for EE {i} is zero")

        # Test that they match
        assert_np_equal(J_anal, J_auto, tol=1e-4)


devices = get_test_devices()


class TestIK(unittest.TestCase):
    pass


add_function_test(TestIK, "test_ik_convergence", test_ik_convergence, devices=devices)
add_function_test(TestIK, "test_position_jacobian", test_position_jacobian, devices=devices)
add_function_test(TestIK, "test_rotation_jacobian", test_rotation_jacobian, devices=devices)
add_function_test(TestIK, "test_joint_limit_jacobian", test_joint_limit_jacobian, devices=devices)
add_function_test(TestIK, "test_multi_target_jacobian", test_multi_target_jacobian, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
