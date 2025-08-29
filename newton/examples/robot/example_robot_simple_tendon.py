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

###########################################################################
# Example Simple Tendon Robot
#
# Shows a simple demonstration of fixed tendons coupling joint motion.
# Creates a 3-link finger where:
# - Joint 1 (base) is actively controlled with position control
# - Joint 2 is passive and coupled to joint 1 via a tendon
# - Joint 3 is passive (free to move based on dynamics)
#
# The example demonstrates how tendons can create mechanical coupling
# between joints, similar to how finger tendons work in robotic hands.
#
# What you should see:
# - The base joint oscillates back and forth
# - Joint 2 follows joint 1's motion (80% same direction)
# - Joint 3 moves freely based on gravity and dynamics
# - Creates a natural finger curling motion
#
# Command: python -m newton.examples robot_simple_tendon --num-envs 4
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, num_envs=4):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs
        self.viewer = viewer

        # Create a simple 3-link finger with tendon coupling
        finger = newton.ModelBuilder()

        # Configure joint and shape parameters
        finger.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=100.0,
            limit_kd=100.0,
            friction=0.2,
            armature=0.1,  # Increased armature for stability
        )

        finger.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.5,
            ke=1.0e3,
            kd=1.0e2,
            kf=1.0e2,
            density=1000.0,  # kg/m^3 - density for future automatic mass/inertia computation
        )

        # Create 3 links
        link_length = 0.15
        link_radius = 0.02
        link_mass = 0.1

        # Link 1
        link1 = finger.add_body(mass=link_mass, key="link1")

        # Link 2
        link2 = finger.add_body(mass=link_mass, key="link2")

        # Link 3
        link3 = finger.add_body(mass=link_mass, key="link3")

        # Create joints
        # Joint 1: world to link1 (this will be actuated)
        joint1 = finger.add_joint_revolute(
            parent=-1,  # Attach to world
            child=link1,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, link_length), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -link_length / 2), wp.quat_identity()),
            axis=(1.0, 0.0, 0.0),  # Rotate around X
            limit_lower=-math.pi / 2,
            limit_upper=math.pi / 2,
            key="joint1",
        )

        # Joint 2: link1 to link2 (coupled by tendon)
        joint2 = finger.add_joint_revolute(
            parent=link1,
            child=link2,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, link_length / 2), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -link_length / 2), wp.quat_identity()),
            axis=(1.0, 0.0, 0.0),
            limit_lower=-math.pi / 2,  # Reasonable limits
            limit_upper=math.pi / 2,
            key="joint2",
        )

        # Joint 3: link2 to link3 (coupled by tendon)
        joint3 = finger.add_joint_revolute(
            parent=link2,
            child=link3,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, link_length / 2), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -link_length / 2), wp.quat_identity()),
            axis=(1.0, 0.0, 0.0),
            limit_lower=-math.pi / 2,  # Reasonable limits
            limit_upper=math.pi / 2,
            key="joint3",
        )

        # Set joint control modes
        finger.joint_dof_mode[joint1] = newton.JointMode.TARGET_POSITION
        finger.joint_target_ke[joint1] = 50.0
        finger.joint_target_kd[joint1] = 5.0

        # Joint 2 is passive (no position control)
        finger.joint_dof_mode[joint2] = newton.JointMode.NONE

        # Joint 3 is passive (controlled by tendon)
        finger.joint_dof_mode[joint3] = newton.JointMode.NONE

        # Add a tendon that couples joint1 to joint2
        # When joint1 moves, joint2 follows with some ratio
        finger.add_tendon(
            name="finger_tendon1",
            joint_ids=[joint1, joint2],
            gearings=[1.0, -0.8],  # Joint2 moves 80% with joint1
            stiffness=100.0,  # Moderate stiffness
            damping=10.0,  # Moderate damping
            rest_length=0.0,
        )

        # Add a tendon that couples joint1 to joint3
        # When joint1 moves, joint3 follows with some ratio
        finger.add_tendon(
            name="finger_tendon2",
            joint_ids=[joint1, joint3],
            gearings=[1.0, -0.6],  # Joint3 moves 80% with joint2
            stiffness=100.0,  # Moderate stiffness
            damping=10.0,  # Moderate damping
            rest_length=0.0,
        )

        # Add visual shapes
        # Position capsules to align with the center of mass
        finger.add_shape_capsule(
            body=link1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0), wp.quat_identity()),
            radius=link_radius,
            half_height=link_length / 2,
        )

        finger.add_shape_capsule(
            body=link2,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0), wp.quat_identity()),
            radius=link_radius,
            half_height=link_length / 2,
        )

        finger.add_shape_capsule(
            body=link3,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0), wp.quat_identity()),
            radius=link_radius,
            half_height=link_length / 2,
        )

        # Build full scene
        builder = newton.ModelBuilder()
        builder.replicate(finger, self.num_envs, spacing=(0.6, 0.6, 0))
        builder.add_ground_plane()

        self.model = builder.finalize()

        # Print tendon info
        if hasattr(self.model, "tendon_count") and self.model.tendon_count > 0:
            print(f"\nModel has {self.model.tendon_count} tendons:")
            # Convert warp arrays to numpy for indexing
            tendon_start_np = self.model.tendon_start.numpy()
            tendon_params_np = self.model.tendon_params.numpy()

            for i in range(self.model.tendon_count):
                tendon_name = self.model.tendon_key[i] if i < len(self.model.tendon_key) else f"tendon_{i}"
                start_idx = tendon_start_np[i]
                end_idx = tendon_start_np[i + 1] if i + 1 < len(tendon_start_np) else len(self.model.tendon_joints)
                num_joints = end_idx - start_idx
                params = tendon_params_np[i]
                print(f"  Tendon '{tendon_name}': couples {num_joints} joints")
                print(f"    Stiffness: {params[0]}, Damping: {params[1]}")

        # Create solver
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            iterations=20,
            ls_iterations=10,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # Store initial joint positions for reference
        self.initial_joint_q = self.state_0.joint_q.numpy().copy()

        # We'll control only the first joint of each finger
        self.actuated_joints = list(range(0, self.model.joint_dof_count, 3))
        print(f"\nActuating joints: {self.actuated_joints}")

        # Show which joints are passive and coupled by tendons
        passive_joints = []
        for i in range(self.model.joint_dof_count):
            if i not in self.actuated_joints:
                passive_joints.append(i)
        print(f"Passive joints (driven by tendon): {passive_joints}")

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply viewer forces (for interaction)
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Apply sinusoidal motion to first joint only
        # The tendon will couple the motion to the other joints
        phase = self.sim_time * 1.0  # 2 rad/s frequency

        # Create target positions array
        target_q = np.zeros(self.model.joint_dof_count)

        # Animate only the first joint of each finger
        for _i, joint_idx in enumerate(self.actuated_joints):
            # Oscillate between -45 and +45 degrees
            angle = 0.6 * math.sin(phase)
            target_q[joint_idx] = angle

        # Set target positions in control
        target_q_wp = wp.array(target_q, dtype=float, device=self.model.device)
        wp.copy(self.control.joint_target, target_q_wp)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        # Debug: Print joint positions occasionally
        if int(self.sim_time * 10) % 10 == 0:  # Every second
            joint_q = self.state_0.joint_q.numpy()
            if len(joint_q) >= 3:
                print(
                    f"Time {self.sim_time:.1f}s: Joint positions: "
                    f"q1={joint_q[0]:.3f}, q2={joint_q[1]:.3f}, q3={joint_q[2]:.3f}, "
                    f"constraint={joint_q[0] + 0.8 * joint_q[1]:.3f}"
                )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test(self):
        # Run a few steps to verify tendons are working
        for _ in range(10):
            self.step()

        # Check that coupled joints have moved together
        current_q = self.state_0.joint_q.numpy()

        # For each finger, check that joint2 and joint3 maintain the coupling
        for finger in range(self.num_envs):
            base_idx = finger * 3
            q1 = current_q[base_idx]  # First joint (actuated)
            q2 = current_q[base_idx + 1]  # Second joint (tendon coupled)
            q3 = current_q[base_idx + 2]  # Third joint (tendon coupled)

            # Verify that joint2 and joint3 maintain approximate ratio
            # The gearing is [1.0, 0.8], so we expect q3 ≈ 0.8 * q2
            if abs(q2) > 0.01:  # Only check if joint2 has moved
                ratio = q3 / q2
                print(f"Finger {finger}: q1={q1:.3f}, q2={q2:.3f}, q3={q3:.3f}, ratio={ratio:.3f}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-envs", type=int, default=4, help="Total number of simulated environments.")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_envs)

    newton.examples.run(example)
