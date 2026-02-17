# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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
# Example Basic Kinematic Platform
#
# Demonstrates kinematic joints in MuJoCo using KinematicMode.
#
# A rotating kinematic platform (free joint, VELOCITY mode) with
# two-link capsule articulations dropped on top. Each articulation has
# a kinematic hinge (POSITION mode) that oscillates, so the capsule
# pairs wave back and forth while riding the spinning platform.
#
# Command: uv run -m newton.examples basic_kinematic_platform
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples
from newton import KinematicMode

PLATFORM_ANGULAR_VEL = 1.0  # rad/s around Z axis
HINGE_AMPLITUDE = 0.8  # radians
HINGE_FREQUENCY = 1.0  # Hz
NUM_CREATURES = 8


@wp.kernel
def update_hinge_targets(
    hinge_q_starts: wp.array(dtype=wp.int32),
    sim_time: wp.array(dtype=wp.float32),
    amplitude: float,
    frequency: float,
    # output
    kinematic_target: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    t = sim_time[0]
    phase = float(i) * 0.5  # stagger the oscillations
    # Position target: desired hinge angle
    omega = 2.0 * 3.14159265 * frequency
    kinematic_target[hinge_q_starts[i]] = amplitude * wp.sin(omega * t + phase)


@wp.kernel
def advance_time(sim_time: wp.array(dtype=wp.float32), dt: float):
    sim_time[0] = sim_time[0] + dt


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder()

        # -- Kinematic rotating platform --
        platform_body = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.05), q=wp.quat_identity()),
            mass=10.0,
            armature=0.5,
        )
        builder.add_shape_box(
            platform_body,
            hx=2.0,
            hy=2.0,
            hz=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(mu=1.0),
        )
        platform_joint = builder.add_joint_free(
            platform_body,
            kinematic=KinematicMode.VELOCITY,
        )
        builder.add_articulation([platform_joint], key="platform")

        # -- Two-link capsule creatures dropped on the platform --

        self.hinge_joints = []
        capsule_radius = 0.08
        capsule_half_length = 0.2
        drop_height = 1.0
        spawn_radius = 0.8
        # Capsule shape extends along Z by default; rotate to extend along X (body-local)
        capsule_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi * 0.5)

        for ci in range(NUM_CREATURES):
            angle = 2.0 * math.pi * ci / NUM_CREATURES
            # Orient each creature radially outward
            body_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(angle))
            dx = math.cos(angle)
            dy = math.sin(angle)

            # First capsule (inner)
            p0 = wp.vec3(spawn_radius * dx, spawn_radius * dy, drop_height)
            b0 = builder.add_link(xform=wp.transform(p=p0, q=body_rot), armature=0.5)
            builder.add_shape_capsule(
                b0,
                radius=capsule_radius,
                half_height=capsule_half_length,
                xform=wp.transform(q=capsule_rot),
            )

            j_free = builder.add_joint_free(b0)

            # Second capsule (outer), offset along the radial direction
            offset = capsule_half_length * 2.0
            p1 = wp.vec3((spawn_radius + offset) * dx, (spawn_radius + offset) * dy, drop_height)
            b1 = builder.add_link(xform=wp.transform(p=p1, q=body_rot), armature=0.5)
            builder.add_shape_capsule(
                b1,
                radius=capsule_radius,
                half_height=capsule_half_length,
                xform=wp.transform(q=capsule_rot),
            )

            # Hinge between the two capsules (along body-local X which points radially)
            j_hinge = builder.add_joint_revolute(
                parent=b0,
                child=b1,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=wp.transform(p=wp.vec3(capsule_half_length, 0.0, 0.0)),
                child_xform=wp.transform(p=wp.vec3(-capsule_half_length, 0.0, 0.0)),
                kinematic=KinematicMode.POSITION,
            )
            self.hinge_joints.append(j_hinge)

            # Set initial hinge angle to match the t=0 target (avoid velocity spike)
            phase = ci * 0.5
            hinge_q_start = builder.joint_q_start[j_hinge]
            builder.joint_q[hinge_q_start] = HINGE_AMPLITUDE * math.sin(phase)

            builder.add_articulation([j_free, j_hinge])

        # Ground plane
        builder.add_ground_plane()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=200)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Set the platform angular velocity target
        platform_q_start = int(self.model.joint_q_start.numpy()[platform_joint])
        target = self.control.kinematic_target.numpy()
        target[platform_q_start + 5] = PLATFORM_ANGULAR_VEL
        self.control.kinematic_target = wp.array(target, dtype=wp.float32, device=self.model.device)

        # Collect hinge q_start indices for the kernel
        q_starts = self.model.joint_q_start.numpy()
        hinge_q_starts = [int(q_starts[j]) for j in self.hinge_joints]
        self.hinge_q_starts_wp = wp.array(hinge_q_starts, dtype=wp.int32, device=self.model.device)

        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.model.device)

        # Initialize hinge position targets to match t=0 values (avoid jump on first step)
        for ci, qs in enumerate(hinge_q_starts):
            phase = ci * 0.5
            target[qs] = HINGE_AMPLITUDE * math.sin(phase)
        self.control.kinematic_target = wp.array(target, dtype=wp.float32, device=self.model.device)

        self.viewer.set_model(self.model)

        self.viewer._paused = True

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # Update all kinematic hinge targets
            wp.launch(
                update_hinge_targets,
                dim=NUM_CREATURES,
                inputs=[
                    self.hinge_q_starts_wp,
                    self.sim_time_wp,
                    HINGE_AMPLITUDE,
                    HINGE_FREQUENCY,
                ],
                outputs=[self.control.kinematic_target],
                device=self.model.device,
            )

            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            wp.launch(advance_time, dim=1, inputs=[self.sim_time_wp, self.sim_dt])

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        # Check all bodies are above ground
        for i in range(self.model.body_count):
            z = float(body_q[i][2])
            assert z > -0.5, f"Body {i} fell through ground: z={z:.4f}"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
