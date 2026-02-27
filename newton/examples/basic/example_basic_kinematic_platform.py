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
# Demonstrates kinematic joints in MuJoCo via high armature (1e10).
#
# A kinematic body is emulated by setting very high armature on its joint
# DOFs. The solver sees real velocity for proper contact resolution, but
# external forces can't meaningfully accelerate the DOF (1/armature ≈ 0
# in the inverse mass matrix).
#
# The scene has a rotating platform (kinematic free joint) with two-link
# capsule creatures dropped on top. Each creature has a kinematic hinge
# that oscillates back and forth.
#
# Command: uv run -m newton.examples basic_kinematic_platform
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples

KINEMATIC_ARMATURE = 1.0e10
PLATFORM_ANGULAR_VEL = 1.0  # rad/s around Z axis
HINGE_AMPLITUDE = 0.8  # radians
HINGE_FREQUENCY = 1.0  # Hz
NUM_CREATURES = 8


@wp.kernel
def set_kinematic_velocities(
    platform_qd_start: int,
    platform_wz: float,
    hinge_qd_starts: wp.array(dtype=wp.int32),
    hinge_q_starts: wp.array(dtype=wp.int32),
    sim_time: wp.array(dtype=wp.float32),
    amplitude: float,
    frequency: float,
    inv_dt: float,
    joint_q: wp.array(dtype=wp.float32),
    # output
    joint_qd: wp.array(dtype=wp.float32),
):
    """Write target velocities directly to joint_qd for kinematic joints.

    Two approaches are shown:
    - Platform: direct velocity control (constant angular velocity).
    - Hinges: position-based control. We compute the desired position, then
      derive the velocity as (target_pos - current_pos) / dt. This way the
      user only needs to specify where the joint should be, and the velocity
      is computed automatically.

    For free joints (6 DOF), position-based control would require quaternion
    difference -> angular velocity conversion. For scalar joints like revolute
    and prismatic, it's just a simple difference divided by dt.
    """
    i = wp.tid()

    if i == 0:
        # Platform: direct velocity control — constant angular velocity around Z
        # Free joint qd layout: [vx, vy, vz, wx, wy, wz]
        joint_qd[platform_qd_start + 0] = 0.0
        joint_qd[platform_qd_start + 1] = 0.0
        joint_qd[platform_qd_start + 2] = 0.0
        joint_qd[platform_qd_start + 3] = 0.0
        joint_qd[platform_qd_start + 4] = 0.0
        joint_qd[platform_qd_start + 5] = platform_wz

    # Hinge: position-based control via velocity = (target_pos - current_pos) / dt
    t = sim_time[0]
    phase = float(i) * 0.5
    omega = 2.0 * 3.14159265 * frequency
    target_pos = amplitude * wp.sin(omega * t + phase)
    current_pos = joint_q[hinge_q_starts[i]]
    joint_qd[hinge_qd_starts[i]] = (target_pos - current_pos) * inv_dt


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
        Dof = newton.ModelBuilder.JointDofConfig

        # -- Kinematic rotating platform (high armature on free joint DOFs) --
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
        platform_joint = builder.add_joint(
            joint_type=newton.JointType.FREE,
            parent=-1,
            child=platform_body,
            linear_axes=[
                Dof(axis=newton.Axis.X, armature=KINEMATIC_ARMATURE),
                Dof(axis=newton.Axis.Y, armature=KINEMATIC_ARMATURE),
                Dof(axis=newton.Axis.Z, armature=KINEMATIC_ARMATURE),
            ],
            angular_axes=[
                Dof(axis=newton.Axis.X, armature=KINEMATIC_ARMATURE),
                Dof(axis=newton.Axis.Y, armature=KINEMATIC_ARMATURE),
                Dof(axis=newton.Axis.Z, armature=KINEMATIC_ARMATURE),
            ],
        )
        q_start = builder.joint_q_start[platform_joint]
        builder.joint_q[q_start : q_start + 7] = list(builder.body_q[platform_body])
        builder.add_articulation([platform_joint], key="platform")

        # -- Two-link capsule creatures dropped on the platform --
        self.hinge_joints = []
        capsule_radius = 0.08
        capsule_half_length = 0.2
        drop_height = 1.0
        spawn_radius = 0.8
        capsule_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi * 0.5)

        for ci in range(NUM_CREATURES):
            angle = 2.0 * math.pi * ci / NUM_CREATURES
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

            # Second capsule (outer)
            offset = capsule_half_length * 2.0
            p1 = wp.vec3((spawn_radius + offset) * dx, (spawn_radius + offset) * dy, drop_height)
            b1 = builder.add_link(xform=wp.transform(p=p1, q=body_rot), armature=0.5)
            builder.add_shape_capsule(
                b1,
                radius=capsule_radius,
                half_height=capsule_half_length,
                xform=wp.transform(q=capsule_rot),
            )

            # Kinematic hinge (high armature on the revolute DOF)
            j_hinge = builder.add_joint_revolute(
                parent=b0,
                child=b1,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=wp.transform(p=wp.vec3(capsule_half_length, 0.0, 0.0)),
                child_xform=wp.transform(p=wp.vec3(-capsule_half_length, 0.0, 0.0)),
                armature=KINEMATIC_ARMATURE,
            )
            self.hinge_joints.append(j_hinge)

            # Set initial hinge angle to match the t=0 target (avoids velocity spike)
            phase = ci * 0.5
            hinge_q_start = builder.joint_q_start[j_hinge]
            builder.joint_q[hinge_q_start] = HINGE_AMPLITUDE * math.sin(phase)

            builder.add_articulation([j_free, j_hinge])

        builder.add_ground_plane()

        self.model = builder.finalize()

        # Collect joint offsets for the kernel
        qd_starts = self.model.joint_qd_start.numpy()
        q_starts = self.model.joint_q_start.numpy()
        self.platform_qd_start = int(qd_starts[platform_joint])
        hinge_qd_starts = [int(qd_starts[j]) for j in self.hinge_joints]
        hinge_q_starts = [int(q_starts[j]) for j in self.hinge_joints]
        self.hinge_qd_starts_wp = wp.array(hinge_qd_starts, dtype=wp.int32, device=self.model.device)
        self.hinge_q_starts_wp = wp.array(hinge_q_starts, dtype=wp.int32, device=self.model.device)

        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=200)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.model.device)

        self.viewer.set_model(self.model)

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

            # Write kinematic velocities directly to joint_qd before the solver step
            wp.launch(
                set_kinematic_velocities,
                dim=NUM_CREATURES,
                inputs=[
                    self.platform_qd_start,
                    PLATFORM_ANGULAR_VEL,
                    self.hinge_qd_starts_wp,
                    self.hinge_q_starts_wp,
                    self.sim_time_wp,
                    HINGE_AMPLITUDE,
                    HINGE_FREQUENCY,
                    1.0 / self.sim_dt,
                    self.state_0.joint_q,
                ],
                outputs=[self.state_0.joint_qd],
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
        for i in range(self.model.body_count):
            z = float(body_q[i][2])
            assert z > -0.5, f"Body {i} fell through ground: z={z:.4f}"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
