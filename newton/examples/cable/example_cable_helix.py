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
# Example Cable Helix
#
# Builds 6 helix-like cables that rise in Z. The cables are arranged
# side-by-side along Y and all share the same helical shape.
# - Left 3 cables: untwisted with increasing bend stiffness
# - Right 3 cables: pre-twisted with increasing bend stiffness
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def create_helix_geometry(
        self,
        pos: wp.vec3 | None = None,
        num_elements: int = 40,
        radius: float = 1.0,
        height: float = 6.0,
        turns: float = 2.0,
        twisting_angle: float = 0.0,
    ):
        """
        Create a helix-like geometry rising in Z with parallel transport quaternions.

        Args:
            pos: World position to offset the helix base.
            num_elements: Number of cable segments (edges).
            radius: Helix radius in the XY plane.
            height: Total rise along Z from start to end.
            turns: Number of helical turns from start to end.
            twisting_angle: Total twist around the local tangent, distributed along the cable.

        Returns:
            tuple: (points, edge_indices, quaternions)
        """
        if pos is None:
            pos = wp.vec3()

        # Parameter along the helix
        t_vals = np.linspace(0.0, 2.0 * np.pi * turns, num_elements + 1, dtype=np.float32)

        # Generate points along a helix: x = R cos t, y = R sin t, z = k t
        z_step = height / (num_elements)
        points = []
        for i, t in enumerate(t_vals):
            x = radius * np.cos(float(t))
            y = radius * np.sin(float(t))
            z = i * z_step
            points.append(pos + wp.vec3(x, y, z))

        # Edge indices for consecutive points
        edge_indices = []
        for i in range(num_elements):
            edge_indices.extend([i, i + 1])
        edge_indices = np.array(edge_indices, dtype=np.int32)

        # Build quaternions using parallel transport and cumulative twist
        edge_q = []
        if num_elements > 0:
            # Start with local capsule axis +Z
            local_axis = wp.vec3(0.0, 0.0, 1.0)
            from_direction = local_axis

            angle_step = twisting_angle / num_elements if num_elements > 0 else 0.0
            cumulative_twist = 0.0

            for i in range(num_elements):
                p0 = points[i]
                p1 = points[i + 1]

                to_direction = wp.normalize(p1 - p0)
                dq_dir = wp.quat_between_vectors(from_direction, to_direction)

                if i == 0:
                    base_quaternion = dq_dir
                else:
                    base_quaternion = wp.mul(dq_dir, edge_q[i - 1])

                if twisting_angle != 0.0:
                    cumulative_twist = cumulative_twist + angle_step
                    twist_rot = wp.quat_from_axis_angle(to_direction, cumulative_twist)
                    final_quaternion = wp.mul(twist_rot, base_quaternion)
                else:
                    final_quaternion = base_quaternion

                edge_q.append(final_quaternion)
                from_direction = to_direction

        return points, edge_indices, edge_q

    def __init__(self, viewer):
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 20
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Cable parameters
        self.num_elements = 50
        self.helix_radius = 1.5
        self.helix_height = 3.0
        self.helix_turns = 3.0
        self.cable_radius = 0.015
        self.total_twist = 2.0 * np.pi  # one full rotation of twist

        self.viewer = viewer

        builder = newton.ModelBuilder()

        # Set default material properties BEFORE adding any shapes
        builder.default_shape_cfg.ke = 1.0e4  # Contact stiffness
        builder.default_shape_cfg.kd = 1.0e-1  # Contact damping
        builder.default_shape_cfg.mu = 1.0e2  # Frictionless

        y_separation = 5.0
        num_cables = 6

        stretch_stiffness = 1.0e18
        initial_bend_stiffness = 5.0e2
        bend_stiffness_scale = 10.0

        # Build 6 helix cables side-by-side along Y
        for i in range(num_cables):
            if i < 3:
                # Left group: untwisted with increasing stiffness
                twist_angle = 0.0
                bend_stiffness = initial_bend_stiffness * (bend_stiffness_scale ** (i))
            else:
                # Right group: same total twist for all three, applied gradually per segment
                twist_angle = float(self.total_twist)
                bend_stiffness = initial_bend_stiffness * (bend_stiffness_scale ** (i - 3))

            y_pos = (i - (num_cables - 1) / 2.0) * y_separation
            start_pos = wp.vec3(0.0, y_pos, 0.5)

            points, edges, quats = self.create_helix_geometry(
                pos=start_pos,
                num_elements=self.num_elements,
                radius=self.helix_radius,
                height=self.helix_height,
                turns=self.helix_turns,
                twisting_angle=twist_angle,
            )

            rod_bodies, rod_joints = builder.add_rod_mesh(
                positions=points,
                quaternions=quats,
                radius=self.cable_radius,
                bend_stiffness=bend_stiffness,
                bend_damping=1.0e-4,
                stretch_stiffness=stretch_stiffness,
                stretch_damping=1.0e-4,
                key=f"helix_{i}",
            )

        # Add ground
        builder.add_ground_plane()

        # Color model
        builder.color()

        # Finalize model
        self.model = builder.finalize()

        # Solver
        self.solver = newton.solvers.SolverAVBD(
            self.model,
            iterations=self.sim_iterations,
            friction_epsilon=0.1,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

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
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test(self):
        pass


if __name__ == "__main__":
    viewer, args = newton.examples.init()

    example = Example(viewer)
    newton.examples.run(example)
