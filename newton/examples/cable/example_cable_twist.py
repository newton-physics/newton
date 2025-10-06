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
# Example Cable Twist Propagation
#
# Demonstrates twist propagation along cables routed through a zigzag path
# on the ground plane. The example spawns six cables arranged into two groups:
# - Group A (indices 0..2): no initial twist; first segment is continuously spun
# - Group B (indices 3..5): pre-applied total twist distributed along the cable
#
# The zigzag routing introduces multiple 90-degree turns, exercising twist
# transport across bends.
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def spin_first_capsules_kernel(
    body_indices: wp.array(dtype=wp.int32),
    twist_rates: wp.array(dtype=float),  # radians per second per body
    dt: float,
    body_q: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    body_id = body_indices[tid]

    t = body_q[body_id]
    pos = wp.transform_get_translation(t)
    rot = wp.transform_get_rotation(t)

    # Local capsule axis is +Z in body frame; convert to world axis
    axis_world = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
    angle = twist_rates[tid] * dt
    dq = wp.quat_from_axis_angle(axis_world, angle)
    rot_new = wp.mul(dq, rot)

    body_q[body_id] = wp.transform(pos, rot_new)


class Example:
    def create_cable_geometry_with_turns(
        self, pos: wp.vec3 | None = None, num_elements=16, length=6.4, twisting_angle=0.0
    ):
        """
        Create a zigzag cable route and per-segment orientations via parallel transport.

        The path lies on the XY-plane with three sharp 90-degree turns.
        Path order used here: +Y -> +X -> -Y -> +X.

        We use parallel transport to maintain smooth reference frames across
        turns, and (optionally) accumulate a total twist around the local
        capsule (+Z) axis along the cable.

        Args:
            pos: Starting position of the cable (default: origin)
            num_elements: Number of cable segments/elements
            length: Total length of the cable
            twisting_angle: Total twist around capsule axis in radians (0 = no twist)

        Returns:
            tuple: (points, edge_indices, quaternions)
        """
        if pos is None:
            pos = wp.vec3()

        # Calculate segment length from total length (consistent with cable_bend.py)
        segment_length = length / num_elements

        # Create zigzag path: +Y -> +X -> -Y -> +X (3 turns)
        num_points = num_elements + 1
        points = []

        segments_per_leg = num_elements // 4  # 4 legs in the zigzag

        for i in range(num_points):
            if i <= segments_per_leg:
                # Leg 1: go in +Y direction
                x = 0.0
                y = i * segment_length
            elif i <= 2 * segments_per_leg:
                # Leg 2: go in +X direction
                x = (i - segments_per_leg) * segment_length
                y = segments_per_leg * segment_length
            elif i <= 3 * segments_per_leg:
                # Leg 3: go in -Y direction
                x = segments_per_leg * segment_length
                y = segments_per_leg * segment_length - (i - 2 * segments_per_leg) * segment_length
            else:
                # Leg 4: go in +X direction
                x = segments_per_leg * segment_length + (i - 3 * segments_per_leg) * segment_length
                y = 0.0

            z = 0.0
            points.append(pos + wp.vec3(x, y, z))

        # Create edge indices connecting consecutive points
        edge_indices = []
        for i in range(num_elements):
            edge_indices.extend([i, i + 1])
        edge_indices = np.array(edge_indices, dtype=np.int32)

        # Create quaternions using parallel transport with cumulative twist distribution
        edge_q = []
        if num_elements > 0:
            # Capsule internal axis is +Z
            local_axis = wp.vec3(0.0, 0.0, 1.0)

            # Parallel transport: maintain smooth rotational continuity
            from_direction = local_axis

            # Distribute total twist cumulatively along the cable
            angle_step = twisting_angle / num_elements if num_elements > 0 else 0.0
            cumulative_twist = 0.0

            for i in range(num_elements):
                p0 = points[i]
                p1 = points[i + 1]

                # Current segment direction
                to_direction = wp.normalize(p1 - p0)

                # Directional transport
                dq_dir = wp.quat_between_vectors(from_direction, to_direction)

                if i == 0:
                    base_quaternion = dq_dir
                else:
                    base_quaternion = wp.mul(dq_dir, edge_q[i - 1])

                # Apply cumulative twist around the current segment direction
                if twisting_angle != 0.0:
                    cumulative_twist = cumulative_twist + angle_step
                    twist_rot = wp.quat_from_axis_angle(to_direction, cumulative_twist)
                    final_quaternion = wp.mul(twist_rot, base_quaternion)
                else:
                    final_quaternion = base_quaternion

                edge_q.append(final_quaternion)

                # Update transport direction
                from_direction = to_direction

        return points, edge_indices, edge_q

    def __init__(self, viewer):
        # Setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10  # More substeps for high stiffness stability
        self.sim_iterations = 10  # More iterations for high stiffness stability
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Cable parameters
        self.num_elements = 64  # More segments for smooth zigzag
        self.segment_length = 0.1
        self.cable_length = self.num_elements * self.segment_length  # Total length = 6.4

        self.cable_radius = 0.02
        self.bend_stiffness = 1.0e4  # Cable joint bend stiffness
        self.total_twist = 2.0 * np.pi  # 360 degrees total twist around capsule axis

        self.viewer = viewer

        # Create builder for the simulation
        builder = newton.ModelBuilder()

        # Set default material properties BEFORE adding any shapes
        builder.default_shape_cfg.ke = 1.0e4  # Contact stiffness (softer for cable flexibility)
        builder.default_shape_cfg.kd = 1.0e-1  # Contact damping
        builder.default_shape_cfg.mu = 1.0e1  # Friction coefficient

        kinematic_body_indices = []

        # Build 6 cables: same zigzag route pattern, starts spaced along Y
        y_separation = 3.0
        num_cables = 6

        stretch_stiffness = 1.0e6
        initial_bend_stiffness = 1.0e1
        bend_stiffness_scale = 10.0
        start_z_pos = 1.0
        self.cable_bodies_list = []
        self.first_bodies = []

        for i in range(num_cables):
            # Left group (0..2): untwisted, increasing stiffness left-to-right
            # Right group (3..5): pre-twisted, increasing stiffness left-to-right
            if i < 3:
                initial_twist = 0.0
                bend_stiffness = initial_bend_stiffness * (bend_stiffness_scale**i)
            else:
                initial_twist = self.total_twist
                bend_stiffness = initial_bend_stiffness * (bend_stiffness_scale ** (i - 3))

            # Place all cables parallel along X, spaced along Y
            y_pos = (i - (num_cables - 1) / 2.0) * y_separation
            start_pos = wp.vec3(-self.cable_length * 0.5, y_pos, start_z_pos)

            points, _, quats = self.create_cable_geometry_with_turns(
                pos=start_pos,
                num_elements=self.num_elements,
                length=self.cable_length,
                twisting_angle=initial_twist,
            )

            rod_bodies, rod_joints = builder.add_rod_mesh(
                positions=points,
                quaternions=quats,
                radius=self.cable_radius,
                bend_stiffness=bend_stiffness,
                bend_damping=1.0e-4,
                stretch_stiffness=stretch_stiffness,
                stretch_damping=1.0e-4,
                key=f"cable_{i}",
            )

            # Fix first body
            first_body = rod_bodies[0]
            builder.body_mass[first_body] = 0.0
            builder.body_inv_mass[first_body] = 0.0
            kinematic_body_indices.append(first_body)

            # Store for later adjustments
            self.cable_bodies_list.append(rod_bodies)
            self.first_bodies.append(first_body)

        # Create array of kinematic body indices
        self.kinematic_bodies = wp.array(kinematic_body_indices, dtype=wp.int32)

        # Add ground plane
        builder.add_ground_plane()

        # Color particles and rigid bodies for VBD solver
        builder.color()

        # Finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverAVBD(self.model, iterations=self.sim_iterations, friction_epsilon=0.1)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # Per-step kinematic twist rates (radians per second) for each first capsule
        twist_rates = np.full(len(kinematic_body_indices), 0.5, dtype=np.float32)
        self.first_twist_rates = wp.array(twist_rates, dtype=wp.float32)

        self.capture()

    def capture(self):
        """Put graph capture into its own function"""
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """Run simulation substeps with kinematic first-capsule rotation"""
        for _ in range(self.sim_substeps):
            # Apply continuous spin to first capsules
            wp.launch(
                kernel=spin_first_capsules_kernel,
                dim=self.kinematic_bodies.shape[0],
                inputs=[self.kinematic_bodies, self.first_twist_rates, self.sim_dt],
                outputs=[self.state_0.body_q],
            )

            # Apply forces from viewer (gravity, etc.)
            self.viewer.apply_forces(self.state_0)

            # Run collision detection and solver step
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance simulation by one frame"""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the simulation state"""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test(self):
        """Test cable twist physics for stability and correctness."""

        # Run simulation for analysis
        test_steps = 50

        for i in range(test_steps):
            self.simulate()

            # Test 1: Check for numerical stability (NaN/inf values and reasonable ranges)
            if self.state_0.body_q is not None and self.state_0.body_qd is not None:
                body_positions = self.state_0.body_q.numpy()
                body_velocities = self.state_0.body_qd.numpy()

                # Check for numerical stability issues
                assert np.isfinite(body_positions).all(), f"Non-finite values in body positions at step {i}"
                assert np.isfinite(body_velocities).all(), f"Non-finite values in body velocities at step {i}"

                # Check for reasonable value ranges (prevent explosive behavior)
                assert (np.abs(body_positions) < 1e3).all(), f"Body positions too large (>1000) at step {i}"
                assert (np.abs(body_velocities) < 5e2).all(), f"Body velocities too large (>500) at step {i}"

        # Test 2: Check cable connectivity (joint constraints)
        if self.state_0.body_q is not None:
            final_positions = self.state_0.body_q.numpy()

            # Test all cables
            for cable_idx, cable_bodies in enumerate(self.cable_bodies_list):
                cable_name = f"cable_{cable_idx}"

                for segment in range(len(cable_bodies) - 1):
                    body1_idx = cable_bodies[segment]
                    body2_idx = cable_bodies[segment + 1]

                    pos1 = final_positions[body1_idx][:3]  # Extract translation part
                    pos2 = final_positions[body2_idx][:3]
                    distance = np.linalg.norm(pos2 - pos1)

                    # Segments should be connected (joint constraint tolerance)
                    expected_distance = self.cable_length / self.num_elements
                    joint_tolerance = expected_distance * 0.15  # Allow 15% stretch max

                    assert distance < expected_distance + joint_tolerance, (
                        f"Cable {cable_name} segments {segment}-{segment + 1} too far apart: {distance:.3f} > {expected_distance + joint_tolerance:.3f}"
                    )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example and run
    example = Example(viewer)
    newton.examples.run(example)
