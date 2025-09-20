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
# Example Cable Bend
#
# Demonstrates cable bending behavior with different stiffness values and
# initial twist states. Shows 6 cables side-by-side:
# - Left 3: Untwisted cables with increasing stiffness (soft to medium to stiff)
# - Right 3: Pre-twisted cables (pi/2 rad) with increasing stiffness
# This comparison reveals how bend stiffness and initial twist affect
# cable dynamics, settling behavior, and physical realism.
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def create_cable_geometry(self, pos: wp.vec3 | None = None, num_elements=10, length=10.0, twisting_angle=0.0):
        """
        Create cable geometry with points, edge indices, and quaternions using parallel transport.

        Uses proper parallel transport to maintain a consistent reference frame along the cable.
        This ensures smooth rotational continuity and physically accurate twist distribution.

        Args:
            pos: The starting position of the cable.
            num_elements: Number of cable elements (edges)
            length: Total length of the cable
            twisting_angle: Total twist angle in radians distributed along the cable

        Returns:
            tuple: (points, edge_indices, quaternions)
        """
        if pos is None:
            pos = wp.vec3()
        # Create points along straight line in x direction.
        num_points = num_elements + 1
        points = []

        for i in range(num_points):
            t = i / num_elements
            x = length * t
            y = 0.0
            z = 0.0
            points.append(pos + wp.vec3(x, y, z))

        # Create edge indices connecting consecutive points
        edge_indices = []
        for i in range(num_elements):
            vertex_0 = i  # First vertex of edge
            vertex_1 = i + 1  # Second vertex of edge
            edge_indices.extend([vertex_0, vertex_1])

        edge_indices = np.array(edge_indices, dtype=np.int32)

        # Create quaternions for each edge using parallel transport
        edge_q = []
        if num_elements > 0:
            # Capsule internal axis is +Z (from capsule code: "internally capsule axis is always +Z")
            local_axis = wp.vec3(0.0, 0.0, 1.0)

            # Parallel transport: maintain smooth rotational continuity along cable
            from_direction = local_axis  # Start with local Z-axis

            # The total twist will be distributed along the cable
            angle_step = twisting_angle / num_elements if num_elements > 0 else 0.0

            for i in range(num_elements):
                p0 = points[i]
                p1 = points[i + 1]

                # Current segment direction
                to_direction = wp.normalize(p1 - p0)

                # Compute rotation from previous direction to current direction
                # This maintains smooth rotational continuity (parallel transport)
                dq = wp.quat_between_vectors(from_direction, to_direction)

                if i == 0:
                    # First segment: just the directional alignment
                    base_quaternion = dq
                else:
                    # Subsequent segments: multiply with previous quaternion (parallel transport)
                    base_quaternion = wp.mul(dq, edge_q[i - 1])

                # Apply incremental twist around the current segment direction
                if twisting_angle != 0.0:
                    twist_increment = angle_step
                    twist_rot = wp.quat_from_axis_angle(to_direction, twist_increment)
                    final_quaternion = wp.mul(twist_rot, base_quaternion)
                else:
                    final_quaternion = base_quaternion

                edge_q.append(final_quaternion)

                # Update for next iteration (parallel transport)
                from_direction = to_direction

        return points, edge_indices, edge_q

    def __init__(self, viewer):
        # Setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 20
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Setup simulation parameters
        self.num_elements = 50
        self.cable_length = 5.0
        cable_radius = 0.01

        initial_stiffness = 1.0e-1
        stiffness_scale = 100

        self.viewer = viewer

        # Create builder for the simulation
        builder = newton.ModelBuilder()

        kinematic_body_indices = []

        y_separation = 2.0
        self.num_cables = 6

        # Create 6 cables in a row along the y-axis, centered around origin
        stiffness = initial_stiffness
        for i in range(self.num_cables):
            # Center cables around origin: -5, -3, -1, 1, 3, 5
            y_pos = (i - (self.num_cables - 1) / 2) * y_separation

            # First 3 are untwisted, next 3 are twisted
            if i < 3:
                initial_twist = 0.0
                stiffness = initial_stiffness * stiffness_scale**i

            else:
                initial_twist = np.pi / 2
                stiffness = initial_stiffness * stiffness_scale ** (i - 3)

            # Center cable in X direction: start at -half_length
            start_x = -self.cable_length / 2.0

            cable_points, _, cable_edge_q = self.create_cable_geometry(
                pos=wp.vec3(start_x, y_pos, 2.0),
                num_elements=self.num_elements,
                length=self.cable_length,
                twisting_angle=initial_twist,
            )

            rod_bodies, rod_joints = builder.add_rod_mesh(
                positions=cable_points,
                quaternions=cable_edge_q,
                radius=cable_radius,
                stiffness=stiffness,
                damping=0.0,
                key=f"cable_{i}",
            )

            # Fix the first body to make it kinematic
            first_body = rod_bodies[0]
            builder.body_mass[first_body] = 0.0
            builder.body_inv_mass[first_body] = 0.0
            kinematic_body_indices.append(first_body)

            stiffness *= stiffness_scale

        # Create array of kinematic body indices
        self.kinematic_bodies = wp.array(kinematic_body_indices, dtype=wp.int32)

        # Add ground plane
        builder.add_ground_plane()

        # Color particles and rigid bodies for VBD solver
        builder.color()

        # Finalize model
        self.model = builder.finalize()

        # Set collision contact parameters
        self.model.soft_contact_ke = 1.0e2  # Contact spring stiffness
        self.model.soft_contact_kd = 1.0e1  # Contact damping
        self.model.soft_contact_mu = 1.0  # Contact friction coefficient
        self.model.soft_contact_restitution = 0.0  # Restitution

        self.solver = newton.solvers.SolverVBD(self.model, iterations=self.sim_iterations, friction_epsilon=0.1)

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
            self.state_0.clear_forces()

            # Apply forces to the model (needed for gravity and external forces)
            self.viewer.apply_forces(self.state_0)

            # Collide for contact detection
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states
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
        """Test cable bending simulation for stability and correctness."""

        # Use instance variables for consistency with initialization
        segment_length = self.cable_length / self.num_elements

        # Run simulation for a reasonable number of steps
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
                assert (np.abs(body_velocities) < 1e2).all(), f"Body velocities too large (>100) at step {i}"

        # Test 2: Check cable connectivity (joint constraints)
        if self.state_0.body_q is not None:
            final_positions = self.state_0.body_q.numpy()
            for cable_idx in range(self.num_cables):
                start_body = cable_idx * self.num_elements
                for segment in range(self.num_elements - 1):
                    body1_idx = start_body + segment
                    body2_idx = start_body + segment + 1

                    pos1 = final_positions[body1_idx][:3]  # Extract translation part
                    pos2 = final_positions[body2_idx][:3]
                    distance = np.linalg.norm(pos2 - pos1)

                    # Segments should be connected (joint constraint tolerance)
                    expected_distance = segment_length
                    joint_tolerance = expected_distance * 0.1  # Allow 10% stretch max
                    assert distance < expected_distance + joint_tolerance, (
                        f"Cable {cable_idx} segments {segment}-{segment + 1} too far apart: {distance:.3f} > {expected_distance + joint_tolerance:.3f}"
                    )

            # Test 3: Check ground interaction
            # Cables should not penetrate ground significantly (z=0)
            ground_tolerance = 0.05  # Small penetration allowed due to penalty-based contacts
            min_z = np.min(final_positions[:, 2])  # Z positions (Newton uses Z-up)
            assert min_z > -ground_tolerance, f"Cable penetrated ground too much: min_z = {min_z:.3f}"

            # Test 4: Check height range - cables should hang between initial height and ground
            initial_height = 2.0  # Cables start at z=2.0
            max_z = np.max(final_positions[:, 2])  # Z positions
            assert max_z <= initial_height + 0.1, (
                f"Cable rose above initial height: max_z = {max_z:.3f} > {initial_height + 0.1:.3f}"
            )
            assert min_z >= -ground_tolerance, f"Cable fell below ground: min_z = {min_z:.3f} < {-ground_tolerance:.3f}"

            # Test 5: Basic physics check - cables should hang down due to gravity
            # Compare first and last segment positions of first cable
            first_segment_z = final_positions[0, 2]
            last_segment_z = final_positions[self.num_elements - 1, 2]
            assert last_segment_z < first_segment_z, (
                f"Cable not hanging properly: last segment z={last_segment_z:.3f} should be < first segment z={first_segment_z:.3f}"
            )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example and run
    example = Example(viewer)

    newton.examples.run(example)
