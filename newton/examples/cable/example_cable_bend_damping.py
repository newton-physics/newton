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
# Example Cable Bend Damping
#
# Demonstrates the effect of bend damping on cable dynamics.
# Shows 5 cables side-by-side with increasing damping values (from low to high).
# All cables have identical bend stiffness, differing only in damping coefficients.
# This illustrates how damping affects settling time, oscillation, and overshoot.
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def create_cable_geometry(self, pos: wp.vec3 | None = None, num_elements=10, length=10.0, twisting_angle=0.0):
        """Create a straight cable geometry with parallel-transported quaternions.

        Uses proper parallel transport to maintain a consistent reference frame along the cable.
        This ensures smooth rotational continuity and physically accurate twist distribution.

        Args:
            pos: Starting position of the cable (default: origin).
            num_elements: Number of cable segments (num_points = num_elements + 1).
            length: Total cable length.
            twisting_angle: Total twist in radians distributed uniformly along the cable.

        Returns:
            Tuple of (points, edge_indices, quaternions):
            - points: List of capsule center positions (num_elements + 1).
            - edge_indices: Flattened array of edge connectivity (2*num_elements).
            - quaternions: List of capsule orientations using parallel transport (num_elements).
        """
        if pos is None:
            pos = wp.vec3()

        # Create points along straight line in X direction
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
            # Capsule internal axis is +Z
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

    def __init__(self, viewer, args=None):
        # Setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 1
        self.update_step_interval = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        # Cable parameters
        self.num_elements = 50
        segment_length = 0.1
        self.cable_length = self.num_elements * segment_length
        cable_radius = 0.01

        # Identical stiffness for all cables; only damping varies
        bend_stiffness = 1.0e2
        stretch_stiffness = 1.0e6

        # Damping sweep (increasing)
        bend_damping_values = [1.0e-3, 1.0e-2, 1.0e-1, 1.0e0]
        self.num_cables = len(bend_damping_values)

        # Create builder for the simulation
        builder = newton.ModelBuilder()

        # Set default material properties before adding any shapes
        builder.default_shape_cfg.ke = 1.0e2  # Contact stiffness
        builder.default_shape_cfg.kd = 1.0e1  # Contact damping
        builder.default_shape_cfg.mu = 1.0  # Friction coefficient

        kinematic_body_indices = []
        self.cable_bodies_list: list[list[int]] = []

        y_separation = 0.5

        # Create 5 cables in a row along the y-axis, centered around origin
        for i, bend_damping in enumerate(bend_damping_values):
            # Center cables around origin: vary by y_separation
            y_pos = (i - (self.num_cables - 1) / 2.0) * y_separation

            # All cables have identical stiffness with varying damping
            # Center cable in X direction: start at -half_length
            start_x = -self.cable_length / 2.0

            cable_points, _, cable_edge_q = self.create_cable_geometry(
                pos=wp.vec3(start_x, y_pos, 4.0),
                num_elements=self.num_elements,
                length=self.cable_length,
                twisting_angle=0.0,
            )

            rod_bodies, _rod_joints = builder.add_rod(
                positions=cable_points,
                quaternions=cable_edge_q,
                radius=cable_radius,
                bend_stiffness=bend_stiffness,
                bend_damping=bend_damping,
                stretch_stiffness=stretch_stiffness,
                stretch_damping=1.0e-4,
                key=f"cable_damp_{i}",
            )

            # Fix the first body to make it kinematic
            first_body = rod_bodies[0]
            builder.body_mass[first_body] = 0.0
            builder.body_inv_mass[first_body] = 0.0
            kinematic_body_indices.append(first_body)

            # Store full body index list for each cable for robust testing
            self.cable_bodies_list.append(rod_bodies)

        # Create array of kinematic body indices
        self.kinematic_bodies = wp.array(kinematic_body_indices, dtype=wp.int32)

        # # Add ground plane
        # builder.add_ground_plane()

        # Color particles and rigid bodies for VBD solver
        builder.color()

        # Finalize model
        self.model = builder.finalize()

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
        # update_step_history = True
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces to the model
            self.viewer.apply_forces(self.state_0)

            # Decide whether to refresh solver history (anchors used for long-range damping)
            # and recompute contacts on this substep, using a configurable cadence.
            update_step_history = (substep % self.update_step_interval) == 0

            # Collide for contact detection
            if update_step_history:
                self.contacts = self.model.collide(self.state_0)

            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
                update_step_history=update_step_history,
            )

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
        """Test cable damping simulation for stability and correctness."""

        # Use instance variables for consistency with initialization
        segment_length = self.cable_length / self.num_elements

        # Check final state after viewer has run 100 frames (no additional simulation needed)
        if self.state_0.body_q is not None and self.state_0.body_qd is not None:
            body_positions = self.state_0.body_q.numpy()
            body_velocities = self.state_0.body_qd.numpy()

            # Test 1: Check for numerical stability (NaN/inf values and reasonable ranges)
            assert np.isfinite(body_positions).all(), "Non-finite values in body positions"
            assert np.isfinite(body_velocities).all(), "Non-finite values in body velocities"
            assert (np.abs(body_positions) < 1e3).all(), "Body positions too large (>1000)"
            assert (np.abs(body_velocities) < 5e2).all(), "Body velocities too large (>500)"

            # Test 2: Check cable connectivity (joint constraints)
            for cable_idx, rod_bodies in enumerate(self.cable_bodies_list):
                for segment in range(len(rod_bodies) - 1):
                    body1_idx = rod_bodies[segment]
                    body2_idx = rod_bodies[segment + 1]

                    pos1 = body_positions[body1_idx][:3]  # Extract translation part
                    pos2 = body_positions[body2_idx][:3]
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
            min_z = np.min(body_positions[:, 2])  # Z positions (Newton uses Z-up)
            assert min_z > -ground_tolerance, f"Cable penetrated ground too much: min_z = {min_z:.3f}"

            # Test 4: Check height range - cables should hang between initial height and ground
            initial_height = 4.0  # Cables start at z=4.0
            max_z = np.max(body_positions[:, 2])  # Z positions
            assert max_z <= initial_height + 0.1, (
                f"Cable rose above initial height: max_z = {max_z:.3f} > {initial_height + 0.1:.3f}"
            )
            assert min_z >= -ground_tolerance, f"Cable fell below ground: min_z = {min_z:.3f} < {-ground_tolerance:.3f}"

            # Test 5: Basic physics check - cables should hang down due to gravity
            # Compare first and last segment positions of first cable
            first_segment_z = body_positions[0, 2]
            last_segment_z = body_positions[self.num_elements - 1, 2]
            assert last_segment_z < first_segment_z, (
                f"Cable not hanging properly: last segment z={last_segment_z:.3f} should be < first segment z={first_segment_z:.3f}"
            )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
