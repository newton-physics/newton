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
# Shows 4 cables side-by-side with increasing damping values (from low to high).
# All cables have identical bend stiffness, differing only in damping coefficients.
# This illustrates how damping affects settling time, oscillation, and overshoot.
#
# It also demonstrates the use of `solver.set_rigid_history_update()` to maintain
# smooth damping behavior by controlling the history update frequency across substeps,
# even with few iterations.
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args=None):
        # Setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 1
        self.update_step_interval = 2
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

        # Create cables in a row along the y-axis, centered around origin
        for i, bend_damping in enumerate(bend_damping_values):
            # Center cables around origin: vary by y_separation
            y_pos = (i - (self.num_cables - 1) / 2.0) * y_separation

            # All cables have identical stiffness with varying damping
            # Center cable in X direction: start at -half_length
            start_x = -self.cable_length / 2.0

            cable_points, cable_edge_q = newton.utils.create_straight_cable_points_and_quaternions(
                start=wp.vec3(start_x, y_pos, 4.0),
                direction=wp.vec3(1.0, 0.0, 0.0),
                length=float(self.cable_length),
                num_segments=int(self.num_elements),
                twist_total=0.0,
            )

            rod_bodies, _rod_joints = builder.add_rod(
                positions=cable_points,
                quaternions=cable_edge_q,
                radius=cable_radius,
                bend_stiffness=bend_stiffness,
                bend_damping=bend_damping,
                stretch_stiffness=stretch_stiffness,
                stretch_damping=1.0e-4,
                label=f"cable_damp_{i}",
            )

            # Fix the first body to make it kinematic
            first_body = rod_bodies[0]
            builder.body_mass[first_body] = 0.0
            builder.body_inv_mass[first_body] = 0.0
            builder.body_inertia[first_body] = wp.mat33(0.0)
            builder.body_inv_inertia[first_body] = wp.mat33(0.0)
            kinematic_body_indices.append(first_body)

            # Store full body index list for each cable for robust testing
            self.cable_bodies_list.append(rod_bodies)

        # Create array of kinematic body indices
        self.kinematic_bodies = wp.array(kinematic_body_indices, dtype=wp.int32)

        # Add ground plane
        builder.add_ground_plane()

        # Color particles and rigid bodies for VBD solver
        builder.color()

        # Finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverVBD(self.model, iterations=self.sim_iterations, friction_epsilon=0.1)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Create collision pipeline (default)
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        if self.solver.device.is_cuda:
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
                self.collision_pipeline.collide(self.state_0, self.contacts)

            self.solver.set_rigid_history_update(update_step_history)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
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

    def test_final(self):
        """Test cable damping simulation for stability and correctness (called after simulation)."""

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

            # Test 5: Basic physics check - cables should sag under gravity.
            # Compare the anchored end and free end of the first cable.
            first_cable = self.cable_bodies_list[0]
            first_segment_z = body_positions[first_cable[0], 2]
            last_segment_z = body_positions[first_cable[-1], 2]
            assert last_segment_z < first_segment_z, (
                f"Cable not hanging properly: last segment z={last_segment_z:.3f} should be < first segment z={first_segment_z:.3f}"
            )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
