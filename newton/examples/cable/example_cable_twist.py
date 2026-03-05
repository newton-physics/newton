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
# Example Cable Twist
#
# Demonstrates twist propagation along cables with dynamic spinning.
# Shows 3 cables side-by-side with zigzag paths and increasing bend stiffness.
# The first segment of each cable continuously spins, propagating twist along the cable.
# The zigzag routing introduces multiple 90-degree turns, demonstrating how twist
# is transported through cable joints and across bends.
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
    body_q0: wp.array(dtype=wp.transform),
    body_q1: wp.array(dtype=wp.transform),
):
    """Apply continuous twist to the first segment of each cable."""
    tid = wp.tid()
    body_id = body_indices[tid]

    t = body_q0[body_id]
    pos = wp.transform_get_translation(t)
    rot = wp.transform_get_rotation(t)

    # Local capsule axis is +Z in body frame; convert to world axis
    axis_world = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
    angle = twist_rates[tid] * dt
    dq = wp.quat_from_axis_angle(axis_world, angle)
    rot_new = wp.mul(dq, rot)

    T = wp.transform(pos, rot_new)
    body_q0[body_id] = T
    body_q1[body_id] = T


class Example:
    def create_cable_geometry_with_turns(
        self, pos: wp.vec3 | None = None, num_elements=16, length=6.4, twisting_angle=0.0
    ):
        """Create a zigzag cable route with parallel-transported quaternions.

        Generates a cable path with three sharp 90-degree turns lying on the XY-plane.
        Path order: +Y -> +X -> -Y -> +X. Uses parallel transport to maintain smooth
        reference frames across turns, with optional twist around the local capsule axis.

        Args:
            pos: Starting position of the cable (default: origin).
            num_elements: Number of cable segments (num_points = num_elements + 1).
            length: Total cable length.
            twisting_angle: Total twist in radians around capsule axis (0 = no twist).

        Returns:
            Tuple of (points, quaternions):
            - points: List of polyline points in world space (num_elements + 1).
            - quaternions: Per-segment orientations using parallel transport (num_elements).
        """
        if pos is None:
            pos = wp.vec3()

        if num_elements <= 0:
            raise ValueError("num_elements must be positive")

        # Calculate segment length from total length
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

        edge_q = newton.utils.create_parallel_transport_cable_quaternions(points, twist_total=float(twisting_angle))
        return points, edge_q

    def __init__(self, viewer, args=None):
        # Store viewer and arguments
        self.viewer = viewer
        self.args = args

        # Simulation cadence
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 2
        self.update_step_interval = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Cable parameters
        self.num_elements = 64  # More segments for smooth zigzag turns
        segment_length = 0.1
        self.cable_length = self.num_elements * segment_length
        cable_radius = 0.02

        # Stiffness sweep (increasing) for bend stiffness
        bend_stiffness_values = [1.0e1, 1.0e2, 1.0e3]
        stretch_stiffness = 1.0e6

        # All cables start untwisted, will be spun dynamically
        self.num_cables = len(bend_stiffness_values)

        # Create builder for the simulation
        builder = newton.ModelBuilder()

        # Set default material properties before adding any shapes
        builder.default_shape_cfg.ke = 1.0e4  # Contact stiffness
        builder.default_shape_cfg.kd = 1.0e-1  # Contact damping
        builder.default_shape_cfg.mu = 1.0e0  # Friction coefficient

        kinematic_body_indices = []
        self.cable_bodies_list = []
        self.first_bodies = []

        y_separation = 3.0

        # Create 3 cables in a row along the y-axis, centered around origin
        for i, bend_stiffness in enumerate(bend_stiffness_values):
            # Center cables around origin: vary by y_separation
            y_pos = (i - (self.num_cables - 1) / 2.0) * y_separation

            # All cables are untwisted with zigzag path and increasing stiffness
            # Cables start at ground level (z=0) to lay flat on ground
            start_pos = wp.vec3(-self.cable_length * 0.25, y_pos, cable_radius)

            cable_points, cable_edge_q = self.create_cable_geometry_with_turns(
                pos=start_pos,
                num_elements=self.num_elements,
                length=self.cable_length,
                twisting_angle=0.0,
            )

            rod_bodies, _rod_joints = builder.add_rod(
                positions=cable_points,
                quaternions=cable_edge_q,
                radius=cable_radius,
                bend_stiffness=bend_stiffness,
                bend_damping=1.0e-2,
                stretch_stiffness=stretch_stiffness,
                stretch_damping=1.0e-4,
                label=f"cable_{i}",
            )

            # Fix the first body to make it kinematic
            first_body = rod_bodies[0]
            builder.body_mass[first_body] = 0.0
            builder.body_inv_mass[first_body] = 0.0
            builder.body_inertia[first_body] = wp.mat33(0.0)
            builder.body_inv_inertia[first_body] = wp.mat33(0.0)
            kinematic_body_indices.append(first_body)

            # Store for twist application and testing
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

        self.solver = newton.solvers.SolverVBD(self.model, iterations=self.sim_iterations, friction_epsilon=0.1)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        # Twist rates for first segments (radians per second)
        twist_rates = np.full(len(kinematic_body_indices), 0.5, dtype=np.float32)
        self.first_twist_rates = wp.array(twist_rates, dtype=wp.float32)

        self.capture()

    def capture(self):
        """Capture simulation loop into a CUDA graph for optimal GPU performance."""
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """Execute all simulation substeps for one frame."""
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply continuous spin to first capsules
            wp.launch(
                kernel=spin_first_capsules_kernel,
                dim=self.kinematic_bodies.shape[0],
                inputs=[self.kinematic_bodies, self.first_twist_rates, self.sim_dt],
                outputs=[self.state_0.body_q, self.state_1.body_q],
            )

            # Apply forces to the model
            self.viewer.apply_forces(self.state_0)

            # Decide whether to refresh solver history (anchors used for long-range damping)
            # and recompute contacts on this substep, using a configurable cadence.
            update_step_history = (substep % self.update_step_interval) == 0

            # Collide for contact detection
            if update_step_history:
                self.model.collide(self.state_0, self.contacts)

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
        """Advance simulation by one frame."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the current simulation state to the viewer."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Test cable twist simulation for stability and correctness (called after simulation)."""

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
            for cable_idx, cable_bodies in enumerate(self.cable_bodies_list):
                for segment in range(len(cable_bodies) - 1):
                    body1_idx = cable_bodies[segment]
                    body2_idx = cable_bodies[segment + 1]

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
            # Cables should stay near ground (z~=0) since they start on the ground plane
            ground_tolerance = 0.5  # Larger tolerance for zigzag cables with dynamic spinning
            min_z = np.min(body_positions[:, 2])  # Z positions (Newton uses Z-up)
            assert min_z > -ground_tolerance, f"Cable penetrated ground too much: min_z = {min_z:.3f}"


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
