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
# Demonstrates cable behavior with helical geometry and varying stiffness.
# Creates 3 helix-shaped cables arranged side-by-side along the Y axis,
# each rising vertically with circular cross-section in the XY plane.
# All cables share the same helical shape but have increasing bend stiffness,
# showing how material properties affect settling and deformation.
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
        """Create a helix-shaped cable geometry rising along the Z axis.

        Generates a helical path with parallel-transported quaternions for physically
        consistent capsule orientations. The helix has a circular cross-section in the
        XY plane and rises linearly in Z.

        Args:
            pos: World position offset for the helix base (default: origin).
            num_elements: Number of cable segments (num_points = num_elements + 1).
            radius: Helix radius in the XY plane.
            height: Total vertical rise along Z from start to end.
            turns: Number of complete helical turns (2*pi radians per turn).
            twisting_angle: Total twist in radians around local tangent (distributed uniformly).

        Returns:
            Tuple of (points, edge_indices, quaternions):
            - points: List of capsule center positions (num_elements + 1).
            - edge_indices: Flattened array of edge connectivity (2*num_elements).
            - quaternions: List of capsule orientations using parallel transport (num_elements).
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

        # Build quaternions using parallel transport and incremental twist
        edge_q = []
        if num_elements > 0:
            # Capsule internal axis is +Z
            local_axis = wp.vec3(0.0, 0.0, 1.0)
            from_direction = local_axis

            angle_step = twisting_angle / num_elements if num_elements > 0 else 0.0

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
                    twist_rot = wp.quat_from_axis_angle(to_direction, angle_step)
                    final_quaternion = wp.mul(twist_rot, base_quaternion)
                else:
                    final_quaternion = base_quaternion

                edge_q.append(final_quaternion)
                from_direction = to_direction

        return points, edge_indices, edge_q

    def __init__(self, viewer, args=None):
        # Store viewer and arguments
        self.viewer = viewer
        self.args = args

        # Simulation cadence
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Helix geometry parameters
        self.num_elements = 50
        self.helix_radius = 1.5
        self.helix_height = 3.0
        self.helix_turns = 3.0
        self.cable_radius = 0.015

        builder = newton.ModelBuilder()

        # Set default material properties before adding any shapes
        builder.default_shape_cfg.ke = 1.0e4  # Contact stiffness (used by plane)
        builder.default_shape_cfg.kd = 1.0e-1  # Contact damping
        builder.default_shape_cfg.mu = 1.0e2  # Friction coefficient

        # Stiffness sweep for cables (increasing)
        bend_stiffness_values = [5.0e2, 5.0e3, 5.0e4]
        num_cables = len(bend_stiffness_values)
        self.cable_bodies_list: list[list[int]] = []
        y_separation = 5.0
        stretch_stiffness = 1.0e6

        # Create 3 helix cables side-by-side along the Y axis
        for i, bend_stiffness in enumerate(bend_stiffness_values):
            # Center cables around origin
            y_pos = (i - (num_cables - 1) / 2.0) * y_separation
            start_pos = wp.vec3(0.0, y_pos, 0.5)

            points, _edges, quats = self.create_helix_geometry(
                pos=start_pos,
                num_elements=self.num_elements,
                radius=self.helix_radius,
                height=self.helix_height,
                turns=self.helix_turns,
                twisting_angle=0.0,  # No initial twist
            )

            rod_bodies, _rod_joints = builder.add_rod(
                positions=points,
                quaternions=quats,
                radius=self.cable_radius,
                bend_stiffness=bend_stiffness,
                bend_damping=1.0e-2,
                stretch_stiffness=stretch_stiffness,
                stretch_damping=1.0e-4,
                key=f"helix_{i}",
            )

            # Record the body indices for this cable for robust testing
            self.cable_bodies_list.append(rod_bodies)

        # Add ground plane
        builder.add_ground_plane()

        # Color bodies for VBD solver
        builder.color()

        # Finalize model
        self.model = builder.finalize()

        # Create VBD solver for rigid body simulation
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            friction_epsilon=0.1,
        )

        # Initialize states and contacts
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        self.viewer.set_model(self.model)

        # Optional capture for CUDA
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
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces to the model
            self.viewer.apply_forces(self.state_0)

            # Collide for contact detection
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance simulation by one frame (either via CUDA graph or direct execution)."""
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

    def test(self):
        """Test helix cable simulation for stability and correctness."""
        # Helix dimensions for physical bounds checking
        initial_height = 0.5  # Helices start at z=0.5
        helix_max_height = initial_height + self.helix_height  # Top of helix
        ground_tolerance = 0.1  # Allow some ground penetration (soft contacts)

        # Check final state after viewer has run 100 frames (no additional simulation needed)
        if self.state_0.body_q is not None and self.state_0.body_qd is not None:
            body_positions = self.state_0.body_q.numpy()
            body_velocities = self.state_0.body_qd.numpy()

            # Test 1: Check for numerical stability
            assert np.isfinite(body_positions).all(), "Non-finite positions"
            assert np.isfinite(body_velocities).all(), "Non-finite velocities"

            # Test 2: Check connectivity - cables should maintain joint distances
            joint_tolerance = 0.05  # Allow 5% stretch max for helical geometry

            # Calculate expected segment length from initial helix geometry
            # For a helix with n turns over height h and radius r, the arc length per segment is:
            # s ~= sqrt((2*pi*r * turns / n)^2 + (h / n)^2)
            turns_per_segment = self.helix_turns / self.num_elements
            horizontal_arc = 2.0 * np.pi * self.helix_radius * turns_per_segment
            vertical_rise = self.helix_height / self.num_elements
            expected_segment_length = np.sqrt(horizontal_arc**2 + vertical_rise**2)

            for cable_idx, rod_bodies in enumerate(self.cable_bodies_list):
                for seg_idx in range(len(rod_bodies) - 1):
                    b0 = rod_bodies[seg_idx]
                    b1 = rod_bodies[seg_idx + 1]
                    p0 = body_positions[b0, :3]
                    p1 = body_positions[b1, :3]
                    distance = np.linalg.norm(p1 - p0)

                    assert distance < expected_segment_length * (1.0 + joint_tolerance), (
                        f"Cable {cable_idx} segment {seg_idx} overstretched: "
                        f"distance={distance:.3f} > expected {expected_segment_length:.3f}"
                    )

            # Test 3: Check ground interaction - no excessive penetration
            z_positions = body_positions[:, 2]
            min_z = np.min(z_positions)

            assert min_z > -ground_tolerance, (
                f"Cables penetrated ground too much: min_z={min_z:.3f} < {-ground_tolerance:.3f}"
            )

            # Test 4: Check reasonable height range (helix should settle but not explode)
            max_z = np.max(z_positions)
            assert max_z < helix_max_height + 1.0, (
                f"Cables too high: max_z={max_z:.3f} > expected {helix_max_height + 1.0:.3f}"
            )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
