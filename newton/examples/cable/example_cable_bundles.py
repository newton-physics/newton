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
# Example Cable Bundle
#
# Demonstrates a bundle of cables arranged in a circular cross-section,
# forming a larger cable structure. The bundle contains multiple smaller
# cables that interact with each other and the ground.
#
# Features:
# - Multiple cables arranged in circular pattern (cross-section)
# - One cable in the center, others arranged in rings around it
# - Proper initial spacing to avoid penetration
# - Fixed attachment at one end
# - Free-hanging at the other end
# - Rubber band rings that wrap around the bundle to hold cables together
# - Separate material properties for:
#   * Cable-to-cable: high friction for bundling behavior
#   * Cable-to-ground: stiffer with less friction
#   * Rings: lower friction, more flexible, thinner
# - Interactive control with viewer forces
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def apply_tip_force_kernel(
    body_indices: wp.array(dtype=wp.int32),  # type: ignore
    force_direction: wp.vec3,
    force_magnitude: float,
    body_f: wp.array(dtype=wp.spatial_vector),  # type: ignore
):
    """Apply a force to the tip bodies of the cable bundle."""
    tid = wp.tid()
    body_id = body_indices[tid]

    # Apply force (spatial_vector takes force and torque as two vec3 arguments)
    force_vec = force_direction * force_magnitude
    torque_vec = wp.vec3(0.0, 0.0, 0.0)
    # type: ignore comments suppress linter warnings about wp.spatial_vector usage
    f = wp.spatial_vector(force_vec, torque_vec)  # type: ignore
    body_f[body_id] = body_f[body_id] + f


class Example:
    def create_cable_geometry(self, pos=None, num_elements=20, length=4.0, twisting_angle=0.0, droop=0.0, closed=False):
        """
        Create cable geometry with points, edge indices, and quaternions using parallel transport.

        Args:
            pos: The starting position of the cable.
            num_elements: Number of cable elements (edges)
            length: Total length of the cable
            twisting_angle: Total twist angle in radians distributed along the cable
            droop: Amount of initial droop/sag in the middle (0 = straight)
            closed: If True, connect last point back to first (forms a loop/ring)

        Returns:
            tuple: (points, edge_indices, quaternions)
        """
        if pos is None:
            pos = wp.vec3()

        # For closed cables, we need num_elements + 1 points (last point = first point position repeated)
        num_points = num_elements + 1
        points = []

        for i in range(num_points):
            if closed and i == num_elements:
                # For closed loop, duplicate first point at the end
                t = 0.0
            else:
                t = i / num_elements

            x = length * t
            y = 0.0
            # Add slight droop in the middle for more natural initial state
            z = -droop * np.sin(np.pi * t)
            points.append(pos + wp.vec3(x, y, z))

        # Create edge indices connecting consecutive points
        edge_indices = []
        for i in range(num_elements):
            if closed and i == num_elements - 1:
                # Last edge connects to first point
                edge_indices.extend([i, num_elements])
            else:
                edge_indices.extend([i, i + 1])
        edge_indices = np.array(edge_indices, dtype=np.int32)

        # Create quaternions for each edge using parallel transport
        edge_q = []
        if num_elements > 0:
            local_axis = wp.vec3(0.0, 0.0, 1.0)
            from_direction = local_axis
            angle_step = twisting_angle / num_elements if num_elements > 0 else 0.0

            for i in range(num_elements):
                p0 = points[i]
                p1 = points[i + 1] if i < num_elements - 1 or not closed else points[num_elements]
                to_direction = wp.normalize(p1 - p0)
                dq = wp.quat_between_vectors(from_direction, to_direction)

                if i == 0:
                    base_quaternion = dq
                else:
                    base_quaternion = wp.mul(dq, edge_q[i - 1])

                if twisting_angle != 0.0:
                    twist_increment = angle_step
                    twist_rot = wp.quat_from_axis_angle(to_direction, twist_increment)
                    final_quaternion = wp.mul(twist_rot, base_quaternion)
                else:
                    final_quaternion = base_quaternion

                edge_q.append(final_quaternion)
                from_direction = to_direction

        return points, edge_indices, edge_q

    def create_ring_geometry(self, center_pos, normal_axis="x", radius=0.1, num_segments=24):
        """
        Create a circular ring (like a rubber band) around the bundle.

        This creates a CLOSED cable that forms a circle. The last segment connects
        back to the first point, forming a complete loop.

        Args:
            center_pos: Center position of the ring (wp.vec3)
            normal_axis: Axis perpendicular to ring plane ('x', 'y', or 'z')
            radius: Radius of the ring
            num_segments: Number of segments in the ring (more = smoother circle)

        Returns:
            tuple: (points, edge_indices, quaternions)
                  - points: num_segments + 1 points (last point = first point position)
                  - edge_indices: num_segments edges (last edge connects back to start)
                  - quaternions: num_segments quaternions (one per edge)
        """
        points = []

        # Create circular points in the appropriate plane
        # We need num_segments + 1 points to close the ring (duplicate first point at end)
        for i in range(num_segments + 1):
            # Use modulo to wrap last point back to first position
            angle = 2.0 * np.pi * (i % num_segments) / num_segments

            if normal_axis == "x":
                # Ring in Y-Z plane (perpendicular to X axis)
                y = radius * np.cos(angle)
                z = radius * np.sin(angle)
                point = center_pos + wp.vec3(0.0, y, z)
            elif normal_axis == "y":
                # Ring in X-Z plane (perpendicular to Y axis)
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)
                point = center_pos + wp.vec3(x, 0.0, z)
            else:  # 'z'
                # Ring in X-Y plane (perpendicular to Z axis)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                point = center_pos + wp.vec3(x, y, 0.0)

            points.append(point)

        # Create edge indices connecting consecutive points (including closing edge)
        edge_indices = []
        for i in range(num_segments):
            edge_indices.extend([i, i + 1])  # Last edge (num_segments-1, num_segments) closes the loop
        edge_indices = np.array(edge_indices, dtype=np.int32)

        # Create quaternions using parallel transport around the ring
        edge_q = []
        local_axis = wp.vec3(0.0, 0.0, 1.0)
        from_direction = local_axis

        for i in range(num_segments):
            p0 = points[i]
            p1 = points[i + 1]  # Last iteration: p1 is the duplicated first point

            to_direction = wp.normalize(p1 - p0)
            dq = wp.quat_between_vectors(from_direction, to_direction)

            if i == 0:
                base_quaternion = dq
            else:
                base_quaternion = wp.mul(dq, edge_q[i - 1])

            edge_q.append(base_quaternion)
            from_direction = to_direction

        return points, edge_indices, edge_q

    def create_bundle_positions(self, num_cables, cable_radius, gap_multiplier):
        """
        Create positions for cables in a circular bundle pattern with no initial overlap.

        The bundle cross-section is in the Y-Z plane (perpendicular to cable direction +X).

        Args:
            num_cables: Total number of cables in the bundle
            cable_radius: Radius of individual cables
            gap_multiplier: Gap multiplier relative to touching distance
                           - 1.0 = cables touching (no extra gap)
                           - 1.1 = 10% extra gap (2mm for 20mm radius cables)
                           - 1.5 = 50% extra gap (10mm for 20mm radius cables)

        Returns:
            List of (y, z) offset positions for each cable in the bundle cross-section
        """
        positions = []

        if num_cables == 1:
            # Just center cable
            positions.append((0.0, 0.0))
        else:
            # One cable in center
            positions.append((0.0, 0.0))

            # Arrange remaining cables in rings
            remaining = num_cables - 1

            # Minimum center-to-center distance calculation:
            # Base distance (touching) = 2 * cable_radius
            # With multiplier: distance = 2 * cable_radius * gap_multiplier
            # Example: multiplier=1.1 gives 10% extra space
            min_center_distance = 2.0 * cable_radius * gap_multiplier

            if remaining <= 6:
                # Single ring of cables
                # Place ring at minimum safe distance from center
                ring_radius = min_center_distance

                # Also verify ring cables don't overlap with each other
                # For N cables on a circle of radius R, chord distance = 2*R*sin(π/N)
                # We need: chord_distance >= min_center_distance
                if remaining > 1:
                    chord_distance = 2.0 * ring_radius * np.sin(np.pi / remaining)
                    if chord_distance < min_center_distance:
                        # Increase ring radius to prevent ring cable overlap
                        ring_radius = min_center_distance / (2.0 * np.sin(np.pi / remaining))

                for i in range(remaining):
                    angle = 2.0 * np.pi * i / remaining
                    y = ring_radius * np.cos(angle)
                    z = ring_radius * np.sin(angle)
                    positions.append((y, z))
            else:
                # Two rings: inner ring with 6 cables, outer ring with rest
                inner_count = 6
                outer_count = remaining - inner_count

                # Inner ring - minimum safe distance from center
                inner_radius = min_center_distance
                # Verify inner ring cables don't overlap with each other
                inner_chord = 2.0 * inner_radius * np.sin(np.pi / inner_count)
                if inner_chord < min_center_distance:
                    inner_radius = min_center_distance / (2.0 * np.sin(np.pi / inner_count))

                for i in range(inner_count):
                    angle = 2.0 * np.pi * i / inner_count
                    y = inner_radius * np.cos(angle)
                    z = inner_radius * np.sin(angle)
                    positions.append((y, z))

                # Outer ring - must clear both center and inner ring
                # Distance from center to outer ring
                outer_radius = inner_radius + min_center_distance
                # Verify outer ring cables don't overlap with each other
                outer_chord = 2.0 * outer_radius * np.sin(np.pi / outer_count)
                if outer_chord < min_center_distance:
                    outer_radius = min_center_distance / (2.0 * np.sin(np.pi / outer_count))

                for i in range(outer_count):
                    angle = 2.0 * np.pi * i / outer_count
                    y = outer_radius * np.cos(angle)
                    z = outer_radius * np.sin(angle)
                    positions.append((y, z))

        return positions

    def __init__(self, viewer):
        """
        Initialize cable bundle COMPARISON with THREE bundles side-by-side.

        Creates three identical bundles with only different ring counts:
        - Bundle 0 (y=-0.5): 0 rings
        - Bundle 1 (y=0.0): 2 rings
        - Bundle 2 (y=+0.5): 5 rings
        """
        # Fixed parameters for all bundles
        num_cables = 7
        cable_gap_multiplier = 1.1
        ring_gap_multiplier = 1.0
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 20
        self.sim_iterations = 20  # More iterations for stability with rings
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Bundle parameters
        self.num_cables = num_cables
        self.cable_gap_multiplier = cable_gap_multiplier
        self.num_elements = 40  # Elements per cable
        self.cable_length = 4.0
        self.cable_radius = 0.02  # Individual cable radius (20mm)

        # Physics parameters
        self.bend_stiffness = 1.0e2
        self.stretch_stiffness = 1.0e9

        self.viewer = viewer

        # Create builder for the simulation
        builder = newton.ModelBuilder()

        # Create separate shape configurations for cables and ground
        # Cable-to-cable properties (softer, more friction for bundling behavior)
        cable_shape_cfg = newton.ModelBuilder.ShapeConfig()
        cable_shape_cfg.ke = 1.0e3  # Contact stiffness
        cable_shape_cfg.kd = 1.0e-1  # Contact damping
        cable_shape_cfg.mu = 0.8  # Higher friction for cable-to-cable grip

        # Ground properties (stiffer, less friction)
        ground_shape_cfg = newton.ModelBuilder.ShapeConfig()
        ground_shape_cfg.ke = 1.0e5  # Stiffer ground
        ground_shape_cfg.kd = 1.0e-1  # More damping on ground
        ground_shape_cfg.mu = 0.3  # Less friction on ground

        # Set default to cable properties (will be used for cables)
        builder.default_shape_cfg = cable_shape_cfg

        # Get bundle positions (circular cross-section arrangement with proper spacing)
        bundle_positions = self.create_bundle_positions(self.num_cables, self.cable_radius, self.cable_gap_multiplier)

        # Fixed body indices (first segment of each cable)
        kinematic_body_indices = []
        # Tip body indices (last segment of each cable for control)
        self.tip_body_indices = []

        # Store all cable bodies for visualization/analysis
        self.cable_bodies_list = []

        #  ==================================================================
        # CREATE THREE BUNDLES SIDE-BY-SIDE WITH DIFFERENT RING COUNTS
        # ==================================================================

        bundle_configs = [
            {"y_offset": -1.0, "num_rings": 0, "label": "No Rings"},
            {"y_offset": 0.0, "num_rings": 2, "label": "2 Rings"},
            {"y_offset": 1.0, "num_rings": 5, "label": "5 Rings"},
        ]

        for bundle_id, config in enumerate(bundle_configs):
            # Starting position for THIS bundle
            start_x = 0.0
            start_y = config["y_offset"]  # Side-by-side positioning
            start_z = 2.0

            add_rings = config["num_rings"] > 0
            num_rings_for_bundle = config["num_rings"]

            # Create each cable in THIS bundle
            for i in range(self.num_cables):
                # Get offset for this cable in the bundle cross-section
                # bundle_positions are in a 2D plane, but cables run along +X direction
                # So we need to apply offsets in the Y-Z plane (perpendicular to cable direction)
                offset_y, offset_z = bundle_positions[i]

                # All cables start at the same X position, offsets are perpendicular (Y-Z plane)
                cable_start = wp.vec3(start_x, start_y + offset_y, start_z + offset_z)

                # Create cable geometry (straight initially, will droop naturally due to gravity)
                cable_points, _, cable_edge_q = self.create_cable_geometry(
                    pos=cable_start,
                    num_elements=self.num_elements,
                    length=self.cable_length,
                    twisting_angle=0.0,
                    droop=0.0,  # Start straight, gravity will cause natural droop
                )

                # Add cable to model
                rod_bodies, rod_joints = builder.add_rod_mesh(
                    positions=cable_points,
                    quaternions=cable_edge_q,
                    radius=self.cable_radius,
                    bend_stiffness=self.bend_stiffness,
                    bend_damping=1.0e-4,
                    stretch_stiffness=self.stretch_stiffness,
                    stretch_damping=1.0e-4,
                    key=f"bundle{bundle_id}_cable_{i}",
                )

                # Fix the first body (top of cable bundle)
                first_body = rod_bodies[0]
                builder.body_mass[first_body] = 0.0
                builder.body_inv_mass[first_body] = 0.0
                kinematic_body_indices.append(first_body)

                # Store last body for tip control
                last_body = rod_bodies[-1]
                self.tip_body_indices.append(last_body)

                # Store all bodies for this cable
                self.cable_bodies_list.append(rod_bodies)

            # Add rubber band rings around THIS bundle
            if add_rings:
                # Ring properties (different from cables)
                ring_shape_cfg = newton.ModelBuilder.ShapeConfig()
                ring_shape_cfg.ke = 1.0e3
                ring_shape_cfg.kd = 1.0e-1
                ring_shape_cfg.mu = 1.0e3

                # Ring parameters
                ring_radius_cable = self.cable_radius * 0.5  # Thinner than bundle cables
                ring_segments = 32  # Segments per ring (smoother circle)
                ring_bend_stiffness = 5.0e1  # Less stiff than bundle cables (more flexible)
                ring_stretch_stiffness = 1.0e9  # Higher stiffness to maintain shape better

                # Calculate ring radius to wrap around bundle WITHOUT penetration
                # Find maximum distance from center to outer edge of cables in bundle cross-section
                max_bundle_radius = 0.0
                for y, z in bundle_positions:
                    dist = np.sqrt(y**2 + z**2)
                    max_bundle_radius = max(max_bundle_radius, dist)

                # Bundle outer radius: distance from center to outer surface of outermost cable
                bundle_outer_radius = max_bundle_radius + self.cable_radius

                # Ring geometry calculation:
                # - ring_radius is the centerline of the ring tube
                # - Ring inner surface is at: ring_radius - ring_radius_cable
                # - Ring outer surface is at: ring_radius + ring_radius_cable
                #
                # For perfect fit (multiplier = 1.0):
                #   ring inner surface = bundle outer surface
                #   ring_radius - ring_radius_cable = bundle_outer_radius
                #   ring_radius = bundle_outer_radius + ring_radius_cable
                #
                # With multiplier:
                #   gap = ring_radius_cable * (ring_gap_multiplier - 1.0)
                #   ring_radius = bundle_outer_radius + ring_radius_cable + gap
                # Example: multiplier=1.05 gives 5% clearance relative to ring thickness
                #          multiplier=0.95 gives 5% compression relative to ring thickness
                # Gap scales automatically with ring thickness!
                gap = ring_radius_cable * (ring_gap_multiplier - 1.0)
                ring_radius = bundle_outer_radius + ring_radius_cable + gap

                # Position rings evenly along cable length, avoiding tips
                # Keep 10% margin on each end, distribute rings in the middle 80%
                margin = 0.1 * self.cable_length
                usable_length = self.cable_length - 2 * margin

                # Calculate ring positions
                actual_num_rings = max(2, num_rings_for_bundle) if num_rings_for_bundle >= 2 else num_rings_for_bundle

                ring_center_y = start_y
                ring_center_z = start_z

                # Create rings evenly spaced
                for ring_idx in range(actual_num_rings):
                    # Position calculation: evenly distribute rings in usable length
                    # For 2 rings: at 10% and 90%
                    # For 3 rings: at 10%, 50%, 90%
                    # For 4 rings: at 10%, 36.67%, 63.33%, 90%
                    t = ring_idx / (actual_num_rings - 1) if actual_num_rings > 1 else 0.5
                    ring_x = start_x + margin + t * usable_length

                    # Create ring geometry
                    ring_points, _, ring_q = self.create_ring_geometry(
                        center_pos=wp.vec3(ring_x, ring_center_y, ring_center_z),
                        normal_axis="x",  # Ring in Y-Z plane (perpendicular to cables)
                        radius=ring_radius,
                        num_segments=ring_segments,
                    )

                    # Create ring mesh
                    ring_bodies, ring_joints = builder.add_rod_mesh(
                        positions=ring_points,
                        quaternions=ring_q,
                        radius=ring_radius_cable,
                        bend_stiffness=ring_bend_stiffness,
                        bend_damping=1.0e-4,
                        stretch_stiffness=ring_stretch_stiffness,
                        stretch_damping=1.0e-4,
                        closed=True,  # Create closed loop (ring)
                        key=f"bundle{bundle_id}_ring_{ring_idx}",
                        cfg=ring_shape_cfg,
                    )

        # Create arrays for kinematic and tip bodies
        self.kinematic_bodies = wp.array(kinematic_body_indices, dtype=wp.int32)
        self.tip_bodies = wp.array(self.tip_body_indices, dtype=wp.int32)

        # Add ground plane with separate ground properties
        builder.add_ground_plane(cfg=ground_shape_cfg)

        # Color particles and rigid bodies
        builder.color()

        # Finalize model
        self.model = builder.finalize()

        # Create solver
        self.solver = newton.solvers.SolverAVBD(self.model, iterations=self.sim_iterations, friction_epsilon=0.1)

        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # Control parameters for interactive forces
        self.apply_tip_force = False
        self.tip_force_direction = wp.vec3(0.0, 1.0, 0.0)
        self.tip_force_magnitude = 10.0

        self.capture()

    def capture(self):
        """Capture CUDA graph for faster execution"""
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """Run simulation substeps"""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces from viewer (gravity, etc.)
            self.viewer.apply_forces(self.state_0)

            # Optionally apply force to bundle tips (for interactive control)
            if self.apply_tip_force:
                wp.launch(
                    kernel=apply_tip_force_kernel,
                    dim=self.tip_bodies.shape[0],
                    inputs=[self.tip_bodies, self.tip_force_direction, self.tip_force_magnitude],
                    outputs=[self.state_0.body_f],
                )

            # Collision detection
            self.contacts = self.model.collide(self.state_0)

            # Solver step
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
        pass


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create comparison example with three bundles
    example = Example(viewer=viewer)

    newton.examples.run(example, args)
