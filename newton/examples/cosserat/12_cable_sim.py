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
# Example Cable Catheter in Aorta
#
# Demonstrates a catheter simulation inside an aorta vessel using Newton's
# built-in cable/rod system. The catheter is modeled as a flexible rod with
# one end fixed (kinematic) that can be controlled via keyboard input.
#
# Controls:
#   - Numpad 4/6 or J/L: Move in X direction
#   - Numpad 8/2 or I/K: Move in Y direction
#   - Numpad 9/3 or U/O: Move in Z direction
#
# Command: uv run -m newton.examples.cosserat.12_cable_sim
#
###########################################################################

import math
import os

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton._src.geometry.kernels import triangle_closest_point


@wp.kernel
def compute_static_tri_aabbs_kernel(
    tri_vertices: wp.array(dtype=wp.vec3f),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    # outputs
    lower_bounds: wp.array(dtype=wp.vec3f),
    upper_bounds: wp.array(dtype=wp.vec3f),
):
    """Compute axis-aligned bounding boxes for static triangles."""
    tid = wp.tid()

    i0 = tri_indices[tid, 0]
    i1 = tri_indices[tid, 1]
    i2 = tri_indices[tid, 2]

    v0 = tri_vertices[i0]
    v1 = tri_vertices[i1]
    v2 = tri_vertices[i2]

    lower = wp.vec3f(
        wp.min(wp.min(v0[0], v1[0]), v2[0]),
        wp.min(wp.min(v0[1], v1[1]), v2[1]),
        wp.min(wp.min(v0[2], v1[2]), v2[2]),
    )
    upper = wp.vec3f(
        wp.max(wp.max(v0[0], v1[0]), v2[0]),
        wp.max(wp.max(v0[1], v1[1]), v2[1]),
        wp.max(wp.max(v0[2], v1[2]), v2[2]),
    )

    lower_bounds[tid] = lower
    upper_bounds[tid] = upper


@wp.func
def compute_triangle_normal(v0: wp.vec3f, v1: wp.vec3f, v2: wp.vec3f) -> wp.vec3f:
    """Compute the normal vector of a triangle from three vertices."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = wp.cross(edge1, edge2)
    length = wp.length(normal)
    if length > 1e-8:
        return normal / length
    return wp.vec3f(0.0, 1.0, 0.0)  # fallback


@wp.kernel
def collide_bodies_vs_triangles_bvh_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_inv_mass: wp.array(dtype=float),
    body_indices: wp.array(dtype=wp.int32),
    body_radius: float,
    tri_vertices: wp.array(dtype=wp.vec3f),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    bvh_id: wp.uint64,
    # outputs
    body_q_out: wp.array(dtype=wp.transform),
):
    """
    Body vs triangles collision kernel using BVH broadphase and PBD response.

    Uses a Gauss-Seidel approach: for each body (treated as a sphere), we iterate
    through triangles and update the sphere position locally after each collision.
    Only the final corrected position is written to memory.

    For each body:
    1. Query BVH for triangles within body radius (broadphase)
    2. For each candidate triangle:
       a. Compute closest point on triangle to current sphere position (narrowphase)
       b. If penetrating, immediately update sphere position locally
    3. Write final position to memory
    """
    tid = wp.tid()
    body_idx = body_indices[tid]

    inv_mass = body_inv_mass[body_idx]

    # Get current body transform
    xform = body_q[body_idx]
    pos = wp.transform_get_translation(xform)
    rot = wp.transform_get_rotation(xform)

    # Skip kinematic bodies
    if inv_mass <= 0.0:
        body_q_out[body_idx] = xform
        return

    # Query BVH for triangles within body's bounding sphere
    # Use a larger margin to account for position updates during iteration
    query_margin = body_radius * 2.0
    lower = wp.vec3f(
        pos[0] - query_margin,
        pos[1] - query_margin,
        pos[2] - query_margin,
    )
    upper = wp.vec3f(
        pos[0] + query_margin,
        pos[1] + query_margin,
        pos[2] + query_margin,
    )

    # Broadphase: query BVH for potentially colliding triangles
    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    tri_idx = wp.int32(0)

    # Gauss-Seidel: update position locally after each triangle collision
    while wp.bvh_query_next(query, tri_idx):
        # Get triangle vertex indices
        i0 = tri_indices[tri_idx, 0]
        i1 = tri_indices[tri_idx, 1]
        i2 = tri_indices[tri_idx, 2]

        # Get triangle vertex positions
        v0 = tri_vertices[i0]
        v1 = tri_vertices[i1]
        v2 = tri_vertices[i2]

        # Compute triangle normal (defines front face direction)
        tri_normal = compute_triangle_normal(v0, v1, v2)

        # Signed distance from triangle plane (positive = front side, negative = back side)
        signed_dist = wp.dot(tri_normal, pos - v0)

        # Narrowphase: find closest point on triangle to current sphere position
        closest_p, bary, feature_type = triangle_closest_point(v0, v1, v2, pos)

        # Compute distance from sphere center to closest point
        to_body = pos - closest_p
        dist = wp.length(to_body)

        if signed_dist < 0.0:
            # Sphere is on backface - push it to front side
            # Target position: closest point + normal * body_radius (front side at correct distance)
            target_pos = closest_p + tri_normal * body_radius
            pos = target_pos
        elif dist < body_radius:
            # Sphere is on frontface but penetrating - standard collision response
            penetration = body_radius - dist
            if dist > 1e-8:
                correction_dir = to_body / dist
            else:
                # Sphere center is exactly on the triangle surface
                correction_dir = tri_normal

            # Compute and apply position correction immediately (Gauss-Seidel)
            correction = correction_dir * penetration
            pos = pos + correction

    # Write final corrected position to memory
    body_q_out[body_idx] = wp.transform(pos, rot)


@wp.kernel
def collide_bodies_vs_triangles_bruteforce_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_inv_mass: wp.array(dtype=float),
    body_indices: wp.array(dtype=wp.int32),
    body_radius: float,
    tri_vertices: wp.array(dtype=wp.vec3f),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    num_triangles: int,
    # outputs
    body_q_out: wp.array(dtype=wp.transform),
):
    """
    Body vs triangles collision kernel using brute force iteration.

    Uses a Gauss-Seidel approach: for each body (treated as a sphere), we iterate
    through ALL triangles and update the sphere position locally after each collision.
    Only the final corrected position is written to memory.

    For each body:
    1. Iterate through all triangles (brute force)
    2. For each triangle:
       a. Compute closest point on triangle to current sphere position (narrowphase)
       b. If penetrating, immediately update sphere position locally
    3. Write final position to memory
    """
    tid = wp.tid()
    body_idx = body_indices[tid]

    inv_mass = body_inv_mass[body_idx]

    # Get current body transform
    xform = body_q[body_idx]
    pos = wp.transform_get_translation(xform)
    rot = wp.transform_get_rotation(xform)

    # Skip kinematic bodies
    if inv_mass <= 0.0:
        body_q_out[body_idx] = xform
        return

    # Gauss-Seidel: iterate through ALL triangles (brute force)
    # Use same margin as BVH version for proximity check
    collision_margin = body_radius * 2.0

    for tri_idx in range(num_triangles):
        # Get triangle vertex indices
        i0 = tri_indices[tri_idx, 0]
        i1 = tri_indices[tri_idx, 1]
        i2 = tri_indices[tri_idx, 2]

        # Get triangle vertex positions
        v0 = tri_vertices[i0]
        v1 = tri_vertices[i1]
        v2 = tri_vertices[i2]

        # Narrowphase: find closest point on triangle to current sphere position
        closest_p, bary, feature_type = triangle_closest_point(v0, v1, v2, pos)

        # Compute distance from sphere center to closest point
        to_body = pos - closest_p
        dist = wp.length(to_body)

        # Early-out: skip triangles that are too far away
        if dist > collision_margin:
            continue

        # Compute triangle normal (defines front face direction)
        tri_normal = compute_triangle_normal(v0, v1, v2)

        # Signed distance from triangle plane (positive = front side, negative = back side)
        signed_dist = wp.dot(tri_normal, pos - v0)

        if signed_dist < 0.0:
            # Sphere is on backface - push it to front side
            # Target position: closest point + normal * body_radius (front side at correct distance)
            target_pos = closest_p + tri_normal * body_radius
            pos = target_pos
        elif dist < body_radius:
            # Sphere is on frontface but penetrating - standard collision response
            penetration = body_radius - dist
            if dist > 1e-8:
                correction_dir = to_body / dist
            else:
                # Sphere center is exactly on the triangle surface
                correction_dir = tri_normal

            # Compute and apply position correction immediately (Gauss-Seidel)
            correction = correction_dir * penetration
            pos = pos + correction

    # Write final corrected position to memory
    body_q_out[body_idx] = wp.transform(pos, rot)


class Example:
    def __init__(self, viewer, args=None):
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.gravity = wp.vec3(0.0, 0.0, 0.0)
        
        self.viewer = viewer
        self.args = args

        # Catheter parameters
        self.num_elements = 50
        segment_length = 0.06
        self.cable_length = self.num_elements * segment_length
        cable_radius = 0.015

        # Build the model
        builder = newton.ModelBuilder()

        # Set default material properties
        builder.default_shape_cfg.ke = 1.0e3  # Contact stiffness
        builder.default_shape_cfg.kd = 1.0e2  # Contact damping
        builder.default_shape_cfg.mu = 0.3  # Friction coefficient

        # Load the aorta vessel mesh
        usd_path = os.path.join(os.path.dirname(__file__), "models", "DynamicAorta.usdc")
        usd_stage = Usd.Stage.Open(usd_path)
        mesh_prim = usd_stage.GetPrimAtPath("/root/A4009/A4007/Xueguan_rudong/Dynamic_vessels/Mesh")
        vessel_mesh = newton.usd.get_mesh(mesh_prim)

        # Store mesh data for collision detection
        self.vessel_vertices_np = np.array(vessel_mesh.vertices, dtype=np.float32)
        self.vessel_indices_np = np.array(vessel_mesh.indices, dtype=np.int32).reshape(-1, 3)
        self.num_vessel_triangles = self.vessel_indices_np.shape[0]
        self.mesh_scale = 0.01  # Convert mesh units to simulation units

        # Add the vessel mesh as a static collision shape
        vessel_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e4,
            kd=1.0e2,
            mu=0.1,  # Low friction for blood vessels
            has_shape_collision=True,  # Enable VBD rigid body collisions
            has_particle_collision=False,
        )

        self.mesh_xform = wp.transform(
            wp.vec3(0.0, 0.0, 1.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2.0),
        )

        builder.add_shape_mesh(
            body=-1,  # Static shape
            mesh=vessel_mesh,
            scale=(self.mesh_scale, self.mesh_scale, self.mesh_scale),
            xform=self.mesh_xform,
            cfg=vessel_cfg,
        )

        # Create catheter geometry
        # Position the catheter tip (end) inside the aorta
        tip_pos = wp.vec3(-3.3, -0.5, 1.7)  # Tip position inside the aorta
        cable_points, cable_edge_q = self._create_cable_geometry(
            end_pos=tip_pos,
            num_elements=self.num_elements,
            length=self.cable_length,
        )

        # Rod stiffness parameters (can be modified at runtime via UI)
        self.bend_stiffness = 1.0e1
        self.stretch_stiffness = 1.0e9

        # Add the catheter as a rod using Newton's built-in system
        self.cable_bodies, self.cable_joints = builder.add_rod(
            positions=cable_points,
            quaternions=cable_edge_q,
            radius=cable_radius,
            bend_stiffness=self.bend_stiffness,
            bend_damping=1.0e-1,
            stretch_stiffness=self.stretch_stiffness,
            stretch_damping=0.0,
            key="catheter",
        )

        # Fix the first body to make it kinematic (anchor point)
        first_body = self.cable_bodies[0]
        builder.body_mass[first_body] = 0.0
        builder.body_inv_mass[first_body] = 0.0
        builder.body_inertia[first_body] = wp.mat33(0.0)
        builder.body_inv_inertia[first_body] = wp.mat33(0.0)

        # Add ground plane
        builder.add_ground_plane()

        # Color bodies for VBD solver
        builder.color()

        # Finalize model
        self.model = builder.finalize()
        self.model.set_gravity(self.gravity)
        # Create VBD solver
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            friction_epsilon=0.1,
        )

        # State buffers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Collision pipeline
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        # Keyboard control parameters
        self.move_speed = 1.0  # units per second

        # Graph capture for CUDA acceleration
        self.graph = None

        # Collision settings
        self.use_bvh = True  # Use BVH acceleration (vs brute force)

        # Setup collision detection for cable vs vessel mesh
        self._setup_mesh_collision()

    def _create_cable_geometry(self, end_pos, num_elements, length):
        """Create a straight cable geometry with proper quaternions.

        Args:
            end_pos: End position of the cable (tip).
            num_elements: Number of cable segments.
            length: Total cable length.

        Returns:
            Tuple of (points, quaternions).
        """
        num_points = num_elements + 1
        points = []

        # Calculate start position so cable ends at end_pos
        # Cable extends in negative X direction from end_pos
        start_pos = end_pos - wp.vec3(length, 0.0, 0.0)

        # Create points along X direction from start to end
        for i in range(num_points):
            t = i / num_elements
            x = length * t
            points.append(start_pos + wp.vec3(x, 0.0, 0.0))

        # Create quaternions for each edge (orient along X axis)
        # Capsule internal axis is +Z, so we need to rotate Z to X
        local_axis = wp.vec3(0.0, 0.0, 1.0)
        target_axis = wp.vec3(1.0, 0.0, 0.0)
        base_quat = wp.quat_between_vectors(local_axis, target_axis)

        edge_q = [base_quat] * num_elements

        return points, edge_q

    def _setup_mesh_collision(self):
        """Setup BVH and collision data structures for cable vs vessel mesh collision."""
        device = self.model.device

        # Apply scale and transform to vertices
        scaled_vertices = self.vessel_vertices_np * self.mesh_scale

        # Apply rotation and translation
        transformed_vertices = np.zeros_like(scaled_vertices)
        for i in range(len(scaled_vertices)):
            v = wp.vec3(scaled_vertices[i, 0], scaled_vertices[i, 1], scaled_vertices[i, 2])
            v_transformed = wp.transform_point(self.mesh_xform, v)
            transformed_vertices[i] = [v_transformed[0], v_transformed[1], v_transformed[2]]

        self.vessel_vertices = wp.array(transformed_vertices, dtype=wp.vec3f, device=device)
        self.vessel_indices = wp.array(self.vessel_indices_np, dtype=wp.int32, device=device)

        # Build BVH for efficient collision detection
        self.tri_lower_bounds = wp.zeros(self.num_vessel_triangles, dtype=wp.vec3f, device=device)
        self.tri_upper_bounds = wp.zeros(self.num_vessel_triangles, dtype=wp.vec3f, device=device)
        wp.launch(
            kernel=compute_static_tri_aabbs_kernel,
            dim=self.num_vessel_triangles,
            inputs=[self.vessel_vertices, self.vessel_indices],
            outputs=[self.tri_lower_bounds, self.tri_upper_bounds],
            device=device,
        )
        self.vessel_bvh = wp.Bvh(self.tri_lower_bounds, self.tri_upper_bounds)

        # Create array of cable body indices for collision detection
        self.cable_body_indices = wp.array(self.cable_bodies, dtype=wp.int32, device=device)
        self.num_cable_bodies = len(self.cable_bodies)

        # Get cable radius from the first body's shape
        self.cable_collision_radius = 0.015  # Same as cable_radius used in add_rod

        # Create temporary buffer for collision output
        self.body_q_temp = wp.zeros(self.model.body_count, dtype=wp.transform, device=device)

    def _apply_mesh_collision(self, state):
        """Apply collision response between cable bodies and vessel mesh."""
        # Copy current state to temp buffer (for bodies not in collision)
        wp.copy(self.body_q_temp, state.body_q)

        if self.use_bvh:
            # BVH-accelerated collision detection
            wp.launch(
                kernel=collide_bodies_vs_triangles_bvh_kernel,
                dim=self.num_cable_bodies,
                inputs=[
                    state.body_q,
                    self.model.body_inv_mass,
                    self.cable_body_indices,
                    self.cable_collision_radius,
                    self.vessel_vertices,
                    self.vessel_indices,
                    self.vessel_bvh.id,
                ],
                outputs=[self.body_q_temp],
                device=self.model.device,
            )
        else:
            # Brute force collision detection (iterate through all triangles)
            wp.launch(
                kernel=collide_bodies_vs_triangles_bruteforce_kernel,
                dim=self.num_cable_bodies,
                inputs=[
                    state.body_q,
                    self.model.body_inv_mass,
                    self.cable_body_indices,
                    self.cable_collision_radius,
                    self.vessel_vertices,
                    self.vessel_indices,
                    self.num_vessel_triangles,
                ],
                outputs=[self.body_q_temp],
                device=self.model.device,
            )

        # Copy collision-corrected positions back to state
        wp.copy(state.body_q, self.body_q_temp)

    def _update_rod_stiffness(self):
        """Update rod joint stiffness values in the solver."""
        # Get segment length for normalization (same as used in add_rod)
        segment_length = self.cable_length / self.num_elements

        # Compute effective stiffness (normalized by segment length)
        stretch_ke_eff = self.stretch_stiffness / segment_length
        bend_ke_eff = self.bend_stiffness / segment_length

        # Update the solver's joint_penalty_k_max array directly
        # VBD solver caches stiffness in joint_penalty_k_max indexed by joint_constraint_start
        joint_constraint_start = self.solver.joint_constraint_start.numpy()
        joint_penalty_k_max = self.solver.joint_penalty_k_max.numpy()

        # Update stiffness for each cable joint
        # Cable joints have 2 constraint slots: stretch (idx 0), bend (idx 1)
        for joint_idx in self.cable_joints:
            c_start = joint_constraint_start[joint_idx]
            joint_penalty_k_max[c_start] = stretch_ke_eff  # Stretch constraint
            joint_penalty_k_max[c_start + 1] = bend_ke_eff  # Bend constraint

        # Update the solver array
        self.solver.joint_penalty_k_max.assign(joint_penalty_k_max)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply viewer forces (for interactive manipulation)
            self.viewer.apply_forces(self.state_0)

            # Detect collisions
            self.contacts = self.model.collide(
                self.state_0,
                collision_pipeline=self.collision_pipeline,
            )

            # Step the solver
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )

            # Custom cable vs vessel mesh collision is disabled - using VBD built-in collisions instead
            # self._apply_mesh_collision(self.state_1)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Handle keyboard input
        self._handle_keyboard_input()

        # Run simulation
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def _handle_keyboard_input(self):
        """Move the anchor point (first body) using keyboard controls."""
        if not hasattr(self.viewer, "is_key_down"):
            return

        try:
            import pyglet.window.key as key
        except ImportError:
            return

        # Calculate movement delta
        dx, dy, dz = 0.0, 0.0, 0.0

        # X movement (Numpad 4/6 or J/L)
        if self.viewer.is_key_down(key.NUM_6) or self.viewer.is_key_down(key.L):
            dx += self.move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_4) or self.viewer.is_key_down(key.J):
            dx -= self.move_speed * self.frame_dt

        # Y movement (Numpad 8/2 or I/K)
        if self.viewer.is_key_down(key.NUM_8) or self.viewer.is_key_down(key.I):
            dy += self.move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_2) or self.viewer.is_key_down(key.K):
            dy -= self.move_speed * self.frame_dt

        # Z movement (Numpad 9/3 or U/O)
        if self.viewer.is_key_down(key.NUM_9) or self.viewer.is_key_down(key.U):
            dz += self.move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_3) or self.viewer.is_key_down(key.O):
            dz -= self.move_speed * self.frame_dt

        # Apply movement if any key was pressed
        if dx != 0.0 or dy != 0.0 or dz != 0.0:
            body_q_np = self.state_0.body_q.numpy()
            pos = body_q_np[self.cable_bodies[0]]

            # Update position (body_q stores [x, y, z, qx, qy, qz, qw])
            new_pos = [
                pos[0] + dx,
                pos[1] + dy,
                max(pos[2] + dz, 0.1),  # Keep above ground
            ]

            body_q_np[self.cable_bodies[0], 0] = new_pos[0]
            body_q_np[self.cable_bodies[0], 1] = new_pos[1]
            body_q_np[self.cable_bodies[0], 2] = new_pos[2]

            self.state_0.body_q = wp.array(body_q_np, dtype=wp.transform, device=self.model.device)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Catheter in Aorta Simulation")
        ui.text(f"Cable elements: {self.num_elements}")
        ui.separator()
        ui.text("Controls:")
        ui.text("  Numpad 4/6 or J/L: Move X")
        ui.text("  Numpad 8/2 or I/K: Move Y")
        ui.text("  Numpad 9/3 or U/O: Move Z")
        ui.separator()
        _changed, self.move_speed = ui.slider_float("Move Speed", self.move_speed, 0.1, 5.0)
        ui.separator()
        ui.text("Collision Settings:")
        _changed, self.use_bvh = ui.checkbox("Use BVH Acceleration", self.use_bvh)
        _changed, self.cable_collision_radius = ui.slider_float(
            "Collision Radius", self.cable_collision_radius, 0.001, 0.1
        )
        ui.separator()
        ui.text("Rod Stiffness:")
        # Use log scale for stiffness sliders
        bend_log = np.log10(self.bend_stiffness)
        changed_bend, bend_log = ui.slider_float("Bend Stiffness (log10)", bend_log, -2.0, 4.0)
        if changed_bend:
            self.bend_stiffness = 10.0**bend_log
            self._update_rod_stiffness()
        ui.text(f"  Bend: {self.bend_stiffness:.2e} N*m")

        stretch_log = np.log10(self.stretch_stiffness)
        changed_stretch, stretch_log = ui.slider_float("Stretch Stiffness (log10)", stretch_log, 4.0, 12.0)
        if changed_stretch:
            self.stretch_stiffness = 10.0**stretch_log
            self._update_rod_stiffness()
        ui.text(f"  Stretch: {self.stretch_stiffness:.2e} N/m")

    def test_final(self):
        """Test that the simulation ran correctly."""
        if self.state_0.body_q is not None and self.state_0.body_qd is not None:
            body_positions = self.state_0.body_q.numpy()
            body_velocities = self.state_0.body_qd.numpy()

            # Check for numerical stability
            assert np.isfinite(body_positions).all(), "Non-finite values in body positions"
            assert np.isfinite(body_velocities).all(), "Non-finite values in body velocities"

            # Check reasonable bounds
            assert (np.abs(body_positions[:, :3]) < 100).all(), "Body positions out of bounds"
            assert (np.abs(body_velocities) < 500).all(), "Body velocities too large"

            # Check cable connectivity
            segment_length = self.cable_length / self.num_elements
            for i in range(len(self.cable_bodies) - 1):
                body1_idx = self.cable_bodies[i]
                body2_idx = self.cable_bodies[i + 1]
                pos1 = body_positions[body1_idx, :3]
                pos2 = body_positions[body2_idx, :3]
                distance = np.linalg.norm(pos2 - pos1)
                max_distance = segment_length * 1.5  # Allow some stretch
                assert distance < max_distance, (
                    f"Cable segments {i}-{i + 1} too far apart: {distance:.3f} > {max_distance:.3f}"
                )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
