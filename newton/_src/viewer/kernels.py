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

"""
Warp kernels for simplified Newton viewers.
These kernels handle mesh operations and transformations.
"""

import warp as wp

import newton


@wp.kernel
def compute_pick_state_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_index: int,
    hit_point_world: wp.vec3,
    # output
    pick_body: wp.array(dtype=int),
    pick_state: wp.array(dtype=float),
):
    """
    Initialize the pick state when a body is first picked.

    This kernel stores:
    - The local space attachment point on the body
    - The initial world space target position
    - The original mouse cursor target
    - The current world space picked point on geometry (for visualization)
    """
    if body_index < 0:
        return

    # store body index
    pick_body[0] = body_index

    # Get body transform
    X_wb = body_q[body_index]
    X_bw = wp.transform_inverse(X_wb)

    # Compute local space attachment point from the hit point
    pick_pos_local = wp.transform_point(X_bw, hit_point_world)

    pick_state[0] = pick_pos_local[0]
    pick_state[1] = pick_pos_local[1]
    pick_state[2] = pick_pos_local[2]

    # IMPORTANT: Initialize target to current attachment point position, not hit point
    # This prevents jumps if the body moved between raycast and first force application
    pick_pos_world = wp.transform_point(X_wb, pick_pos_local)

    # store target world (current attachment point position)
    pick_state[3] = pick_pos_world[0]
    pick_state[4] = pick_pos_world[1]
    pick_state[5] = pick_pos_world[2]

    # store original mouse cursor target (where user clicked)
    pick_state[8] = hit_point_world[0]
    pick_state[9] = hit_point_world[1]
    pick_state[10] = hit_point_world[2]

    # store current world space picked point on geometry (for visualization)
    pick_state[11] = pick_pos_world[0]
    pick_state[12] = pick_pos_world[1]
    pick_state[13] = pick_pos_world[2]


@wp.kernel
def apply_picking_force_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    pick_body_arr: wp.array(dtype=int),
    pick_state: wp.array(dtype=float),
    body_com: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
):
    pick_body = pick_body_arr[0]
    if pick_body < 0:
        return

    pick_pos_local = wp.vec3(pick_state[0], pick_state[1], pick_state[2])
    current_target = wp.vec3(pick_state[3], pick_state[4], pick_state[5])
    pick_stiffness = pick_state[6]
    # pick_damping from pick_state[7] is ignored - we compute critical damping

    # Get the original mouse cursor target
    mouse_cursor_target = wp.vec3(pick_state[8], pick_state[9], pick_state[10])

    # Apply smooth movement towards the mouse cursor target
    desired_delta = mouse_cursor_target - current_target
    desired_distance = wp.length(desired_delta)

    # Use adaptive movement speed based on distance (gentle movement)
    max_delta_per_frame = 0.02  # Doubled for more responsive movement

    if desired_distance > 0.001:  # Avoid division by zero
        # Scale movement speed based on distance (gentler scaling)
        adaptive_speed = wp.min(desired_distance * 1.0, max_delta_per_frame * 3.0)
        movement_speed = wp.min(adaptive_speed, desired_distance)  # Don't overshoot

        # Move toward the mouse cursor target
        direction = desired_delta / desired_distance
        pick_target_world = current_target + direction * movement_speed

        # Update the current target in pick_state
        pick_state[3] = pick_target_world[0]
        pick_state[4] = pick_target_world[1]
        pick_state[5] = pick_target_world[2]
    else:
        pick_target_world = current_target

    # Get body properties for stability
    mass = body_mass[pick_body]

    # Compute critical damping to avoid oscillations: c_critical = 2 * sqrt(k * m)
    pick_damping = 2.0 * wp.sqrt(pick_stiffness * mass)

    # world space attachment point
    X_wb = body_q[pick_body]
    pick_pos_world = wp.transform_point(X_wb, pick_pos_local)

    # update current world space picked point on geometry (for visualization)
    pick_state[11] = pick_pos_world[0]
    pick_state[12] = pick_pos_world[1]
    pick_state[13] = pick_pos_world[2]

    # center of mass (corrected calculation)
    com = wp.transform_point(X_wb, body_com[pick_body])

    # get velocity of attachment point
    vel_com = wp.spatial_top(body_qd[pick_body])

    # compute spring force with critical damping (only damp linear velocity, not rotational)
    f = pick_stiffness * (pick_target_world - pick_pos_world) - pick_damping * vel_com

    # Force limiting to prevent instability
    max_force = mass * 10000.0
    force_magnitude = wp.length(f)
    if force_magnitude > max_force:
        f = f * (max_force / force_magnitude)

    # Add velocity damping to linear motion only
    velocity_damping_factor = 50.0 * mass  # Mass-dependent velocity damping
    linear_vel = wp.spatial_top(body_qd[pick_body])
    velocity_damping_force = -velocity_damping_factor * linear_vel

    # Combine spring force with velocity damping
    total_force = f + velocity_damping_force

    # Compute natural torque from off-center force application
    # When pick point != COM, this creates a torque that rotates the object
    t = wp.cross(pick_pos_world - com, total_force)

    # Apply force at pick point with natural torque
    wp.atomic_add(body_f, pick_body, wp.spatial_vector(total_force, t))


@wp.kernel
def update_pick_target_kernel(
    p: wp.vec3,
    d: wp.vec3,
    world_offset: wp.vec3,
    # read-write
    pick_state: wp.array(dtype=float),
):
    # get original mouse cursor target (in physics space)
    original_target = wp.vec3(pick_state[8], pick_state[9], pick_state[10])

    # Add world offset to convert to offset space for distance calculation
    original_target_offset = original_target + world_offset

    # compute distance from ray origin to original target (to maintain depth)
    dist = wp.length(original_target_offset - p)

    # Project new mouse cursor target at the same depth (in offset space)
    new_mouse_target_offset = p + d * dist

    # Convert back to physics space by subtracting world offset
    new_mouse_target = new_mouse_target_offset - world_offset

    # Update the original mouse cursor target (no smoothing here)
    pick_state[8] = new_mouse_target[0]
    pick_state[9] = new_mouse_target[1]
    pick_state[10] = new_mouse_target[2]


@wp.kernel
def update_shape_xforms(
    shape_xforms: wp.array(dtype=wp.transform),
    shape_parents: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    shape_worlds: wp.array(dtype=int, ndim=1),
    world_offsets: wp.array(dtype=wp.vec3, ndim=1),
    world_xforms: wp.array(dtype=wp.transform),
):
    tid = wp.tid()

    shape_xform = shape_xforms[tid]
    shape_parent = shape_parents[tid]

    if shape_parent >= 0:
        world_xform = wp.transform_multiply(body_q[shape_parent], shape_xform)
    else:
        world_xform = shape_xform

    if world_offsets:
        shape_world = shape_worlds[tid]
        if shape_world >= 0 and shape_world < world_offsets.shape[0]:
            offset = world_offsets[shape_world]
            world_xform = wp.transform(world_xform.p + offset, world_xform.q)

    world_xforms[tid] = world_xform


@wp.kernel
def estimate_world_extents(
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_collision_radius: wp.array(dtype=float),
    shape_world: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    num_worlds: int,
    # outputs (num_worlds x 3 arrays for min/max xyz per world)
    world_bounds_min: wp.array(dtype=float, ndim=2),
    world_bounds_max: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()

    # Get shape's world assignment
    world_idx = shape_world[tid]

    # Skip global shapes (world -1) or invalid world indices
    if world_idx < 0 or world_idx >= num_worlds:
        return

    # Get collision radius and skip shapes with unreasonably large radii
    radius = shape_collision_radius[tid]
    if radius > 1.0e5:  # Skip outliers like infinite planes
        return

    # Get shape's world position
    shape_xform = shape_transform[tid]
    shape_parent = shape_body[tid]

    # Compute world transform
    if shape_parent >= 0:
        # Shape attached to body: world_xform = body_xform * shape_xform
        body_xform = body_q[shape_parent]
        world_xform = wp.transform_multiply(body_xform, shape_xform)
    else:
        # Static shape: already in world space
        world_xform = shape_xform

    # Get position and radius
    pos = wp.transform_get_translation(world_xform)
    radius = shape_collision_radius[tid]

    # Update bounds for this world using atomic operations
    min_pos = pos - wp.vec3(radius, radius, radius)
    max_pos = pos + wp.vec3(radius, radius, radius)

    # Atomic min for each component
    wp.atomic_min(world_bounds_min, world_idx, 0, min_pos[0])
    wp.atomic_min(world_bounds_min, world_idx, 1, min_pos[1])
    wp.atomic_min(world_bounds_min, world_idx, 2, min_pos[2])

    # Atomic max for each component
    wp.atomic_max(world_bounds_max, world_idx, 0, max_pos[0])
    wp.atomic_max(world_bounds_max, world_idx, 1, max_pos[1])
    wp.atomic_max(world_bounds_max, world_idx, 2, max_pos[2])


@wp.kernel
def compute_contact_lines(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_world: wp.array(dtype=int),
    world_offsets: wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    line_scale: float,
    # outputs
    line_start: wp.array(dtype=wp.vec3),
    line_end: wp.array(dtype=wp.vec3),
):
    """Create line segments along contact normals for visualization."""
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        line_start[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        line_end[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        return
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        line_start[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        line_end[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        return

    # Get world transforms for both shapes
    body_a = shape_body[shape_a]
    body_b = shape_body[shape_b]
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    if body_a >= 0:
        X_wb_a = body_q[body_a]
    if body_b >= 0:
        X_wb_b = body_q[body_b]

    # Compute world space contact positions
    world_pos0 = wp.transform_point(X_wb_a, contact_point0[tid])
    world_pos1 = wp.transform_point(X_wb_b, contact_point1[tid])
    # Use the midpoint of the contact as the line start
    contact_center = (world_pos0 + world_pos1) * 0.5

    # Apply world offset
    world_a, world_b = shape_world[shape_a], shape_world[shape_b]
    if world_a >= 0 or world_b >= 0:
        contact_center += world_offsets[world_a if world_a >= 0 else world_b]

    # Create line along normal direction
    # Normal points from shape0 to shape1, draw from center in normal direction
    normal = contact_normal[tid]
    line_vector = normal * line_scale

    line_start[tid] = contact_center
    line_end[tid] = contact_center + line_vector


@wp.kernel
def compute_joint_basis_lines(
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    body_world: wp.array(dtype=int),
    world_offsets: wp.array(dtype=wp.vec3),
    shape_collision_radius: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    line_scale: float,
    # outputs - unified buffers for all joint lines
    line_starts: wp.array(dtype=wp.vec3),
    line_ends: wp.array(dtype=wp.vec3),
    line_colors: wp.array(dtype=wp.vec3),
):
    """Create line segments for joint basis vectors for visualization.
    Each joint produces 3 lines (x, y, z axes).
    Thread ID maps to line index: joint_id * 3 + axis_id
    """
    tid = wp.tid()

    # Determine which joint and which axis this thread handles
    joint_id = tid // 3
    axis_id = tid % 3

    # Check if this is a supported joint type
    if joint_id >= len(joint_type):
        line_starts[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        line_ends[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        line_colors[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    joint_t = joint_type[joint_id]
    if joint_t != int(newton.JointType.REVOLUTE) and joint_t != int(newton.JointType.D6):
        # Set NaN for unsupported joints to hide them
        line_starts[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        line_ends[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        line_colors[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    # Get joint transform
    joint_tf = joint_transform[joint_id]
    joint_pos = wp.transform_get_translation(joint_tf)
    joint_rot = wp.transform_get_rotation(joint_tf)

    # Get parent body transform
    parent_body = joint_parent[joint_id]
    if parent_body >= 0:
        parent_tf = body_q[parent_body]
        # Transform joint to world space
        world_pos = wp.transform_point(parent_tf, joint_pos)
        world_rot = wp.mul(wp.transform_get_rotation(parent_tf), joint_rot)
        # Apply world offset
        parent_body_world = body_world[parent_body]
        if parent_body_world >= 0:
            world_pos += world_offsets[parent_body_world]
    else:
        world_pos = joint_pos
        world_rot = joint_rot

    # Determine scale based on child body shapes
    scale_factor = line_scale

    # Create the appropriate basis vector based on axis_id
    if axis_id == 0:  # X-axis (red)
        axis_vec = wp.quat_rotate(world_rot, wp.vec3(1.0, 0.0, 0.0))
        color = wp.vec3(1.0, 0.0, 0.0)
    elif axis_id == 1:  # Y-axis (green)
        axis_vec = wp.quat_rotate(world_rot, wp.vec3(0.0, 1.0, 0.0))
        color = wp.vec3(0.0, 1.0, 0.0)
    else:  # Z-axis (blue)
        axis_vec = wp.quat_rotate(world_rot, wp.vec3(0.0, 0.0, 1.0))
        color = wp.vec3(0.0, 0.0, 1.0)

    # Set line endpoints
    line_starts[tid] = world_pos
    line_ends[tid] = world_pos + axis_vec * scale_factor
    line_colors[tid] = color


@wp.func
def depth_to_color(depth: float, min_depth: float, max_depth: float) -> wp.vec3:
    """Convert depth value to a color using a blue-to-red colormap."""
    # Normalize depth to [0, 1]
    t = wp.clamp((depth - min_depth) / (max_depth - min_depth + 1e-8), 0.0, 1.0)
    # Blue (0,0,1) -> Cyan (0,1,1) -> Green (0,1,0) -> Yellow (1,1,0) -> Red (1,0,0)
    if t < 0.25:
        s = t / 0.25
        return wp.vec3(0.0, s, 1.0)
    elif t < 0.5:
        s = (t - 0.25) / 0.25
        return wp.vec3(0.0, 1.0, 1.0 - s)
    elif t < 0.75:
        s = (t - 0.5) / 0.25
        return wp.vec3(s, 1.0, 0.0)
    else:
        s = (t - 0.75) / 0.25
        return wp.vec3(1.0, 1.0 - s, 0.0)


@wp.kernel
def compute_isosurface_lines(
    triangle_vertices: wp.array(dtype=wp.vec3),
    face_depths: wp.array(dtype=wp.float32),
    num_faces: int,
    min_depth: float,
    max_depth: float,
    penetrating_only: bool,
    line_starts: wp.array(dtype=wp.vec3),
    line_ends: wp.array(dtype=wp.vec3),
    line_colors: wp.array(dtype=wp.vec3),
):
    """Convert triangle vertices to line segments for wireframe rendering."""
    tid = wp.tid()
    if tid >= num_faces:
        return

    # Get the 3 vertices of this triangle
    v0 = triangle_vertices[tid * 3 + 0]
    v1 = triangle_vertices[tid * 3 + 1]
    v2 = triangle_vertices[tid * 3 + 2]

    # Compute color from depth
    depth = face_depths[tid]

    # Skip non-penetrating contacts if requested (only render depth > 0)
    if penetrating_only and depth <= 0.0:
        zero = wp.vec3(0.0, 0.0, 0.0)
        line_starts[tid * 3 + 0] = zero
        line_ends[tid * 3 + 0] = zero
        line_colors[tid * 3 + 0] = zero
        line_starts[tid * 3 + 1] = zero
        line_ends[tid * 3 + 1] = zero
        line_colors[tid * 3 + 1] = zero
        line_starts[tid * 3 + 2] = zero
        line_ends[tid * 3 + 2] = zero
        line_colors[tid * 3 + 2] = zero
        return

    if depth > 0.0:
        color = depth_to_color(depth, min_depth, max_depth)
    else:
        color = wp.vec3(0.0, 0.0, 0.0)

    # Each triangle produces 3 line segments (edges)
    # Edge 0: v0 -> v1
    line_starts[tid * 3 + 0] = v0
    line_ends[tid * 3 + 0] = v1
    line_colors[tid * 3 + 0] = color

    # Edge 1: v1 -> v2
    line_starts[tid * 3 + 1] = v1
    line_ends[tid * 3 + 1] = v2
    line_colors[tid * 3 + 1] = color

    # Edge 2: v2 -> v0
    line_starts[tid * 3 + 2] = v2
    line_ends[tid * 3 + 2] = v0
    line_colors[tid * 3 + 2] = color
