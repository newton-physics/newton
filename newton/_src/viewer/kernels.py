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
    if body_index < 0:
        return

    # store body index
    pick_body[0] = body_index

    # store target world
    pick_state[3] = hit_point_world[0]
    pick_state[4] = hit_point_world[1]
    pick_state[5] = hit_point_world[2]

    # compute and store local space attachment point
    X_wb = body_q[body_index]
    X_bw = wp.transform_inverse(X_wb)
    pick_pos_local = wp.transform_point(X_bw, hit_point_world)

    pick_state[0] = pick_pos_local[0]
    pick_state[1] = pick_pos_local[1]
    pick_state[2] = pick_pos_local[2]


@wp.kernel
def apply_picking_force_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    pick_body_arr: wp.array(dtype=int),
    pick_state: wp.array(dtype=float),
):
    pick_body = pick_body_arr[0]
    if pick_body < 0:
        return

    pick_pos_local = wp.vec3(pick_state[0], pick_state[1], pick_state[2])
    pick_target_world = wp.vec3(pick_state[3], pick_state[4], pick_state[5])
    pick_stiffness = pick_state[6]
    pick_damping = pick_state[7]
    angular_damping = 1.0  # Damping coefficient for angular velocity

    # world space attachment point
    X_wb = body_q[pick_body]
    pick_pos_world = wp.transform_point(X_wb, pick_pos_local)

    # center of mass
    com = wp.transform_get_translation(X_wb)

    # get velocity of attachment point
    omega = wp.spatial_bottom(body_qd[pick_body])
    vel_com = wp.spatial_top(body_qd[pick_body])
    vel_world = vel_com + wp.cross(omega, pick_pos_world - com)

    # compute spring force
    f = pick_stiffness * (pick_target_world - pick_pos_world) - pick_damping * vel_world

    # compute torque with angular damping
    t = wp.cross(pick_pos_world - com, f) - angular_damping * omega

    # apply force and torque
    wp.atomic_add(body_f, pick_body, wp.spatial_vector(f, t))


@wp.kernel
def update_pick_target_kernel(
    p: wp.vec3,
    d: wp.vec3,
    # read-write
    pick_state: wp.array(dtype=float),
):
    # get current target position
    current_target = wp.vec3(pick_state[3], pick_state[4], pick_state[5])

    # compute distance from ray origin to current target
    dist = wp.length(current_target - p)

    # project new target onto sphere with same radius
    new_target = p + d * dist

    pick_state[3] = new_target[0]
    pick_state[4] = new_target[1]
    pick_state[5] = new_target[2]


@wp.kernel
def update_shape_xforms(
    shape_xforms: wp.array(dtype=wp.transform),
    shape_parents: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    world_xforms: wp.array(dtype=wp.transform),
):
    tid = wp.tid()

    shape_xform = shape_xforms[tid]
    shape_parent = shape_parents[tid]

    if shape_parent >= 0:
        world_xform = wp.transform_multiply(body_q[shape_parent], shape_xform)
    else:
        world_xform = shape_xform

    world_xforms[tid] = world_xform


@wp.kernel
def compute_contact_lines(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
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
