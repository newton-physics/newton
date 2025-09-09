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

from __future__ import annotations

import warp as wp


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
    body_com: wp.array(dtype=wp.vec3),
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
    com = wp.transform_point(X_wb, body_com[pick_body])

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
    pick_camera_front: wp.vec3,
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
def compute_contact_points(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    # outputs
    contact_pos0: wp.array(dtype=wp.vec3),
    contact_pos1: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        contact_pos0[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        contact_pos1[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        return
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        contact_pos0[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        contact_pos1[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
        return

    body_a = shape_body[shape_a]
    body_b = shape_body[shape_b]
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    if body_a >= 0:
        X_wb_a = body_q[body_a]
    if body_b >= 0:
        X_wb_b = body_q[body_b]

    contact_pos0[tid] = wp.transform_point(X_wb_a, contact_point0[tid])
    contact_pos1[tid] = wp.transform_point(X_wb_b, contact_point1[tid])
