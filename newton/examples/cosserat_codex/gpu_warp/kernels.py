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

"""Warp kernels used by the GPU-resident Cosserat solver."""

from __future__ import annotations

import warp as wp

from newton.examples.cosserat_codex import warp_cosserat_codex as base


@wp.func
def _warp_quat_correction_g(q: wp.quat, dtheta: wp.vec3) -> wp.quat:
    norm_sq = dtheta.x * dtheta.x + dtheta.y * dtheta.y + dtheta.z * dtheta.z
    if norm_sq < 1.0e-20:
        return q
    corr_x = 0.5 * (q.w * dtheta.x + q.z * dtheta.y - q.y * dtheta.z)
    corr_y = 0.5 * (-q.z * dtheta.x + q.w * dtheta.y + q.x * dtheta.z)
    corr_z = 0.5 * (q.y * dtheta.x - q.x * dtheta.y + q.w * dtheta.z)
    corr_w = 0.5 * (-q.x * dtheta.x - q.y * dtheta.y - q.z * dtheta.z)
    q_new = wp.quat(q.x + corr_x, q.y + corr_y, q.z + corr_z, q.w + corr_w)
    return base._warp_quat_normalize(q_new)


@wp.func
def _warp_jacobian_dot(
    jacobian: wp.array(dtype=wp.float32),
    edge: int,
    col: int,
    dl0: float,
    dl1: float,
    dl2: float,
    dl3: float,
    dl4: float,
    dl5: float,
) -> float:
    return (
        jacobian[base._warp_jacobian_index(edge, 0, col)] * dl0
        + jacobian[base._warp_jacobian_index(edge, 1, col)] * dl1
        + jacobian[base._warp_jacobian_index(edge, 2, col)] * dl2
        + jacobian[base._warp_jacobian_index(edge, 3, col)] * dl3
        + jacobian[base._warp_jacobian_index(edge, 4, col)] * dl4
        + jacobian[base._warp_jacobian_index(edge, 5, col)] * dl5
    )


@wp.kernel
def _warp_apply_direct_corrections(
    predicted_positions: wp.array(dtype=wp.vec3),
    predicted_orientations: wp.array(dtype=wp.quat),
    inv_masses: wp.array(dtype=wp.float32),
    quat_inv_masses: wp.array(dtype=wp.float32),
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
    delta_lambda: wp.array(dtype=wp.float32),
    lambda_sum: wp.array(dtype=wp.float32),
    n_edges: int,
    max_delta_out: wp.array(dtype=wp.float32),
    max_corr_out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid != 0:
        return

    max_delta = float(0.0)
    max_corr = float(0.0)

    for edge in range(n_edges):
        base_idx = edge * 6
        dl0 = delta_lambda[base_idx + 0]
        dl1 = delta_lambda[base_idx + 1]
        dl2 = delta_lambda[base_idx + 2]
        dl3 = delta_lambda[base_idx + 3]
        dl4 = delta_lambda[base_idx + 4]
        dl5 = delta_lambda[base_idx + 5]

        lambda_sum[base_idx + 0] = lambda_sum[base_idx + 0] + dl0
        lambda_sum[base_idx + 1] = lambda_sum[base_idx + 1] + dl1
        lambda_sum[base_idx + 2] = lambda_sum[base_idx + 2] + dl2
        lambda_sum[base_idx + 3] = lambda_sum[base_idx + 3] + dl3
        lambda_sum[base_idx + 4] = lambda_sum[base_idx + 4] + dl4
        lambda_sum[base_idx + 5] = lambda_sum[base_idx + 5] + dl5

        abs_dl = wp.abs(dl0)
        if abs_dl > max_delta:
            max_delta = abs_dl
        abs_dl = wp.abs(dl1)
        if abs_dl > max_delta:
            max_delta = abs_dl
        abs_dl = wp.abs(dl2)
        if abs_dl > max_delta:
            max_delta = abs_dl
        abs_dl = wp.abs(dl3)
        if abs_dl > max_delta:
            max_delta = abs_dl
        abs_dl = wp.abs(dl4)
        if abs_dl > max_delta:
            max_delta = abs_dl
        abs_dl = wp.abs(dl5)
        if abs_dl > max_delta:
            max_delta = abs_dl

        inv_m0 = inv_masses[edge]
        inv_m1 = inv_masses[edge + 1]

        if inv_m0 > 0.0:
            dp0_x = _warp_jacobian_dot(jacobian_pos, edge, 0, dl0, dl1, dl2, dl3, dl4, dl5)
            dp0_y = _warp_jacobian_dot(jacobian_pos, edge, 1, dl0, dl1, dl2, dl3, dl4, dl5)
            dp0_z = _warp_jacobian_dot(jacobian_pos, edge, 2, dl0, dl1, dl2, dl3, dl4, dl5)
            dp0 = wp.vec3(dp0_x * inv_m0, dp0_y * inv_m0, dp0_z * inv_m0)
            predicted_positions[edge] = predicted_positions[edge] + dp0
            corr = wp.sqrt(dp0.x * dp0.x + dp0.y * dp0.y + dp0.z * dp0.z)
            if corr > max_corr:
                max_corr = corr

        if inv_m1 > 0.0:
            dp1_x = _warp_jacobian_dot(jacobian_pos, edge, 3, dl0, dl1, dl2, dl3, dl4, dl5)
            dp1_y = _warp_jacobian_dot(jacobian_pos, edge, 4, dl0, dl1, dl2, dl3, dl4, dl5)
            dp1_z = _warp_jacobian_dot(jacobian_pos, edge, 5, dl0, dl1, dl2, dl3, dl4, dl5)
            dp1 = wp.vec3(dp1_x * inv_m1, dp1_y * inv_m1, dp1_z * inv_m1)
            predicted_positions[edge + 1] = predicted_positions[edge + 1] + dp1
            corr = wp.sqrt(dp1.x * dp1.x + dp1.y * dp1.y + dp1.z * dp1.z)
            if corr > max_corr:
                max_corr = corr

        if quat_inv_masses[edge] > 0.0:
            dtheta0 = wp.vec3(
                _warp_jacobian_dot(jacobian_rot, edge, 0, dl0, dl1, dl2, dl3, dl4, dl5),
                _warp_jacobian_dot(jacobian_rot, edge, 1, dl0, dl1, dl2, dl3, dl4, dl5),
                _warp_jacobian_dot(jacobian_rot, edge, 2, dl0, dl1, dl2, dl3, dl4, dl5),
            )
            corr = wp.sqrt(dtheta0.x * dtheta0.x + dtheta0.y * dtheta0.y + dtheta0.z * dtheta0.z)
            if corr > max_corr:
                max_corr = corr
            predicted_orientations[edge] = _warp_quat_correction_g(predicted_orientations[edge], dtheta0)

        if quat_inv_masses[edge + 1] > 0.0:
            dtheta1 = wp.vec3(
                _warp_jacobian_dot(jacobian_rot, edge, 3, dl0, dl1, dl2, dl3, dl4, dl5),
                _warp_jacobian_dot(jacobian_rot, edge, 4, dl0, dl1, dl2, dl3, dl4, dl5),
                _warp_jacobian_dot(jacobian_rot, edge, 5, dl0, dl1, dl2, dl3, dl4, dl5),
            )
            corr = wp.sqrt(dtheta1.x * dtheta1.x + dtheta1.y * dtheta1.y + dtheta1.z * dtheta1.z)
            if corr > max_corr:
                max_corr = corr
            predicted_orientations[edge + 1] = _warp_quat_correction_g(predicted_orientations[edge + 1], dtheta1)

    max_delta_out[0] = max_delta
    max_corr_out[0] = max_corr


@wp.kernel
def _warp_constraint_max(
    constraint_values: wp.array(dtype=wp.float32),
    n_edges: int,
    out_max: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid != 0:
        return
    max_val = float(0.0)
    for edge in range(n_edges):
        base_idx = edge * 6
        norm_sq = float(0.0)
        for j in range(6):
            val = constraint_values[base_idx + j]
            norm_sq += val * val
        norm = wp.sqrt(norm_sq)
        if norm > max_val:
            max_val = norm
    out_max[0] = max_val


@wp.kernel
def _warp_zero_float(arr: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    arr[tid] = 0.0


@wp.kernel
def _warp_zero_2d(arr: wp.array2d(dtype=wp.float32), rows: int, cols: int):
    tid = wp.tid()
    if tid < rows * cols:
        row = tid // cols
        col = tid - row * cols
        arr[row, col] = 0.0


@wp.kernel
def _warp_copy_with_offset(
    src: wp.array(dtype=wp.vec3),
    offset: wp.vec3,
    start: int,
    dst: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    dst[start + i] = src[i] + offset


@wp.kernel
def _warp_copy_from_offset(
    src: wp.array(dtype=wp.vec3),
    offset: wp.vec3,
    start: int,
    dst: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    dst[i] = src[start + i] - offset


@wp.kernel
def _warp_build_segment_lines(
    positions: wp.array(dtype=wp.vec3),
    offset: wp.vec3,
    start: int,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    idx = start + i
    starts[idx] = positions[i] + offset
    ends[idx] = positions[i + 1] + offset


@wp.kernel
def _warp_apply_floor_collisions(
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    min_z: float,
    restitution: float,
):
    i = wp.tid()
    pos = positions[i]
    if pos.z < min_z:
        clamped = wp.vec3(pos.x, pos.y, min_z)
        positions[i] = clamped
        predicted[i] = clamped
        vel = velocities[i]
        if vel.z < 0.0:
            velocities[i] = wp.vec3(vel.x, vel.y, -restitution * vel.z)


@wp.kernel
def _warp_apply_root_translation(
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dx: float,
    dy: float,
    dz: float,
):
    tid = wp.tid()
    if tid != 0:
        return
    pos = positions[0]
    new_pos = wp.vec3(pos.x + dx, pos.y + dy, pos.z + dz)
    positions[0] = new_pos
    predicted[0] = new_pos
    velocities[0] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _warp_zero_root_velocities(
    velocities: wp.array(dtype=wp.vec3),
    angular_velocities: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid != 0:
        return
    velocities[0] = wp.vec3(0.0, 0.0, 0.0)
    angular_velocities[0] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _warp_set_root_orientation(
    orientations: wp.array(dtype=wp.quat),
    predicted: wp.array(dtype=wp.quat),
    prev: wp.array(dtype=wp.quat),
    q: wp.quat,
):
    tid = wp.tid()
    if tid != 0:
        return
    orientations[0] = q
    predicted[0] = q
    prev[0] = q


@wp.kernel
def _warp_update_velocities_from_positions(
    old_positions: wp.array(dtype=wp.vec3),
    new_positions: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    dt: float,
    velocities: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if inv_masses[tid] == 0.0:
        velocities[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    velocities[tid] = (new_positions[tid] - old_positions[tid]) / dt


@wp.func
def _closest_point_on_edge(
    point: wp.vec3,
    edge_start: wp.vec3,
    edge_end: wp.vec3,
) -> wp.vec2:
    """Return (t, distance_sq) where t is parameter along edge and distance_sq is squared distance."""
    edge = edge_end - edge_start
    edge_len_sq = wp.dot(edge, edge)
    if edge_len_sq < 1.0e-12:
        # Degenerate edge - return midpoint
        return wp.vec2(0.5, wp.length_sq(point - edge_start))
    t = wp.dot(point - edge_start, edge) / edge_len_sq
    # Clamp t to [0, 1] for closest point on segment
    t_clamped = wp.clamp(t, 0.0, 1.0)
    closest = edge_start + t_clamped * edge
    dist_sq = wp.length_sq(point - closest)
    return wp.vec2(t, dist_sq)


@wp.kernel
def _warp_apply_track_sliding(
    positions: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    track_start: wp.vec3,
    track_end: wp.vec3,
    stiffness: float,
    start_idx: int,
    end_idx: int,
):
    """
    Constrain particles to slide along a track (line segment).

    For each particle between start_idx and end_idx, project it onto the track
    and apply a correction scaled by stiffness. Only applies correction if the
    particle's projection is interior to the track (0 < t < 1).
    """
    tid = wp.tid()
    idx = start_idx + tid
    if idx >= end_idx:
        return

    # Skip fixed particles
    if inv_masses[idx] <= 0.0:
        return

    pos = positions[idx]
    result = _closest_point_on_edge(pos, track_start, track_end)
    t = result.x
    dist_sq = result.y

    # Only apply constraint if particle projects to interior of track
    # and has some distance from the track
    if t > 0.0 and t < 1.0 and dist_sq > 1.0e-10:
        # Compute closest point on track
        edge = track_end - track_start
        closest = track_start + t * edge
        # Correction vector from particle to track
        correction = closest - pos
        # Apply correction scaled by stiffness
        new_pos = pos + correction * stiffness
        positions[idx] = new_pos
        predicted_positions[idx] = new_pos


__all__ = [
    "_warp_apply_direct_corrections",
    "_warp_apply_floor_collisions",
    "_warp_apply_root_translation",
    "_warp_apply_track_sliding",
    "_warp_build_segment_lines",
    "_warp_constraint_max",
    "_warp_copy_from_offset",
    "_warp_copy_with_offset",
    "_warp_set_root_orientation",
    "_warp_zero_2d",
    "_warp_zero_float",
    "_warp_zero_root_velocities",
    "_warp_update_velocities_from_positions",
]
