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

"""Warp kernels for collision, constraint corrections, and utility operations."""

from __future__ import annotations

import warp as wp

from .kernels_math import _warp_jacobian_index, _warp_quat_normalize

# ==============================================================================
# Utility operations
# ==============================================================================


@wp.kernel
def _warp_zero_float(arr: wp.array(dtype=wp.float32)):
    """Zero out a float array."""
    tid = wp.tid()
    arr[tid] = 0.0


@wp.kernel
def _warp_zero_2d(arr: wp.array2d(dtype=wp.float32), rows: int, cols: int):
    """Zero out a 2D float array."""
    tid = wp.tid()
    if tid < rows * cols:
        row = tid // cols
        col = tid - row * cols
        arr[row, col] = 0.0


@wp.kernel
def _warp_zero_vec3(arr: wp.array(dtype=wp.vec3)):
    """Zero out a vec3 array."""
    tid = wp.tid()
    arr[tid] = wp.vec3(0.0, 0.0, 0.0)


# ==============================================================================
# Floor collision
# ==============================================================================


@wp.kernel
def _warp_apply_floor_collisions(
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    min_z: float,
    restitution: float,
):
    """Apply floor collision constraints."""
    i = wp.tid()
    pred = predicted[i]
    if pred.z < min_z:
        clamped = wp.vec3(pred.x, pred.y, min_z)
        predicted[i] = clamped
        vel = velocities[i]
        if vel.z < 0.0:
            velocities[i] = wp.vec3(vel.x, vel.y, -restitution * vel.z)


# ==============================================================================
# Root control
# ==============================================================================


@wp.kernel
def _warp_apply_root_translation(
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dx: float,
    dy: float,
    dz: float,
):
    """Apply translation to the root particle."""
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
    """Zero out root particle velocities."""
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
    """Set the root particle orientation."""
    tid = wp.tid()
    if tid != 0:
        return
    orientations[0] = q
    predicted[0] = q
    prev[0] = q


# ==============================================================================
# Constraint max computation
# ==============================================================================


@wp.kernel
def _warp_constraint_max(
    constraint_values: wp.array(dtype=wp.float32),
    n_edges: int,
    out_max: wp.array(dtype=wp.float32),
):
    """Compute maximum constraint violation norm (parallel)."""
    edge = wp.tid()
    if edge >= n_edges:
        return
    base_idx = edge * 6
    norm_sq = float(0.0)
    for j in range(6):
        val = constraint_values[base_idx + j]
        norm_sq += val * val
    norm = wp.sqrt(norm_sq)
    wp.atomic_max(out_max, 0, norm)


# ==============================================================================
# Inverse inertia world-frame computation
# ==============================================================================


@wp.kernel
def _warp_compute_inv_inertia_world(
    orientations: wp.array(dtype=wp.quat),
    quat_inv_masses: wp.array(dtype=wp.float32),
    inv_inertia_local_diag: wp.vec3,
    inv_inertia_out: wp.array(dtype=wp.float32),
):
    """Compute inverse inertia tensor in world frame from local frame diagonal."""
    i = wp.tid()
    base = i * 9

    if quat_inv_masses[i] <= 0.0:
        for j in range(9):
            inv_inertia_out[base + j] = 0.0
        return

    q = orientations[i]
    xx = q.x * q.x
    yy = q.y * q.y
    zz = q.z * q.z
    xy = q.x * q.y
    xz = q.x * q.z
    yz = q.y * q.z
    wx = q.w * q.x
    wy = q.w * q.y
    wz = q.w * q.z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    d0 = inv_inertia_local_diag.x
    d1 = inv_inertia_local_diag.y
    d2 = inv_inertia_local_diag.z

    inv_inertia_out[base + 0] = r00 * d0 * r00 + r01 * d1 * r01 + r02 * d2 * r02
    inv_inertia_out[base + 1] = r00 * d0 * r10 + r01 * d1 * r11 + r02 * d2 * r12
    inv_inertia_out[base + 2] = r00 * d0 * r20 + r01 * d1 * r21 + r02 * d2 * r22
    inv_inertia_out[base + 3] = r10 * d0 * r00 + r11 * d1 * r01 + r12 * d2 * r02
    inv_inertia_out[base + 4] = r10 * d0 * r10 + r11 * d1 * r11 + r12 * d2 * r12
    inv_inertia_out[base + 5] = r10 * d0 * r20 + r11 * d1 * r21 + r12 * d2 * r22
    inv_inertia_out[base + 6] = r20 * d0 * r00 + r21 * d1 * r01 + r22 * d2 * r02
    inv_inertia_out[base + 7] = r20 * d0 * r10 + r21 * d1 * r11 + r22 * d2 * r12
    inv_inertia_out[base + 8] = r20 * d0 * r20 + r21 * d1 * r21 + r22 * d2 * r22


@wp.kernel
def _warp_compute_inv_inertia_world_batched(
    orientations: wp.array(dtype=wp.quat),
    quat_inv_masses: wp.array(dtype=wp.float32),
    inv_inertia_local_diag: wp.array(dtype=wp.vec3),
    particle_rod_id: wp.array(dtype=wp.int32),
    inv_inertia_out: wp.array(dtype=wp.float32),
):
    """Compute inverse inertia tensors for all rods in a single launch."""
    i = wp.tid()
    base = i * 9

    if quat_inv_masses[i] <= 0.0:
        for j in range(9):
            inv_inertia_out[base + j] = 0.0
        return

    rod_id = particle_rod_id[i]
    inv_inertia_local = inv_inertia_local_diag[rod_id]

    q = orientations[i]
    xx = q.x * q.x
    yy = q.y * q.y
    zz = q.z * q.z
    xy = q.x * q.y
    xz = q.x * q.z
    yz = q.y * q.z
    wx = q.w * q.x
    wy = q.w * q.y
    wz = q.w * q.z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    d0 = inv_inertia_local.x
    d1 = inv_inertia_local.y
    d2 = inv_inertia_local.z

    inv_inertia_out[base + 0] = r00 * d0 * r00 + r01 * d1 * r01 + r02 * d2 * r02
    inv_inertia_out[base + 1] = r00 * d0 * r10 + r01 * d1 * r11 + r02 * d2 * r12
    inv_inertia_out[base + 2] = r00 * d0 * r20 + r01 * d1 * r21 + r02 * d2 * r22
    inv_inertia_out[base + 3] = r10 * d0 * r00 + r11 * d1 * r01 + r12 * d2 * r02
    inv_inertia_out[base + 4] = r10 * d0 * r10 + r11 * d1 * r11 + r12 * d2 * r12
    inv_inertia_out[base + 5] = r10 * d0 * r20 + r11 * d1 * r21 + r12 * d2 * r22
    inv_inertia_out[base + 6] = r20 * d0 * r00 + r21 * d1 * r01 + r22 * d2 * r02
    inv_inertia_out[base + 7] = r20 * d0 * r10 + r21 * d1 * r11 + r22 * d2 * r12
    inv_inertia_out[base + 8] = r20 * d0 * r20 + r21 * d1 * r21 + r22 * d2 * r22


# ==============================================================================
# Quaternion correction helper
# ==============================================================================


@wp.func
def _warp_quat_correction_g(q: wp.quat, dtheta: wp.vec3) -> wp.quat:
    """Apply a small angular correction to a quaternion."""
    norm_sq = dtheta.x * dtheta.x + dtheta.y * dtheta.y + dtheta.z * dtheta.z
    if norm_sq < 1.0e-20:
        return q
    corr_x = 0.5 * (q.w * dtheta.x + q.z * dtheta.y - q.y * dtheta.z)
    corr_y = 0.5 * (-q.z * dtheta.x + q.w * dtheta.y + q.x * dtheta.z)
    corr_z = 0.5 * (q.y * dtheta.x - q.x * dtheta.y + q.w * dtheta.z)
    corr_w = 0.5 * (-q.x * dtheta.x - q.y * dtheta.y - q.z * dtheta.z)
    q_new = wp.quat(q.x + corr_x, q.y + corr_y, q.z + corr_z, q.w + corr_w)
    return _warp_quat_normalize(q_new)


# ==============================================================================
# Jacobian dot product helper
# ==============================================================================


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
    """Compute dot product of Jacobian column with delta lambda."""
    return (
        jacobian[_warp_jacobian_index(edge, 0, col)] * dl0
        + jacobian[_warp_jacobian_index(edge, 1, col)] * dl1
        + jacobian[_warp_jacobian_index(edge, 2, col)] * dl2
        + jacobian[_warp_jacobian_index(edge, 3, col)] * dl3
        + jacobian[_warp_jacobian_index(edge, 4, col)] * dl4
        + jacobian[_warp_jacobian_index(edge, 5, col)] * dl5
    )


@wp.func
def _inv_inertia_mul_vec_kernels(inv_inertia: wp.array(dtype=wp.float32), particle_idx: int, v: wp.vec3) -> wp.vec3:
    """Multiply inverse inertia tensor (3x3) by a vector."""
    base_idx = particle_idx * 9
    return wp.vec3(
        inv_inertia[base_idx + 0] * v.x + inv_inertia[base_idx + 1] * v.y + inv_inertia[base_idx + 2] * v.z,
        inv_inertia[base_idx + 3] * v.x + inv_inertia[base_idx + 4] * v.y + inv_inertia[base_idx + 5] * v.z,
        inv_inertia[base_idx + 6] * v.x + inv_inertia[base_idx + 7] * v.y + inv_inertia[base_idx + 8] * v.z,
    )


# ==============================================================================
# Direct correction application (parallel version)
# ==============================================================================


@wp.kernel
def _warp_compute_corrections_parallel(
    predicted_positions: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    quat_inv_masses: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
    delta_lambda: wp.array(dtype=wp.float32),
    lambda_sum: wp.array(dtype=wp.float32),
    n_edges: int,
    pos_corrections: wp.array(dtype=wp.vec3),
    rot_corrections: wp.array(dtype=wp.vec3),
    max_delta_out: wp.array(dtype=wp.float32),
    max_corr_out: wp.array(dtype=wp.float32),
):
    """Compute position and rotation corrections per edge (parallel phase 1)."""
    edge = wp.tid()
    if edge >= n_edges:
        return

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

    local_max_delta = wp.max(
        wp.max(wp.max(wp.abs(dl0), wp.abs(dl1)), wp.max(wp.abs(dl2), wp.abs(dl3))),
        wp.max(wp.abs(dl4), wp.abs(dl5)),
    )
    wp.atomic_max(max_delta_out, 0, local_max_delta)

    inv_m0 = inv_masses[edge]
    inv_m1 = inv_masses[edge + 1]

    if inv_m0 > 0.0:
        dp0_x = _warp_jacobian_dot(jacobian_pos, edge, 0, dl0, dl1, dl2, dl3, dl4, dl5)
        dp0_y = _warp_jacobian_dot(jacobian_pos, edge, 1, dl0, dl1, dl2, dl3, dl4, dl5)
        dp0_z = _warp_jacobian_dot(jacobian_pos, edge, 2, dl0, dl1, dl2, dl3, dl4, dl5)
        dp0 = wp.vec3(dp0_x * inv_m0, dp0_y * inv_m0, dp0_z * inv_m0)
        wp.atomic_add(pos_corrections, edge, dp0)

    if inv_m1 > 0.0:
        dp1_x = _warp_jacobian_dot(jacobian_pos, edge, 3, dl0, dl1, dl2, dl3, dl4, dl5)
        dp1_y = _warp_jacobian_dot(jacobian_pos, edge, 4, dl0, dl1, dl2, dl3, dl4, dl5)
        dp1_z = _warp_jacobian_dot(jacobian_pos, edge, 5, dl0, dl1, dl2, dl3, dl4, dl5)
        dp1 = wp.vec3(dp1_x * inv_m1, dp1_y * inv_m1, dp1_z * inv_m1)
        wp.atomic_add(pos_corrections, edge + 1, dp1)

    if quat_inv_masses[edge] > 0.0:
        j_t0_delta = wp.vec3(
            _warp_jacobian_dot(jacobian_rot, edge, 0, dl0, dl1, dl2, dl3, dl4, dl5),
            _warp_jacobian_dot(jacobian_rot, edge, 1, dl0, dl1, dl2, dl3, dl4, dl5),
            _warp_jacobian_dot(jacobian_rot, edge, 2, dl0, dl1, dl2, dl3, dl4, dl5),
        )
        dtheta0 = _inv_inertia_mul_vec_kernels(inv_inertia, edge, j_t0_delta)
        wp.atomic_add(rot_corrections, edge, dtheta0)

    if quat_inv_masses[edge + 1] > 0.0:
        j_t1_delta = wp.vec3(
            _warp_jacobian_dot(jacobian_rot, edge, 3, dl0, dl1, dl2, dl3, dl4, dl5),
            _warp_jacobian_dot(jacobian_rot, edge, 4, dl0, dl1, dl2, dl3, dl4, dl5),
            _warp_jacobian_dot(jacobian_rot, edge, 5, dl0, dl1, dl2, dl3, dl4, dl5),
        )
        dtheta1 = _inv_inertia_mul_vec_kernels(inv_inertia, edge + 1, j_t1_delta)
        wp.atomic_add(rot_corrections, edge + 1, dtheta1)


@wp.kernel
def _warp_compute_corrections_parallel_batched(
    predicted_positions: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    quat_inv_masses: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
    delta_lambda: wp.array(dtype=wp.float32),
    lambda_sum: wp.array(dtype=wp.float32),
    rod_offsets: wp.array(dtype=wp.int32),
    edge_offsets: wp.array(dtype=wp.int32),
    edge_rod_id: wp.array(dtype=wp.int32),
    pos_corrections: wp.array(dtype=wp.vec3),
    rot_corrections: wp.array(dtype=wp.vec3),
    max_delta_out: wp.array(dtype=wp.float32),
    max_corr_out: wp.array(dtype=wp.float32),
):
    """Compute position and rotation corrections for all rods in a single launch.

    This batched version uses rod_offsets and edge_offsets to map global edge
    indices to the correct particle indices in concatenated arrays.
    """
    global_edge = wp.tid()
    rod_id = edge_rod_id[global_edge]
    local_edge = global_edge - edge_offsets[rod_id]
    p0_idx = rod_offsets[rod_id] + local_edge
    p1_idx = p0_idx + 1

    base_idx = global_edge * 6
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

    local_max_delta = wp.max(
        wp.max(wp.max(wp.abs(dl0), wp.abs(dl1)), wp.max(wp.abs(dl2), wp.abs(dl3))),
        wp.max(wp.abs(dl4), wp.abs(dl5)),
    )
    wp.atomic_max(max_delta_out, 0, local_max_delta)

    inv_m0 = inv_masses[p0_idx]
    inv_m1 = inv_masses[p1_idx]

    if inv_m0 > 0.0:
        dp0_x = _warp_jacobian_dot(jacobian_pos, global_edge, 0, dl0, dl1, dl2, dl3, dl4, dl5)
        dp0_y = _warp_jacobian_dot(jacobian_pos, global_edge, 1, dl0, dl1, dl2, dl3, dl4, dl5)
        dp0_z = _warp_jacobian_dot(jacobian_pos, global_edge, 2, dl0, dl1, dl2, dl3, dl4, dl5)
        dp0 = wp.vec3(dp0_x * inv_m0, dp0_y * inv_m0, dp0_z * inv_m0)
        wp.atomic_add(pos_corrections, p0_idx, dp0)

    if inv_m1 > 0.0:
        dp1_x = _warp_jacobian_dot(jacobian_pos, global_edge, 3, dl0, dl1, dl2, dl3, dl4, dl5)
        dp1_y = _warp_jacobian_dot(jacobian_pos, global_edge, 4, dl0, dl1, dl2, dl3, dl4, dl5)
        dp1_z = _warp_jacobian_dot(jacobian_pos, global_edge, 5, dl0, dl1, dl2, dl3, dl4, dl5)
        dp1 = wp.vec3(dp1_x * inv_m1, dp1_y * inv_m1, dp1_z * inv_m1)
        wp.atomic_add(pos_corrections, p1_idx, dp1)

    if quat_inv_masses[p0_idx] > 0.0:
        j_t0_delta = wp.vec3(
            _warp_jacobian_dot(jacobian_rot, global_edge, 0, dl0, dl1, dl2, dl3, dl4, dl5),
            _warp_jacobian_dot(jacobian_rot, global_edge, 1, dl0, dl1, dl2, dl3, dl4, dl5),
            _warp_jacobian_dot(jacobian_rot, global_edge, 2, dl0, dl1, dl2, dl3, dl4, dl5),
        )
        dtheta0 = _inv_inertia_mul_vec_kernels(inv_inertia, p0_idx, j_t0_delta)
        wp.atomic_add(rot_corrections, p0_idx, dtheta0)

    if quat_inv_masses[p1_idx] > 0.0:
        j_t1_delta = wp.vec3(
            _warp_jacobian_dot(jacobian_rot, global_edge, 3, dl0, dl1, dl2, dl3, dl4, dl5),
            _warp_jacobian_dot(jacobian_rot, global_edge, 4, dl0, dl1, dl2, dl3, dl4, dl5),
            _warp_jacobian_dot(jacobian_rot, global_edge, 5, dl0, dl1, dl2, dl3, dl4, dl5),
        )
        dtheta1 = _inv_inertia_mul_vec_kernels(inv_inertia, p1_idx, j_t1_delta)
        wp.atomic_add(rot_corrections, p1_idx, dtheta1)


@wp.kernel
def _warp_apply_accumulated_corrections(
    predicted_positions: wp.array(dtype=wp.vec3),
    predicted_orientations: wp.array(dtype=wp.quat),
    pos_corrections: wp.array(dtype=wp.vec3),
    rot_corrections: wp.array(dtype=wp.vec3),
    n_particles: int,
):
    """Apply accumulated corrections (parallel phase 2)."""
    particle = wp.tid()
    if particle >= n_particles:
        return
    predicted_positions[particle] = predicted_positions[particle] + pos_corrections[particle]
    predicted_orientations[particle] = _warp_quat_correction_g(
        predicted_orientations[particle], rot_corrections[particle]
    )


@wp.kernel
def _warp_merge_delta_lambda(
    stretch_dl: wp.array(dtype=wp.float32),
    darboux_dl: wp.array(dtype=wp.float32),
    combined_dl: wp.array(dtype=wp.float32),
    n_edges: int,
):
    """Merge split delta_lambda arrays into combined 6-vector format."""
    edge = wp.tid()
    if edge >= n_edges:
        return
    for i in range(3):
        combined_dl[edge * 6 + i] = stretch_dl[edge * 3 + i]
    for i in range(3):
        combined_dl[edge * 6 + i + 3] = darboux_dl[edge * 3 + i]
