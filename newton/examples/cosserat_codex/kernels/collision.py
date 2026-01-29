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

"""Warp kernels for collision, constraints, and utility operations.

This module contains kernels for:
- Floor collision detection and response
- Direct constraint correction application
- Track sliding and concentric constraints
- Utility operations (copy, zero, etc.)
"""

from __future__ import annotations

import warp as wp

from .math import _warp_jacobian_index, _warp_quat_normalize


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
def _warp_copy_with_offset(
    src: wp.array(dtype=wp.vec3),
    offset: wp.vec3,
    start: int,
    dst: wp.array(dtype=wp.vec3),
):
    """Copy vectors with offset to destination starting at given index."""
    i = wp.tid()
    dst[start + i] = src[i] + offset


@wp.kernel
def _warp_copy_from_offset(
    src: wp.array(dtype=wp.vec3),
    offset: wp.vec3,
    start: int,
    dst: wp.array(dtype=wp.vec3),
):
    """Copy vectors from source starting at given index, subtracting offset."""
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
    """Build line segment endpoints for visualization."""
    i = wp.tid()
    idx = start + i
    starts[idx] = positions[i] + offset
    ends[idx] = positions[i + 1] + offset


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
    """Apply floor collision constraints.
    
    Clamps particles above the floor and reflects velocity with restitution.
    
    Args:
        positions: Current positions (updated in-place).
        predicted: Predicted positions (updated in-place).
        velocities: Velocities (updated in-place).
        min_z: Minimum Z coordinate (floor + radius).
        restitution: Coefficient of restitution.
    """
    i = wp.tid()
    pos = positions[i]
    if pos.z < min_z:
        clamped = wp.vec3(pos.x, pos.y, min_z)
        positions[i] = clamped
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


@wp.kernel
def _warp_update_velocities_from_positions(
    old_positions: wp.array(dtype=wp.vec3),
    new_positions: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    dt: float,
    velocities: wp.array(dtype=wp.vec3),
):
    """Update velocities from position change."""
    tid = wp.tid()
    if inv_masses[tid] == 0.0:
        velocities[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    velocities[tid] = (new_positions[tid] - old_positions[tid]) / dt


# ==============================================================================
# Constraint max computation
# ==============================================================================


@wp.kernel
def _warp_constraint_max(
    constraint_values: wp.array(dtype=wp.float32),
    n_edges: int,
    out_max: wp.array(dtype=wp.float32),
):
    """Compute maximum constraint violation norm."""
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
# Direct correction application
# ==============================================================================


@wp.kernel
def _warp_apply_direct_corrections(
    predicted_positions: wp.array(dtype=wp.vec3),
    predicted_orientations: wp.array(dtype=wp.quat),
    inv_masses: wp.array(dtype=wp.float32),
    quat_inv_masses: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
    delta_lambda: wp.array(dtype=wp.float32),
    lambda_sum: wp.array(dtype=wp.float32),
    n_edges: int,
    max_delta_out: wp.array(dtype=wp.float32),
    max_corr_out: wp.array(dtype=wp.float32),
):
    """Apply position and rotation corrections from deltaLambda.

    Position correction: inv_mass * J_pos^T * deltaLambda
    Rotation correction: inv_inertia * J_rot^T * deltaLambda

    Uses actual inverse masses to correctly handle locked particles (inv_mass=0).
    
    Args:
        predicted_positions: Predicted positions (updated in-place).
        predicted_orientations: Predicted orientations (updated in-place).
        inv_masses: Inverse masses.
        quat_inv_masses: Inverse rotational masses.
        inv_inertia: Inverse inertia tensors.
        jacobian_pos: Position Jacobians.
        jacobian_rot: Rotation Jacobians.
        delta_lambda: Lagrange multiplier increments.
        lambda_sum: Accumulated Lagrange multipliers (updated in-place).
        n_edges: Number of edges.
        max_delta_out: Output max delta lambda norm.
        max_corr_out: Output max correction magnitude.
    """
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

        # Use actual inverse masses from the array
        inv_m0 = inv_masses[edge]
        inv_m1 = inv_masses[edge + 1]

        # Position correction for particle 0: inv_m0 * J_p0^T * deltaLambda
        if inv_m0 > 0.0:
            dp0_x = _warp_jacobian_dot(jacobian_pos, edge, 0, dl0, dl1, dl2, dl3, dl4, dl5)
            dp0_y = _warp_jacobian_dot(jacobian_pos, edge, 1, dl0, dl1, dl2, dl3, dl4, dl5)
            dp0_z = _warp_jacobian_dot(jacobian_pos, edge, 2, dl0, dl1, dl2, dl3, dl4, dl5)
            dp0 = wp.vec3(dp0_x * inv_m0, dp0_y * inv_m0, dp0_z * inv_m0)
            predicted_positions[edge] = predicted_positions[edge] + dp0
            corr = wp.sqrt(dp0.x * dp0.x + dp0.y * dp0.y + dp0.z * dp0.z)
            if corr > max_corr:
                max_corr = corr

        # Position correction for particle 1: inv_m1 * J_p1^T * deltaLambda
        if inv_m1 > 0.0:
            dp1_x = _warp_jacobian_dot(jacobian_pos, edge, 3, dl0, dl1, dl2, dl3, dl4, dl5)
            dp1_y = _warp_jacobian_dot(jacobian_pos, edge, 4, dl0, dl1, dl2, dl3, dl4, dl5)
            dp1_z = _warp_jacobian_dot(jacobian_pos, edge, 5, dl0, dl1, dl2, dl3, dl4, dl5)
            dp1 = wp.vec3(dp1_x * inv_m1, dp1_y * inv_m1, dp1_z * inv_m1)
            predicted_positions[edge + 1] = predicted_positions[edge + 1] + dp1
            corr = wp.sqrt(dp1.x * dp1.x + dp1.y * dp1.y + dp1.z * dp1.z)
            if corr > max_corr:
                max_corr = corr

        # Rotation correction for particle 0: inv_I0 * J_t0^T * deltaLambda
        if quat_inv_masses[edge] > 0.0:
            # Compute J_t0^T * deltaLambda
            j_t0_delta = wp.vec3(
                _warp_jacobian_dot(jacobian_rot, edge, 0, dl0, dl1, dl2, dl3, dl4, dl5),
                _warp_jacobian_dot(jacobian_rot, edge, 1, dl0, dl1, dl2, dl3, dl4, dl5),
                _warp_jacobian_dot(jacobian_rot, edge, 2, dl0, dl1, dl2, dl3, dl4, dl5),
            )
            # Multiply by inverse inertia tensor
            dtheta0 = _inv_inertia_mul_vec_kernels(inv_inertia, edge, j_t0_delta)
            corr = wp.sqrt(dtheta0.x * dtheta0.x + dtheta0.y * dtheta0.y + dtheta0.z * dtheta0.z)
            if corr > max_corr:
                max_corr = corr
            predicted_orientations[edge] = _warp_quat_correction_g(predicted_orientations[edge], dtheta0)

        # Rotation correction for particle 1: inv_I1 * J_t1^T * deltaLambda
        if quat_inv_masses[edge + 1] > 0.0:
            # Compute J_t1^T * deltaLambda
            j_t1_delta = wp.vec3(
                _warp_jacobian_dot(jacobian_rot, edge, 3, dl0, dl1, dl2, dl3, dl4, dl5),
                _warp_jacobian_dot(jacobian_rot, edge, 4, dl0, dl1, dl2, dl3, dl4, dl5),
                _warp_jacobian_dot(jacobian_rot, edge, 5, dl0, dl1, dl2, dl3, dl4, dl5),
            )
            # Multiply by inverse inertia tensor
            dtheta1 = _inv_inertia_mul_vec_kernels(inv_inertia, edge + 1, j_t1_delta)
            corr = wp.sqrt(dtheta1.x * dtheta1.x + dtheta1.y * dtheta1.y + dtheta1.z * dtheta1.z)
            if corr > max_corr:
                max_corr = corr
            predicted_orientations[edge + 1] = _warp_quat_correction_g(predicted_orientations[edge + 1], dtheta1)

    max_delta_out[0] = max_delta
    max_corr_out[0] = max_corr


# ==============================================================================
# Track sliding constraint
# ==============================================================================


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
    """Constrain particles to slide along a track (line segment).

    For each particle between start_idx and end_idx, project it onto the track
    and apply a correction scaled by stiffness. Only applies correction if the
    particle's projection is interior to the track (0 < t < 1).
    
    Args:
        positions: Current positions (updated in-place).
        predicted_positions: Predicted positions (updated in-place).
        inv_masses: Inverse masses.
        track_start: Start of the track line segment.
        track_end: End of the track line segment.
        stiffness: Constraint stiffness.
        start_idx: First particle index to constrain.
        end_idx: Last particle index (exclusive).
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


@wp.kernel
def _warp_set_root_on_track(
    positions: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    track_start: wp.vec3,
    track_end: wp.vec3,
    insertion: float,
    root_idx: int,
):
    """Set the root particle position along the track based on insertion depth.

    The insertion value is clamped to [0, track_length] and the root particle
    is positioned at that distance from track_start along the track direction.
    
    Args:
        positions: Positions (updated in-place).
        predicted_positions: Predicted positions (updated in-place).
        velocities: Velocities (updated in-place).
        track_start: Start of the track.
        track_end: End of the track.
        insertion: Insertion depth along the track.
        root_idx: Index of the root particle.
    """
    tid = wp.tid()
    if tid != 0:
        return

    track_vec = track_end - track_start
    track_length = wp.length(track_vec)

    if track_length < 1.0e-10:
        return

    track_dir = track_vec / track_length

    # Clamp insertion to valid range
    clamped_insertion = wp.clamp(insertion, 0.0, track_length)

    # Compute new root position
    new_pos = track_start + track_dir * clamped_insertion

    positions[root_idx] = new_pos
    predicted_positions[root_idx] = new_pos
    velocities[root_idx] = wp.vec3(0.0, 0.0, 0.0)


# ==============================================================================
# Concentric constraint
# ==============================================================================


@wp.kernel
def _warp_apply_concentric_constraint(
    positions_a: wp.array(dtype=wp.vec3),
    predicted_a: wp.array(dtype=wp.vec3),
    inv_masses_a: wp.array(dtype=wp.float32),
    num_points_a: int,
    positions_b: wp.array(dtype=wp.vec3),
    predicted_b: wp.array(dtype=wp.vec3),
    inv_masses_b: wp.array(dtype=wp.float32),
    num_points_b: int,
    rest_lengths_a: wp.array(dtype=wp.float32),
    rest_lengths_b: wp.array(dtype=wp.float32),
    insertion_diff: float,
    stiffness: float,
    weight_a: float,
    weight_b: float,
    use_inv_mass_sq: int,
    start_particle: int,
):
    """Constrain two rods to be concentric (like guidewire inside catheter).

    For each particle in rod A (inner rod), compute its corresponding point on rod B's
    centerline (outer rod) using arclength-based parametrization and apply bilateral
    corrections to make them concentric.

    The correspondence is computed purely from insertion difference and arclengths,
    not from closest point search. This models the physical behavior where the inner
    rod slides through the outer rod based on relative insertion depths.

    Args:
        positions_a: Current positions of rod A particles.
        predicted_a: Predicted positions of rod A particles.
        inv_masses_a: Inverse masses of rod A particles.
        num_points_a: Number of particles in rod A.
        positions_b: Current positions of rod B particles.
        predicted_b: Predicted positions of rod B particles.
        inv_masses_b: Inverse masses of rod B particles.
        num_points_b: Number of particles in rod B.
        rest_lengths_a: Rest lengths of rod A segments.
        rest_lengths_b: Rest lengths of rod B segments.
        insertion_diff: Difference in insertion depth (insertion_a - insertion_b).
        stiffness: Constraint stiffness [0, 1].
        weight_a: Weight for rod A corrections (inner rod).
        weight_b: Weight for rod B corrections (outer rod).
        use_inv_mass_sq: If 1, use squared barycentric weights in denominator.
        start_particle: First particle to apply constraint to.
    """
    tid = wp.tid()
    i = tid  # particle index in rod A

    if i >= num_points_a:
        return

    if i < start_particle:
        return

    # Skip fixed particles
    if inv_masses_a[i] <= 0.0:
        return

    # Compute arclength from root for particle i in rod A
    # Add insertion_diff to account for relative positioning
    arclength_a = float(insertion_diff)
    for k in range(i):
        arclength_a = arclength_a + rest_lengths_a[k]

    # Skip if arclength is negative (particle is before rod B's root)
    if arclength_a < 0.0:
        return

    # Find segment on rod B that contains this arclength
    # Iterate through segments accumulating rest lengths
    j = int(0)
    prev_arclength_b = float(0.0)
    arclength_b = float(0.0)

    # Find the segment j where prev_arclength_b <= arclength_a < arclength_b
    while j < num_points_b - 1:
        arclength_b = prev_arclength_b + rest_lengths_b[j]
        if arclength_b > arclength_a:
            break
        prev_arclength_b = arclength_b
        j = j + 1

    # Check if we're past the end of rod B
    if j >= num_points_b - 1:
        return

    # Compute parametric position t within segment j
    segment_length = arclength_b - prev_arclength_b
    if segment_length < 1.0e-10:
        return

    t = (arclength_a - prev_arclength_b) / segment_length
    t = wp.clamp(t, 0.0, 1.0)

    # Compute the corresponding point on rod B
    p_b0 = predicted_b[j]
    p_b1 = predicted_b[j + 1]
    target_point = p_b0 + t * (p_b1 - p_b0)

    # Compute correction direction
    pos_a = predicted_a[i]
    dp = pos_a - target_point
    dp_len = wp.length(dp)

    # Skip very small corrections
    if dp_len < 1.0e-8:
        return

    dp_normalized = dp / dp_len

    # Barycentric weights for distributing correction to rod B vertices
    b0 = 1.0 - t
    b1 = t

    # Compute scaling factor based on weight distribution
    # The denominator accounts for how correction is distributed
    if use_inv_mass_sq == 1:
        denom = weight_a + weight_b * b0 * b0 + weight_b * b1 * b1
    else:
        denom = weight_a + weight_b * b0 + weight_b * b1

    if denom < 1.0e-10:
        return

    # Constraint magnitude
    s = dp_len / denom * stiffness

    # Ease out at tip of rod B (last segment)
    if j == num_points_b - 2:
        s = s * (1.0 - t)

    # Apply corrections
    # Rod A (inner) moves toward rod B's centerline
    if inv_masses_a[i] > 0.0:
        correction_a = dp_normalized * s * weight_a
        new_pos_a = pos_a - correction_a
        positions_a[i] = new_pos_a
        predicted_a[i] = new_pos_a

    # Rod B (outer) moves toward rod A's particle
    # Distributed between the two vertices of the segment
    if inv_masses_b[j] > 0.0:
        correction_b0 = dp_normalized * s * b0 * weight_b
        new_pos_b0 = p_b0 + correction_b0
        positions_b[j] = new_pos_b0
        predicted_b[j] = new_pos_b0

    if inv_masses_b[j + 1] > 0.0:
        correction_b1 = dp_normalized * s * b1 * weight_b
        new_pos_b1 = p_b1 + correction_b1
        positions_b[j + 1] = new_pos_b1
        predicted_b[j + 1] = new_pos_b1


# ==============================================================================
# Concentric constraint v2 - Inner rod stays on outer rod centerline
# ==============================================================================


@wp.func
def _compute_arclength_to_segment(
    target_arclength: float,
    rest_lengths: wp.array(dtype=wp.float32),
    num_segments: int,
) -> wp.vec2:
    """Find segment index and local t-value for a given arc-length.

    Given a target arc-length along a rod, find which segment contains that point
    and the parametric t-value within that segment.

    Args:
        target_arclength: Arc-length from rod root.
        rest_lengths: Array of rest lengths for each segment.
        num_segments: Total number of segments.

    Returns:
        vec2(segment_index, t_value). If target is past end, returns (num_segments-1, 1.0).
        If target is negative, returns (-1, 0.0).
    """
    if target_arclength < 0.0:
        return wp.vec2(-1.0, 0.0)

    cumulative = float(0.0)
    for seg in range(num_segments):
        seg_len = rest_lengths[seg]
        next_cumulative = cumulative + seg_len

        if target_arclength <= next_cumulative:
            # Found the segment
            if seg_len > 1.0e-10:
                t = (target_arclength - cumulative) / seg_len
            else:
                t = 0.0
            return wp.vec2(float(seg), t)

        cumulative = next_cumulative

    # Past the end of the rod
    return wp.vec2(float(num_segments - 1), 1.0)


@wp.kernel
def _warp_apply_concentric_constraint_v2(
    # Outer rod (catheter) - the centerline we constrain to
    outer_positions: wp.array(dtype=wp.vec3),
    outer_predicted: wp.array(dtype=wp.vec3),
    outer_inv_masses: wp.array(dtype=wp.float32),
    outer_num_points: int,
    outer_rest_lengths: wp.array(dtype=wp.float32),
    # Inner rod (guidewire) - constrained to stay on outer centerline
    inner_positions: wp.array(dtype=wp.vec3),
    inner_predicted: wp.array(dtype=wp.vec3),
    inner_inv_masses: wp.array(dtype=wp.float32),
    inner_num_points: int,
    inner_rest_lengths: wp.array(dtype=wp.float32),
    # Constraint parameters
    insertion_diff: float,  # insertion_inner - insertion_outer
    stiffness: float,
    weight_inner: float,  # weight for inner rod corrections
    weight_outer: float,  # weight for outer rod corrections
    start_particle: int,  # first particle to apply constraint to
    end_particle: int,  # last particle to apply constraint to (-1 for all)
):
    """Constrain inner rod to stay on outer rod's centerline.

    For each particle on the inner rod (e.g., guidewire), compute the exact
    corresponding point on the outer rod's centerline (e.g., catheter) using
    arc-length parametrization and apply PBD-style corrections.

    The correspondence uses the insertion difference to map inner rod particles
    to outer rod centerline positions. This models a guidewire sliding through
    a catheter based on their relative insertion depths.

    Math:
        For inner particle i at arc-length s_inner from inner rod root:
        - Corresponding outer arc-length: s_outer = s_inner + insertion_diff
        - Find segment j on outer rod containing s_outer
        - Compute t = (s_outer - cumsum[j]) / segment_length
        - Target point P = outer[j] * (1-t) + outer[j+1] * t
        - Apply PBD correction to minimize |inner[i] - P|

    Args:
        outer_positions: Current positions of outer rod particles.
        outer_predicted: Predicted positions of outer rod particles.
        outer_inv_masses: Inverse masses of outer rod particles.
        outer_num_points: Number of particles in outer rod.
        outer_rest_lengths: Rest lengths of outer rod segments.
        inner_positions: Current positions of inner rod particles.
        inner_predicted: Predicted positions of inner rod particles.
        inner_inv_masses: Inverse masses of inner rod particles.
        inner_num_points: Number of particles in inner rod.
        inner_rest_lengths: Rest lengths of inner rod segments.
        insertion_diff: insertion_inner - insertion_outer.
        stiffness: Constraint stiffness [0, 1].
        weight_inner: Weight for inner rod corrections.
        weight_outer: Weight for outer rod corrections.
        start_particle: First inner particle index to constrain.
        end_particle: Last inner particle index to constrain (-1 = all).
    """
    tid = wp.tid()
    i = tid  # inner rod particle index

    if i >= inner_num_points:
        return

    if i < start_particle:
        return

    effective_end = inner_num_points
    if end_particle >= 0 and end_particle < inner_num_points:
        effective_end = end_particle + 1
    if i >= effective_end:
        return

    # Skip fixed particles (root)
    w_i = inner_inv_masses[i]
    if w_i <= 0.0:
        return

    # Compute arc-length from root for inner particle i
    inner_arclength = float(0.0)
    for k in range(i):
        inner_arclength = inner_arclength + inner_rest_lengths[k]

    # Compute corresponding arc-length on outer rod
    # insertion_diff = insertion_inner - insertion_outer
    # If inner is more inserted (positive), the corresponding outer point is further along
    outer_arclength = inner_arclength + insertion_diff

    # Skip if outside outer rod
    if outer_arclength < 0.0:
        return

    num_outer_segments = outer_num_points - 1
    if num_outer_segments <= 0:
        return

    # Find segment and t-value on outer rod
    seg_t = _compute_arclength_to_segment(outer_arclength, outer_rest_lengths, num_outer_segments)
    j = int(seg_t[0])
    t = seg_t[1]

    # Check bounds
    if j < 0 or j >= num_outer_segments:
        return

    # Skip if at very tip of outer rod (ease out)
    if j == num_outer_segments - 1 and t > 0.99:
        return

    # Clamp t
    t = wp.clamp(t, 0.0, 1.0)

    # Compute target point on outer rod centerline
    p_out_0 = outer_predicted[j]
    p_out_1 = outer_predicted[j + 1]
    target = p_out_0 * (1.0 - t) + p_out_1 * t

    # Current inner particle position
    p_inner = inner_predicted[i]

    # Constraint violation: distance from centerline
    delta = p_inner - target
    delta_len = wp.length(delta)

    # Skip tiny violations
    if delta_len < 1.0e-8:
        return

    delta_norm = delta / delta_len

    # Barycentric weights for outer rod vertices
    b0 = 1.0 - t
    b1 = t

    # Get inverse masses for outer rod vertices
    w_out_0 = outer_inv_masses[j]
    w_out_1 = outer_inv_masses[j + 1]

    # Effective inverse mass (PBD-style)
    # Inner contributes w_i * weight_inner
    # Outer vertex 0 contributes w_out_0 * b0^2 * weight_outer
    # Outer vertex 1 contributes w_out_1 * b1^2 * weight_outer
    w_eff = w_i * weight_inner + w_out_0 * b0 * b0 * weight_outer + w_out_1 * b1 * b1 * weight_outer

    if w_eff < 1.0e-10:
        return

    # PBD correction magnitude
    lambda_val = delta_len / w_eff * stiffness

    # Ease out constraint at outer rod tip
    if j == num_outer_segments - 1:
        lambda_val = lambda_val * (1.0 - t)

    # Apply corrections
    # Inner rod moves toward target (negative direction along delta)
    if w_i > 0.0:
        correction_inner = delta_norm * lambda_val * w_i * weight_inner
        new_inner = p_inner - correction_inner
        inner_positions[i] = new_inner
        inner_predicted[i] = new_inner

    # Outer rod moves away from target (positive direction along delta)
    # Distributed between vertices based on barycentric weights
    if w_out_0 > 0.0:
        correction_0 = delta_norm * lambda_val * w_out_0 * b0 * weight_outer
        new_out_0 = p_out_0 + correction_0
        outer_positions[j] = new_out_0
        outer_predicted[j] = new_out_0

    if w_out_1 > 0.0:
        correction_1 = delta_norm * lambda_val * w_out_1 * b1 * weight_outer
        new_out_1 = p_out_1 + correction_1
        outer_positions[j + 1] = new_out_1
        outer_predicted[j + 1] = new_out_1


# ==============================================================================
# Inverse inertia computation
# ==============================================================================


@wp.kernel
def _warp_compute_inv_inertia_world(
    orientations: wp.array(dtype=wp.quat),
    quat_inv_masses: wp.array(dtype=wp.float32),
    inv_inertia_local_diag: wp.vec3,
    inv_inertia_out: wp.array(dtype=wp.float32),
):
    """Compute inverse inertia tensor in world frame from local frame diagonal.
    
    For a cylindrical rod segment, the local inertia tensor is diagonal:
    I_local = diag(I_perp, I_perp, I_axial)
    
    The world-frame inverse inertia is: inv_I_world = R * inv_I_local * R^T
    where R is the rotation matrix from the quaternion orientation.
    
    Args:
        orientations: Quaternion orientations per particle.
        quat_inv_masses: Inverse rotational mass mask (0 for locked particles).
        inv_inertia_local_diag: Diagonal of inverse inertia in local frame (vec3).
        inv_inertia_out: Output array of 9 floats per particle (row-major 3x3 matrix).
    """
    i = wp.tid()
    
    # Start of this particle's 3x3 matrix in the flat array
    base = i * 9
    
    if quat_inv_masses[i] <= 0.0:
        # Locked particle - zero inverse inertia (infinite inertia)
        for j in range(9):
            inv_inertia_out[base + j] = 0.0
        return
    
    q = orientations[i]
    
    # Extract rotation matrix from quaternion (column vectors)
    # R = [r0 | r1 | r2] where r0, r1, r2 are column vectors
    xx = q.x * q.x
    yy = q.y * q.y
    zz = q.z * q.z
    xy = q.x * q.y
    xz = q.x * q.z
    yz = q.y * q.z
    wx = q.w * q.x
    wy = q.w * q.y
    wz = q.w * q.z
    
    # Rotation matrix (row-major)
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)
    
    # Compute R * diag(inv_I) * R^T
    # Let d = inv_inertia_local_diag = (d0, d1, d2)
    # Result[i,j] = sum_k R[i,k] * d[k] * R[j,k]
    d0 = inv_inertia_local_diag.x
    d1 = inv_inertia_local_diag.y
    d2 = inv_inertia_local_diag.z
    
    # Row 0
    inv_inertia_out[base + 0] = r00 * d0 * r00 + r01 * d1 * r01 + r02 * d2 * r02  # [0,0]
    inv_inertia_out[base + 1] = r00 * d0 * r10 + r01 * d1 * r11 + r02 * d2 * r12  # [0,1]
    inv_inertia_out[base + 2] = r00 * d0 * r20 + r01 * d1 * r21 + r02 * d2 * r22  # [0,2]
    
    # Row 1
    inv_inertia_out[base + 3] = r10 * d0 * r00 + r11 * d1 * r01 + r12 * d2 * r02  # [1,0]
    inv_inertia_out[base + 4] = r10 * d0 * r10 + r11 * d1 * r11 + r12 * d2 * r12  # [1,1]
    inv_inertia_out[base + 5] = r10 * d0 * r20 + r11 * d1 * r21 + r12 * d2 * r22  # [1,2]
    
    # Row 2
    inv_inertia_out[base + 6] = r20 * d0 * r00 + r21 * d1 * r01 + r22 * d2 * r02  # [2,0]
    inv_inertia_out[base + 7] = r20 * d0 * r10 + r21 * d1 * r11 + r22 * d2 * r12  # [2,1]
    inv_inertia_out[base + 8] = r20 * d0 * r20 + r21 * d1 * r21 + r22 * d2 * r22  # [2,2]


__all__ = [
    # Utility operations
    "_warp_zero_float",
    "_warp_zero_2d",
    "_warp_copy_with_offset",
    "_warp_copy_from_offset",
    "_warp_build_segment_lines",
    # Floor collision
    "_warp_apply_floor_collisions",
    # Root control
    "_warp_apply_root_translation",
    "_warp_zero_root_velocities",
    "_warp_set_root_orientation",
    "_warp_update_velocities_from_positions",
    # Constraint diagnostics
    "_warp_constraint_max",
    # Direct corrections
    "_warp_apply_direct_corrections",
    # Track sliding
    "_warp_apply_track_sliding",
    "_warp_set_root_on_track",
    # Concentric constraint
    "_warp_apply_concentric_constraint",
    # Inverse inertia
    "_warp_compute_inv_inertia_world",
]
