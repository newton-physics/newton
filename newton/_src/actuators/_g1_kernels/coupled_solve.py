# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

wp.set_module_options({"block_dim": 64, "enable_backward": False})
Vec43 = wp.types.vector(length=43, dtype=wp.float64)
Vec44 = wp.types.vector(length=44, dtype=wp.float64)


@wp.func
def force(
    q: wp.float64,
    qd: wp.float64,
    tq: wp.float64,
    tqd: wp.float64,
    ff: wp.float64,
    kp: wp.float64,
    kd: wp.float64,
    bias: wp.float64,
):
    return bias + ff + kp * (tq - q) + kd * (tqd - qd)


@wp.kernel(enable_backward=False)
def solve(
    group_size: wp.array[wp.int32],
    group_art: wp.array[wp.int32],
    group_slot: wp.array2d[wp.int32],
    group_local: wp.array2d[wp.int32],
    inverse_blocks: wp.array3d[wp.float32],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    target_vel_indices: wp.array[wp.uint32],
    positions: wp.array[wp.float32],
    velocities: wp.array[wp.float32],
    target_pos: wp.array[wp.float32],
    target_vel: wp.array[wp.float32],
    feedforward: wp.array[wp.float32],
    params: wp.array2d[wp.float32],
    h: float,
    max_iters: int,
    residual_tol: float,
    update_tol: float,
    fd_epsilon: float,
    derivative_floor: float,
    warm_zero: int,
    clamp_params: wp.array2d[wp.float32],
    computed_efforts: wp.array[wp.float32],
    applied_efforts: wp.array[wp.float32],
):
    group, row = wp.tid()
    count = group_size[group]
    art = group_art[group]
    hd = wp.float64(h)
    q0 = wp.float64(0.0)
    v0 = wp.float64(0.0)
    tq = wp.float64(0.0)
    tv = wp.float64(0.0)
    ff = wp.float64(0.0)
    kp = wp.float64(0.0)
    kd = wp.float64(0.0)
    bias = wp.float64(0.0)
    limit = wp.float64(0.0)
    impulse = wp.float64(0.0)
    slot = int(0)
    inverse_row = Vec43()
    if row < count:
        slot = group_slot[group, row]
        local_row = group_local[group, row]
        q0 = wp.float64(positions[pos_indices[slot]])
        v0 = wp.float64(velocities[vel_indices[slot]])
        tq = wp.float64(target_pos[target_pos_indices[slot]])
        tv = wp.float64(target_vel[target_vel_indices[slot]])
        if feedforward:
            ff = wp.float64(feedforward[target_vel_indices[slot]])
        kp = wp.float64(params[slot, 0])
        kd = wp.float64(params[slot, 1])
        bias = wp.float64(params[slot, 2])
        limit = wp.float64(clamp_params[slot, 0])
        for column in range(count):
            inverse_row[column] = wp.float64(inverse_blocks[art, local_row, group_local[group, column]])
        if warm_zero == 0:
            impulse = hd * wp.clamp(force(q0, v0, tq, tv, ff, kp, kd, bias), -limit, limit)
    inverse = wp.tile(inverse_row)
    for _iteration in range(max_iters):
        impulses = wp.tile(impulse)
        velocity = v0
        for column in range(count):
            velocity += wp.tile_extract(inverse, column, row) * wp.tile_extract(impulses, column)
        position = q0 + hd * velocity
        effort = wp.clamp(force(position, velocity, tq, tv, ff, kp, kd, bias), -limit, limit)
        residual = impulse - hd * effort
        eps = wp.float64(fd_epsilon) * (wp.float64(1.0) + wp.abs(velocity))
        perturbed = wp.clamp(force(position + hd * eps, velocity + eps, tq, tv, ff, kp, kd, bias), -limit, limit)
        slope = (perturbed - effort) / eps
        residuals = wp.tile(residual * residual)
        residual_norm = wp.float64(0.0)
        for column in range(count):
            residual_norm += wp.tile_extract(residuals, column)
        if residual_norm < wp.float64(residual_tol) * wp.float64(residual_tol):
            break
        jacobian_row = Vec44()
        for column in range(count):
            entry = -hd * slope * wp.tile_extract(inverse, column, row)
            if row == column:
                entry += wp.float64(1.0)
            jacobian_row[column] = entry
        jacobian_row[43] = -residual
        jacobian = wp.tile(jacobian_row)
        diagonal = wp.float64(0.0)
        for pivot_index in range(count):
            pivot = wp.tile_extract(jacobian, pivot_index, pivot_index)
            if wp.abs(pivot) < wp.float64(derivative_floor):
                pivot = wp.where(pivot < wp.float64(0.0), -wp.float64(derivative_floor), wp.float64(derivative_floor))
            if row == pivot_index:
                diagonal = pivot
            factor = wp.float64(0.0)
            if row > pivot_index and row < count:
                factor = wp.tile_extract(jacobian, pivot_index, row) / pivot
            factors = wp.tile_reshape(wp.tile(factor), shape=(1, 64))
            factors_broadcast = wp.tile_broadcast(factors, shape=(44, 64))
            pivot_row = wp.tile_view(jacobian, offset=(0, pivot_index), shape=(44, 1))
            pivot_broadcast = wp.tile_broadcast(pivot_row, shape=(44, 64))
            jacobian -= factors_broadcast * pivot_broadcast
        deltas = wp.tile_zeros(shape=(43,), dtype=wp.float64, storage="shared")
        delta = wp.float64(0.0)
        update_norm = wp.float64(0.0)
        for backward in range(count):
            pivot_index = count - 1 - backward
            if row == pivot_index:
                total = wp.tile_extract(jacobian, 43, row)
                for column in range(pivot_index + 1, count):
                    total -= wp.tile_extract(jacobian, column, row) * wp.tile_extract(deltas, column)
                delta = total / diagonal
                impulse += delta
            wp.tile_scatter_masked(deltas, pivot_index, delta, row == pivot_index)
            delta_value = wp.tile_extract(deltas, pivot_index)
            update_norm += delta_value * delta_value
        if update_norm < wp.float64(update_tol) * wp.float64(update_tol):
            break
    impulses = wp.tile(impulse)
    velocity = v0
    for column in range(count):
        velocity += wp.tile_extract(inverse, column, row) * wp.tile_extract(impulses, column)
    if row < count:
        position = q0 + hd * velocity
        computed_efforts[slot] = wp.float32(force(position, velocity, tq, tv, ff, kp, kd, bias))
        applied_efforts[slot] = wp.float32(wp.clamp(impulse / hd, -limit, limit))
