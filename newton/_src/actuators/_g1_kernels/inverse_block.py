# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

wp.set_module_options({"block_dim": 64, "enable_backward": False})

Vector49 = wp.types.vector(length=49, dtype=wp.float32)


@wp.kernel(enable_backward=False)
def inverse_kernel(
    H: wp.array3d[wp.float32],
    counts: wp.array[wp.int32],
    inverse: wp.array3d[wp.float32],
):
    batch, lane = wp.tid()
    count = counts[batch]
    factor = wp.tile_zeros(shape=(64, 65), dtype=wp.float32, storage="shared")
    for column in range(count):
        value = float(0.0)
        if lane >= column and lane < count:
            value = H[batch, lane, column]
            for inner in range(column):
                value -= factor[lane, inner] * factor[column, inner]
            if lane == column:
                value = wp.max(value, 1.1920928955078125e-7 * wp.max(wp.abs(H[batch, column, column]), 1.0))
                value = wp.sqrt(value)
        factor[lane, column] = value
        root = factor[column, column]
        if lane > column and lane < count:
            value = value / root
        factor[lane, column] = value
    if lane < count:
        solution = Vector49(0.0)
        for row in range(count):
            value = float(0.0)
            if row == lane:
                value = 1.0
            for inner in range(row):
                value -= factor[row, inner] * solution[inner]
            solution[row] = value / factor[row, row]
        for reverse in range(count):
            row = count - 1 - reverse
            value = solution[row]
            for inner in range(row + 1, count):
                value -= factor[inner, row] * solution[inner]
            solution[row] = value / factor[row, row]
        for row in range(count):
            inverse[batch, row, lane] = solution[row]
