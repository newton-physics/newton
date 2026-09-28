# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Blocked forward responses for compact DVI operators."""

from functools import cache

import warp as wp

from ...linalg.factorize.llt_blocked_rcm import get_float32_array_offset_ptr
from .kernels import _compact_schur_fits
from .sparse_kernels import _subgroup_sum_32

wp.set_module_options({"enable_backward": False})


@cache
def make_response_kernel():
    """Compute four columns of ``L^-1 P D C`` per 128-thread CUDA block.

    The unpacked RCM factor uses 32-row tiles. Its symbolic pattern lets
    independent column groups skip zero tiles while sharing factor loads.
    Both the whitened responses and the compact-world coupling are stored
    contiguously with row stride ``nu``.
    """
    block_size = 32
    width = 4

    @wp.kernel
    def response(
        dim: wp.array[wp.int32],
        njc: wp.array[wp.int32],
        mio: wp.array[wp.int32],
        vio: wp.array[wp.int32],
        preconditioner: wp.array[wp.float32],
        factor: wp.array[wp.float32],
        permutation: wp.array[wp.int32],
        response_mio: wp.array[wp.int32],
        response_stride: wp.array[wp.int32],
        coupling: wp.array[wp.float32],
        output: wp.array[wp.float32],
        tpo: wp.array[wp.int32],
        pattern: wp.array[wp.int32],
        columns_per_world: wp.int32,
    ):
        task, lane = wp.tid()
        groups = (columns_per_world + width - 1) // width
        world = task // groups
        column = (task % groups) * width
        n = njc[world]
        nu = dim[world] - n
        if column >= nu or not _compact_schur_fits(n, nu, response_stride[world]):
            return
        matrix = wp.array(ptr=get_float32_array_offset_ptr(factor, mio[world]), shape=(n, n), dtype=wp.float32)
        result = wp.array(
            ptr=get_float32_array_offset_ptr(output, response_mio[world]), shape=(n, nu), dtype=wp.float32
        )
        tiles = (n + block_size - 1) // block_size
        for i in range(0, n, block_size):
            rhs = wp.tile_zeros(shape=(block_size, width), dtype=wp.float32, storage="shared")
            local_row = lane // width
            local_col = lane % width
            active = local_row < block_size and i + local_row < n and column + local_col < nu
            value = wp.float32(0.0)
            if active:
                row = permutation[vio[world] + i + local_row]
                value = preconditioner[vio[world] + row] * coupling[response_mio[world] + row * nu + column + local_col]
            wp.tile_scatter_masked(rhs, local_row, local_col, value, active)
            diagonal = wp.tile_load(matrix, shape=(block_size, block_size), offset=(i, i))
            for j in range(0, i, block_size):
                if pattern[tpo[world] + (i // block_size) * tiles + j // block_size] != 0:
                    block = wp.tile_load(matrix, shape=(block_size, block_size), offset=(i, j))
                    previous = wp.tile_load(result, shape=(block_size, width), offset=(j, column))
                    wp.tile_matmul(block, previous, rhs, alpha=-1.0)
            wp.tile_lower_solve_inplace(diagonal, rhs)
            wp.tile_store(result, rhs, offset=(i, column))

    return response


@wp.kernel
def _update_forward_bilateral_rhs(
    dim: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    bvio: wp.array[wp.int32],
    rio: wp.array[wp.int32],
    response: wp.array[wp.float32],
    initial: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
    y: wp.array[wp.float32],
):
    """Update cached ``L^-1 P b`` by subtracting whitened responses times the impulse change."""
    world, row, lane = wp.tid()
    n = njc[world]
    if row >= n:
        return
    nu = dim[world] - n
    offset = vio[world] + n
    value = wp.float32(0.0)
    for column in range(lane, nu, 32):
        delta = lambdas[offset + column] - initial[offset + column]
        value += response[rio[world] + row * nu + column] * delta
    value = _subgroup_sum_32(value)
    if lane == 0:
        y[bvio[world] + row] -= value


@wp.kernel
def _add_forward_bilateral_gradient(
    dim: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    bvio: wp.array[wp.int32],
    rio: wp.array[wp.int32],
    response: wp.array[wp.float32],
    y: wp.array[wp.float32],
    gradient: wp.array[wp.float32],
):
    """Add the eliminated bilateral gradient using ``Y^T y``."""
    world, column, lane = wp.tid()
    n = njc[world]
    nu = dim[world] - n
    if column >= nu:
        return
    value = wp.float32(0.0)
    for row in range(lane, n, 32):
        value += response[rio[world] + row * nu + column] * y[bvio[world] + row]
    value = _subgroup_sum_32(value)
    if lane == 0:
        gradient[vio[world] + n + column] += value
