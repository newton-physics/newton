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

"""Warp kernels for linear solvers.

This module contains kernels for solving linear systems:
- Tiled Cholesky for small systems
- Block Thomas for tridiagonal systems
- Banded Cholesky for banded systems
"""

from __future__ import annotations

import warp as wp

from .constants import BAND_KD, TILE
from .kernels_math import (
    _block_column,
    _block_column_3x3,
    _block_column_offset,
    _block_mul,
    _block_mul_vec,
    _block_set_column,
    _block_set_column_3x3,
    _block_set_column_offset,
    _block_solve,
    _block_sub,
    _load_block,
    _load_block_3x3,
    _load_block_offset,
    _load_vec,
    _load_vec3_block,
    _load_vec_offset,
    _mat33_cholesky,
    _mat33_cholesky_solve,
    _mat33_mul,
    _mat33_mul_vec3,
    _mat33_sub,
    _mat33_transpose,
    _store_vec,
    _store_vec3_block,
    _store_vec_offset,
)


@wp.kernel
def _warp_cholesky_solve_tile(
    A: wp.array2d(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
):
    """Solve a system using tiled Cholesky decomposition.

    Uses Warp's tile operations for efficient small matrix solves.

    Args:
        A: Dense SPD matrix (TILE x TILE).
        b: RHS vector (TILE).
        x: Output solution vector (TILE).
    """
    a_tile = wp.tile_load(A, shape=(TILE, TILE))
    b_tile = wp.tile_load(b, shape=TILE)
    L = wp.tile_cholesky(a_tile)
    x_tile = wp.tile_cholesky_solve(L, b_tile)
    wp.tile_store(x, x_tile)


@wp.kernel
def _warp_block_thomas_solve(
    diag_blocks: wp.array(dtype=wp.float32),
    offdiag_blocks: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    n_edges: int,
    c_blocks: wp.array(dtype=wp.float32),
    d_prime: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
):
    """Solve a block-tridiagonal system using Thomas algorithm.

    Solves (D - L - L^T) x = b where:
    - D is block diagonal (6x6 blocks)
    - L is block lower triangular

    Args:
        diag_blocks: Diagonal blocks (36 floats per edge).
        offdiag_blocks: Off-diagonal blocks (36 floats per edge).
        rhs: RHS vector (6 per edge).
        n_edges: Number of edges (blocks).
        c_blocks: Workspace for intermediate matrices.
        d_prime: Workspace for intermediate RHS.
        x: Output solution vector.
    """
    tid = wp.tid()
    if tid != 0:
        return
    if n_edges <= 0:
        return

    A0, B0, C0, D0 = _load_block(diag_blocks, 0)
    b0, b1 = _load_vec(rhs, 0)

    if n_edges > 1:
        for col in range(6):
            u0, u1 = _block_column(offdiag_blocks, 1, col)
            x0, x1 = _block_solve(A0, B0, C0, D0, u0, u1)
            _block_set_column(c_blocks, 0, col, x0, x1)
    else:
        zero = wp.vec3(0.0, 0.0, 0.0)
        for col in range(6):
            _block_set_column(c_blocks, 0, col, zero, zero)

    d0, d1 = _block_solve(A0, B0, C0, D0, b0, b1)
    _store_vec(d_prime, 0, d0, d1)

    for i in range(1, n_edges):
        DiA, DiB, DiC, DiD = _load_block(diag_blocks, i)
        # Load L_i^T (super-diagonal stored in offdiag_blocks)
        LtA, LtB, LtC, LtD = _load_block(offdiag_blocks, i)
        CpA, CpB, CpC, CpD = _load_block(c_blocks, i - 1)

        # Transpose to get L_i: [A B; C D]^T = [A^T C^T; B^T D^T]
        LiA = _mat33_transpose(LtA)
        LiB = _mat33_transpose(LtC)  # Swapped: B position gets C^T
        LiC = _mat33_transpose(LtB)  # Swapped: C position gets B^T
        LiD = _mat33_transpose(LtD)

        # Schur complement: T_i = D_i - L_i * C_{i-1}
        LCA, LCB, LCC, LCD = _block_mul(LiA, LiB, LiC, LiD, CpA, CpB, CpC, CpD)
        TiA, TiB, TiC, TiD = _block_sub(DiA, DiB, DiC, DiD, LCA, LCB, LCC, LCD)

        # RHS update: b'_i = b_i - L_i * d'_{i-1}
        bi0, bi1 = _load_vec(rhs, i)
        dp0, dp1 = _load_vec(d_prime, i - 1)
        ld0, ld1 = _block_mul_vec(LiA, LiB, LiC, LiD, dp0, dp1)
        bi0 = bi0 - ld0
        bi1 = bi1 - ld1

        if i < n_edges - 1:
            for col in range(6):
                u0, u1 = _block_column(offdiag_blocks, i + 1, col)
                x0, x1 = _block_solve(TiA, TiB, TiC, TiD, u0, u1)
                _block_set_column(c_blocks, i, col, x0, x1)
        else:
            zero = wp.vec3(0.0, 0.0, 0.0)
            for col in range(6):
                _block_set_column(c_blocks, i, col, zero, zero)

        di0, di1 = _block_solve(TiA, TiB, TiC, TiD, bi0, bi1)
        _store_vec(d_prime, i, di0, di1)

    dn0, dn1 = _load_vec(d_prime, n_edges - 1)
    _store_vec(x, n_edges - 1, dn0, dn1)
    for i in range(n_edges - 2, -1, -1):
        CiA, CiB, CiC, CiD = _load_block(c_blocks, i)
        xn0, xn1 = _load_vec(x, i + 1)
        cx0, cx1 = _block_mul_vec(CiA, CiB, CiC, CiD, xn0, xn1)
        di0, di1 = _load_vec(d_prime, i)
        xi0 = di0 - cx0
        xi1 = di1 - cx1
        _store_vec(x, i, xi0, xi1)


@wp.kernel
def _warp_block_thomas_solve_batched(
    diag_blocks: wp.array(dtype=wp.float32),
    offdiag_blocks: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    edge_offsets: wp.array(dtype=wp.int32),
    n_rods: int,
    c_blocks: wp.array(dtype=wp.float32),
    d_prime: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
):
    """Solve block-tridiagonal systems for multiple rods in parallel.

    This batched version launches with dim=n_rods, where each GPU thread
    independently solves the Thomas algorithm for one rod. Since different
    rods are completely independent, this enables inter-rod parallelism.

    Solves (D - L - L^T) x = b where:
    - D is block diagonal (6x6 blocks)
    - L is block lower triangular

    Args:
        diag_blocks: Concatenated diagonal blocks for all rods (36 floats per edge).
        offdiag_blocks: Concatenated off-diagonal blocks for all rods (36 floats per edge).
        rhs: Concatenated RHS vectors for all rods (6 per edge).
        edge_offsets: Cumulative edge offsets [n_rods + 1]. Rod i has edges
                     [edge_offsets[i], edge_offsets[i+1]).
        n_rods: Number of rods to process.
        c_blocks: Concatenated workspace for intermediate matrices.
        d_prime: Concatenated workspace for intermediate RHS.
        x: Concatenated output solution vector.
    """
    rod_id = wp.tid()
    if rod_id >= n_rods:
        return

    # Get this rod's edge range
    edge_start = edge_offsets[rod_id]
    edge_end = edge_offsets[rod_id + 1]
    n_edges = edge_end - edge_start

    if n_edges <= 0:
        return

    # Forward elimination phase
    # Load first block
    A0, B0, C0, D0 = _load_block_offset(diag_blocks, edge_start, 0)
    b0, b1 = _load_vec_offset(rhs, edge_start, 0)

    # Compute c_0 = D_0^{-1} * L_1^T (only if we have more than one block)
    if n_edges > 1:
        for col in range(6):
            u0, u1 = _block_column_offset(offdiag_blocks, edge_start, 1, col)
            x0, x1 = _block_solve(A0, B0, C0, D0, u0, u1)
            _block_set_column_offset(c_blocks, edge_start, 0, col, x0, x1)
    else:
        zero = wp.vec3(0.0, 0.0, 0.0)
        for col in range(6):
            _block_set_column_offset(c_blocks, edge_start, 0, col, zero, zero)

    # Compute d'_0 = D_0^{-1} * b_0
    d0, d1 = _block_solve(A0, B0, C0, D0, b0, b1)
    _store_vec_offset(d_prime, edge_start, 0, d0, d1)

    # Forward pass: eliminate lower triangular part
    for i in range(1, n_edges):
        # Load D_i (diagonal block)
        DiA, DiB, DiC, DiD = _load_block_offset(diag_blocks, edge_start, i)

        # Load L_i^T (super-diagonal stored in offdiag_blocks)
        LtA, LtB, LtC, LtD = _load_block_offset(offdiag_blocks, edge_start, i)

        # Load C_{i-1}
        CpA, CpB, CpC, CpD = _load_block_offset(c_blocks, edge_start, i - 1)

        # Transpose to get L_i: [A B; C D]^T = [A^T C^T; B^T D^T]
        LiA = _mat33_transpose(LtA)
        LiB = _mat33_transpose(LtC)  # Swapped: B position gets C^T
        LiC = _mat33_transpose(LtB)  # Swapped: C position gets B^T
        LiD = _mat33_transpose(LtD)

        # Schur complement: T_i = D_i - L_i * C_{i-1}
        LCA, LCB, LCC, LCD = _block_mul(LiA, LiB, LiC, LiD, CpA, CpB, CpC, CpD)
        TiA, TiB, TiC, TiD = _block_sub(DiA, DiB, DiC, DiD, LCA, LCB, LCC, LCD)

        # RHS update: b'_i = b_i - L_i * d'_{i-1}
        bi0, bi1 = _load_vec_offset(rhs, edge_start, i)
        dp0, dp1 = _load_vec_offset(d_prime, edge_start, i - 1)
        ld0, ld1 = _block_mul_vec(LiA, LiB, LiC, LiD, dp0, dp1)
        bi0 = bi0 - ld0
        bi1 = bi1 - ld1

        # Compute c_i = T_i^{-1} * L_{i+1}^T (if not last block)
        if i < n_edges - 1:
            for col in range(6):
                u0, u1 = _block_column_offset(offdiag_blocks, edge_start, i + 1, col)
                x0, x1 = _block_solve(TiA, TiB, TiC, TiD, u0, u1)
                _block_set_column_offset(c_blocks, edge_start, i, col, x0, x1)
        else:
            zero = wp.vec3(0.0, 0.0, 0.0)
            for col in range(6):
                _block_set_column_offset(c_blocks, edge_start, i, col, zero, zero)

        # Compute d'_i = T_i^{-1} * b'_i
        di0, di1 = _block_solve(TiA, TiB, TiC, TiD, bi0, bi1)
        _store_vec_offset(d_prime, edge_start, i, di0, di1)

    # Back substitution: x_{n-1} = d'_{n-1}
    dn0, dn1 = _load_vec_offset(d_prime, edge_start, n_edges - 1)
    _store_vec_offset(x, edge_start, n_edges - 1, dn0, dn1)

    # Back substitution: x_i = d'_i - c_i * x_{i+1}
    for i in range(n_edges - 2, -1, -1):
        CiA, CiB, CiC, CiD = _load_block_offset(c_blocks, edge_start, i)
        xn0, xn1 = _load_vec_offset(x, edge_start, i + 1)
        cx0, cx1 = _block_mul_vec(CiA, CiB, CiC, CiD, xn0, xn1)
        di0, di1 = _load_vec_offset(d_prime, edge_start, i)
        xi0 = di0 - cx0
        xi1 = di1 - cx1
        _store_vec_offset(x, edge_start, i, xi0, xi1)


@wp.kernel
def _warp_block_thomas_solve_3x3(
    diag_blocks: wp.array(dtype=wp.float32),
    offdiag_blocks: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    n_edges: int,
    c_blocks: wp.array(dtype=wp.float32),
    d_prime: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
):
    """Solve a 3x3 block-tridiagonal system using Thomas algorithm.

    This is a simplified version for the split Thomas solver that operates
    on 3x3 blocks instead of 6x6 blocks. It can be used for either the
    stretch or darboux subsystem independently.

    Solves (D - L - L^T) x = b where:
    - D is block diagonal (3x3 blocks)
    - L is block lower triangular (3x3 blocks)

    Args:
        diag_blocks: Diagonal blocks (9 floats per edge).
        offdiag_blocks: Off-diagonal blocks (9 floats per edge).
        rhs: RHS vector (3 per edge).
        n_edges: Number of edges (blocks).
        c_blocks: Workspace for intermediate matrices (9 floats per edge).
        d_prime: Workspace for intermediate RHS (3 floats per edge).
        x: Output solution vector (3 floats per edge).
    """
    tid = wp.tid()
    if tid != 0:
        return
    if n_edges <= 0:
        return

    # === Forward Elimination ===

    # First block
    D0 = _load_block_3x3(diag_blocks, 0)
    b0 = _load_vec3_block(rhs, 0)
    L0 = _mat33_cholesky(D0)

    # Compute c_0 = D_0^{-1} * L_1^T (if more than one block)
    if n_edges > 1:
        # Solve D_0 * C_0 = L_1^T column by column
        for col in range(3):
            u = _block_column_3x3(offdiag_blocks, 1, col)
            v = _mat33_cholesky_solve(L0, u)
            _block_set_column_3x3(c_blocks, 0, col, v)
    else:
        zero = wp.vec3(0.0, 0.0, 0.0)
        for col in range(3):
            _block_set_column_3x3(c_blocks, 0, col, zero)

    # d'_0 = D_0^{-1} * b_0
    d0 = _mat33_cholesky_solve(L0, b0)
    _store_vec3_block(d_prime, 0, d0)

    # Forward pass for remaining blocks
    for i in range(1, n_edges):
        # Load D_i (diagonal block)
        Di = _load_block_3x3(diag_blocks, i)
        # Load L_i^T (stored in offdiag_blocks)
        Lti = _load_block_3x3(offdiag_blocks, i)
        # Load C_{i-1}
        Cp = _load_block_3x3(c_blocks, i - 1)

        # Transpose to get L_i
        Li = _mat33_transpose(Lti)

        # Schur complement: T_i = D_i - L_i * C_{i-1}
        LC = _mat33_mul(Li, Cp)
        Ti = _mat33_sub(Di, LC)

        # Cholesky of T_i
        Li_factor = _mat33_cholesky(Ti)

        # RHS update: b'_i = b_i - L_i * d'_{i-1}
        bi = _load_vec3_block(rhs, i)
        dp = _load_vec3_block(d_prime, i - 1)
        ld = _mat33_mul_vec3(Li, dp)
        bi = bi - ld

        # c_i = T_i^{-1} * L_{i+1}^T (if not last block)
        if i < n_edges - 1:
            for col in range(3):
                u = _block_column_3x3(offdiag_blocks, i + 1, col)
                v = _mat33_cholesky_solve(Li_factor, u)
                _block_set_column_3x3(c_blocks, i, col, v)
        else:
            zero = wp.vec3(0.0, 0.0, 0.0)
            for col in range(3):
                _block_set_column_3x3(c_blocks, i, col, zero)

        # d'_i = T_i^{-1} * b'_i
        di = _mat33_cholesky_solve(Li_factor, bi)
        _store_vec3_block(d_prime, i, di)

    # === Back Substitution ===

    # x_{n-1} = d'_{n-1}
    xn = _load_vec3_block(d_prime, n_edges - 1)
    _store_vec3_block(x, n_edges - 1, xn)

    # x_i = d'_i - C_i * x_{i+1}
    for i in range(n_edges - 2, -1, -1):
        Ci = _load_block_3x3(c_blocks, i)
        x_next = _load_vec3_block(x, i + 1)
        cx = _mat33_mul_vec3(Ci, x_next)
        di = _load_vec3_block(d_prime, i)
        xi = di - cx
        _store_vec3_block(x, i, xi)


@wp.kernel
def _warp_spbsv_u11_1rhs(
    n: int,
    ab: wp.array2d(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
):
    """Solve a banded SPD system using Cholesky decomposition.

    In-place factorization and solve matching LAPACK spbsv.

    Args:
        n: System size.
        ab: Banded matrix in LAPACK format (factored in-place).
        b: RHS vector (solution stored in-place).
    """
    tid = wp.tid()
    if tid != 0:
        return

    # Relative tolerance for regularization
    REL_TOL = float(1.0e-8)
    ABS_FLOOR = float(1.0e-12)

    # Track maximum diagonal for relative thresholding
    max_diag = float(0.0)

    for j in range(n):
        sum_val = float(0.0)
        kmax = j if j < BAND_KD else BAND_KD
        for k in range(1, kmax + 1):
            u = ab[BAND_KD - k, j]
            sum_val += u * u

        ajj = ab[BAND_KD, j] - sum_val

        # Update max diagonal before any modification (use original diagonal for scaling reference)
        orig_diag = ab[BAND_KD, j]
        if orig_diag > max_diag:
            max_diag = orig_diag

        # Relative thresholding: tolerance scales with matrix magnitude
        tol = REL_TOL * max_diag
        if tol < ABS_FLOOR:
            tol = ABS_FLOOR

        if ajj <= tol:
            ajj = tol

        ujj = wp.sqrt(ajj)
        ab[BAND_KD, j] = ujj

        imax = (n - j - 1) if (n - j - 1) < BAND_KD else BAND_KD
        for i in range(1, imax + 1):
            dot = float(0.0)
            k2max = BAND_KD - i
            if k2max > j:
                k2max = j
            if k2max < 0:
                k2max = 0
            for k in range(1, k2max + 1):
                dot += ab[BAND_KD - k, j] * ab[BAND_KD - i - k, j + i]
            aji = ab[BAND_KD - i, j + i] - dot
            ab[BAND_KD - i, j + i] = aji / ujj

    for i in range(n):
        sum_val = float(0.0)
        k0 = 0 if i < BAND_KD else i - BAND_KD
        for k in range(k0, i):
            sum_val += ab[BAND_KD + k - i, i] * b[k]
        b[i] = (b[i] - sum_val) / ab[BAND_KD, i]

    for i in range(n - 1, -1, -1):
        sum_val = float(0.0)
        k1 = i + BAND_KD if i + BAND_KD < n else n - 1
        for k in range(i + 1, k1 + 1):
            sum_val += ab[BAND_KD + i - k, k] * b[k]
        b[i] = (b[i] - sum_val) / ab[BAND_KD, i]


@wp.kernel
def _warp_solve_blocks_jacobi(
    diag_blocks: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    delta_lambda: wp.array(dtype=wp.float32),
    n_edges: int,
):
    """Solve each 6x6 diagonal block independently (Block Jacobi iteration).

    This is a highly parallel solver that ignores off-diagonal coupling between
    edges. Each thread solves one 6x6 block system: D_i * Δλ_i = b_i.

    Advantages:
    - Maximum parallelism (one thread per edge)
    - Simple implementation
    - Good for GPUs with many cores

    Disadvantages:
    - Ignores coupling between adjacent edges
    - May require more XPBD iterations to converge

    Args:
        diag_blocks: Diagonal blocks (36 floats per edge).
        rhs: RHS vector (6 per edge).
        delta_lambda: Output solution vector (6 per edge).
        n_edges: Number of edges (blocks).
    """
    edge = wp.tid()
    if edge >= n_edges:
        return

    # Load diagonal block and RHS for this edge
    A, B, C, D = _load_block(diag_blocks, edge)
    b0, b1 = _load_vec(rhs, edge)

    # Solve 6x6 block system using Cholesky
    x0, x1 = _block_solve(A, B, C, D, b0, b1)

    # Store result
    _store_vec(delta_lambda, edge, x0, x1)


__all__ = [
    "_warp_block_thomas_solve",
    "_warp_block_thomas_solve_3x3",
    "_warp_block_thomas_solve_batched",
    "_warp_cholesky_solve_tile",
    "_warp_solve_blocks_jacobi",
    "_warp_spbsv_u11_1rhs",
]
