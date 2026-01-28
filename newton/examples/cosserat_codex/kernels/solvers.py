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

from newton.examples.cosserat_codex.constants import BAND_KD, TILE

from .math import (
    _block_mul,
    _block_mul_vec,
    _block_row,
    _block_set_column,
    _block_solve,
    _block_sub,
    _load_block,
    _load_vec,
    _store_vec,
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
            u0, u1 = _block_row(offdiag_blocks, 1, col)
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
        LiA, LiB, LiC, LiD = _load_block(offdiag_blocks, i)
        CpA, CpB, CpC, CpD = _load_block(c_blocks, i - 1)

        LCA, LCB, LCC, LCD = _block_mul(LiA, LiB, LiC, LiD, CpA, CpB, CpC, CpD)
        TiA, TiB, TiC, TiD = _block_sub(DiA, DiB, DiC, DiD, LCA, LCB, LCC, LCD)

        bi0, bi1 = _load_vec(rhs, i)
        dp0, dp1 = _load_vec(d_prime, i - 1)
        ld0, ld1 = _block_mul_vec(LiA, LiB, LiC, LiD, dp0, dp1)
        bi0 = bi0 - ld0
        bi1 = bi1 - ld1

        if i < n_edges - 1:
            for col in range(6):
                u0, u1 = _block_row(offdiag_blocks, i + 1, col)
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
def _warp_spbsv_u11_1rhs_iter_ref(
    n: int,
    ab: wp.array2d(dtype=wp.float32),       # Will be overwritten with Cholesky factor U
    b: wp.array(dtype=wp.float32),           # RHS, overwritten with solution x
    ab_orig: wp.array2d(dtype=wp.float32),  # Original matrix (preserved for residual computation)
    b_orig: wp.array(dtype=wp.float32),      # Original RHS (preserved for residual computation)
    r: wp.array(dtype=wp.float32),           # Workspace for residual/correction
    max_iters: int,                          # Number of refinement iterations (typically 1-3)
):
    """Banded Cholesky solver with iterative refinement for improved accuracy.
    
    Uses double-precision accumulation for residual computation to recover
    accuracy lost during single-precision factorization.
    
    Args:
        n: System size.
        ab: Banded matrix (factored in-place).
        b: RHS vector (solution stored in-place).
        ab_orig: Original matrix preserved for residual computation.
        b_orig: Original RHS preserved for residual computation.
        r: Workspace for residual computation.
        max_iters: Number of refinement iterations.
    """
    tid = wp.tid()
    if tid != 0:
        return

    # Relative tolerance for regularization
    REL_TOL = float(1.0e-8)
    ABS_FLOOR = float(1.0e-12)
    
    # Track maximum diagonal for relative thresholding
    max_diag = float(0.0)

    # ========================================================================
    # 1) In-place Cholesky factorization: AB -> U
    # ========================================================================
    for j in range(n):
        sum_val = float(0.0)
        kmax = j if j < BAND_KD else BAND_KD
        for k in range(1, kmax + 1):
            u = ab[BAND_KD - k, j]
            sum_val += u * u

        ajj = ab[BAND_KD, j] - sum_val
        
        # Update max diagonal before any modification
        orig_diag = ab[BAND_KD, j]
        if orig_diag > max_diag:
            max_diag = orig_diag
        
        # Relative thresholding
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

    # ========================================================================
    # 2) Initial solve: forward substitution U^T y = b
    # ========================================================================
    for i in range(n):
        sum_val = float(0.0)
        k0 = 0 if i < BAND_KD else i - BAND_KD
        for k in range(k0, i):
            sum_val += ab[BAND_KD + k - i, i] * b[k]
        b[i] = (b[i] - sum_val) / ab[BAND_KD, i]

    # ========================================================================
    # 3) Initial solve: backward substitution U x = y
    # ========================================================================
    for i in range(n - 1, -1, -1):
        sum_val = float(0.0)
        k1 = i + BAND_KD if i + BAND_KD < n else n - 1
        for k in range(i + 1, k1 + 1):
            sum_val += ab[BAND_KD + i - k, k] * b[k]
        b[i] = (b[i] - sum_val) / ab[BAND_KD, i]

    # ========================================================================
    # 4) Iterative refinement loop
    # ========================================================================
    for iteration in range(max_iters):
        # --------------------------------------------------------------------
        # 4a) Compute residual: r = b_orig - A_orig * x
        #     Use double precision accumulation for better accuracy
        # --------------------------------------------------------------------
        for i in range(n):
            # Start with original RHS (cast to double for accumulation)
            acc = wp.float64(b_orig[i])
            
            # Subtract A_orig[i,:] * x using banded structure
            # For symmetric banded matrix: A(i,j) = A(j,i)
            # Diagonal element
            acc -= wp.float64(ab_orig[BAND_KD, i]) * wp.float64(b[i])
            
            # Off-diagonal elements (both upper and lower by symmetry)
            j_start = 0 if i < BAND_KD else i - BAND_KD
            j_end = i + BAND_KD + 1 if i + BAND_KD < n else n
            
            for j in range(j_start, i):
                # A(i,j) where j < i: stored at ab_orig[BAND_KD + j - i, i]
                aij = wp.float64(ab_orig[BAND_KD + j - i, i])
                acc -= aij * wp.float64(b[j])
            
            for j in range(i + 1, j_end):
                # A(i,j) where j > i: stored at ab_orig[BAND_KD + i - j, j]
                aij = wp.float64(ab_orig[BAND_KD + i - j, j])
                acc -= aij * wp.float64(b[j])
            
            r[i] = wp.float32(acc)
        
        # --------------------------------------------------------------------
        # 4b) Solve for correction: U^T U d = r (reuse factorization)
        # --------------------------------------------------------------------
        # Forward substitution: U^T d = r
        for i in range(n):
            sum_val = float(0.0)
            k0 = 0 if i < BAND_KD else i - BAND_KD
            for k in range(k0, i):
                sum_val += ab[BAND_KD + k - i, i] * r[k]
            r[i] = (r[i] - sum_val) / ab[BAND_KD, i]

        # Backward substitution: U d = (U^T)^{-1} r
        for i in range(n - 1, -1, -1):
            sum_val = float(0.0)
            k1 = i + BAND_KD if i + BAND_KD < n else n - 1
            for k in range(i + 1, k1 + 1):
                sum_val += ab[BAND_KD + i - k, k] * r[k]
            r[i] = (r[i] - sum_val) / ab[BAND_KD, i]

        # --------------------------------------------------------------------
        # 4c) Update solution: x = x + d
        # --------------------------------------------------------------------
        for i in range(n):
            b[i] = b[i] + r[i]


__all__ = [
    "_warp_cholesky_solve_tile",
    "_warp_block_thomas_solve",
    "_warp_spbsv_u11_1rhs",
    "_warp_spbsv_u11_1rhs_iter_ref",
]
