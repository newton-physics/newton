# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct solver kernels for Cosserat rod simulations.

Contains Thomas algorithm (O(n) tridiagonal) and Cholesky decomposition solvers.
"""

import warp as wp


# Warp tile configuration
BLOCK_DIM = 128
TILE = 32  # 32x32 tile size for Cholesky


@wp.kernel
def thomas_solve_kernel(
    diag: wp.array(dtype=float),
    off_diag: wp.array(dtype=float),
    rhs: wp.array(dtype=float),
    num_constraints: int,
    # workspace
    c_prime: wp.array(dtype=float),
    d_prime: wp.array(dtype=float),
    # output
    x: wp.array(dtype=float),
):
    """Thomas algorithm (TDMA) for symmetric tridiagonal systems.

    Solves Ax = b where A is tridiagonal with:
        - diag[i] = main diagonal
        - off_diag[i] = sub/super diagonal (A[i,i+1] = A[i+1,i])

    Algorithm is O(n) - forward elimination then back substitution.

    Args:
        diag: Main diagonal of the system matrix.
        off_diag: Sub/super diagonal (symmetric).
        rhs: Right-hand side vector.
        num_constraints: Number of constraints (system size).
        c_prime: Workspace for forward elimination.
        d_prime: Workspace for forward elimination.
        x: Output solution vector.
    """
    n = num_constraints

    # Forward elimination
    # c'[0] = c[0] / d[0], where c[0] = off_diag[0]
    c_prime[0] = off_diag[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n):
        # a[i] = off_diag[i-1] (sub-diagonal element)
        a_i = off_diag[i - 1]

        # denom = d[i] - a[i] * c'[i-1]
        denom = diag[i] - a_i * c_prime[i - 1]

        # c'[i] = c[i] / denom (only if not last row)
        if i < n - 1:
            c_prime[i] = off_diag[i] / denom

        # d'[i] = (b[i] - a[i] * d'[i-1]) / denom
        d_prime[i] = (rhs[i] - a_i * d_prime[i - 1]) / denom

    # Back substitution
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]


@wp.kernel
def cholesky_solve_kernel(
    A: wp.array2d(dtype=float),
    b: wp.array1d(dtype=float),
    # output
    x: wp.array1d(dtype=float),
):
    """Solve Ax = b using tile Cholesky decomposition.

    Uses Warp's tile-based Cholesky factorization for efficient GPU solving.
    Matrix A must be symmetric positive definite and fit in a TILE x TILE tile.

    Args:
        A: System matrix (TILE x TILE).
        b: Right-hand side vector (TILE).
        x: Output solution vector (TILE).
    """
    a_tile = wp.tile_load(A, shape=(TILE, TILE))
    b_tile = wp.tile_load(b, shape=TILE)

    # Cholesky factorization: A = L L^T
    L = wp.tile_cholesky(a_tile)

    # Solve: L L^T x = b
    x_tile = wp.tile_cholesky_solve(L, b_tile)

    wp.tile_store(x, x_tile)
