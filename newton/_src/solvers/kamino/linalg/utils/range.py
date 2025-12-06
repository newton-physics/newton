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

"""KAMINO: Linear Algebra Utilities: Operations to check if a rhs vector lies within the range of a matrix"""

import numpy as np

###
# Module interface
###

__all__ = [
    "in_range_via_gaussian_elimination",
    "in_range_via_left_nullspace",
    "in_range_via_projection",
    "in_range_via_rank",
    "in_range_via_residual",
]

###
# Utilities
###


def _svd_rank(s: np.ndarray, shape: tuple, rcond: float | None = None):
    """
    Determine numerical rank from singular values using a pinv-like threshold.
    """
    m, n = shape
    if rcond is None:
        rcond = float(np.finfo(s.dtype).eps)
    tol = rcond * max(m, n) * (s[0] if s.size else 0.0)
    return int(np.sum(s > tol)), float(tol)


###
# Functions
###


def in_range_via_rank(A: np.ndarray, b: np.ndarray) -> bool:
    """
    b is in range(A) iff rank(A) == rank([A|b])
    """
    b = b.reshape(-1, 1)
    rank_A = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(np.hstack([A, b]))
    return rank_A == rank_Ab


def in_range_via_residual(A: np.ndarray, b: np.ndarray) -> bool:
    """
    Solve min_x ||Ax - b||_2 and test if residual is ~0.
    Tolerance scales with numerical precision and conditioning of A.
    """
    b = b.reshape(-1, 1)

    # Least-squares solution
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ x
    r_norm = float(np.linalg.norm(r))

    # Compute singular values for scaling the tolerance
    s = np.linalg.svd(A, compute_uv=False)

    # Scale-aware tolerance: eps * max(n) * sigma_max(A) * ||b||
    eps = float(np.finfo(s.dtype).eps)
    sigma_max = float(s[0]) if s.size else float(1.0)
    tol = eps * max(A.shape) * sigma_max * float(np.linalg.norm(b))
    return r_norm <= tol, r_norm, float(tol), x.ravel()


def in_range_via_left_nullspace(U: np.ndarray, s: np.ndarray, b: np.ndarray, shape: tuple, rcond: float | None = None):
    """
    b is in range(A) iff U0^T b ≈ 0, where U0 are left singular vectors for zero sigmas.
    Returns (bool, residual_norm, tol).
    """
    r, _ = _svd_rank(s, shape, rcond)

    U0 = U[:, r:]  # left-nullspace basis (empty if full rank)
    if U0.size == 0:
        return True, 0.0, 0.0

    w = U0.T @ b
    res = float(np.linalg.norm(w))
    norm_b = float(np.linalg.norm(b))
    eps = float(np.finfo(b.dtype).eps)
    tol_b = eps * float(max(shape)) * norm_b
    return res <= tol_b, res, tol_b


def in_range_via_projection(U: np.ndarray, s: np.ndarray, b: np.ndarray, shape: tuple, rcond: float | None = None):
    """
    Project b onto span(U_r) and measure the leftover: ||(I - U_r U_r^T) b||.
    Returns (bool, distance_to_range, tol, b_proj).
    """
    r, _ = _svd_rank(s, shape, rcond)

    Ur = U[:, :r]
    b_proj = Ur @ (Ur.T @ b) if r > 0 else np.zeros_like(b)
    residual = b - b_proj
    norm_b = float(np.linalg.norm(b))
    dist = float(np.linalg.norm(residual))
    eps = float(np.finfo(b.dtype).eps)
    tol_b = eps * float(max(shape)) * norm_b
    return dist <= tol_b, dist, tol_b, b_proj


def in_range_via_gaussian_elimination(A: np.ndarray, b: np.ndarray, tol: float = 1e-12):
    """
    Check if b is in the range (column space) of A by forming the augmented
    matrix Ab = [A | b] and performing Gaussian elimination without pivoting.

    Parameters
    ----------
    A : (m, n) ndarray
        Coefficient matrix.
    b : (m,) or (m,1) ndarray
        Right-hand side vector.
    tol : float
        Threshold for treating values as zero (numerical tolerance).

    Returns
    -------
    in_range : bool
        True iff rank(A) == rank([A|b]) under Gaussian elimination w/o pivoting.
    ranks : tuple[int, int]
        (rank_A, rank_Ab) computed from the row-echelon form obtained w/o pivoting.
    UAb : ndarray
        The upper-triangular (row-echelon-like) matrix after elimination on [A|b]
        (useful for debugging/inspection).

    Notes
    -----
    - No row swaps (no pivoting) are used, per the requirement.
    - This procedure is less numerically robust than pivoted elimination.
    - Rank is computed as the number of nonzero rows (by `tol`) in the echelon form.
    """
    tol = A.dtype.type(tol)
    if b.ndim == 1:
        b = b[:, None]
    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b must have the same number of rows.")
    if A.dtype != b.dtype:
        raise ValueError("A and b must have the same dtype.")

    # Form augmented matrix [A | b]
    UAb = np.concatenate([A, b], axis=1)
    m, n_aug = UAb.shape
    n = n_aug - 1  # number of columns in A portion

    # Gaussian elimination without pivoting
    # (Equivalent to LU factorization steps without P; we only keep the U-like result.)
    for k in range(min(m, n)):
        pivot = UAb[k, k]
        if abs(pivot) <= tol:
            # No row swap allowed; skip elimination for this column
            continue
        for i in range(k + 1, m):
            factor = UAb[i, k] / pivot
            # subtract factor * row k from row i (only on the trailing part for efficiency)
            UAb[i, k:n_aug] -= factor * UAb[k, k:n_aug]

    # Helper: count nonzero rows under tolerance
    def rank_from_row_echelon(M, tol):
        # A row is nonzero if any absolute entry exceeds tol
        return int(np.sum(np.any(np.abs(M) > tol, axis=1)))

    # Rank of A: evaluate on the left block after the same row ops
    rank_A = rank_from_row_echelon(UAb[:, :n], tol)
    # Rank of augmented matrix
    rank_Ab = rank_from_row_echelon(UAb, tol)

    return (rank_A == rank_Ab), (rank_A, rank_Ab), UAb
