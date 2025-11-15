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

"""Unit-test utilities for generating random linear system and factorization problems."""

import numpy as np
import scipy.linalg
import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.types import FloatArrayLike, float32, int32
from newton._src.solvers.kamino.linalg.utils.rand import (
    random_rhs_for_matrix,
    random_spd_matrix,
    random_symmetric_matrix,
)

###
# Classes
###


class RandomProblemLLT:
    def __init__(
        self,
        seed: int = 42,
        dims: list[int] | int | None = None,
        maxdims: list[int] | int | None = None,
        A: list[np.ndarray] | None = None,
        b: list[np.ndarray] | None = None,
        np_dtype=np.float32,
        wp_dtype=float32,
        device: Devicelike = None,
        upper: bool = False,
    ):
        # Check input data to ensure they are indeed lists of numpy arrays
        if A is not None:
            if not isinstance(A, list) or not all(isinstance(a, np.ndarray) for a in A):
                raise TypeError("A must be a list of numpy arrays.")
            dims = [a.shape[0] for a in A]  # Update dims based on provided A
        if b is not None:
            if not isinstance(b, list) or not all(isinstance(b_i, np.ndarray) for b_i in b):
                raise TypeError("b must be a list of numpy arrays.")

        # Ensure the problem dimensions are valid
        if isinstance(dims, int):
            dims = [dims]
        elif isinstance(dims, list):
            if not all(isinstance(d, int) and d > 0 for d in dims):
                raise ValueError("All dimensions must be positive integers.")
        else:
            raise TypeError("Dimensions must be an integer or a list of integers.")

        # Ensure the max problem dimensions are valid if provided, otherwise set them to dims
        if maxdims is not None:
            if isinstance(maxdims, int):
                maxdims = [maxdims] * len(dims)
            elif isinstance(maxdims, list):
                if not all(isinstance(md, int) and md > 0 for md in maxdims):
                    raise ValueError("All max dimensions must be positive integers.")
                if len(maxdims) != len(dims):
                    raise ValueError("maxdims must have the same length as dims.")
                if any(md < d for md, d in zip(maxdims, dims, strict=False)):
                    raise ValueError("All maxdims must be greater than or equal to dims.")
            else:
                raise TypeError("maxdims must be an integer or a list of integers.")
        else:
            maxdims = dims

        # Cache the problem configurations
        self.num_blocks: int = len(dims)
        self.dims: list[int] = dims
        self.maxdims: list[int] = maxdims
        self.seed: int = seed
        self.np_dtype = np_dtype
        self.wp_dtype = wp_dtype
        self.device: Devicelike = device

        # Declare the lists of reference problem data
        self.A_np: list[np.ndarray] = []
        self.b_np: list[np.ndarray] = []
        self.X_np: list[np.ndarray] = []
        self.y_np: list[np.ndarray] = []
        self.x_np: list[np.ndarray] = []

        # Declare the warp arrays of concatenated problem data
        self.maxdim_wp: wp.array | None = None
        self.dim_wp: wp.array | None = None
        self.mio_wp: wp.array | None = None
        self.vio_wp: wp.array | None = None
        self.A_wp: wp.array | None = None
        self.b_wp: wp.array | None = None

        # Initialize the flattened problem data
        A_sizes = [n * n for n in self.maxdims]
        A_offsets = [0] + [sum(A_sizes[:i]) for i in range(1, len(A_sizes) + 1)]
        A_flat_size = sum(A_sizes)
        A_flat = np.full(shape=(A_flat_size,), fill_value=np.inf, dtype=np_dtype)
        b_sizes = list(self.maxdims)
        b_offsets = [0] + [sum(b_sizes[:i]) for i in range(1, len(b_sizes) + 1)]
        b_flat_size = sum(b_sizes)
        b_flat = np.full(shape=(b_flat_size,), fill_value=np.inf, dtype=np_dtype)

        # Generate randomized problem data
        for i, (n, nmax) in enumerate(zip(self.dims, self.maxdims, strict=False)):
            # Generate a random SPD matrix if not provided
            if A is None:
                A_mat = random_spd_matrix(dim=n, seed=self.seed, dtype=np_dtype)
            else:
                A_mat = A[i]
            # Generate a random RHS vector if not provided
            if b is None:
                b_vec = random_rhs_for_matrix(A_mat)
            else:
                b_vec = b[i]
            # Compute the Cholesky decomposition using numpy
            X_mat = np.linalg.cholesky(A_mat, upper=upper)
            # Compute final and intermediate reference solutions
            if upper:
                y_vec, x_vec = _solve_cholesky_upper_numpy(X_mat, b_vec)
            else:
                y_vec, x_vec = _solve_cholesky_lower_numpy(X_mat, b_vec)
            # Store the reference data
            self.A_np.append(A_mat)
            self.b_np.append(b_vec)
            self.X_np.append(X_mat)
            self.y_np.append(y_vec)
            self.x_np.append(x_vec)
            # Flatten the matrix and store it in the A_flat array
            A_start = A_offsets[len(self.A_np) - 1]
            # Fill the flattened array row-wise to account for dim <= maxdim
            if n == nmax:
                A_end = A_offsets[len(self.A_np)]
                A_flat[A_start:A_end] = A_mat.flat
            else:
                A_end = A_start + n * n
                A_flat[A_start:A_end] = A_mat.flat
            # Flatten the vector and store it in the b_flat array
            b_start = b_offsets[len(self.b_np) - 1]
            b_end = b_start + n
            b_flat[b_start:b_end] = b_vec

        # Construct the warp arrays
        with wp.ScopedDevice(self.device):
            self.maxdim_wp = wp.array(self.maxdims, dtype=int32)
            self.dim_wp = wp.array(self.dims, dtype=int32)
            self.mio_wp = wp.array(A_offsets[: self.num_blocks], dtype=int32)
            self.vio_wp = wp.array(b_offsets[: self.num_blocks], dtype=int32)
            self.A_wp = wp.array(A_flat, dtype=float32)
            self.b_wp = wp.array(b_flat, dtype=float32)

    def __str__(self) -> str:
        return (
            f"RandomProblemLLT("
            f"\nseed: {self.seed}"
            f"\nnum_blocks: {self.num_blocks}"
            f"\nnp_dtype: {self.np_dtype}"
            f"\nwp_dtype: {self.wp_dtype}"
            f"\ndims: {self.dims}"
            f"\nmaxdims: {self.maxdims}"
            f"\ndevice: {self.device}"
            f"\nA_np (shape): {[A.shape for A in self.A_np]}"
            f"\nb_np (shape): {[b.shape for b in self.b_np]}"
            f"\nX_np (shape): {[X.shape for X in self.X_np]}"
            f"\ny_np (shape): {[y.shape for y in self.y_np]}"
            f"\nx_np (shape): {[x.shape for x in self.x_np]}"
            f"\nmaxdim_wp: {self.maxdim_wp.numpy()}"
            f"\ndim_wp: {self.dim_wp.numpy()}"
            f"\nmio_wp: {self.mio_wp.numpy()}"
            f"\nvio_wp: {self.vio_wp.numpy()}"
            f"\nA_wp (shape): {self.A_wp.shape}"
            f"\nb_wp (shape): {self.b_wp.shape}"
            f"\n)"
        )


class RandomProblemLDLT:
    def __init__(
        self,
        seed: int = 42,
        dims: list[int] | int | None = None,
        maxdims: list[int] | int | None = None,
        ranks: list[int] | int | None = None,
        eigenvalues: FloatArrayLike | None = None,
        A: list[np.ndarray] | None = None,
        b: list[np.ndarray] | None = None,
        np_dtype=np.float32,
        wp_dtype=float32,
        device: Devicelike = None,
        lower: bool = True,
    ):
        # Check input data to ensure they are indeed lists of numpy arrays
        if A is not None:
            if not isinstance(A, list) or not all(isinstance(a, np.ndarray) for a in A):
                raise TypeError("A must be a list of numpy arrays.")
            dims = [a.shape[0] for a in A]  # Update dims based on provided A
        if b is not None:
            if not isinstance(b, list) or not all(isinstance(b_i, np.ndarray) for b_i in b):
                raise TypeError("b must be a list of numpy arrays.")

        # Ensure the problem dimensions are valid
        if isinstance(dims, int):
            dims = [dims]
        elif isinstance(dims, list):
            if not all(isinstance(d, int) for d in dims):
                raise ValueError("All dimensions must be integers.")
        else:
            raise TypeError("Dimensions must be an integer or a list of integers.")

        # Ensure the max problem dimensions are valid if provided, otherwise set them to dims
        if maxdims is not None:
            if isinstance(maxdims, int):
                maxdims = [maxdims] * len(dims)
            elif isinstance(maxdims, list):
                if not all(isinstance(md, int) for md in maxdims):
                    raise ValueError("All max dimensions must be integers.")
                if len(maxdims) != len(dims):
                    raise ValueError("maxdims must have the same length as dims.")
            else:
                raise TypeError("maxdims must be an integer or a list of integers.")
        else:
            maxdims = dims

        # Ensure the rank dimensions are valid
        if ranks is not None:
            if isinstance(ranks, int):
                ranks = [ranks]
            elif isinstance(ranks, list):
                if not all(isinstance(r, int) for r in ranks):
                    raise ValueError("All ranks must be integers.")
            else:
                raise TypeError("Ranks must be an integer or a list of integers.")
        else:
            ranks = [None] * len(dims)

        # Ensure the eigenvalues are valid
        if eigenvalues is not None:
            if not isinstance(eigenvalues, list) or not all(isinstance(ev, int | float) for ev in eigenvalues):
                raise TypeError("Eigenvalues must be a list of numbers.")
        else:
            eigenvalues = [None] * len(dims)

        # Cache the problem configurations
        self.num_blocks: int = len(dims)
        self.dims: list[int] = dims
        self.maxdims: list[int] = maxdims
        self.seed: int = seed
        self.np_dtype = np_dtype
        self.wp_dtype = wp_dtype
        self.device: Devicelike = device

        # Declare the lists of reference problem data
        self.A_np: list[np.ndarray] = []
        self.b_np: list[np.ndarray] = []
        self.X_np: list[np.ndarray] = []
        self.D_np: list[np.ndarray] = []
        self.P_np: list[np.ndarray] = []
        self.z_np: list[np.ndarray] = []
        self.y_np: list[np.ndarray] = []
        self.x_np: list[np.ndarray] = []

        # Declare the warp arrays of concatenated problem data
        self.maxdim_wp: wp.array | None = None
        self.dim_wp: wp.array | None = None
        self.mio_wp: wp.array | None = None
        self.vio_wp: wp.array | None = None
        self.A_wp: wp.array | None = None
        self.b_wp: wp.array | None = None

        # Initialize the flattened problem data
        A_sizes = [n * n for n in self.maxdims]
        A_offsets = [0] + [sum(A_sizes[:i]) for i in range(1, len(A_sizes) + 1)]
        A_flat_size = sum(A_sizes)
        A_flat = np.ndarray(shape=(A_flat_size,), dtype=np_dtype)
        b_sizes = list(self.maxdims)
        b_offsets = [0] + [sum(b_sizes[:i]) for i in range(1, len(b_sizes) + 1)]
        b_flat_size = sum(b_sizes)
        b_flat = np.ndarray(shape=(b_flat_size,), dtype=np_dtype)

        # Generate randomized problem data
        for i, (n, nmax) in enumerate(zip(self.dims, self.maxdims, strict=False)):
            # Generate a random SPD matrix if not provided
            if A is None:
                A_mat = random_symmetric_matrix(
                    dim=n, seed=self.seed, rank=ranks[i], eigenvalues=eigenvalues[i], dtype=np_dtype
                )
            else:
                A_mat = A[i]
            # Generate a random RHS vector if not provided
            if b is None:
                b_vec = random_rhs_for_matrix(A_mat)
            else:
                b_vec = b[i]
            # Compute the LDLT decomposition using numpy
            X_mat, D_mat, P_mat = scipy.linalg.ldl(A_mat, lower=lower)
            # Compute final and intermediate reference solutions
            if lower:
                z_vec, y_vec, x_vec = _solve_ldlt_lower_numpy(X_mat, D_mat, P_mat, b_vec)
            else:
                z_vec, y_vec, x_vec = _solve_ldlt_upper_numpy(X_mat, D_mat, P_mat, b_vec)
            # Store the reference data
            self.A_np.append(A_mat)
            self.b_np.append(b_vec)
            self.X_np.append(X_mat)
            self.D_np.append(D_mat)
            self.P_np.append(P_mat)
            self.z_np.append(z_vec)
            self.y_np.append(y_vec)
            self.x_np.append(x_vec)
            # Flatten the matrix and store it in the A_flat array
            A_start = A_offsets[len(self.A_np) - 1]
            # Fill the flattened array row-wise to account for dim <= maxdim
            if n == nmax:
                A_end = A_offsets[len(self.A_np)]
                A_flat[A_start:A_end] = A_mat.flat
            else:
                A_end = A_start + n * n
                A_flat[A_start:A_end] = A_mat.flat
            # Flatten the vector and store it in the b_flat array
            b_start = b_offsets[len(self.b_np) - 1]
            b_end = b_offsets[len(self.b_np)]
            b_flat[b_start:b_end] = b_vec

        # Construct the warp arrays
        with wp.ScopedDevice(self.device):
            self.maxdim_wp = wp.array(self.maxdims, dtype=int32)
            self.dim_wp = wp.array(self.dims, dtype=int32)
            self.mio_wp = wp.array(A_offsets[: self.num_blocks], dtype=int32)
            self.vio_wp = wp.array(b_offsets[: self.num_blocks], dtype=int32)
            self.A_wp = wp.array(A_flat, dtype=float32)
            self.b_wp = wp.array(b_flat, dtype=float32)

    def __str__(self) -> str:
        return (
            f"RandomProblemLDLT("
            f"\nseed: {self.seed}"
            f"\nnum_blocks: {self.num_blocks}"
            f"\nnp_dtype: {self.np_dtype}"
            f"\nwp_dtype: {self.wp_dtype}"
            f"\ndims: {self.dims}"
            f"\nmaxdims: {self.maxdims}"
            f"\ndevice: {self.device}"
            f"\nA_np (shape): {[A.shape for A in self.A_np]}"
            f"\nb_np (shape): {[b.shape for b in self.b_np]}"
            f"\nX_np (shape): {[X.shape for X in self.X_np]}"
            f"\nD_np (shape): {[D.shape for D in self.D_np]}"
            f"\nP_np (shape): {[P.shape for P in self.P_np]}"
            f"\nz_np (shape): {[z.shape for z in self.z_np]}"
            f"\ny_np (shape): {[y.shape for y in self.y_np]}"
            f"\nx_np (shape): {[x.shape for x in self.x_np]}"
            f"\nmaxdim_wp: {self.maxdim_wp.numpy()}"
            f"\ndim_wp: {self.dim_wp.numpy()}"
            f"\nmio_wp: {self.mio_wp.numpy()}"
            f"\nvio_wp: {self.vio_wp.numpy()}"
            f"\nA_wp (shape): {self.A_wp.shape}"
            f"\nb_wp (shape): {self.b_wp.shape}"
            f"\n)"
        )


###
# Utilities
###


def _solve_cholesky_lower_numpy(L: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the linear system Ax = b using Cholesky decomposition.

    Args:
        A (np.ndarray): The input matrix (must be symmetric positive definite).
        b (np.ndarray): The RHS vector.

    Returns:
        np.ndarray: The solution vector x.
    """
    y = scipy.linalg.solve_triangular(L, b, lower=True)
    x = scipy.linalg.solve_triangular(L.T, y, lower=False)
    return y, x


def _solve_cholesky_upper_numpy(U: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the linear system Ax = b using Cholesky decomposition.

    Args:
        A (np.ndarray): The input matrix (must be symmetric positive definite).
        b (np.ndarray): The RHS vector.

    Returns:
        np.ndarray: The solution vector x.
    """
    y = scipy.linalg.solve_triangular(U.T, b, lower=True)
    x = scipy.linalg.solve_triangular(U, y, lower=False)
    return y, x


def _solve_ldlt_lower_numpy(
    L: np.ndarray, D: np.ndarray, P: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the linear system Ax = b using LDL^T decomposition.

    Args:
        L (np.ndarray): The lower triangular matrix from the LDL^T decomposition.
        D (np.ndarray): The diagonal matrix from the LDL^T decomposition.
        P (np.ndarray): The permutation index array from the LDL^T decomposition.
        b (np.ndarray): The RHS vector.

    Returns:
        np.ndarray: The solution vector x.
    """
    PL = L[P, :]
    z = scipy.linalg.solve_triangular(PL, b[P], lower=True)
    y = z / np.diag(D)
    x = scipy.linalg.solve_triangular(PL.T, y, lower=False)
    x = x[np.argsort(P)]
    return z, y, x


def _solve_ldlt_upper_numpy(
    U: np.ndarray, D: np.ndarray, P: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the linear system Ax = b using LDL^T decomposition.

    Args:
        U (np.ndarray): The upper triangular matrix from the LDL^T decomposition.
        D (np.ndarray): The diagonal matrix from the LDL^T decomposition.
        P (np.ndarray): The permutation index array from the LDL^T decomposition.
        b (np.ndarray): The RHS vector.

    Returns:
        np.ndarray: The solution vector x.
    """
    PU = U[P, :]
    z = scipy.linalg.solve_triangular(PU.T, b[P], lower=True)
    y = z / np.diag(D)
    x = scipy.linalg.solve_triangular(PU, y, lower=False)
    x = x[np.argsort(P)]
    return z, y, x
