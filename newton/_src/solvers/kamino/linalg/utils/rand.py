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

"""KAMINO: Linear Algebra Utilities: Random matrix/rhs generation"""

import numpy as np

from ...core.types import FloatArrayLike

###
# Module interface
###

__all__ = [
    "eigenvalues_from_distribution",
    "random_rhs_for_matrix",
    "random_spd_matrix",
    "random_symmetric_matrix",
]


###
# Functions
###


def eigenvalues_from_distribution(
    size: int,
    num_pos: int | float = 0,
    num_pos_eps: int | float = 0,
    num_zero: int | float = 0,
    num_neg_eps: int | float = 0,
    num_neg: int | float = 0,
    max_pos: float = 1e2,
    min_pos: float = 1e-2,
    eps_val: float = 1e-6,
    max_neg: float = -1e-2,
    min_neg: float = -1e2,
    dtype: np.dtype = np.float64,
    seed: int | None = None,
    shuffle: bool = False,
) -> np.ndarray:
    """
    Creates an array of eigen-values based on a specified distribution.

    Notes:
        - Default max/min/eps values are set in order to generate a moderately broad spectrum.
        - The num_* arguments can be int (count) or float (percentage of size).
        - The final counts are adjusted to sum to 'size'.

    Args:
        size (int): The total size of the eigenvalue distribution.
        num_pos (int | float): The number of positive eigenvalues (count or percentage).
        num_pos_eps (int | float): The number of positive epsilon eigenvalues (count or percentage).
        num_zero (int | float): The number of zero eigenvalues (count or percentage).
        num_neg_eps (int | float): The number of negative epsilon eigenvalues (count or percentage).
        num_neg (int | float): The number of negative eigenvalues (count or percentage).
        max_pos (float): The maximum value for positive eigenvalues.
        min_pos (float): The minimum value for positive eigenvalues.
        eps_val (float): The value for epsilon eigenvalues.
        max_neg (float): The maximum value for negative eigenvalues.
        min_neg (float): The minimum value for negative eigenvalues.
        dtype (np.dtype): The data type for the eigenvalues.
        shuffle (bool): Whether to shuffle the eigenvalues.

    Returns:
        np.ndarray: The generated eigenvalue array.
    """

    # Helper to convert count/percentage to int
    def resolve_count(val):
        if isinstance(val, float):
            return int(round(val * size))
        return int(val)

    # Interpret args as either counts or percentages
    counts = {
        "num_pos": resolve_count(num_pos),
        "num_pos_eps": resolve_count(num_pos_eps),
        "num_zero": resolve_count(num_zero),
        "num_neg_eps": resolve_count(num_neg_eps),
        "num_neg": resolve_count(num_neg),
    }

    # Check total counts and correct if necessary
    total = sum(counts.values())

    # If all counts are zero, assign all eigenvalues as positive
    if total == 0:
        counts["num_pos"] = size

    # Otherwise, adjust counts to match 'size'
    elif total != size:
        # Distribute the difference to the largest group
        diff = size - total
        # Find the key with the largest count
        if counts:
            max_key = max(counts, key=lambda k: counts[k])
            counts[max_key] += diff

    # Generate the distribution of eigenvalues according to the specified counts
    eigenvalues_pos = np.linspace(max_pos, min_pos, num=counts["num_pos"]) if counts["num_pos"] > 0 else np.array([])
    eigenvalues_pos_eps = np.array([eps_val] * counts["num_pos_eps"]) if counts["num_pos_eps"] > 0 else np.array([])
    eigenvalues_zero = np.zeros(counts["num_zero"]) if counts["num_zero"] > 0 else np.array([])
    eigenvalues_neg_eps = np.array([-eps_val] * counts["num_neg_eps"]) if counts["num_neg_eps"] > 0 else np.array([])
    eigenvalues_neg = np.linspace(max_neg, min_neg, num=counts["num_neg"]) if counts["num_neg"] > 0 else np.array([])

    # Concatenate all eigenvalues into a single array of target dtype
    eigenvalues = np.concatenate(
        [
            eigenvalues_pos.astype(dtype),
            eigenvalues_pos_eps.astype(dtype),
            eigenvalues_zero.astype(dtype),
            eigenvalues_neg_eps.astype(dtype),
            eigenvalues_neg.astype(dtype),
        ]
    )

    # Optionally shuffle the eigenvalues to randomize their order
    if shuffle:
        # Set the random seed if specified and valid
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError("seed must be a int.")
        # Initialize the random number generator
        rng = np.random.default_rng(seed)
        # Shuffle the eigenvalues in-place
        rng.shuffle(eigenvalues)

    # Finally return the constructed eigenvalues array
    return eigenvalues


def random_symmetric_matrix(
    dim: int,
    dtype: np.dtype = np.float64,
    scale: float | None = None,
    seed: int | None = None,
    rank: int | None = None,
    eigenvalues: FloatArrayLike | None = None,
    return_source: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Generate a random symmetric matrix of size (dim, dim).

    Args:
    - dim (int): The size of the matrix.
    - dtype (data-type, optional): Data type of the matrix (default is float64).
    - scale (float, optional): Scale factor for the matrix (default is 1.0).
    - seed (int, optional): Seed for the random number generator.
    - rank (int, optional): Rank of the matrix (must be <= dim).
    - eigenvalues (array-like, optional): Eigenvalues for the matrix (must be of length dim).
    - return_source (bool, optional): Whether to return the source matrix (default is False).

    Returns:
    - A (ndarray): A (dim, dim) symmetric matrix.
    """
    # Set the random seed if specified and valid
    if seed is not None:
        if not isinstance(seed, int):
            raise TypeError("seed must be a int.")

    # Initialize the random number generator
    rng = np.random.default_rng(seed)

    # Set a default unit scale if unspecified
    if scale is None:
        scale = 1.0
        sqrt_scale = 1.0
    # Otherwise, check if scale is a float
    else:
        if not isinstance(scale, float):
            raise TypeError("scale must be a float.")
        sqrt_scale = np.sqrt(scale)
    scale = dtype(scale)
    sqrt_scale = dtype(sqrt_scale)

    # Generate a symmetric matrix of random rank and eigenvalues, if unspecified
    if eigenvalues is None and rank is None:
        X = scale * rng.standard_normal((dim, dim)).astype(dtype)
        # Make a symmetric matrix from the source random matrix
        A = dtype(0.5) * (X + X.T)

    # If eigenvalues are specified these take precedence
    elif eigenvalues is not None:
        if len(eigenvalues) != dim:
            raise ValueError("The number of eigenvalues must match the matrix dimension.")

        # Generate random square matrix
        if np.all(eigenvalues == eigenvalues[0]):
            X = rng.standard_normal((dim, dim)).astype(dtype)
        else:
            X, _ = np.linalg.qr(rng.standard_normal((dim, dim)).astype(dtype))
        # Diagonal matrix of eigenvalues
        D = np.diag(eigenvalues)
        # A = X * D * X^T
        A = scale * (X @ D @ X.T)
        # Additional step to ensure symmetry
        A = dtype(0.5) * (A + A.T)

    # Otherwise generate a symmetric matrix of specified rank
    elif rank is not None:
        if rank > dim:
            raise ValueError("Rank must not exceed matrix dimension.")
        # Generate random rectangular matrix
        X = sqrt_scale * rng.standard_normal((dim, rank)).astype(dtype)
        # Make a rank-deficient symmetric matrix
        A = X @ X.T
        # Additional step to ensure symmetry
        A = dtype(0.5) * (A + A.T)

    # Optionally return both final and source matrices
    if return_source:
        return A, X
    return A


def random_spd_matrix(
    dim: int,
    dtype: np.dtype = np.float64,
    scale: float | None = None,
    eta: float | None = None,
    seed: int | None = None,
    return_source: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Generate a symmetric positive definite (SPD) matrix of shape (n, n).

    Args:
    - n (int): The size of the matrix.
    - dtype (data-type, optional): Data type of the matrix (default is float64).
    - scale (float, optional): Scale factor for the matrix (default is 1.0).
    - seed (int, optional): Seed for the random number generator.
    - return_source (bool, optional): Whether to return the source matrix (default is False).

    Returns:
    - A (ndarray): An n x n symmetric positive definite matrix.
    """
    # Set the random seed if specified and valid
    if seed is not None:
        if not isinstance(seed, int):
            raise TypeError("seed must be a int.")

    # Initialize the random number generator
    rng = np.random.default_rng(seed)

    # Set a default unit scale if unspecified
    if scale is None:
        scale = 1.0
        sqrt_scale = 1.0
    # Otherwise, check if scale is a float
    else:
        if not isinstance(scale, float):
            raise TypeError("scale must be a float.")
        sqrt_scale = np.sqrt(scale)
    scale = dtype(scale)
    sqrt_scale = dtype(sqrt_scale)

    # Set a default diagonal regularizer `eta` value if unspecified
    if eta is None:
        eta = dim
    elif not isinstance(eta, float):
        raise TypeError("eta must be a float.")
    eta = dtype(eta)

    # Generate a random matrix
    X = sqrt_scale * rng.standard_normal((dim, dim)).astype(dtype)

    # Construct symmetric positive definite matrix: A.T @ A + eta * I
    A = X.T @ X + eta * np.eye(dim, dtype=dtype)

    # Ensure the matrix is symmetric
    A = dtype(0.5) * (A + A.T)

    # Optionally return both final and source matrices
    if return_source:
        return A, X
    return A


def random_rhs_for_matrix(
    A: np.ndarray, scale: float | None = None, seed: int | None = None, return_source: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Generate a random RHS vector b that is in the range space of A.

    Args:
        A (np.ndarray): The input matrix.
        scale (float): Scale factor for the random vector (default is 1.0).
        return_source (bool): Whether to return the source vector used to generate the RHS (default is False).

    Returns:
        np.ndarray: A random RHS vector b in the range space of A.
    """
    # Set the random seed if specified and valid
    if seed is not None:
        if not isinstance(seed, int):
            raise TypeError("seed must be a int.")

    # Initialize the random number generator
    rng = np.random.default_rng(seed)

    # Set a default unit scale if unspecified
    if scale is None:
        scale = 1.0
    # Otherwise, check if scale is a float
    else:
        if not isinstance(scale, float):
            raise TypeError("scale must be a float.")
    scale = A.dtype.type(scale)

    # Generate a random vector x and compute b = A @ x
    x = scale * rng.standard_normal((A.shape[1],)).astype(A.dtype)
    b = A @ x

    # Optionally return both final and source vectors
    if return_source:
        return b, x
    return b
