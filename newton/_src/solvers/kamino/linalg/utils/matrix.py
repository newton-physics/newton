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

"""KAMINO: Linear Algebra Utilities: Matrix properties"""

from enum import IntEnum

import numpy as np

###
# Module interface
###

__all__ = [
    "DEFAULT_MATRIX_SYMMETRY_EPS",
    "MatrixComparison",
    "MatrixSign",
    "SquareSymmetricMatrixProperties",
    "assert_is_square_matrix",
    "assert_is_symmetric_matrix",
    "is_square_matrix",
    "is_symmetric_matrix",
    "symmetry_error_norm_l2",
]


###
# Constants
###

DEFAULT_MATRIX_SYMMETRY_EPS = 1e-10
"""A global constant to configure the tolerance on matrix symmetry checks."""

MAXLOG_FP64 = 709.782
"""Maximum log value for float64 to avoid overflow in exp()."""

###
# Types
###


class MatrixSign(IntEnum):
    ZeroSign = 0
    Indefinite = 1
    PositiveSemiDef = 2
    NegativeSemiDef = 3
    PositiveDef = 4
    NegativeDef = 5


###
# Utilities
###


def _safe_slogdet(A: np.ndarray) -> tuple[float, float, float]:
    sign, logabsdet = np.linalg.slogdet(A)
    sign = float(sign)
    logabsdet = float(logabsdet)
    if sign != 0.0 and logabsdet < MAXLOG_FP64:
        det = sign * np.exp(logabsdet)
    elif sign == 0.0:
        det = 0.0
    else:
        det = np.inf
    return sign, logabsdet, det


def _make_tolerance(tol: float | None = None, dtype: np.dtype = np.float64):
    eps = np.finfo(dtype).eps
    if tol is None:
        tol = dtype.type(eps)
    else:
        if not isinstance(tol, float | np.float32 | np.float64):
            raise ValueError("tolerance 'tol' must be a `float`, `np.float32`, or `np.float64` value.")
    return dtype.type(max(tol, eps))


def is_square_matrix(A: np.ndarray) -> bool:
    return A.shape[0] == A.shape[1]


def is_symmetric_matrix(A: np.ndarray, tol: float | None = None) -> bool:
    tol = _make_tolerance(tol=tol, dtype=A.dtype)
    return np.allclose(A, A.T, atol=tol, rtol=0.0)


def symmetry_error_norm_l2(A: np.ndarray) -> float:
    return np.linalg.norm(A - A.T, ord=2)


def assert_is_square_matrix(A: np.ndarray):
    if not is_square_matrix(A):
        raise ValueError("Matrix is not square.")


def assert_is_symmetric_matrix(A: np.ndarray):
    eps = max(_make_tolerance(dtype=A.dtype), A.dtype.type(DEFAULT_MATRIX_SYMMETRY_EPS))
    if not is_symmetric_matrix(A, tol=eps):
        error = symmetry_error_norm_l2(A)
        raise ValueError(f"Matrix is not symmetric within tolerance {eps}, with error (inf-norm): {error}")


###
# Matrix Properties
###


class RectangularMatrixProperties:
    def __init__(self, matrix: np.ndarray | None = None):
        self.matrix: np.ndarray
        """Reference to the original matrix."""

        # Matrix statistics
        self.min: float = np.inf
        """The minimum element of the matrix."""
        self.max: float = np.inf
        """The maximum element of the matrix."""
        self.mean: float = np.inf
        """The mean of the matrix elements."""
        self.std: float = np.inf
        """The standard deviation of the matrix elements."""

        # Matrix dimensions
        self.shape: tuple[int, int] = (0, 0)
        """The matrix shape (rows, cols)."""

        # Matrix properties
        self.rank: int = 0
        """The matrix rank compute using `numpy.linalg.matrix_rank()`."""

        # Matrix norms
        self.norm_fro: float = np.inf
        """The Frobenius norm compute using `numpy.linalg.norm()`."""
        self.norm_inf: float = np.inf
        """The infinity norm compute using `numpy.linalg.norm()`."""

        # SVD properties
        self.sigma_min: float = np.inf
        """The smallest singular value."""
        self.sigma_max: float = np.inf
        """The largest singular value."""
        self.sigma_cond: float = np.inf
        """The condition number defined via the ratio of max/min singular values."""

        # Caches
        self.sigmas: np.ndarray | None = None
        self.U: np.ndarray | None = None
        self.Vt: np.ndarray | None = None

        # Compute matrix properties if specified
        if matrix is not None:
            self.compute(matrix)

    def compute(self, matrix: np.ndarray):
        """
        Compute the properties of the rectangular matrix.

        Args:
            matrix (np.ndarray): The input matrix to analyze.
            tol (float, optional): The tolerance for numerical stability.

        Raises:
            TypeError: If the input matrix is not a numpy array.
            ValueError: If the input matrix is not 2D.
        """
        # Check if the matrix is valid type and dimensions
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if matrix.ndim != 2:
            raise ValueError("Input must be a 2D matrix.")

        # Capture the reference to the target matrix
        self.matrix = matrix

        # Then compute statistics over the coefficients
        self.min = np.min(self.matrix)
        self.max = np.max(self.matrix)
        self.mean = np.mean(self.matrix)
        self.std = np.std(self.matrix)

        # Extract additional properties using numpy operations
        self.shape = self.matrix.shape
        self.rank = np.linalg.matrix_rank(self.matrix)

        # Compute matrix norms
        self.norm_fro = np.linalg.norm(self.matrix, ord="fro")
        self.norm_inf = np.linalg.norm(self.matrix, ord=np.inf)

        # Extract the matrix singular values
        self.U, self.sigmas, self.Vt = np.linalg.svd(self.matrix, full_matrices=True, compute_uv=True, hermitian=False)
        self.sigma_min = self.sigmas[-1]
        self.sigma_max = self.sigmas[0]
        self.sigma_cond = self.sigma_max / self.sigma_min

    def __str__(self) -> str:
        return (
            f"Type:\n"
            f"   shape: {self.matrix.shape}\n"
            f"   dtype: {self.matrix.dtype}\n"
            f"Statistics:\n"
            f"   min: {self.min}\n"
            f"   max: {self.max}\n"
            f"  mean: {self.mean}\n"
            f"   std: {self.std}\n"
            f"Basics:\n"
            f"   rank: {self.rank}\n"
            f"Norms:\n"
            f"   fro: {self.norm_fro}\n"
            f"   inf: {self.norm_inf}\n"
            f"SVD:\n"
            f"   sigma min: {self.sigma_min}\n"
            f"   sigma max: {self.sigma_max}\n"
            f"  sigma cond: {self.sigma_cond}\n"
        )


class SquareSymmetricMatrixProperties:
    def __init__(self, matrix: np.ndarray | None = None, tol: float | None = None):
        self.matrix: np.ndarray
        """Reference to the original matrix."""

        # Matrix statistics
        self.min: float = np.inf
        """The minimum element of the matrix."""
        self.max: float = np.inf
        """The maximum element of the matrix."""
        self.mean: float = np.inf
        """The mean of the matrix elements."""
        self.std: float = np.inf
        """The standard deviation of the matrix elements."""

        # Matrix dimensions
        """The matrix shape (rows, cols)."""
        self.dim: int = 0
        """The matrix dimension (number of rows/columns) when square."""
        self.symmetry_error: float = np.inf
        """The error measure of symmetry for the matrix."""

        # Matrix properties
        self.rank: int = 0
        """The matrix rank computed using `numpy.linalg.matrix_rank()`."""
        self.trace: float = np.inf
        """The matrix trace computed using `numpy.trace()`."""
        self.cond: float = np.inf
        """The matrix condition number computed using `numpy.linalg.cond()`."""
        self.signdet: float = np.inf
        """The matrix determinant sign computed using `numpy.linalg.slogdet()`."""
        self.logabsdet: float = np.inf
        """The matrix log absolute determinant computed using `numpy.linalg.slogdet()`."""
        self.det: float = np.inf
        """The matrix determinant computed as `sign * exp(logabsdet)`."""

        # Matrix norms
        self.norm_fro: float = np.inf
        """The Frobenius norm computed using `numpy.linalg.norm()`."""
        self.norm_inf: float = np.inf
        """The infinity norm computed using `numpy.linalg.norm()`."""

        # Spectral properties
        self.lambda_min: float = np.inf
        """The smallest eigenvalue."""
        self.lambda_max: float = np.inf
        """The largest eigenvalue."""
        self.lambda_cond: float = np.inf
        """The condition number defined via the ratio of max/min eigenvalues."""

        # SVD properties
        self.sigma_min: float = np.inf
        """The smallest singular value."""
        self.sigma_max: float = np.inf
        """The largest singular value."""
        self.sigma_cond: float = np.inf
        """The condition number defined via the ratio of max/min singular values."""

        # Convexity properties
        self.m: float = np.inf
        """The strong convexity parameter, defined as `m(A) := max(0, lambda_min(A))`."""
        self.L: float = np.inf
        """The Lipschitz constant, defined as the spectral radius `L(A) := rho(A) := abs(lambda_max(A))`."""
        self.kappa: float = np.inf
        """The condition number, defined as `kappa(A) := L(A) / m(A)`."""

        # Matrix properties
        self.is_square: bool = False
        self.is_symmetric: bool = False
        self.is_positive_definite: bool = False
        self.is_positive_semi_definite: bool = False

        # Caches
        self.lambdas: np.ndarray | None = None
        self.sigmas: np.ndarray | None = None
        self.U: np.ndarray | None = None
        self.Vt: np.ndarray | None = None

        # Compute matrix properties if specified
        if matrix is not None:
            self.compute(matrix, tol)

    def compute(self, matrix: np.ndarray, tol: float | None = None):
        """
        Compute the properties of the matrix.

        Args:
            matrix (np.ndarray): The input matrix to analyze.
            tol (float, optional): The tolerance for numerical stability.

        Raises:
            TypeError: If the input matrix is not a numpy array.
            ValueError: If the input matrix is not 2D.
        """
        # Check if the matrix is valid type and dimensions
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if matrix.ndim != 2:
            raise ValueError("Input must be a 2D matrix.")

        # Extract the epsilon value either from the specified tolerance or on the dtype
        if tol is not None:
            if not isinstance(tol, float):
                raise TypeError("tolerance parameter `tol` must be `float` type.")
            eps = tol
            eps_relaxed = tol
            eps_symmetry = max(tol, DEFAULT_MATRIX_SYMMETRY_EPS)
        else:
            eps = float(np.finfo(matrix.dtype).eps)
            eps_relaxed = 1e3 * eps
            eps_symmetry = max(eps_relaxed, DEFAULT_MATRIX_SYMMETRY_EPS)

        # Capture the reference to the target matrix
        self.matrix = matrix

        # First extract the basic properties of the matrix
        self.is_square = self.matrix.shape[0] == self.matrix.shape[1]
        self.is_symmetric = is_symmetric_matrix(self.matrix, tol=eps_symmetry)
        self.symmetry_error = symmetry_error_norm_l2(self.matrix)

        # Then compute statistics over the coefficients
        self.min = np.min(self.matrix)
        self.max = np.max(self.matrix)
        self.mean = np.mean(self.matrix)
        self.std = np.std(self.matrix)

        # Extract additional properties using numpy operations
        self.dim = self.matrix.shape[0]
        self.rank = np.linalg.matrix_rank(self.matrix)
        self.trace = np.trace(self.matrix)
        self.cond = np.linalg.cond(self.matrix)

        # Compute the determinant from the signed log-determinant
        self.signdet, self.logabsdet, self.det = _safe_slogdet(self.matrix)

        # Compute matrix norms
        self.norm_fro = np.linalg.norm(self.matrix, ord="fro")
        self.norm_inf = np.linalg.norm(self.matrix, ord=np.inf)

        # Extract the matrix eigenvalues
        self.lambdas = np.linalg.eigvals(self.matrix).real
        self.lambda_min = self.lambdas.min()
        self.lambda_max = self.lambdas.max()
        self.lambda_cond = self.lambda_max / self.lambda_min

        # Extract the matrix singular values
        self.U, self.sigmas, self.Vt = np.linalg.svd(self.matrix, full_matrices=True, compute_uv=True, hermitian=False)
        self.sigma_min = self.sigmas[-1]
        self.sigma_max = self.sigmas[0]
        self.sigma_cond = self.sigma_max / self.sigma_min

        # Compute the convexity parameters
        # self.m = np.abs(self.lambda_min)
        self.m = max(0.0, self.lambda_min)
        self.L = np.abs(self.lambda_max)
        self.kappa = self.L / self.m if self.m > 0 else np.inf

        # Determine the definiteness of the matrix
        self.is_positive_definite = np.all(self.lambdas > eps_relaxed)
        self.is_positive_semi_definite = np.all(self.lambdas > eps)

    def __str__(self) -> str:
        return (
            f"Type:\n"
            f"   shape: {self.matrix.shape}\n"
            f"   dtype: {self.matrix.dtype}\n"
            f"Info:\n"
            f"  square: {self.is_square}\n"
            f"   symm.: {self.is_symmetric} (err={self.symmetry_error})\n"
            f"     PSD: {self.is_positive_semi_definite}\n"
            f"      PD: {self.is_positive_definite}\n"
            f"Statistics:\n"
            f"   min: {self.min}\n"
            f"   max: {self.max}\n"
            f"  mean: {self.mean}\n"
            f"   std: {self.std}\n"
            f"Basics:\n"
            f"     rank: {self.rank}\n"
            f"    trace: {self.trace}\n"
            f"     cond: {self.cond}\n"
            f"sign(det): {self.signdet}\n"
            f" log|det|: {self.logabsdet}\n"
            f"      det: {self.det}\n"
            f"Norms:\n"
            f"   fro: {self.norm_fro}\n"
            f"   inf: {self.norm_inf}\n"
            f"Spectral:\n"
            f"   lambda min: {self.lambda_min}\n"
            f"   lambda max: {self.lambda_max}\n"
            f"  lambda cond: {self.lambda_cond}\n"
            f"SVD:\n"
            f"   sigma min: {self.sigma_min}\n"
            f"   sigma max: {self.sigma_max}\n"
            f"  sigma cond: {self.sigma_cond}\n"
            f"Convexity:\n"
            f"      L: {self.L}\n"
            f"      m: {self.m}\n"
            f"  kappa: {self.kappa}\n"
        )


class MatrixComparison:
    def __init__(self, A: np.ndarray, B: np.ndarray, tol: float = 0.0):
        self.A = A
        self.B = B

        self.eps = np.finfo(self.A.dtype).eps

        self.E = self.A - self.B
        self.E_clip = self._error_clipped(tol=tol)

        # Compute all metrics
        self.frobenius_error = self._frobenius_error()
        self.max_element_error = self._max_element_error()
        self.relative_frobenius_error = self._relative_frobenius_error()
        self.svd_error = self._svd_error()
        self.relative_determinant_error = self._relative_determinant_error()
        self.norm_1_error = self._norm_1_error()
        self.norm_2_error = self._norm_2_error()
        self.norm_inf_error = self._norm_inf_error()

    def save(
        self,
        path: str,
        title: str = "Matrix",
        name_A: str = "A",
        name_B: str = "B",
        symbol_A: str = "A",
        symbol_B: str = "B",
    ):
        """Save error visualizations to the specified path."""
        import os  # noqa: PLC0415

        from newton._src.solvers.kamino.utils.sparse import sparseview  # noqa: PLC0415

        os.makedirs(path, exist_ok=True)
        sparseview(self.A, title=f"{title} {name_A}", path=os.path.join(path, f"{symbol_A}.png"))
        sparseview(self.B, title=f"{title} {name_B}", path=os.path.join(path, f"{symbol_B}.png"))
        sparseview(self.E, title=f"{title} Error", path=os.path.join(path, f"{symbol_A}_err.png"))
        sparseview(self.E_clip, title=f"{title} Error (Clipped)", path=os.path.join(path, f"{symbol_A}_err_clip.png"))

    def _error_clipped(self, tol: float = 0.0):
        """Clip small errors to zero for visualization."""
        if tol > 0.0:
            eps = tol
        else:
            eps = self.eps
        E_clip = np.zeros_like(self.E)
        for i in range(self.E.shape[0]):
            for j in range(self.E.shape[1]):
                if np.abs(self.E[i, j]) < eps:
                    E_clip[i, j] = 0.0
                else:
                    E_clip[i, j] = self.E[i, j]
        return E_clip

    def _frobenius_error(self):
        return np.linalg.norm(self.E, "fro")

    def _max_element_error(self):
        return np.max(np.abs(self.E))

    def _relative_frobenius_error(self):
        return np.linalg.norm(self.E, "fro") / np.linalg.norm(self.A, "fro")

    def _svd_error(self):
        # Singular value decomposition error
        S_A = np.linalg.svd(self.A, compute_uv=False)
        S_B = np.linalg.svd(self.B, compute_uv=False)
        return np.linalg.norm(S_A - S_B) / np.linalg.norm(S_A)

    def _relative_determinant_error(self):
        det_A = np.linalg.det(self.A)
        det_B = np.linalg.det(self.B)
        if det_A == 0:
            return np.abs(det_A - det_B)  # Just the difference if det(A) is zero
        return np.abs(det_A - det_B) / np.abs(det_A)

    def _norm_1_error(self):
        return np.linalg.norm(self.E, ord=1)

    def _norm_2_error(self):
        return np.linalg.norm(self.E, ord=2)

    def _norm_inf_error(self):
        return np.linalg.norm(self.E, ord=np.inf)

    def __str__(self) -> str:
        return (
            f"Frobenius Error           : {self.frobenius_error}\n"
            f"Max Element Error         : {self.max_element_error}\n"
            f"Relative Frobenius Error  : {self.relative_frobenius_error}\n"
            f"SVD Error                 : {self.svd_error}\n"
            f"Relative Determinant Error: {self.relative_determinant_error}\n"
            f"1-Norm Error              : {self.norm_1_error}\n"
            f"2-Norm Error              : {self.norm_2_error}\n"
            f"Inf-Norm Error            : {self.norm_inf_error}\n"
        )
