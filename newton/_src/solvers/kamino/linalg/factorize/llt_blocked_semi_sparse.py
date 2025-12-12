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

"""KAMINO: Linear Algebra: Blocked Semi-Sparse LLT (i.e. Cholesky) factorization using Warp's Tile API."""

from functools import cache

import numpy as np
import warp as wp

###
# Module interface
###

__all__ = ["SemiSparseBlockCholeskySolverBatched"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


def cuthill_mckee_ordering(M):
    """
    Given a symmetric binary matrix M (0/1, shape n x n), returns a permutation array P
    such that reordering both the rows and columns of M by P (i.e., M[P][:, P]) produces
    a matrix with reduced bandwidth according to the Cuthill-McKee algorithm (a minimal fill-in heuristic).
    """
    n = M.shape[0]
    visited = np.zeros(n, dtype=bool)
    degrees = M.sum(axis=1)
    order = []

    for start in np.argsort(degrees):
        if not visited[start]:
            queue = [start]
            visited[start] = True
            while queue:
                node = queue.pop(0)
                order.append(node)
                # Find unvisited neighbors
                neighbors = np.where((M[node] != 0) & (~visited))[0]
                # Sort neighbors by degree (ascending)
                neighbors = neighbors[np.argsort(degrees[neighbors])]
                for neighbor in neighbors:
                    visited[neighbor] = True
                queue.extend(neighbors)

    # For minimal fill-in, use reverse Cuthill-McKee
    P = order[::-1]
    return np.array(P, dtype=int)


def compute_inverse_ordering(ordering):
    """
    Computes the inverse permutation of the given ordering.

    Args:
        ordering (np.ndarray): The permutation array used for reordering (length n).

    Returns:
        inv_ordering (np.ndarray): The inverse permutation array.
    """
    inv_ordering = np.empty_like(ordering)
    inv_ordering[ordering] = np.arange(len(ordering))
    return inv_ordering


@wp.kernel
def reorder_rows_kernel(
    src: wp.array3d(dtype=float),
    dst: wp.array3d(dtype=float),
    ordering: wp.array(dtype=int, ndim=2),
    n_rows_arr: wp.array(dtype=int, ndim=1),
    n_cols_arr: wp.array(dtype=int, ndim=1),
    batch_mask: wp.array(dtype=int, ndim=1),
):
    batch_id, i, j = wp.tid()  # 2D launch: (n_rows, n_cols)
    n_rows = n_rows_arr[batch_id]
    n_cols = n_cols_arr[batch_id]
    if i < n_rows and j < n_cols and batch_mask[batch_id] != 0:
        src_row = ordering[batch_id, i]
        src_col = ordering[batch_id, j]
        dst[batch_id, i, j] = src[batch_id, src_row, src_col]


@wp.kernel
def reorder_rows_kernel_col_vector(
    src: wp.array3d(dtype=float),
    dst: wp.array3d(dtype=float),
    ordering: wp.array(dtype=int, ndim=2),
    n_rows_arr: wp.array(dtype=int, ndim=1),
    batch_mask: wp.array(dtype=int, ndim=1),
):
    batch_id, i = wp.tid()
    n_rows = n_rows_arr[batch_id]
    if i < n_rows and batch_mask[batch_id] != 0:
        src_row = ordering[batch_id, i]
        # For column vectors (2d arrays with shape (n, 1)), just copy columns directly
        dst[batch_id, i, 0] = src[batch_id, src_row, 0]


def to_binary_matrix(M):
    return (M != 0).astype(int)


def sparsity_to_tiles(sparsity_matrix, tile_size):
    """
    Given a 2D 0/1 sparsity matrix and a tile size,
    returns a 2D 0/1 matrix indicating which tiles are nonzero.
    """
    n_rows, n_cols = sparsity_matrix.shape
    n_tile_rows = (n_rows + tile_size - 1) // tile_size
    n_tile_cols = (n_cols + tile_size - 1) // tile_size
    tile_matrix = np.zeros((n_tile_rows, n_tile_cols), dtype=int)
    for i in range(n_rows):
        for j in range(n_cols):
            if sparsity_matrix[i, j] != 0:
                tile_row = i // tile_size
                tile_col = j // tile_size
                tile_matrix[tile_row, tile_col] = 1
    return tile_matrix


def symbolic_cholesky_dense(M, tile_size):
    """
    Given a symmetric 0/1 matrix M, returns the block sparsity pattern (lower-triangular, bool)
    for the Cholesky factor L of M, using a block Cholesky symbolic analysis.
    The output is a 2D 0/1 matrix of shape (n_tiles, n_tiles) indicating which tiles of L are nonzero.

    This implementation follows the classical symbolic block Cholesky fill-in algorithm:
    For each block row i, and for each block column j < i, if M[i, j] is nonzero or
    there exists a k < j such that both L[i, k] and L[j, k] are nonzero, then L[i, j] is nonzero.
    """

    # Dimensions
    n = M.shape[0]
    n_tiles = (n + tile_size - 1) // tile_size

    # Compute block sparsity pattern of M
    M_tile_pattern = sparsity_to_tiles(M, tile_size)

    # Initialize L_tile_pattern as strictly lower triangle of M_tile_pattern
    L_tile_pattern = np.zeros((n_tiles, n_tiles), dtype=bool)
    for i in range(n_tiles):
        for j in range(i + 1):
            L_tile_pattern[i, j] = bool(M_tile_pattern[i, j])

    # Symbolic block Cholesky fill-in
    for j in range(n_tiles):
        for i in range(j + 1, n_tiles):
            if not L_tile_pattern[i, j]:
                # Check for fill-in: does there exist k < j such that L[i, k] and L[j, k] are nonzero?
                for k in range(j):
                    if L_tile_pattern[i, k] and L_tile_pattern[j, k]:
                        L_tile_pattern[i, j] = True
                        break

    # Only lower triangle is relevant for Cholesky
    L_tile_pattern = np.tril(L_tile_pattern, k=0)
    return L_tile_pattern.astype(np.int32)


@cache
def create_blocked_cholesky_kernel(block_size: int):
    @wp.kernel
    def blocked_cholesky_kernel(
        A_batched: wp.array(dtype=float, ndim=3),
        L_batched: wp.array(dtype=float, ndim=3),
        L_tile_pattern_batched: wp.array(dtype=int, ndim=3),
        active_matrix_size_arr: wp.array(dtype=int, ndim=1),
        batch_mask: wp.array(dtype=int, ndim=1),
    ):
        """
        Batched Cholesky factorization of symmetric positive definite matrices in blocks.
        For each matrix A in batch, computes lower-triangular L where A = L L^T.

        Args:
            A_batched: Input SPD matrices (batch_size, n, n)
            L_batched: Output Cholesky factors (batch_size, n, n)
            L_tile_pattern_batched: Sparsity pattern for L tiles (1=nonzero, 0=zero)
            active_matrix_size_arr: Size of each active matrix in batch
            batch_mask: Flag for each matrix in the batch, indicating whether to process it (0 = skip)

        Notes:
            - Parallel processing across batch dimension
            - Block size = block_size x block_size
            - Uses tile patterns to skip zero blocks
            - A must support block reading
        """
        batch_id, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

        if batch_mask[batch_id] == 0:
            return

        A = A_batched[batch_id]
        L = L_batched[batch_id]
        L_tile_pattern = L_tile_pattern_batched[batch_id]

        active_matrix_size = active_matrix_size_arr[batch_id]

        # Round up active_matrix_size to next multiple of block_size
        n = ((active_matrix_size + block_size - 1) // block_size) * block_size

        # Process the matrix in blocks along its leading dimension.
        for k in range(0, n, block_size):
            end = k + block_size

            # Check if this diagonal tile is nonzero
            tile_k = k // block_size

            # Load current diagonal block A[k:end, k:end]
            # and update with contributions from previously computed blocks.
            A_kk_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(k, k), storage="shared")
            # The following pads the matrix if it is not divisible by block_size
            if k + block_size > active_matrix_size:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block

                for i in range(num_iterations):
                    linear_index = tid_block + i * num_threads_per_block
                    linear_index = linear_index % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = A_kk_tile[row, col]
                    if k + row >= active_matrix_size or k + col >= active_matrix_size:
                        value = wp.where(row == col, float(1), float(0))
                    A_kk_tile[row, col] = value

            if k > 0:
                for j in range(0, k, block_size):
                    tile_j = j // block_size
                    # Only update if both L_tile_pattern[tile_k, tile_j] is nonzero
                    if L_tile_pattern[tile_k, tile_j] == 0:
                        continue
                    L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j))
                    L_block_T = wp.tile_transpose(L_block)
                    L_L_T_block = wp.tile_matmul(L_block, L_block_T)
                    A_kk_tile -= L_L_T_block

            # Compute the Cholesky factorization for the block
            L_kk_tile = wp.tile_cholesky(A_kk_tile)
            wp.tile_store(L, L_kk_tile, offset=(k, k))

            # Process the blocks below the current block
            for i in range(end, n, block_size):
                tile_i = i // block_size

                # Only store result if L_tile_pattern[tile_i, tile_k] is nonzero
                if L_tile_pattern[tile_i, tile_k] == 0:
                    continue

                A_ik_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(i, k), storage="shared")
                # The following if pads the matrix if it is not divisible by block_size
                if i + block_size > active_matrix_size or k + block_size > active_matrix_size:
                    num_tile_elements = block_size * block_size
                    num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block

                    for ii in range(num_iterations):
                        linear_index = tid_block + ii * num_threads_per_block
                        linear_index = linear_index % num_tile_elements
                        row = linear_index // block_size
                        col = linear_index % block_size
                        value = A_ik_tile[row, col]
                        if i + row >= active_matrix_size or k + col >= active_matrix_size:
                            value = wp.where(i + row == k + col, float(1), float(0))
                        A_ik_tile[row, col] = value

                if k > 0:
                    for j in range(0, k, block_size):
                        tile_j = j // block_size
                        # Only update if both L_tile_pattern[tile_i, tile_j] and L_tile_pattern[tile_k, tile_j] are nonzero
                        if L_tile_pattern[tile_i, tile_j] == 0 or L_tile_pattern[tile_k, tile_j] == 0:
                            continue
                        L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j))
                        L_2_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j))
                        L_T_tile = wp.tile_transpose(L_2_tile)
                        L_L_T_tile = wp.tile_matmul(L_tile, L_T_tile)
                        A_ik_tile -= L_L_T_tile

                t = wp.tile_transpose(A_ik_tile)
                tmp = wp.tile_lower_solve(L_kk_tile, t)
                sol_tile = wp.tile_transpose(tmp)

                wp.tile_store(L, sol_tile, offset=(i, k))

    return blocked_cholesky_kernel


@cache
def create_blocked_cholesky_solve_kernel(block_size: int):
    @wp.kernel
    def blocked_cholesky_solve_kernel(
        L_batched: wp.array(dtype=float, ndim=3),
        L_tile_pattern_batched: wp.array(dtype=int, ndim=3),
        b_batched: wp.array(dtype=float, ndim=3),
        x_batched: wp.array(dtype=float, ndim=3),
        y_batched: wp.array(dtype=float, ndim=3),
        active_matrix_size_arr: wp.array(dtype=int, ndim=1),
        batch_mask: wp.array(dtype=int, ndim=1),
    ):
        """
        Batched blocked Cholesky solver kernel. For each batch, solves A x = b using L L^T = A.
        Uses forward/backward substitution with block size optimization.
        """

        batch_id, _tid_block = wp.tid()

        if batch_mask[batch_id] == 0:
            return

        L = L_batched[batch_id]
        b = b_batched[batch_id]
        x = x_batched[batch_id]
        y = y_batched[batch_id]
        L_tile_pattern = L_tile_pattern_batched[batch_id]
        active_matrix_size = active_matrix_size_arr[batch_id]

        # Round up active_matrix_size to next multiple of block_size
        n = ((active_matrix_size + block_size - 1) // block_size) * block_size

        # Forward substitution: solve L y = b
        for i in range(0, n, block_size):
            tile_i = i // block_size
            # Only process if diagonal block is present
            if L_tile_pattern[tile_i, tile_i] == 0:
                continue

            i_end = i + block_size
            rhs_tile = wp.tile_load(b, shape=(block_size, 1), offset=(i, 0))
            if i > 0:
                for j in range(0, i, block_size):
                    tile_j = j // block_size
                    # Only process if L_tile_pattern[tile_i, tile_j] is nonzero
                    if L_tile_pattern[tile_i, tile_j] == 0:
                        continue
                    L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j))
                    y_block = wp.tile_load(y, shape=(block_size, 1), offset=(j, 0))
                    Ly_block = wp.tile_matmul(L_block, y_block)
                    rhs_tile -= Ly_block
            L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, i))
            y_tile = wp.tile_lower_solve(L_tile, rhs_tile)
            wp.tile_store(y, y_tile, offset=(i, 0))

        # Backward substitution: solve L^T x = y
        for i in range(n - block_size, -1, -block_size):
            tile_i = i // block_size
            # Only process if diagonal block is present
            if L_tile_pattern[tile_i, tile_i] == 0:
                continue

            i_start = i
            i_end = i_start + block_size
            rhs_tile = wp.tile_load(y, shape=(block_size, 1), offset=(i_start, 0))
            if i_end < n:
                for j in range(i_end, n, block_size):
                    tile_j = j // block_size
                    # Only process if L_tile_pattern[tile_j, tile_i] is nonzero
                    if L_tile_pattern[tile_j, tile_i] == 0:
                        continue
                    L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(j, i_start))
                    L_T_tile = wp.tile_transpose(L_tile)
                    x_tile = wp.tile_load(x, shape=(block_size, 1), offset=(j, 0))
                    L_T_x_tile = wp.tile_matmul(L_T_tile, x_tile)
                    rhs_tile -= L_T_x_tile
            L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i_start, i_start))
            x_tile = wp.tile_upper_solve(wp.tile_transpose(L_tile), rhs_tile)
            wp.tile_store(x, x_tile, offset=(i_start, 0))

    return blocked_cholesky_solve_kernel


class SemiSparseBlockCholeskySolverBatched:
    """
    Batched solver for linear systems using block Cholesky factorization.
    "Semi-sparse" because it uses dense storage but exploits sparsity by tracking zero tiles
    to skip unnecessary computation. Handles multiple systems in parallel with fixed matrix size.
    """

    def __init__(self, num_batches: int, max_num_equations: int, block_size=16, device="cuda", enable_reordering=True):
        self.num_batches = num_batches
        self.max_num_equations = max_num_equations
        self.device = device

        self.num_threads_per_block_factorize = 128
        self.num_threads_per_block_solve = 64
        self.active_matrix_size_int = -1

        self.block_size = block_size
        self.cholesky_kernel = create_blocked_cholesky_kernel(block_size)
        self.solve_kernel = create_blocked_cholesky_solve_kernel(block_size)

        # Allocate workspace arrays for factorization and solve
        # Compute padded size rounded up to next multiple of block size
        self.padded_num_equations = (
            (self.max_num_equations + self.block_size - 1) // self.block_size
        ) * self.block_size

        self.A_swizzled = wp.zeros(
            shape=(num_batches, self.padded_num_equations, self.padded_num_equations), dtype=float, device=self.device
        )
        self.L = wp.zeros(
            shape=(num_batches, self.padded_num_equations, self.padded_num_equations), dtype=float, device=self.device
        )
        self.y = wp.zeros(
            shape=(num_batches, self.padded_num_equations, 1), dtype=float, device=self.device
        )  # temp memory
        self.result_swizzled = wp.zeros(
            shape=(num_batches, self.padded_num_equations, 1), dtype=float, device=self.device
        )  # temp memory
        self.rhs_swizzled = wp.zeros(
            shape=(num_batches, self.padded_num_equations, 1), dtype=float, device=self.device
        )  # temp memory

        self.num_tiles = (self.padded_num_equations + self.block_size - 1) // self.block_size
        self.L_tile_pattern = wp.zeros(
            shape=(num_batches, self.num_tiles, self.num_tiles), dtype=int, device=self.device
        )

        self.enable_reordering = enable_reordering

    def capture_sparsity_pattern(
        self,
        A: np.ndarray,  # 3D array (batch_size, n, n)
        A_reorder_size: np.ndarray,  # 1D array (batch_size)
    ):
        """
        Captures sparsity pattern and computes fill-reducing ordering for batched matrices.

        Args:
            A: Input SPD matrices of shape (batch_size, n, n), as float arrays or directly as binary 0/1 matrices
            indicating the sparsity pattern (float arrays will be converted to binary automatically).
            A_reorder_size: Per-batch size of top-left block to reorder for sparsity

        Computes Cuthill-McKee ordering on top-left block, analyzes symbolic Cholesky factorization,
        and stores tile-level sparsity patterns. Tiles beyond A_reorder_size are treated as dense.
        """

        batch_size = A.shape[0]

        # Convert to binary
        A = to_binary_matrix(A)
        A = A[:, : self.max_num_equations, : self.max_num_equations]

        # Initialize arrays for each batch
        orderings = np.zeros((batch_size, self.max_num_equations), dtype=np.int32)
        inverse_orderings = np.zeros((batch_size, self.max_num_equations), dtype=np.int32)
        L_tile_patterns = np.zeros((batch_size, self.num_tiles, self.num_tiles), dtype=np.int32)

        # Process each batch independently
        for batch_id in range(batch_size):
            # Call cuthill_mckee_ordering on the binary version of A and store both orderings
            reorder_size = A_reorder_size[batch_id]
            ordering = cuthill_mckee_ordering(A[batch_id, :reorder_size, :reorder_size])

            # Append sequential indices for remaining rows/cols
            remaining_indices = np.arange(reorder_size, self.max_num_equations)
            ordering = np.concatenate([ordering, remaining_indices])
            orderings[batch_id] = ordering

            inverse_ordering = compute_inverse_ordering(ordering)
            inverse_orderings[batch_id] = inverse_ordering

            # Reorder A and then extract the sparsity patterns
            if self.enable_reordering:
                A_reordered = A[batch_id][ordering][:, ordering]
            else:
                A_reordered = A[batch_id]

            L_tile_pattern_np = symbolic_cholesky_dense(A_reordered, self.block_size)

            # Set all tiles after A_reorder_size to 1
            reorder_tile_idx = reorder_size // self.block_size  # Conservative: Round down
            for i in range(reorder_tile_idx, self.num_tiles):
                for j in range(min(i + 1, self.num_tiles)):  # Only set lower triangular part
                    L_tile_pattern_np[i, j] = 1

            L_tile_patterns[batch_id] = L_tile_pattern_np

        # Convert to warp arrays on the correct device
        self.ordering = wp.array(orderings, dtype=int, device=self.device)
        self.inverse_ordering = wp.array(inverse_orderings, dtype=int, device=self.device)
        self.L_tile_pattern = wp.array(L_tile_patterns, dtype=int, device=self.device)

    def factorize(
        self,
        A: wp.array(dtype=float, ndim=3),
        num_active_equations: wp.array(dtype=int, ndim=1),
        batch_mask: wp.array(dtype=int, ndim=1),
    ):
        """
        Computes the Cholesky factorization of a symmetric positive definite matrix A in blocks.
        It returns a lower-triangular matrix L such that A = L L^T.

        Args:
            A: Input SPD matrices of shape (batch_size, n, n).
            num_active_equations: Size of the top-left block to factorize for each matrix in the batch.
            batch_mask: Flag for each matrix in the batch, indicating whether to process it (0 = skip)
        """

        self.num_active_equations = num_active_equations

        # Reorder A and store in self.A_reordered
        if self.enable_reordering:
            wp.launch(
                reorder_rows_kernel,
                dim=[self.num_batches, self.max_num_equations, self.max_num_equations],
                inputs=[
                    A,
                    self.A_swizzled,
                    self.ordering,
                    num_active_equations,
                    num_active_equations,
                    batch_mask,
                ],
                device=self.device,
            )
            A = self.A_swizzled

        wp.launch_tiled(
            self.cholesky_kernel,
            dim=self.num_batches,
            inputs=[A, self.L, self.L_tile_pattern, num_active_equations, batch_mask],
            block_dim=self.num_threads_per_block_factorize,
            device=self.device,
        )

    def solve(
        self,
        rhs: wp.array(dtype=float, ndim=3),
        result: wp.array(dtype=float, ndim=3),
        batch_mask: wp.array(dtype=int, ndim=1),
    ):
        """
        Solves A x = b given the Cholesky factor L (A = L L^T) using
        blocked forward and backward substitution.

        Args:
            rhs: Input right-hand-side matrices of shape (batch_size, n, p).
            result: Output solution matrices of shape (batch_size, n, p).
            batch_mask: Flag for each matrix in the batch, indicating whether to process it (0 = skip)
        """

        R = result
        if self.enable_reordering:
            R = self.result_swizzled
            wp.launch(
                reorder_rows_kernel_col_vector,
                dim=[self.num_batches, self.max_num_equations],
                inputs=[rhs, self.rhs_swizzled, self.ordering, self.num_active_equations, batch_mask],
                device=self.device,
            )

            rhs = self.rhs_swizzled

        # Then solve the system using blocked_cholesky_solve kernel
        wp.launch_tiled(
            self.solve_kernel,
            dim=self.num_batches,
            inputs=[self.L, self.L_tile_pattern, rhs, R, self.y, self.num_active_equations, batch_mask],
            block_dim=self.num_threads_per_block_solve,
            device=self.device,
        )

        if self.enable_reordering:
            # Undo reordering
            wp.launch(
                reorder_rows_kernel_col_vector,
                dim=[self.num_batches, self.max_num_equations],
                inputs=[R, result, self.inverse_ordering, self.num_active_equations, batch_mask],
                device=self.device,
            )
