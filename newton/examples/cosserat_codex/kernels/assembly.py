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

"""Warp kernels for JMJT matrix assembly.

This module contains kernels for assembling the JMJT matrix in dense,
banded, and block-tridiagonal formats.
"""

from __future__ import annotations

import warp as wp

from newton.examples.cosserat_codex.constants import BAND_KD

from .math import _block_index, _inv_inertia_mul_vec, _warp_jacobian_index


@wp.kernel
def _warp_assemble_jmjt_dense(
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    inv_masses: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    n_dofs: int,
    A: wp.array2d(dtype=wp.float32),
):
    """Assemble JMJT dense matrix with proper mass weighting.

    Constraint i involves particles i (segment 0) and i+1 (segment 1).
    Uses actual inverse masses to correctly handle locked particles (inv_mass=0).
    
    Args:
        jacobian_pos: Position Jacobians.
        jacobian_rot: Rotation Jacobians.
        compliance: Compliance values.
        inv_masses: Inverse masses.
        inv_inertia: Inverse inertia tensors (9 floats per particle).
        n_dofs: Number of DOFs.
        A: Output dense matrix.
    """
    i = wp.tid()
    block_start = 6 * i

    # Use actual inverse masses from the array
    inv_m0 = inv_masses[i]
    inv_m1 = inv_masses[i + 1]

    # Diagonal block
    for row in range(6):
        for col in range(6):
            val = 0.0

            # Position contribution from segment 0 (particle i)
            for k in range(3):
                j_p0_r = jacobian_pos[_warp_jacobian_index(i, row, k)]
                j_p0_c = jacobian_pos[_warp_jacobian_index(i, col, k)]
                val += j_p0_r * inv_m0 * j_p0_c

            # Position contribution from segment 1 (particle i+1)
            for k in range(3):
                j_p1_r = jacobian_pos[_warp_jacobian_index(i, row, k + 3)]
                j_p1_c = jacobian_pos[_warp_jacobian_index(i, col, k + 3)]
                val += j_p1_r * inv_m1 * j_p1_c

            # Rotation contribution from segment 0
            j_t0_r_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, row, 0)],
                jacobian_rot[_warp_jacobian_index(i, row, 1)],
                jacobian_rot[_warp_jacobian_index(i, row, 2)],
            )
            j_t0_c_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col, 0)],
                jacobian_rot[_warp_jacobian_index(i, col, 1)],
                jacobian_rot[_warp_jacobian_index(i, col, 2)],
            )
            inv_I0_j_t0_c = _inv_inertia_mul_vec(inv_inertia, i, j_t0_c_vec)
            val += wp.dot(j_t0_r_vec, inv_I0_j_t0_c)

            # Rotation contribution from segment 1
            j_t1_r_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, row, 3)],
                jacobian_rot[_warp_jacobian_index(i, row, 4)],
                jacobian_rot[_warp_jacobian_index(i, row, 5)],
            )
            j_t1_c_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col, 3)],
                jacobian_rot[_warp_jacobian_index(i, col, 4)],
                jacobian_rot[_warp_jacobian_index(i, col, 5)],
            )
            inv_I1_j_t1_c = _inv_inertia_mul_vec(inv_inertia, i + 1, j_t1_c_vec)
            val += wp.dot(j_t1_r_vec, inv_I1_j_t1_c)

            if row == col:
                val += compliance[i * 6 + row]
            row_idx = block_start + row
            col_idx = block_start + col
            if row_idx < n_dofs and col_idx < n_dofs:
                A[row_idx, col_idx] = val

    # Off-diagonal block: coupling between constraint i-1 and i
    # Shared particle is i (segment 1 of constraint i-1 = segment 0 of constraint i)
    if i > 0:
        prev = i - 1
        prev_block = 6 * prev

        for row in range(6):
            for col in range(6):
                val = 0.0

                # Position contribution using shared particle's inverse mass
                for k in range(3):
                    j_p1_prev = jacobian_pos[_warp_jacobian_index(prev, row, k + 3)]
                    j_p0_cur = jacobian_pos[_warp_jacobian_index(i, col, k)]
                    val += j_p1_prev * inv_m0 * j_p0_cur

                # Rotation contribution
                j_t1_prev_vec = wp.vec3(
                    jacobian_rot[_warp_jacobian_index(prev, row, 3)],
                    jacobian_rot[_warp_jacobian_index(prev, row, 4)],
                    jacobian_rot[_warp_jacobian_index(prev, row, 5)],
                )
                j_t0_cur_vec = wp.vec3(
                    jacobian_rot[_warp_jacobian_index(i, col, 0)],
                    jacobian_rot[_warp_jacobian_index(i, col, 1)],
                    jacobian_rot[_warp_jacobian_index(i, col, 2)],
                )
                inv_I_shared_j_t0_cur = _inv_inertia_mul_vec(inv_inertia, i, j_t0_cur_vec)
                val += wp.dot(j_t1_prev_vec, inv_I_shared_j_t0_cur)

                row_idx = prev_block + row
                col_idx = block_start + col
                if row_idx < n_dofs and col_idx < n_dofs:
                    A[row_idx, col_idx] = val
                row_idx = block_start + col
                col_idx = prev_block + row
                if row_idx < n_dofs and col_idx < n_dofs:
                    A[row_idx, col_idx] = val


@wp.kernel
def _warp_assemble_jmjt_banded(
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    inv_masses: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    n_dofs: int,
    ab: wp.array2d(dtype=wp.float32),
):
    """Assemble JMJT banded matrix with proper mass weighting.

    The JMJT assembly computes J * M^-1 * J^T where M^-1 is block diagonal.
    Uses actual inverse masses to correctly handle locked particles (inv_mass=0).

    Constraint i involves particles i (segment 0) and i+1 (segment 1).
    
    Args:
        jacobian_pos: Position Jacobians.
        jacobian_rot: Rotation Jacobians.
        compliance: Compliance values.
        inv_masses: Inverse masses.
        inv_inertia: Inverse inertia tensors (9 floats per particle).
        n_dofs: Number of DOFs.
        ab: Output banded matrix in LAPACK format.
    """
    i = wp.tid()
    block_start = 6 * i
    if block_start >= n_dofs:
        return

    # Use actual inverse masses from the array
    inv_m0 = inv_masses[i]
    inv_m1 = inv_masses[i + 1]

    # Diagonal block: J0 * M0_inv * J0^T + J1 * M1_inv * J1^T + compliance
    for row in range(6):
        for col in range(6):
            val = 0.0

            # Position contribution from segment 0 (particle i)
            for k in range(3):
                j_p0_r = jacobian_pos[_warp_jacobian_index(i, row, k)]
                j_p0_c = jacobian_pos[_warp_jacobian_index(i, col, k)]
                val += j_p0_r * inv_m0 * j_p0_c

            # Position contribution from segment 1 (particle i+1)
            for k in range(3):
                j_p1_r = jacobian_pos[_warp_jacobian_index(i, row, k + 3)]
                j_p1_c = jacobian_pos[_warp_jacobian_index(i, col, k + 3)]
                val += j_p1_r * inv_m1 * j_p1_c

            # Rotation contribution from segment 0: J_t0_r^T * inv_I0 * J_t0_c
            j_t0_r_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, row, 0)],
                jacobian_rot[_warp_jacobian_index(i, row, 1)],
                jacobian_rot[_warp_jacobian_index(i, row, 2)],
            )
            j_t0_c_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col, 0)],
                jacobian_rot[_warp_jacobian_index(i, col, 1)],
                jacobian_rot[_warp_jacobian_index(i, col, 2)],
            )
            inv_I0_j_t0_c = _inv_inertia_mul_vec(inv_inertia, i, j_t0_c_vec)
            val += wp.dot(j_t0_r_vec, inv_I0_j_t0_c)

            # Rotation contribution from segment 1: J_t1_r^T * inv_I1 * J_t1_c
            j_t1_r_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, row, 3)],
                jacobian_rot[_warp_jacobian_index(i, row, 4)],
                jacobian_rot[_warp_jacobian_index(i, row, 5)],
            )
            j_t1_c_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col, 3)],
                jacobian_rot[_warp_jacobian_index(i, col, 4)],
                jacobian_rot[_warp_jacobian_index(i, col, 5)],
            )
            inv_I1_j_t1_c = _inv_inertia_mul_vec(inv_inertia, i + 1, j_t1_c_vec)
            val += wp.dot(j_t1_r_vec, inv_I1_j_t1_c)

            # Add compliance to diagonal
            if row == col:
                val += compliance[i * 6 + row]

            row_idx = block_start + row
            col_idx = block_start + col
            if row_idx <= col_idx:
                band_row = BAND_KD + row_idx - col_idx
                if band_row >= 0 and band_row <= BAND_KD:
                    ab[band_row, col_idx] = val

    # Off-diagonal block: coupling between constraint i-1 and i
    # Shared particle is i (segment 1 of constraint i-1 = segment 0 of constraint i)
    if i > 0:
        prev = i - 1
        prev_block = block_start - 6

        for row in range(6):
            for col in range(6):
                val = 0.0

                # Position contribution using shared particle's inverse mass
                for k in range(3):
                    j_p1_prev = jacobian_pos[_warp_jacobian_index(prev, row, k + 3)]
                    j_p0_cur = jacobian_pos[_warp_jacobian_index(i, col, k)]
                    val += j_p1_prev * inv_m0 * j_p0_cur

                # Rotation contribution: J_t1_prev^T * inv_I_shared * J_t0_cur
                j_t1_prev_vec = wp.vec3(
                    jacobian_rot[_warp_jacobian_index(prev, row, 3)],
                    jacobian_rot[_warp_jacobian_index(prev, row, 4)],
                    jacobian_rot[_warp_jacobian_index(prev, row, 5)],
                )
                j_t0_cur_vec = wp.vec3(
                    jacobian_rot[_warp_jacobian_index(i, col, 0)],
                    jacobian_rot[_warp_jacobian_index(i, col, 1)],
                    jacobian_rot[_warp_jacobian_index(i, col, 2)],
                )
                inv_I_shared_j_t0_cur = _inv_inertia_mul_vec(inv_inertia, i, j_t0_cur_vec)
                val += wp.dot(j_t1_prev_vec, inv_I_shared_j_t0_cur)

                row_idx = prev_block + row
                col_idx = block_start + col
                band_row = BAND_KD + row_idx - col_idx
                if band_row >= 0 and band_row <= BAND_KD:
                    ab[band_row, col_idx] = val


@wp.kernel
def _warp_assemble_jmjt_blocks(
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    inv_masses: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    n_edges: int,
    diag_blocks: wp.array(dtype=wp.float32),
    offdiag_blocks: wp.array(dtype=wp.float32),
):
    """Assemble JMJT blocks with proper mass weighting for block Thomas solver.

    Uses actual inverse masses to correctly handle locked particles (inv_mass=0).
    Constraint i involves particles i (segment 0) and i+1 (segment 1).
    
    Args:
        jacobian_pos: Position Jacobians.
        jacobian_rot: Rotation Jacobians.
        compliance: Compliance values.
        inv_masses: Inverse masses.
        inv_inertia: Inverse inertia tensors (9 floats per particle).
        n_edges: Number of edges.
        diag_blocks: Output diagonal blocks (36 floats per edge).
        offdiag_blocks: Output off-diagonal blocks (36 floats per edge).
    """
    i = wp.tid()
    if i >= n_edges:
        return
    regularization = 1.0e-6
    block_start = 6 * i

    # Use actual inverse masses from the array
    inv_m0 = inv_masses[i]
    inv_m1 = inv_masses[i + 1]

    # Diagonal block
    for row in range(6):
        for col in range(6):
            val = 0.0

            # Position contribution from segment 0 (particle i)
            for k in range(3):
                j_p0_r = jacobian_pos[_warp_jacobian_index(i, row, k)]
                j_p0_c = jacobian_pos[_warp_jacobian_index(i, col, k)]
                val += j_p0_r * inv_m0 * j_p0_c

            # Position contribution from segment 1 (particle i+1)
            for k in range(3):
                j_p1_r = jacobian_pos[_warp_jacobian_index(i, row, k + 3)]
                j_p1_c = jacobian_pos[_warp_jacobian_index(i, col, k + 3)]
                val += j_p1_r * inv_m1 * j_p1_c

            # Rotation contribution from segment 0
            j_t0_r_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, row, 0)],
                jacobian_rot[_warp_jacobian_index(i, row, 1)],
                jacobian_rot[_warp_jacobian_index(i, row, 2)],
            )
            j_t0_c_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col, 0)],
                jacobian_rot[_warp_jacobian_index(i, col, 1)],
                jacobian_rot[_warp_jacobian_index(i, col, 2)],
            )
            inv_I0_j_t0_c = _inv_inertia_mul_vec(inv_inertia, i, j_t0_c_vec)
            val += wp.dot(j_t0_r_vec, inv_I0_j_t0_c)

            # Rotation contribution from segment 1
            j_t1_r_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, row, 3)],
                jacobian_rot[_warp_jacobian_index(i, row, 4)],
                jacobian_rot[_warp_jacobian_index(i, row, 5)],
            )
            j_t1_c_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col, 3)],
                jacobian_rot[_warp_jacobian_index(i, col, 4)],
                jacobian_rot[_warp_jacobian_index(i, col, 5)],
            )
            inv_I1_j_t1_c = _inv_inertia_mul_vec(inv_inertia, i + 1, j_t1_c_vec)
            val += wp.dot(j_t1_r_vec, inv_I1_j_t1_c)

            if row == col:
                val += compliance[i * 6 + row] + regularization
            diag_blocks[_block_index(i, row, col)] = val

    # Off-diagonal block
    if i == 0:
        for row in range(6):
            for col in range(6):
                offdiag_blocks[_block_index(i, row, col)] = 0.0
        return

    prev = i - 1

    for row in range(6):
        for col in range(6):
            val = 0.0

            # Position contribution using shared particle's inverse mass
            for k in range(3):
                j_p1_prev = jacobian_pos[_warp_jacobian_index(prev, row, k + 3)]
                j_p0_cur = jacobian_pos[_warp_jacobian_index(i, col, k)]
                val += j_p1_prev * inv_m0 * j_p0_cur

            # Rotation contribution
            j_t1_prev_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(prev, row, 3)],
                jacobian_rot[_warp_jacobian_index(prev, row, 4)],
                jacobian_rot[_warp_jacobian_index(prev, row, 5)],
            )
            j_t0_cur_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col, 0)],
                jacobian_rot[_warp_jacobian_index(i, col, 1)],
                jacobian_rot[_warp_jacobian_index(i, col, 2)],
            )
            inv_I_shared_j_t0_cur = _inv_inertia_mul_vec(inv_inertia, i, j_t0_cur_vec)
            val += wp.dot(j_t1_prev_vec, inv_I_shared_j_t0_cur)

            offdiag_blocks[_block_index(i, row, col)] = val


@wp.kernel
def _warp_pad_diagonal(
    A: wp.array2d(dtype=wp.float32),
    n_dofs: int,
    tile: int,
):
    """Pad diagonal with 1s for unused DOFs in tiled solver.
    
    Args:
        A: Dense matrix to pad.
        n_dofs: Number of actual DOFs.
        tile: Tile size.
    """
    i = wp.tid()
    if i >= n_dofs and i < tile:
        A[i, i] = 1.0


__all__ = [
    "_warp_assemble_jmjt_dense",
    "_warp_assemble_jmjt_banded",
    "_warp_assemble_jmjt_blocks",
    "_warp_pad_diagonal",
]
