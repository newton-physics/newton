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

from .constants import BAND_KD
from .kernels_math import _block_index, _block_index_3x3, _inv_inertia_mul_vec, _warp_jacobian_index


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
                val += compliance[i * 6 + row] + 1.0e-6
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

            # Add compliance and regularization to diagonal
            if row == col:
                val += compliance[i * 6 + row] + 1.0e-6

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


@wp.kernel
def _warp_assemble_jmjt_blocks_batched(
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    inv_masses: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    rod_offsets: wp.array(dtype=wp.int32),
    edge_offsets: wp.array(dtype=wp.int32),
    edge_rod_id: wp.array(dtype=wp.int32),
    diag_blocks: wp.array(dtype=wp.float32),
    offdiag_blocks: wp.array(dtype=wp.float32),
):
    """Assemble JMJT blocks for all rods in a single launch.

    This batched version processes all edges across all rods, using
    rod_offsets and edge_offsets to map global indices to local ones.

    Args:
        jacobian_pos: Concatenated position Jacobians.
        jacobian_rot: Concatenated rotation Jacobians.
        compliance: Concatenated compliance values.
        inv_masses: Concatenated inverse masses.
        inv_inertia: Concatenated inverse inertia tensors (9 floats per particle).
        rod_offsets: Cumulative point offsets [n_rods + 1].
        edge_offsets: Cumulative edge offsets [n_rods + 1].
        edge_rod_id: Rod index for each edge.
        diag_blocks: Output diagonal blocks (36 floats per edge).
        offdiag_blocks: Output off-diagonal blocks (36 floats per edge).
    """
    global_edge = wp.tid()
    rod_id = edge_rod_id[global_edge]

    # Local edge index within this rod
    local_edge = global_edge - edge_offsets[rod_id]

    # Global particle indices for this edge
    rod_start = rod_offsets[rod_id]
    p0_idx = rod_start + local_edge
    p1_idx = p0_idx + 1

    regularization = 1.0e-6
    i = global_edge

    # Use actual inverse masses from the array
    inv_m0 = inv_masses[p0_idx]
    inv_m1 = inv_masses[p1_idx]

    # Diagonal block
    for row in range(6):
        for col in range(6):
            val = 0.0

            # Position contribution from segment 0 (particle p0_idx)
            for k in range(3):
                j_p0_r = jacobian_pos[_warp_jacobian_index(i, row, k)]
                j_p0_c = jacobian_pos[_warp_jacobian_index(i, col, k)]
                val += j_p0_r * inv_m0 * j_p0_c

            # Position contribution from segment 1 (particle p1_idx)
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
            inv_I0_j_t0_c = _inv_inertia_mul_vec(inv_inertia, p0_idx, j_t0_c_vec)
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
            inv_I1_j_t1_c = _inv_inertia_mul_vec(inv_inertia, p1_idx, j_t1_c_vec)
            val += wp.dot(j_t1_r_vec, inv_I1_j_t1_c)

            if row == col:
                val += compliance[i * 6 + row] + regularization
            diag_blocks[_block_index(i, row, col)] = val

    # Off-diagonal block (coupling with previous edge within same rod)
    if local_edge == 0:
        # First edge in rod has no off-diagonal coupling
        for row in range(6):
            for col in range(6):
                offdiag_blocks[_block_index(i, row, col)] = 0.0
        return

    prev = global_edge - 1

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
            inv_I_shared_j_t0_cur = _inv_inertia_mul_vec(inv_inertia, p0_idx, j_t0_cur_vec)
            val += wp.dot(j_t1_prev_vec, inv_I_shared_j_t0_cur)

            offdiag_blocks[_block_index(i, row, col)] = val


@wp.kernel
def _warp_compute_inv_inertia_world_batched(
    orientations: wp.array(dtype=wp.quat),
    quat_inv_masses: wp.array(dtype=wp.float32),
    inv_inertia_local_diag: wp.array(dtype=wp.vec3),
    particle_rod_id: wp.array(dtype=wp.int32),
    inv_inertia_out: wp.array(dtype=wp.float32),
):
    """Compute inverse inertia tensors for all rods in a single launch.

    This batched version processes all particles across all rods.

    Args:
        orientations: Concatenated quaternion orientations.
        quat_inv_masses: Concatenated inverse rotational mass mask.
        inv_inertia_local_diag: Per-rod local inverse inertia diagonal [n_rods].
        particle_rod_id: Rod index for each particle.
        inv_inertia_out: Output array of 9 floats per particle.
    """
    i = wp.tid()
    base = i * 9

    if quat_inv_masses[i] <= 0.0:
        # Locked particle - zero inverse inertia (infinite inertia)
        for j in range(9):
            inv_inertia_out[base + j] = 0.0
        return

    rod_id = particle_rod_id[i]
    inv_inertia_local = inv_inertia_local_diag[rod_id]

    q = orientations[i]

    # Extract rotation matrix from quaternion
    xx = q.x * q.x
    yy = q.y * q.y
    zz = q.z * q.z
    xy = q.x * q.y
    xz = q.x * q.z
    yz = q.y * q.z
    wx = q.w * q.x
    wy = q.w * q.y
    wz = q.w * q.z

    # Rotation matrix (row-major)
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    # Compute R * diag(inv_I) * R^T
    d0 = inv_inertia_local.x
    d1 = inv_inertia_local.y
    d2 = inv_inertia_local.z

    # Row 0
    inv_inertia_out[base + 0] = r00 * d0 * r00 + r01 * d1 * r01 + r02 * d2 * r02
    inv_inertia_out[base + 1] = r00 * d0 * r10 + r01 * d1 * r11 + r02 * d2 * r12
    inv_inertia_out[base + 2] = r00 * d0 * r20 + r01 * d1 * r21 + r02 * d2 * r22

    # Row 1
    inv_inertia_out[base + 3] = r10 * d0 * r00 + r11 * d1 * r01 + r12 * d2 * r02
    inv_inertia_out[base + 4] = r10 * d0 * r10 + r11 * d1 * r11 + r12 * d2 * r12
    inv_inertia_out[base + 5] = r10 * d0 * r20 + r11 * d1 * r21 + r12 * d2 * r22

    # Row 2
    inv_inertia_out[base + 6] = r20 * d0 * r00 + r21 * d1 * r01 + r22 * d2 * r02
    inv_inertia_out[base + 7] = r20 * d0 * r10 + r21 * d1 * r11 + r22 * d2 * r12
    inv_inertia_out[base + 8] = r20 * d0 * r20 + r21 * d1 * r21 + r22 * d2 * r22


@wp.kernel
def _warp_assemble_stretch_blocks(
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    inv_masses: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    n_edges: int,
    diag_blocks: wp.array(dtype=wp.float32),
    offdiag_blocks: wp.array(dtype=wp.float32),
):
    """Assemble 3x3 block-tridiagonal system for stretch constraints.

    This kernel assembles the JMJT matrix for the stretch (position-based
    inextensibility) constraints only. The stretch constraint depends on
    both positions AND rotations (since c = p + R*offset), so both
    Jacobian contributions are included.

    A_stretch = J_pos * M^{-1} * J_pos^T + J_rot_stretch * I^{-1} * J_rot_stretch^T + C_stretch

    Where J_rot_stretch uses rows 0-2 (stretch rows) of jacobian_rot.

    Args:
        jacobian_pos: Position Jacobians (36 floats per edge).
        jacobian_rot: Rotation Jacobians (36 floats per edge).
        compliance: Compliance values (6 per edge, rows 0-2 used for stretch).
        inv_masses: Inverse masses per particle.
        inv_inertia: Inverse inertia tensors (9 floats per particle).
        n_edges: Number of edges.
        diag_blocks: Output diagonal blocks (9 floats per edge).
        offdiag_blocks: Output off-diagonal blocks (9 floats per edge).
    """
    i = wp.tid()
    if i >= n_edges:
        return

    regularization = 1.0e-6

    # Inverse masses for particles connected by this edge
    inv_m0 = inv_masses[i]
    inv_m1 = inv_masses[i + 1]

    # === Diagonal block ===
    for row in range(3):
        for col in range(3):
            val = float(0.0)

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

            # Rotation contribution from segment 0 (stretch rows 0-2)
            j_r0_r_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, row, 0)],
                jacobian_rot[_warp_jacobian_index(i, row, 1)],
                jacobian_rot[_warp_jacobian_index(i, row, 2)],
            )
            j_r0_c_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col, 0)],
                jacobian_rot[_warp_jacobian_index(i, col, 1)],
                jacobian_rot[_warp_jacobian_index(i, col, 2)],
            )
            inv_I0_j = _inv_inertia_mul_vec(inv_inertia, i, j_r0_c_vec)
            val += wp.dot(j_r0_r_vec, inv_I0_j)

            # Rotation contribution from segment 1 (stretch rows 0-2)
            j_r1_r_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, row, 3)],
                jacobian_rot[_warp_jacobian_index(i, row, 4)],
                jacobian_rot[_warp_jacobian_index(i, row, 5)],
            )
            j_r1_c_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col, 3)],
                jacobian_rot[_warp_jacobian_index(i, col, 4)],
                jacobian_rot[_warp_jacobian_index(i, col, 5)],
            )
            inv_I1_j = _inv_inertia_mul_vec(inv_inertia, i + 1, j_r1_c_vec)
            val += wp.dot(j_r1_r_vec, inv_I1_j)

            # Add compliance and regularization on diagonal
            if row == col:
                val += compliance[i * 6 + row] + regularization

            diag_blocks[_block_index_3x3(i, row, col)] = val

    # === Off-diagonal block ===
    if i == 0:
        # First edge has no off-diagonal coupling
        for row in range(3):
            for col in range(3):
                offdiag_blocks[_block_index_3x3(i, row, col)] = 0.0
        return

    prev = i - 1

    for row in range(3):
        for col in range(3):
            val = float(0.0)

            # Position contribution: prev edge's segment 1 (particle i) to
            # current edge's segment 0 (also particle i)
            for k in range(3):
                j_p1_prev = jacobian_pos[_warp_jacobian_index(prev, row, k + 3)]
                j_p0_cur = jacobian_pos[_warp_jacobian_index(i, col, k)]
                val += j_p1_prev * inv_m0 * j_p0_cur

            # Rotation contribution: prev edge's stretch rotation Jacobian (seg 1)
            # to current edge's stretch rotation Jacobian (seg 0)
            j_r1_prev_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(prev, row, 3)],
                jacobian_rot[_warp_jacobian_index(prev, row, 4)],
                jacobian_rot[_warp_jacobian_index(prev, row, 5)],
            )
            j_r0_cur_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col, 0)],
                jacobian_rot[_warp_jacobian_index(i, col, 1)],
                jacobian_rot[_warp_jacobian_index(i, col, 2)],
            )
            inv_I_shared_j = _inv_inertia_mul_vec(inv_inertia, i, j_r0_cur_vec)
            val += wp.dot(j_r1_prev_vec, inv_I_shared_j)

            offdiag_blocks[_block_index_3x3(i, row, col)] = val


@wp.kernel
def _warp_assemble_darboux_blocks(
    jacobian_rot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    n_edges: int,
    diag_blocks: wp.array(dtype=wp.float32),
    offdiag_blocks: wp.array(dtype=wp.float32),
):
    """Assemble 3x3 block-tridiagonal system for darboux (bend/twist) constraints.

    This kernel assembles the JMJT matrix for the darboux constraints only.
    Darboux constraints depend purely on rotations (quaternion relative twist).

    A_darboux = J_rot_darboux * I^{-1} * J_rot_darboux^T + C_darboux

    Where J_rot_darboux uses rows 3-5 (darboux rows) of jacobian_rot.

    Args:
        jacobian_rot: Rotation Jacobians (36 floats per edge).
        compliance: Compliance values (6 per edge, rows 3-5 used for darboux).
        inv_inertia: Inverse inertia tensors (9 floats per particle).
        n_edges: Number of edges.
        diag_blocks: Output diagonal blocks (9 floats per edge).
        offdiag_blocks: Output off-diagonal blocks (9 floats per edge).
    """
    i = wp.tid()
    if i >= n_edges:
        return

    regularization = 1.0e-6

    # === Diagonal block ===
    for row in range(3):
        for col in range(3):
            val = float(0.0)

            # Rotation contribution from segment 0 (darboux rows 3-5)
            j_r0_r_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, row + 3, 0)],
                jacobian_rot[_warp_jacobian_index(i, row + 3, 1)],
                jacobian_rot[_warp_jacobian_index(i, row + 3, 2)],
            )
            j_r0_c_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col + 3, 0)],
                jacobian_rot[_warp_jacobian_index(i, col + 3, 1)],
                jacobian_rot[_warp_jacobian_index(i, col + 3, 2)],
            )
            inv_I0_j = _inv_inertia_mul_vec(inv_inertia, i, j_r0_c_vec)
            val += wp.dot(j_r0_r_vec, inv_I0_j)

            # Rotation contribution from segment 1 (darboux rows 3-5)
            j_r1_r_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, row + 3, 3)],
                jacobian_rot[_warp_jacobian_index(i, row + 3, 4)],
                jacobian_rot[_warp_jacobian_index(i, row + 3, 5)],
            )
            j_r1_c_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col + 3, 3)],
                jacobian_rot[_warp_jacobian_index(i, col + 3, 4)],
                jacobian_rot[_warp_jacobian_index(i, col + 3, 5)],
            )
            inv_I1_j = _inv_inertia_mul_vec(inv_inertia, i + 1, j_r1_c_vec)
            val += wp.dot(j_r1_r_vec, inv_I1_j)

            # Add compliance and regularization on diagonal
            if row == col:
                val += compliance[i * 6 + row + 3] + regularization

            diag_blocks[_block_index_3x3(i, row, col)] = val

    # === Off-diagonal block ===
    if i == 0:
        # First edge has no off-diagonal coupling
        for row in range(3):
            for col in range(3):
                offdiag_blocks[_block_index_3x3(i, row, col)] = 0.0
        return

    prev = i - 1

    for row in range(3):
        for col in range(3):
            val = float(0.0)

            # Rotation contribution: prev edge's darboux rotation Jacobian (seg 1)
            # to current edge's darboux rotation Jacobian (seg 0)
            j_r1_prev_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(prev, row + 3, 3)],
                jacobian_rot[_warp_jacobian_index(prev, row + 3, 4)],
                jacobian_rot[_warp_jacobian_index(prev, row + 3, 5)],
            )
            j_r0_cur_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col + 3, 0)],
                jacobian_rot[_warp_jacobian_index(i, col + 3, 1)],
                jacobian_rot[_warp_jacobian_index(i, col + 3, 2)],
            )
            # Shared particle is particle i
            inv_I_shared_j = _inv_inertia_mul_vec(inv_inertia, i, j_r0_cur_vec)
            val += wp.dot(j_r1_prev_vec, inv_I_shared_j)

            offdiag_blocks[_block_index_3x3(i, row, col)] = val


__all__ = [
    "_warp_assemble_darboux_blocks",
    "_warp_assemble_jmjt_banded",
    "_warp_assemble_jmjt_blocks",
    "_warp_assemble_jmjt_blocks_batched",
    "_warp_assemble_jmjt_dense",
    "_warp_assemble_stretch_blocks",
    "_warp_compute_inv_inertia_world_batched",
    "_warp_pad_diagonal",
]
