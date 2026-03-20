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

"""Warp math functions for Cosserat rod simulation.

Quaternion operations, matrix operations, and block-tridiagonal helpers
used by the direct solver.
"""

from __future__ import annotations

import warp as wp

# ==============================================================================
# Quaternion operations
# ==============================================================================


@wp.func
def _warp_quat_mul(q1: wp.quat, q2: wp.quat) -> wp.quat:
    """Multiply two quaternions."""
    return wp.quat(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
    )


@wp.func
def _warp_quat_conjugate(q: wp.quat) -> wp.quat:
    """Compute the conjugate of a quaternion."""
    return wp.quat(-q.x, -q.y, -q.z, q.w)


@wp.func
def _warp_quat_normalize(q: wp.quat) -> wp.quat:
    """Normalize a quaternion to unit length."""
    norm = wp.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm < 1.0e-8:
        return q
    inv = 1.0 / norm
    return wp.quat(q.x * inv, q.y * inv, q.z * inv, q.w * inv)


@wp.func
def _warp_quat_rotate_vector(q: wp.quat, v: wp.vec3) -> wp.vec3:
    """Rotate a vector by a quaternion."""
    tx = 2.0 * (q.y * v.z - q.z * v.y)
    ty = 2.0 * (q.z * v.x - q.x * v.z)
    tz = 2.0 * (q.x * v.y - q.y * v.x)
    return wp.vec3(
        v.x + q.w * tx + q.y * tz - q.z * ty,
        v.y + q.w * ty + q.z * tx - q.x * tz,
        v.z + q.w * tz + q.x * ty - q.y * tx,
    )


# ==============================================================================
# Jacobian indexing
# ==============================================================================


@wp.func
def _warp_jacobian_index(edge: int, row: int, col: int) -> int:
    """Compute flat index for Jacobian storage (6x6 per edge)."""
    return edge * 36 + row * 6 + col


# ==============================================================================
# 3x3 Matrix operations
# ==============================================================================


@wp.func
def _mat33_add(a: wp.mat33, b: wp.mat33) -> wp.mat33:
    """Add two 3x3 matrices."""
    return wp.mat33(
        a[0, 0] + b[0, 0],
        a[0, 1] + b[0, 1],
        a[0, 2] + b[0, 2],
        a[1, 0] + b[1, 0],
        a[1, 1] + b[1, 1],
        a[1, 2] + b[1, 2],
        a[2, 0] + b[2, 0],
        a[2, 1] + b[2, 1],
        a[2, 2] + b[2, 2],
    )


@wp.func
def _mat33_sub(a: wp.mat33, b: wp.mat33) -> wp.mat33:
    """Subtract two 3x3 matrices."""
    return wp.mat33(
        a[0, 0] - b[0, 0],
        a[0, 1] - b[0, 1],
        a[0, 2] - b[0, 2],
        a[1, 0] - b[1, 0],
        a[1, 1] - b[1, 1],
        a[1, 2] - b[1, 2],
        a[2, 0] - b[2, 0],
        a[2, 1] - b[2, 1],
        a[2, 2] - b[2, 2],
    )


@wp.func
def _mat33_mul(a: wp.mat33, b: wp.mat33) -> wp.mat33:
    """Multiply two 3x3 matrices."""
    return wp.mat33(
        a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] + a[0, 2] * b[2, 0],
        a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1] + a[0, 2] * b[2, 1],
        a[0, 0] * b[0, 2] + a[0, 1] * b[1, 2] + a[0, 2] * b[2, 2],
        a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0] + a[1, 2] * b[2, 0],
        a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1] + a[1, 2] * b[2, 1],
        a[1, 0] * b[0, 2] + a[1, 1] * b[1, 2] + a[1, 2] * b[2, 2],
        a[2, 0] * b[0, 0] + a[2, 1] * b[1, 0] + a[2, 2] * b[2, 0],
        a[2, 0] * b[0, 1] + a[2, 1] * b[1, 1] + a[2, 2] * b[2, 1],
        a[2, 0] * b[0, 2] + a[2, 1] * b[1, 2] + a[2, 2] * b[2, 2],
    )


@wp.func
def _mat33_mul_vec3(a: wp.mat33, v: wp.vec3) -> wp.vec3:
    """Multiply a 3x3 matrix by a 3D vector."""
    return wp.vec3(
        a[0, 0] * v[0] + a[0, 1] * v[1] + a[0, 2] * v[2],
        a[1, 0] * v[0] + a[1, 1] * v[1] + a[1, 2] * v[2],
        a[2, 0] * v[0] + a[2, 1] * v[1] + a[2, 2] * v[2],
    )


@wp.func
def _mat33_transpose(a: wp.mat33) -> wp.mat33:
    """Transpose a 3x3 matrix."""
    return wp.mat33(
        a[0, 0],
        a[1, 0],
        a[2, 0],
        a[0, 1],
        a[1, 1],
        a[2, 1],
        a[0, 2],
        a[1, 2],
        a[2, 2],
    )


@wp.func
def _mat33_cholesky(a: wp.mat33) -> wp.mat33:
    """Compute the Cholesky decomposition of a 3x3 SPD matrix."""
    eps = 1.0e-9
    l00 = wp.sqrt(wp.max(a[0, 0], eps))
    l10 = a[1, 0] / l00
    l20 = a[2, 0] / l00
    l11 = wp.sqrt(wp.max(a[1, 1] - l10 * l10, eps))
    l21 = (a[2, 1] - l20 * l10) / l11
    l22 = wp.sqrt(wp.max(a[2, 2] - l20 * l20 - l21 * l21, eps))
    return wp.mat33(l00, 0.0, 0.0, l10, l11, 0.0, l20, l21, l22)


@wp.func
def _mat33_solve_lower(L: wp.mat33, b: wp.vec3) -> wp.vec3:
    """Solve L * x = b where L is lower triangular."""
    y0 = b[0] / L[0, 0]
    y1 = (b[1] - L[1, 0] * y0) / L[1, 1]
    y2 = (b[2] - L[2, 0] * y0 - L[2, 1] * y1) / L[2, 2]
    return wp.vec3(y0, y1, y2)


@wp.func
def _mat33_solve_upper(L: wp.mat33, b: wp.vec3) -> wp.vec3:
    """Solve L^T * x = b where L is lower triangular."""
    x2 = b[2] / L[2, 2]
    x1 = (b[1] - L[2, 1] * x2) / L[1, 1]
    x0 = (b[0] - L[1, 0] * x1 - L[2, 0] * x2) / L[0, 0]
    return wp.vec3(x0, x1, x2)


@wp.func
def _mat33_cholesky_solve(L: wp.mat33, b: wp.vec3) -> wp.vec3:
    """Solve L * L^T * x = b given Cholesky factor L."""
    y = _mat33_solve_lower(L, b)
    return _mat33_solve_upper(L, y)


# ==============================================================================
# 3x3 Block storage operations (for split Thomas solver)
# ==============================================================================


@wp.func
def _block_index_3x3(block: int, row: int, col: int) -> int:
    """Compute flat index for 3x3 block storage."""
    return block * 9 + row * 3 + col


@wp.func
def _load_block_3x3(blocks: wp.array(dtype=wp.float32), block: int) -> wp.mat33:
    """Load a 3x3 block from flat storage."""
    base = block * 9
    return wp.mat33(
        blocks[base + 0],
        blocks[base + 1],
        blocks[base + 2],
        blocks[base + 3],
        blocks[base + 4],
        blocks[base + 5],
        blocks[base + 6],
        blocks[base + 7],
        blocks[base + 8],
    )


@wp.func
def _load_vec3_block(values: wp.array(dtype=wp.float32), block: int) -> wp.vec3:
    """Load a 3-vector from block storage."""
    base = block * 3
    return wp.vec3(values[base], values[base + 1], values[base + 2])


@wp.func
def _store_vec3_block(values: wp.array(dtype=wp.float32), block: int, v: wp.vec3):
    """Store a 3-vector to block storage."""
    base = block * 3
    values[base] = v[0]
    values[base + 1] = v[1]
    values[base + 2] = v[2]


@wp.func
def _block_column_3x3(blocks: wp.array(dtype=wp.float32), block: int, col: int) -> wp.vec3:
    """Load a column from a 3x3 block as a 3-vector."""
    base = block * 9
    return wp.vec3(
        blocks[base + 0 * 3 + col],
        blocks[base + 1 * 3 + col],
        blocks[base + 2 * 3 + col],
    )


@wp.func
def _block_set_column_3x3(blocks: wp.array(dtype=wp.float32), block: int, col: int, v: wp.vec3):
    """Store a 3-vector as a column in a 3x3 block."""
    base = block * 9
    blocks[base + 0 * 3 + col] = v[0]
    blocks[base + 1 * 3 + col] = v[1]
    blocks[base + 2 * 3 + col] = v[2]


# ==============================================================================
# Inverse inertia operations
# ==============================================================================


@wp.func
def _inv_inertia_mul_vec(inv_inertia: wp.array(dtype=wp.float32), particle_idx: int, v: wp.vec3) -> wp.vec3:
    """Multiply inverse inertia tensor (3x3) by a vector."""
    base = particle_idx * 9
    return wp.vec3(
        inv_inertia[base + 0] * v.x + inv_inertia[base + 1] * v.y + inv_inertia[base + 2] * v.z,
        inv_inertia[base + 3] * v.x + inv_inertia[base + 4] * v.y + inv_inertia[base + 5] * v.z,
        inv_inertia[base + 6] * v.x + inv_inertia[base + 7] * v.y + inv_inertia[base + 8] * v.z,
    )


# ==============================================================================
# Block-tridiagonal indexing and operations (6x6 blocks)
# ==============================================================================


@wp.func
def _block_index(block: int, row: int, col: int) -> int:
    """Compute flat index for 6x6 block storage."""
    return block * 36 + row * 6 + col


@wp.func
def _load_block(blocks: wp.array(dtype=wp.float32), block: int) -> tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
    """Load a 6x6 block as four 3x3 matrices."""
    base = block * 36
    A = wp.mat33(
        blocks[base + 0],
        blocks[base + 1],
        blocks[base + 2],
        blocks[base + 6],
        blocks[base + 7],
        blocks[base + 8],
        blocks[base + 12],
        blocks[base + 13],
        blocks[base + 14],
    )
    B = wp.mat33(
        blocks[base + 3],
        blocks[base + 4],
        blocks[base + 5],
        blocks[base + 9],
        blocks[base + 10],
        blocks[base + 11],
        blocks[base + 15],
        blocks[base + 16],
        blocks[base + 17],
    )
    C = wp.mat33(
        blocks[base + 18],
        blocks[base + 19],
        blocks[base + 20],
        blocks[base + 24],
        blocks[base + 25],
        blocks[base + 26],
        blocks[base + 30],
        blocks[base + 31],
        blocks[base + 32],
    )
    D = wp.mat33(
        blocks[base + 21],
        blocks[base + 22],
        blocks[base + 23],
        blocks[base + 27],
        blocks[base + 28],
        blocks[base + 29],
        blocks[base + 33],
        blocks[base + 34],
        blocks[base + 35],
    )
    return A, B, C, D


@wp.func
def _load_block_offset(
    blocks: wp.array(dtype=wp.float32), edge_offset: int, local_block: int
) -> tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
    """Load a 6x6 block with global offset for batched operations."""
    return _load_block(blocks, edge_offset + local_block)


@wp.func
def _load_vec(values: wp.array(dtype=wp.float32), block: int) -> tuple[wp.vec3, wp.vec3]:
    """Load a 6-vector as two 3-vectors."""
    base = block * 6
    v0 = wp.vec3(values[base + 0], values[base + 1], values[base + 2])
    v1 = wp.vec3(values[base + 3], values[base + 4], values[base + 5])
    return v0, v1


@wp.func
def _load_vec_offset(values: wp.array(dtype=wp.float32), edge_offset: int, local_block: int) -> tuple[wp.vec3, wp.vec3]:
    """Load a 6-vector with global offset for batched operations."""
    return _load_vec(values, edge_offset + local_block)


@wp.func
def _store_vec(values: wp.array(dtype=wp.float32), block: int, v0: wp.vec3, v1: wp.vec3):
    """Store two 3-vectors as a 6-vector."""
    base = block * 6
    values[base + 0] = v0[0]
    values[base + 1] = v0[1]
    values[base + 2] = v0[2]
    values[base + 3] = v1[0]
    values[base + 4] = v1[1]
    values[base + 5] = v1[2]


@wp.func
def _store_vec_offset(
    values: wp.array(dtype=wp.float32),
    edge_offset: int,
    local_block: int,
    v0: wp.vec3,
    v1: wp.vec3,
):
    """Store two 3-vectors as a 6-vector with global offset."""
    _store_vec(values, edge_offset + local_block, v0, v1)


@wp.func
def _block_column(blocks: wp.array(dtype=wp.float32), block: int, col: int) -> tuple[wp.vec3, wp.vec3]:
    """Load a column from a 6x6 block as two 3-vectors."""
    base = block * 36
    v0 = wp.vec3(blocks[base + 0 * 6 + col], blocks[base + 1 * 6 + col], blocks[base + 2 * 6 + col])
    v1 = wp.vec3(blocks[base + 3 * 6 + col], blocks[base + 4 * 6 + col], blocks[base + 5 * 6 + col])
    return v0, v1


@wp.func
def _block_column_offset(
    blocks: wp.array(dtype=wp.float32), edge_offset: int, local_block: int, col: int
) -> tuple[wp.vec3, wp.vec3]:
    """Load a column from a 6x6 block with global offset."""
    return _block_column(blocks, edge_offset + local_block, col)


@wp.func
def _block_set_column(blocks: wp.array(dtype=wp.float32), block: int, col: int, v0: wp.vec3, v1: wp.vec3):
    """Store two 3-vectors as a column in a 6x6 block."""
    base = block * 36
    blocks[base + 0 * 6 + col] = v0[0]
    blocks[base + 1 * 6 + col] = v0[1]
    blocks[base + 2 * 6 + col] = v0[2]
    blocks[base + 3 * 6 + col] = v1[0]
    blocks[base + 4 * 6 + col] = v1[1]
    blocks[base + 5 * 6 + col] = v1[2]


@wp.func
def _block_set_column_offset(
    blocks: wp.array(dtype=wp.float32),
    edge_offset: int,
    local_block: int,
    col: int,
    v0: wp.vec3,
    v1: wp.vec3,
):
    """Store two 3-vectors as a column in a 6x6 block with global offset."""
    _block_set_column(blocks, edge_offset + local_block, col, v0, v1)


@wp.func
def _block_mul(
    A: wp.mat33,
    B: wp.mat33,
    C: wp.mat33,
    D: wp.mat33,
    E: wp.mat33,
    F: wp.mat33,
    G: wp.mat33,
    H: wp.mat33,
) -> tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
    """Multiply two 2x2 block matrices (each block is 3x3)."""
    return (
        _mat33_add(_mat33_mul(A, E), _mat33_mul(B, G)),
        _mat33_add(_mat33_mul(A, F), _mat33_mul(B, H)),
        _mat33_add(_mat33_mul(C, E), _mat33_mul(D, G)),
        _mat33_add(_mat33_mul(C, F), _mat33_mul(D, H)),
    )


@wp.func
def _block_sub(
    A: wp.mat33,
    B: wp.mat33,
    C: wp.mat33,
    D: wp.mat33,
    E: wp.mat33,
    F: wp.mat33,
    G: wp.mat33,
    H: wp.mat33,
) -> tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
    """Subtract two 2x2 block matrices (each block is 3x3)."""
    return (_mat33_sub(A, E), _mat33_sub(B, F), _mat33_sub(C, G), _mat33_sub(D, H))


@wp.func
def _block_mul_vec(
    A: wp.mat33,
    B: wp.mat33,
    C: wp.mat33,
    D: wp.mat33,
    v0: wp.vec3,
    v1: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
    """Multiply a 2x2 block matrix by a 2-block vector."""
    top = _mat33_mul_vec3(A, v0) + _mat33_mul_vec3(B, v1)
    bot = _mat33_mul_vec3(C, v0) + _mat33_mul_vec3(D, v1)
    return top, bot


@wp.func
def _block_solve(
    A: wp.mat33,
    B: wp.mat33,
    C: wp.mat33,
    D: wp.mat33,
    b0: wp.vec3,
    b1: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
    """Solve a 6x6 SPD block system ``[[A, B], [C, D]]`` via Cholesky.

    ``B`` is accepted for API symmetry with :func:`_load_block` but unused
    because the solver assumes SPD structure where ``B = C^T``.
    """
    L11 = _mat33_cholesky(A)
    c0 = wp.vec3(C[0, 0], C[0, 1], C[0, 2])
    c1 = wp.vec3(C[1, 0], C[1, 1], C[1, 2])
    c2 = wp.vec3(C[2, 0], C[2, 1], C[2, 2])
    y0 = _mat33_solve_lower(L11, c0)
    y1 = _mat33_solve_lower(L11, c1)
    y2 = _mat33_solve_lower(L11, c2)
    L21 = wp.mat33(y0[0], y0[1], y0[2], y1[0], y1[1], y1[2], y2[0], y2[1], y2[2])
    L21_t = _mat33_transpose(L21)
    S = _mat33_sub(D, _mat33_mul(L21, L21_t))
    L22 = _mat33_cholesky(S)
    yb0 = _mat33_solve_lower(L11, b0)
    tmp = b1 - _mat33_mul_vec3(L21, yb0)
    yb1 = _mat33_solve_lower(L22, tmp)
    x1 = _mat33_solve_upper(L22, yb1)
    x0 = _mat33_solve_upper(L11, yb0 - _mat33_mul_vec3(L21_t, x1))
    return x0, x1
