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

"""Direct Cosserat rod: native reference + NumPy/Warp scaffold.

This example runs two direct rods side-by-side:
- Reference rod: full native DefKitAdv.dll pipeline.
- Candidate rod: hybrid pipeline where steps can be replaced by NumPy or Warp.

Command:
    uv run python newton/examples/cosserat_codex/warp_cosserat_codex.py --dll-path "C:\\path\\to\\DefKitAdv.dll"
"""

from __future__ import annotations

import atexit
import ctypes
import math
import os

import numpy as np
import warp as wp

import newton
import newton.examples

# Warp tile configuration for direct solve
BLOCK_DIM = 128
TILE = 64

# Banded Cholesky layout (matches spbsv_u11_1rhs in C++)
BAND_KD = 11
BAND_LDAB = 34

DIRECT_SOLVE_WARP_BLOCK_THOMAS = "warp_block_thomas"
DIRECT_SOLVE_WARP_BANDED_CHOLESKY = "warp_banded_cholesky"
DIRECT_SOLVE_CPU_NUMPY = "cpu_numpy"
DIRECT_SOLVE_BACKENDS = (
    DIRECT_SOLVE_WARP_BLOCK_THOMAS,
    DIRECT_SOLVE_WARP_BANDED_CHOLESKY,
    DIRECT_SOLVE_CPU_NUMPY,
)


class BtVector3(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("w", ctypes.c_float),
    ]


class BtQuaternion(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("w", ctypes.c_float),
    ]


def _as_ptr(array: np.ndarray, ctype):
    if array.dtype != np.float32:
        raise TypeError(f"Expected float32 array, got {array.dtype}")
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError("Expected C-contiguous array")
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def _as_float_ptr(array: np.ndarray):
    if array.dtype != np.float32:
        raise TypeError(f"Expected float32 array, got {array.dtype}")
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError("Expected C-contiguous array")
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    norm = np.linalg.norm(axis)
    if norm < 1.0e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = axis / norm
    half = angle * 0.5
    s = math.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)], dtype=np.float32)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


@wp.func
def _warp_quat_mul(q1: wp.quat, q2: wp.quat) -> wp.quat:
    return wp.quat(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
    )


@wp.func
def _warp_quat_conjugate(q: wp.quat) -> wp.quat:
    return wp.quat(-q.x, -q.y, -q.z, q.w)


@wp.func
def _warp_quat_normalize(q: wp.quat) -> wp.quat:
    norm = wp.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm < 1.0e-8:
        return q
    inv = 1.0 / norm
    return wp.quat(q.x * inv, q.y * inv, q.z * inv, q.w * inv)


@wp.kernel
def _warp_predict_positions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    gravity: wp.vec3,
    dt: float,
    damping: float,
    predicted: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    inv_mass = inv_masses[i]
    if inv_mass > 0.0:
        v = velocities[i] + (forces[i] * inv_mass + gravity) * dt
        v = v * (1.0 - damping)
        velocities[i] = v
        predicted[i] = positions[i] + v * dt
    else:
        velocities[i] = wp.vec3(0.0, 0.0, 0.0)
        predicted[i] = positions[i]


@wp.kernel
def _warp_integrate_positions(
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    dt: float,
):
    i = wp.tid()
    inv_mass = inv_masses[i]
    if inv_mass > 0.0:
        v = (predicted[i] - positions[i]) * (1.0 / dt)
        velocities[i] = v
        positions[i] = predicted[i]


@wp.kernel
def _warp_predict_rotations(
    orientations: wp.array(dtype=wp.quat),
    angular_velocities: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    quat_inv_masses: wp.array(dtype=wp.float32),
    dt: float,
    damping: float,
    predicted: wp.array(dtype=wp.quat),
):
    i = wp.tid()
    inv_mass = quat_inv_masses[i]
    if inv_mass > 0.0:
        half_dt = 0.5 * dt
        w = angular_velocities[i] + torques[i] * inv_mass * dt
        w = w * (1.0 - damping)
        angular_velocities[i] = w
        q = orientations[i]
        omega_q = wp.quat(w.x, w.y, w.z, 0.0)
        qdot = _warp_quat_mul(omega_q, q)
        q_pred = wp.quat(
            q.x + qdot.x * half_dt,
            q.y + qdot.y * half_dt,
            q.z + qdot.z * half_dt,
            q.w + qdot.w * half_dt,
        )
        predicted[i] = _warp_quat_normalize(q_pred)
    else:
        angular_velocities[i] = wp.vec3(0.0, 0.0, 0.0)
        predicted[i] = orientations[i]


@wp.kernel
def _warp_integrate_rotations(
    orientations: wp.array(dtype=wp.quat),
    predicted: wp.array(dtype=wp.quat),
    prev_orientations: wp.array(dtype=wp.quat),
    angular_velocities: wp.array(dtype=wp.vec3),
    quat_inv_masses: wp.array(dtype=wp.float32),
    dt: float,
):
    i = wp.tid()
    if quat_inv_masses[i] > 0.0:
        q = orientations[i]
        rel = _warp_quat_mul(predicted[i], _warp_quat_conjugate(q))
        angular_velocities[i] = wp.vec3(rel.x, rel.y, rel.z) * (2.0 / dt)
        prev_orientations[i] = q
        orientations[i] = predicted[i]


@wp.func
def _warp_quat_rotate_vector(q: wp.quat, v: wp.vec3) -> wp.vec3:
    tx = 2.0 * (q.y * v.z - q.z * v.y)
    ty = 2.0 * (q.z * v.x - q.x * v.z)
    tz = 2.0 * (q.x * v.y - q.y * v.x)
    return wp.vec3(
        v.x + q.w * tx + q.y * tz - q.z * ty,
        v.y + q.w * ty + q.z * tx - q.x * tz,
        v.z + q.w * tz + q.x * ty - q.y * tx,
    )


@wp.func
def _warp_jacobian_index(edge: int, row: int, col: int) -> int:
    return edge * 36 + row * 6 + col


@wp.kernel
def _warp_prepare_compliance(
    rest_lengths: wp.array(dtype=wp.float32),
    bend_stiffness: wp.array(dtype=wp.vec3),
    young_modulus: float,
    torsion_modulus: float,
    dt: float,
    compliance: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    L = rest_lengths[i]
    dt2 = dt * dt
    eps = 1.0e-10

    k_bend1_ref = young_modulus * bend_stiffness[i].x
    k_bend2_ref = young_modulus * bend_stiffness[i].y
    k_twist_ref = torsion_modulus * bend_stiffness[i].z

    k_bend1_eff = k_bend1_ref * L
    k_bend2_eff = k_bend2_ref * L
    k_twist_eff = k_twist_ref * L

    stretch_compliance = 1.0e-10

    base = i * 6
    compliance[base + 0] = stretch_compliance
    compliance[base + 1] = stretch_compliance
    compliance[base + 2] = stretch_compliance
    compliance[base + 3] = 1.0 / (k_bend1_eff * dt2 + eps)
    compliance[base + 4] = 1.0 / (k_bend2_eff * dt2 + eps)
    compliance[base + 5] = 1.0 / (k_twist_eff * dt2 + eps)


@wp.kernel
def _warp_update_constraints_direct(
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    rest_lengths: wp.array(dtype=wp.float32),
    rest_darboux: wp.array(dtype=wp.vec3),
    constraint_values: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    p0 = positions[i]
    p1 = positions[i + 1]
    q0 = orientations[i]
    q1 = orientations[i + 1]

    half_L = 0.5 * rest_lengths[i]
    r0_world = _warp_quat_rotate_vector(q0, wp.vec3(0.0, 0.0, half_L))
    r1_world = _warp_quat_rotate_vector(q1, wp.vec3(0.0, 0.0, -half_L))

    c0 = p0 + r0_world
    c1 = p1 + r1_world
    stretch_error = c0 - c1

    q_rel = _warp_quat_mul(_warp_quat_conjugate(q0), q1)
    omega = wp.vec3(q_rel.x, q_rel.y, q_rel.z)
    darboux_error = omega - rest_darboux[i]

    base = i * 6
    constraint_values[base + 0] = stretch_error.x
    constraint_values[base + 1] = stretch_error.y
    constraint_values[base + 2] = stretch_error.z
    constraint_values[base + 3] = darboux_error.x
    constraint_values[base + 4] = darboux_error.y
    constraint_values[base + 5] = darboux_error.z


@wp.kernel
def _warp_compute_jacobians_direct(
    orientations: wp.array(dtype=wp.quat),
    rest_lengths: wp.array(dtype=wp.float32),
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    q0 = orientations[i]
    q1 = orientations[i + 1]

    half_L = 0.5 * rest_lengths[i]
    r0 = _warp_quat_rotate_vector(q0, wp.vec3(0.0, 0.0, half_L))
    r1 = _warp_quat_rotate_vector(q1, wp.vec3(0.0, 0.0, -half_L))

    # Identity for J_pos
    for d in range(3):
        jacobian_pos[_warp_jacobian_index(i, d, d)] = 1.0
        jacobian_pos[_warp_jacobian_index(i, d, d + 3)] = -1.0

    # -skew(r0) and skew(r1) for stretch rotation Jacobians
    jacobian_rot[_warp_jacobian_index(i, 0, 0)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 0, 1)] = r0.z
    jacobian_rot[_warp_jacobian_index(i, 0, 2)] = -r0.y

    jacobian_rot[_warp_jacobian_index(i, 1, 0)] = -r0.z
    jacobian_rot[_warp_jacobian_index(i, 1, 1)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 1, 2)] = r0.x

    jacobian_rot[_warp_jacobian_index(i, 2, 0)] = r0.y
    jacobian_rot[_warp_jacobian_index(i, 2, 1)] = -r0.x
    jacobian_rot[_warp_jacobian_index(i, 2, 2)] = 0.0

    jacobian_rot[_warp_jacobian_index(i, 0, 3)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 0, 4)] = -r1.z
    jacobian_rot[_warp_jacobian_index(i, 0, 5)] = r1.y

    jacobian_rot[_warp_jacobian_index(i, 1, 3)] = r1.z
    jacobian_rot[_warp_jacobian_index(i, 1, 4)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 1, 5)] = -r1.x

    jacobian_rot[_warp_jacobian_index(i, 2, 3)] = -r1.y
    jacobian_rot[_warp_jacobian_index(i, 2, 4)] = r1.x
    jacobian_rot[_warp_jacobian_index(i, 2, 5)] = 0.0

    x0 = q0.x
    y0 = q0.y
    z0 = q0.z
    w0 = q0.w
    x1 = q1.x
    y1 = q1.y
    z1 = q1.z
    w1 = q1.w

    j0_r0 = wp.vec4(-w1, -z1, y1, x1)
    j0_r1 = wp.vec4(z1, -w1, -x1, y1)
    j0_r2 = wp.vec4(-y1, x1, -w1, z1)

    g0_c0 = wp.vec4(0.5 * w0, -0.5 * z0, 0.5 * y0, -0.5 * x0)
    g0_c1 = wp.vec4(0.5 * z0, 0.5 * w0, -0.5 * x0, -0.5 * y0)
    g0_c2 = wp.vec4(-0.5 * y0, 0.5 * x0, 0.5 * w0, -0.5 * z0)

    j1_r0 = wp.vec4(w0, z0, -y0, -x0)
    j1_r1 = wp.vec4(-z0, w0, x0, -y0)
    j1_r2 = wp.vec4(y0, -x0, w0, -z0)

    g1_c0 = wp.vec4(0.5 * w1, -0.5 * z1, 0.5 * y1, -0.5 * x1)
    g1_c1 = wp.vec4(0.5 * z1, 0.5 * w1, -0.5 * x1, -0.5 * y1)
    g1_c2 = wp.vec4(-0.5 * y1, 0.5 * x1, 0.5 * w1, -0.5 * z1)

    jacobian_rot[_warp_jacobian_index(i, 3, 0)] = wp.dot(j0_r0, g0_c0)
    jacobian_rot[_warp_jacobian_index(i, 3, 1)] = wp.dot(j0_r0, g0_c1)
    jacobian_rot[_warp_jacobian_index(i, 3, 2)] = wp.dot(j0_r0, g0_c2)

    jacobian_rot[_warp_jacobian_index(i, 4, 0)] = wp.dot(j0_r1, g0_c0)
    jacobian_rot[_warp_jacobian_index(i, 4, 1)] = wp.dot(j0_r1, g0_c1)
    jacobian_rot[_warp_jacobian_index(i, 4, 2)] = wp.dot(j0_r1, g0_c2)

    jacobian_rot[_warp_jacobian_index(i, 5, 0)] = wp.dot(j0_r2, g0_c0)
    jacobian_rot[_warp_jacobian_index(i, 5, 1)] = wp.dot(j0_r2, g0_c1)
    jacobian_rot[_warp_jacobian_index(i, 5, 2)] = wp.dot(j0_r2, g0_c2)

    jacobian_rot[_warp_jacobian_index(i, 3, 3)] = wp.dot(j1_r0, g1_c0)
    jacobian_rot[_warp_jacobian_index(i, 3, 4)] = wp.dot(j1_r0, g1_c1)
    jacobian_rot[_warp_jacobian_index(i, 3, 5)] = wp.dot(j1_r0, g1_c2)

    jacobian_rot[_warp_jacobian_index(i, 4, 3)] = wp.dot(j1_r1, g1_c0)
    jacobian_rot[_warp_jacobian_index(i, 4, 4)] = wp.dot(j1_r1, g1_c1)
    jacobian_rot[_warp_jacobian_index(i, 4, 5)] = wp.dot(j1_r1, g1_c2)

    jacobian_rot[_warp_jacobian_index(i, 5, 3)] = wp.dot(j1_r2, g1_c0)
    jacobian_rot[_warp_jacobian_index(i, 5, 4)] = wp.dot(j1_r2, g1_c1)
    jacobian_rot[_warp_jacobian_index(i, 5, 5)] = wp.dot(j1_r2, g1_c2)


@wp.kernel
def _warp_assemble_jmjt_dense(
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    n_dofs: int,
    A: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    block_start = 6 * i

    for row in range(6):
        for col in range(6):
            val = 0.0
            for k in range(3):
                j_p0_r = jacobian_pos[_warp_jacobian_index(i, row, k)]
                j_p0_c = jacobian_pos[_warp_jacobian_index(i, col, k)]
                j_p1_r = jacobian_pos[_warp_jacobian_index(i, row, k + 3)]
                j_p1_c = jacobian_pos[_warp_jacobian_index(i, col, k + 3)]
                j_t0_r = jacobian_rot[_warp_jacobian_index(i, row, k)]
                j_t0_c = jacobian_rot[_warp_jacobian_index(i, col, k)]
                j_t1_r = jacobian_rot[_warp_jacobian_index(i, row, k + 3)]
                j_t1_c = jacobian_rot[_warp_jacobian_index(i, col, k + 3)]
                val += (
                    j_p0_r * j_p0_c
                    + j_p1_r * j_p1_c
                    + j_t0_r * j_t0_c
                    + j_t1_r * j_t1_c
                )
            if row == col:
                val += compliance[i * 6 + row]
            row_idx = block_start + row
            col_idx = block_start + col
            if row_idx < n_dofs and col_idx < n_dofs:
                A[row_idx, col_idx] = val

    if i > 0:
        prev = i - 1
        prev_block = 6 * prev
        for row in range(6):
            for col in range(6):
                val = 0.0
                for k in range(3):
                    j_p1_prev = jacobian_pos[_warp_jacobian_index(prev, row, k + 3)]
                    j_p0_cur = jacobian_pos[_warp_jacobian_index(i, col, k)]
                    j_t1_prev = jacobian_rot[_warp_jacobian_index(prev, row, k + 3)]
                    j_t0_cur = jacobian_rot[_warp_jacobian_index(i, col, k)]
                    val += j_p1_prev * j_p0_cur + j_t1_prev * j_t0_cur

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
    n_dofs: int,
    ab: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    block_start = 6 * i
    if block_start >= n_dofs:
        return

    for row in range(6):
        for col in range(6):
            val = 0.0
            for k in range(3):
                j_p0_r = jacobian_pos[_warp_jacobian_index(i, row, k)]
                j_p0_c = jacobian_pos[_warp_jacobian_index(i, col, k)]
                j_p1_r = jacobian_pos[_warp_jacobian_index(i, row, k + 3)]
                j_p1_c = jacobian_pos[_warp_jacobian_index(i, col, k + 3)]
                j_t0_r = jacobian_rot[_warp_jacobian_index(i, row, k)]
                j_t0_c = jacobian_rot[_warp_jacobian_index(i, col, k)]
                j_t1_r = jacobian_rot[_warp_jacobian_index(i, row, k + 3)]
                j_t1_c = jacobian_rot[_warp_jacobian_index(i, col, k + 3)]
                val += (
                    j_p0_r * j_p0_c
                    + j_p1_r * j_p1_c
                    + j_t0_r * j_t0_c
                    + j_t1_r * j_t1_c
                )
            if row == col:
                val += compliance[i * 6 + row]

            row_idx = block_start + row
            col_idx = block_start + col
            if row_idx <= col_idx:
                band_row = BAND_KD + row_idx - col_idx
                if band_row >= 0 and band_row <= BAND_KD:
                    ab[band_row, col_idx] = val

    if i > 0:
        prev = i - 1
        prev_block = block_start - 6
        for row in range(6):
            for col in range(6):
                val = 0.0
                for k in range(3):
                    j_p1_prev = jacobian_pos[_warp_jacobian_index(prev, row, k + 3)]
                    j_p0_cur = jacobian_pos[_warp_jacobian_index(i, col, k)]
                    j_t1_prev = jacobian_rot[_warp_jacobian_index(prev, row, k + 3)]
                    j_t0_cur = jacobian_rot[_warp_jacobian_index(i, col, k)]
                    val += j_p1_prev * j_p0_cur + j_t1_prev * j_t0_cur

                row_idx = prev_block + row
                col_idx = block_start + col
                band_row = BAND_KD + row_idx - col_idx
                if band_row >= 0 and band_row <= BAND_KD:
                    ab[band_row, col_idx] = val


@wp.kernel
def _warp_build_rhs(
    constraint_values: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lambda_sum: wp.array(dtype=wp.float32),
    n_dofs: int,
    rhs: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if i < n_dofs:
        rhs[i] = -constraint_values[i] - compliance[i] * lambda_sum[i]
    else:
        rhs[i] = 0.0


@wp.kernel
def _warp_pad_diagonal(
    A: wp.array2d(dtype=wp.float32),
    n_dofs: int,
    tile: int,
):
    i = wp.tid()
    if i >= n_dofs and i < tile:
        A[i, i] = 1.0


@wp.kernel
def _warp_cholesky_solve_tile(
    A: wp.array2d(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
):
    a_tile = wp.tile_load(A, shape=(TILE, TILE))
    b_tile = wp.tile_load(b, shape=TILE)
    L = wp.tile_cholesky(a_tile)
    x_tile = wp.tile_cholesky_solve(L, b_tile)
    wp.tile_store(x, x_tile)


@wp.func
def _mat33_add(a: wp.mat33, b: wp.mat33) -> wp.mat33:
    return wp.mat33(
        a[0, 0] + b[0, 0], a[0, 1] + b[0, 1], a[0, 2] + b[0, 2],
        a[1, 0] + b[1, 0], a[1, 1] + b[1, 1], a[1, 2] + b[1, 2],
        a[2, 0] + b[2, 0], a[2, 1] + b[2, 1], a[2, 2] + b[2, 2],
    )


@wp.func
def _mat33_sub(a: wp.mat33, b: wp.mat33) -> wp.mat33:
    return wp.mat33(
        a[0, 0] - b[0, 0], a[0, 1] - b[0, 1], a[0, 2] - b[0, 2],
        a[1, 0] - b[1, 0], a[1, 1] - b[1, 1], a[1, 2] - b[1, 2],
        a[2, 0] - b[2, 0], a[2, 1] - b[2, 1], a[2, 2] - b[2, 2],
    )


@wp.func
def _mat33_mul(a: wp.mat33, b: wp.mat33) -> wp.mat33:
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
    return wp.vec3(
        a[0, 0] * v[0] + a[0, 1] * v[1] + a[0, 2] * v[2],
        a[1, 0] * v[0] + a[1, 1] * v[1] + a[1, 2] * v[2],
        a[2, 0] * v[0] + a[2, 1] * v[1] + a[2, 2] * v[2],
    )


@wp.func
def _mat33_transpose(a: wp.mat33) -> wp.mat33:
    return wp.mat33(
        a[0, 0], a[1, 0], a[2, 0],
        a[0, 1], a[1, 1], a[2, 1],
        a[0, 2], a[1, 2], a[2, 2],
    )


@wp.func
def _mat33_cholesky(a: wp.mat33) -> wp.mat33:
    eps = 1.0e-9
    l00 = wp.sqrt(wp.max(a[0, 0], eps))
    l10 = a[1, 0] / l00
    l20 = a[2, 0] / l00

    l11 = wp.sqrt(wp.max(a[1, 1] - l10 * l10, eps))
    l21 = (a[2, 1] - l20 * l10) / l11

    l22 = wp.sqrt(wp.max(a[2, 2] - l20 * l20 - l21 * l21, eps))

    return wp.mat33(
        l00, 0.0, 0.0,
        l10, l11, 0.0,
        l20, l21, l22,
    )


@wp.func
def _mat33_solve_lower(L: wp.mat33, b: wp.vec3) -> wp.vec3:
    y0 = b[0] / L[0, 0]
    y1 = (b[1] - L[1, 0] * y0) / L[1, 1]
    y2 = (b[2] - L[2, 0] * y0 - L[2, 1] * y1) / L[2, 2]
    return wp.vec3(y0, y1, y2)


@wp.func
def _mat33_solve_upper(L: wp.mat33, b: wp.vec3) -> wp.vec3:
    x2 = b[2] / L[2, 2]
    x1 = (b[1] - L[2, 1] * x2) / L[1, 1]
    x0 = (b[0] - L[1, 0] * x1 - L[2, 0] * x2) / L[0, 0]
    return wp.vec3(x0, x1, x2)


@wp.func
def _mat33_inverse(a: wp.mat33) -> wp.mat33:
    det = (
        a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
        - a[0, 1] * (a[1, 0] * a[2, 2] - a[1, 2] * a[2, 0])
        + a[0, 2] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0])
    )
    if wp.abs(det) < 1.0e-9:
        det = 1.0e-9
    inv_det = 1.0 / det
    return wp.mat33(
        (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]) * inv_det,
        (a[0, 2] * a[2, 1] - a[0, 1] * a[2, 2]) * inv_det,
        (a[0, 1] * a[1, 2] - a[0, 2] * a[1, 1]) * inv_det,
        (a[1, 2] * a[2, 0] - a[1, 0] * a[2, 2]) * inv_det,
        (a[0, 0] * a[2, 2] - a[0, 2] * a[2, 0]) * inv_det,
        (a[0, 2] * a[1, 0] - a[0, 0] * a[1, 2]) * inv_det,
        (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0]) * inv_det,
        (a[0, 1] * a[2, 0] - a[0, 0] * a[2, 1]) * inv_det,
        (a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]) * inv_det,
    )


@wp.func
def _block_index(block: int, row: int, col: int) -> int:
    return block * 36 + row * 6 + col


@wp.func
def _vec_index(block: int, row: int) -> int:
    return block * 6 + row


@wp.func
def _load_block(blocks: wp.array(dtype=wp.float32), block: int) -> tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
    base = block * 36
    A = wp.mat33(
        blocks[base + 0], blocks[base + 1], blocks[base + 2],
        blocks[base + 6], blocks[base + 7], blocks[base + 8],
        blocks[base + 12], blocks[base + 13], blocks[base + 14],
    )
    B = wp.mat33(
        blocks[base + 3], blocks[base + 4], blocks[base + 5],
        blocks[base + 9], blocks[base + 10], blocks[base + 11],
        blocks[base + 15], blocks[base + 16], blocks[base + 17],
    )
    C = wp.mat33(
        blocks[base + 18], blocks[base + 19], blocks[base + 20],
        blocks[base + 24], blocks[base + 25], blocks[base + 26],
        blocks[base + 30], blocks[base + 31], blocks[base + 32],
    )
    D = wp.mat33(
        blocks[base + 21], blocks[base + 22], blocks[base + 23],
        blocks[base + 27], blocks[base + 28], blocks[base + 29],
        blocks[base + 33], blocks[base + 34], blocks[base + 35],
    )
    return A, B, C, D


@wp.func
def _store_block(
    blocks: wp.array(dtype=wp.float32),
    block: int,
    A: wp.mat33,
    B: wp.mat33,
    C: wp.mat33,
    D: wp.mat33,
):
    base = block * 36
    blocks[base + 0] = A[0, 0]
    blocks[base + 1] = A[0, 1]
    blocks[base + 2] = A[0, 2]
    blocks[base + 3] = B[0, 0]
    blocks[base + 4] = B[0, 1]
    blocks[base + 5] = B[0, 2]
    blocks[base + 6] = A[1, 0]
    blocks[base + 7] = A[1, 1]
    blocks[base + 8] = A[1, 2]
    blocks[base + 9] = B[1, 0]
    blocks[base + 10] = B[1, 1]
    blocks[base + 11] = B[1, 2]
    blocks[base + 12] = A[2, 0]
    blocks[base + 13] = A[2, 1]
    blocks[base + 14] = A[2, 2]
    blocks[base + 15] = B[2, 0]
    blocks[base + 16] = B[2, 1]
    blocks[base + 17] = B[2, 2]
    blocks[base + 18] = C[0, 0]
    blocks[base + 19] = C[0, 1]
    blocks[base + 20] = C[0, 2]
    blocks[base + 21] = D[0, 0]
    blocks[base + 22] = D[0, 1]
    blocks[base + 23] = D[0, 2]
    blocks[base + 24] = C[1, 0]
    blocks[base + 25] = C[1, 1]
    blocks[base + 26] = C[1, 2]
    blocks[base + 27] = D[1, 0]
    blocks[base + 28] = D[1, 1]
    blocks[base + 29] = D[1, 2]
    blocks[base + 30] = C[2, 0]
    blocks[base + 31] = C[2, 1]
    blocks[base + 32] = C[2, 2]
    blocks[base + 33] = D[2, 0]
    blocks[base + 34] = D[2, 1]
    blocks[base + 35] = D[2, 2]


@wp.func
def _load_vec(values: wp.array(dtype=wp.float32), block: int) -> tuple[wp.vec3, wp.vec3]:
    base = block * 6
    v0 = wp.vec3(values[base + 0], values[base + 1], values[base + 2])
    v1 = wp.vec3(values[base + 3], values[base + 4], values[base + 5])
    return v0, v1


@wp.func
def _store_vec(values: wp.array(dtype=wp.float32), block: int, v0: wp.vec3, v1: wp.vec3):
    base = block * 6
    values[base + 0] = v0[0]
    values[base + 1] = v0[1]
    values[base + 2] = v0[2]
    values[base + 3] = v1[0]
    values[base + 4] = v1[1]
    values[base + 5] = v1[2]


@wp.func
def _block_column(
    blocks: wp.array(dtype=wp.float32), block: int, col: int
) -> tuple[wp.vec3, wp.vec3]:
    base = block * 36
    v0 = wp.vec3(
        blocks[base + 0 * 6 + col],
        blocks[base + 1 * 6 + col],
        blocks[base + 2 * 6 + col],
    )
    v1 = wp.vec3(
        blocks[base + 3 * 6 + col],
        blocks[base + 4 * 6 + col],
        blocks[base + 5 * 6 + col],
    )
    return v0, v1


@wp.func
def _block_set_column(
    blocks: wp.array(dtype=wp.float32), block: int, col: int, v0: wp.vec3, v1: wp.vec3
):
    base = block * 36
    blocks[base + 0 * 6 + col] = v0[0]
    blocks[base + 1 * 6 + col] = v0[1]
    blocks[base + 2 * 6 + col] = v0[2]
    blocks[base + 3 * 6 + col] = v1[0]
    blocks[base + 4 * 6 + col] = v1[1]
    blocks[base + 5 * 6 + col] = v1[2]


@wp.func
def _block_row(
    blocks: wp.array(dtype=wp.float32), block: int, row: int
) -> tuple[wp.vec3, wp.vec3]:
    base = block * 36 + row * 6
    v0 = wp.vec3(
        blocks[base + 0],
        blocks[base + 1],
        blocks[base + 2],
    )
    v1 = wp.vec3(
        blocks[base + 3],
        blocks[base + 4],
        blocks[base + 5],
    )
    return v0, v1


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
    return (
        _mat33_sub(A, E),
        _mat33_sub(B, F),
        _mat33_sub(C, G),
        _mat33_sub(D, H),
    )


@wp.func
def _block_mul_vec(
    A: wp.mat33,
    B: wp.mat33,
    C: wp.mat33,
    D: wp.mat33,
    v0: wp.vec3,
    v1: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
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
    L11 = _mat33_cholesky(A)
    c0 = wp.vec3(C[0, 0], C[0, 1], C[0, 2])
    c1 = wp.vec3(C[1, 0], C[1, 1], C[1, 2])
    c2 = wp.vec3(C[2, 0], C[2, 1], C[2, 2])
    y0 = _mat33_solve_lower(L11, c0)
    y1 = _mat33_solve_lower(L11, c1)
    y2 = _mat33_solve_lower(L11, c2)
    L21 = wp.mat33(
        y0[0], y0[1], y0[2],
        y1[0], y1[1], y1[2],
        y2[0], y2[1], y2[2],
    )
    L21_t = _mat33_transpose(L21)
    S = _mat33_sub(D, _mat33_mul(L21, L21_t))
    L22 = _mat33_cholesky(S)

    yb0 = _mat33_solve_lower(L11, b0)
    tmp = b1 - _mat33_mul_vec3(L21, yb0)
    yb1 = _mat33_solve_lower(L22, tmp)

    x1 = _mat33_solve_upper(L22, yb1)
    x0 = _mat33_solve_upper(L11, yb0 - _mat33_mul_vec3(L21_t, x1))
    return x0, x1


@wp.kernel
def _warp_assemble_jmjt_blocks(
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    n_edges: int,
    diag_blocks: wp.array(dtype=wp.float32),
    offdiag_blocks: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if i >= n_edges:
        return
    regularization = 1.0e-6
    block_start = 6 * i

    for row in range(6):
        for col in range(6):
            val = 0.0
            for k in range(3):
                j_p0_r = jacobian_pos[_warp_jacobian_index(i, row, k)]
                j_p0_c = jacobian_pos[_warp_jacobian_index(i, col, k)]
                j_p1_r = jacobian_pos[_warp_jacobian_index(i, row, k + 3)]
                j_p1_c = jacobian_pos[_warp_jacobian_index(i, col, k + 3)]
                j_t0_r = jacobian_rot[_warp_jacobian_index(i, row, k)]
                j_t0_c = jacobian_rot[_warp_jacobian_index(i, col, k)]
                j_t1_r = jacobian_rot[_warp_jacobian_index(i, row, k + 3)]
                j_t1_c = jacobian_rot[_warp_jacobian_index(i, col, k + 3)]
                val += (
                    j_p0_r * j_p0_c
                    + j_p1_r * j_p1_c
                    + j_t0_r * j_t0_c
                    + j_t1_r * j_t1_c
                )
            if row == col:
                val += compliance[i * 6 + row] + regularization
            diag_blocks[_block_index(i, row, col)] = val

    if i == 0:
        for row in range(6):
            for col in range(6):
                offdiag_blocks[_block_index(i, row, col)] = 0.0
        return

    prev = i - 1
    for row in range(6):
        for col in range(6):
            val = 0.0
            for k in range(3):
                j_p1_prev = jacobian_pos[_warp_jacobian_index(prev, row, k + 3)]
                j_p0_cur = jacobian_pos[_warp_jacobian_index(i, col, k)]
                j_t1_prev = jacobian_rot[_warp_jacobian_index(prev, row, k + 3)]
                j_t0_cur = jacobian_rot[_warp_jacobian_index(i, col, k)]
                val += j_p1_prev * j_p0_cur + j_t1_prev * j_t0_cur
            offdiag_blocks[_block_index(i, row, col)] = val


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
    tid = wp.tid()
    if tid != 0:
        return

    for j in range(n):
        sum_val = float(0.0)
        kmax = j if j < BAND_KD else BAND_KD
        for k in range(1, kmax + 1):
            u = ab[BAND_KD - k, j]
            sum_val += u * u

        ajj = ab[BAND_KD, j] - sum_val
        if ajj <= 1.0e-8:
            ajj = 1.0e-8
            
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

class DefKitDirectLibrary:
    def __init__(self, dll_path: str | None, calling_convention: str):
        self.dll_path = self._resolve_dll_path(dll_path)
        self.dll = self._load_library(self.dll_path, calling_convention)
        self._bind_functions()

    @staticmethod
    def _resolve_dll_path(dll_path: str | None) -> str:
        if dll_path:
            abs_path = os.path.abspath(dll_path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"DefKit DLL not found: {abs_path}")
            return abs_path
        return "DefKitAdv.dll"

    @staticmethod
    def _load_library(dll_path: str, calling_convention: str):
        loader = ctypes.WinDLL if calling_convention == "stdcall" else ctypes.CDLL
        try:
            return loader(dll_path)
        except OSError as exc:
            raise RuntimeError(
                "Failed to load DefKit DLL. Provide --dll-path or add it to PATH. "
                f"Path: {dll_path}. Error: {exc}"
            ) from exc

    def _get_function(self, name: str, argtypes, restype=None):
        try:
            fn = getattr(self.dll, name)
        except AttributeError as exc:
            raise RuntimeError(f"Required symbol '{name}' not found in {self.dll_path}") from exc
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    def _get_optional_function(self, name: str, argtypes, restype=None):
        fn = getattr(self.dll, name, None)
        if fn is None:
            return None
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    def _bind_functions(self):
        self.PredictPositions_native = self._get_function(
            "PredictPositions_native",
            [
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(BtVector3),
            ],
        )
        self.Integrate_native = self._get_function(
            "Integrate_native",
            [
                ctypes.c_float,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(ctypes.c_float),
            ],
        )
        self.PredictRotationsPBD = self._get_function(
            "PredictRotationsPBD",
            [
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(ctypes.c_float),
            ],
        )
        self.IntegrateRotationsPBD = self._get_function(
            "IntegrateRotationsPBD",
            [
                ctypes.c_float,
                ctypes.c_int,
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(ctypes.c_float),
            ],
        )
        self.InitDirectElasticRod = self._get_function(
            "InitDirectElasticRod",
            [
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float,
                ctypes.c_float,
            ],
            restype=ctypes.c_void_p,
        )
        self.PrepareDirectElasticRodConstraints = self._get_function(
            "PrepareDirectElasticRodConstraints",
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float,
                ctypes.c_float,
            ],
        )
        self.UpdateConstraints_DirectElasticRodConstraintsBanded = self._get_function(
            "UpdateConstraints_DirectElasticRodConstraintsBanded",
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(ctypes.c_float),
            ],
        )
        self.ComputeJacobians_DirectElasticRodConstraintsBanded = self._get_function(
            "ComputeJacobians_DirectElasticRodConstraintsBanded",
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(ctypes.c_float),
            ],
        )
        self.AssembleJMJT_DirectElasticRodConstraintsBanded = self._get_function(
            "AssembleJMJT_DirectElasticRodConstraintsBanded",
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(ctypes.c_float),
            ],
        )
        self.ProjectJMJT_DirectElasticRodConstraintsBanded = self._get_function(
            "ProjectJMJT_DirectElasticRodConstraintsBanded",
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
            ],
        )
        self.UpdateDirectElasticRodConstraints = self._get_optional_function(
            "UpdateDirectElasticRodConstraints",
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(ctypes.c_float),
            ],
        )
        self.ProjectDirectElasticRodConstraints = self._get_optional_function(
            "ProjectDirectElasticRodConstraints",
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
            ],
        )
        self.DestroyDirectElasticRod = self._get_function(
            "DestroyDirectElasticRod",
            [ctypes.c_void_p],
        )


class DefKitDirectRodState:
    def __init__(
        self,
        lib: DefKitDirectLibrary,
        num_points: int,
        segment_length: float,
        mass: float,
        particle_height: float,
        rod_radius: float,
        bend_stiffness: float,
        twist_stiffness: float,
        rest_bend_d1: float,
        rest_bend_d2: float,
        rest_twist: float,
        young_modulus: float,
        torsion_modulus: float,
        gravity: np.ndarray,
        lock_root_rotation: bool,
        use_banded: bool,
    ):
        self.lib = lib
        self.num_points = num_points
        self.num_edges = max(0, num_points - 1)
        self.segment_length = segment_length
        self.young_modulus = young_modulus
        self.torsion_modulus = torsion_modulus
        self.rod_radius = rod_radius
        self.supports_non_banded = self.lib.ProjectDirectElasticRodConstraints is not None
        self.use_banded = use_banded or not self.supports_non_banded

        self.positions = np.zeros((num_points, 4), dtype=np.float32)
        self.predicted_positions = np.zeros((num_points, 4), dtype=np.float32)
        self.velocities = np.zeros((num_points, 4), dtype=np.float32)
        self.forces = np.zeros((num_points, 4), dtype=np.float32)

        for i in range(num_points):
            self.positions[i, 0] = i * segment_length
            self.positions[i, 2] = particle_height

        self.predicted_positions[:] = self.positions

        q_align = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32), math.pi * 0.5)
        self.orientations = np.tile(q_align, (num_points, 1)).astype(np.float32)
        self.predicted_orientations = self.orientations.copy()
        self.prev_orientations = self.orientations.copy()

        self.angular_velocities = np.zeros((num_points, 4), dtype=np.float32)
        self.torques = np.zeros((num_points, 4), dtype=np.float32)

        inv_mass_value = 0.0 if mass == 0.0 else 1.0 / mass
        self.inv_masses = np.full(num_points, inv_mass_value, dtype=np.float32)
        self.inv_masses[0] = 0.0
        self._root_inv_mass_unlocked = inv_mass_value

        self.quat_inv_masses = np.full(num_points, 1.0, dtype=np.float32)
        # Match C++ behavior: if inv_mass is 0 (static), rotation is also locked.
        static_mask = (self.inv_masses == 0.0)
        self.quat_inv_masses[static_mask] = 0.0
        self._root_quat_inv_mass_unlocked = np.float32(1.0)
        
        if lock_root_rotation:
            self.quat_inv_masses[0] = 0.0
        self.root_locked = True

        self.rest_lengths = np.full(self.num_edges, segment_length, dtype=np.float32)

        self.rest_darboux = np.zeros((self.num_edges, 4), dtype=np.float32)
        self.set_rest_darboux(rest_bend_d1, rest_bend_d2, rest_twist)

        self.bend_stiffness = np.zeros((self.num_edges, 4), dtype=np.float32)
        self.set_bend_stiffness(bend_stiffness, twist_stiffness)

        self.pos_corrections = np.zeros((num_points, 4), dtype=np.float32)
        self.rot_corrections = np.zeros((num_points, 4), dtype=np.float32)

        self.gravity = np.zeros((1, 4), dtype=np.float32)
        self.set_gravity(gravity)

        self._initial_positions = self.positions.copy()
        self._initial_orientations = self.orientations.copy()

        self.rod_ptr = self.lib.InitDirectElasticRod(
            ctypes.c_int(self.num_points),
            _as_ptr(self.positions, BtVector3),
            _as_ptr(self.orientations, BtQuaternion),
            ctypes.c_float(rod_radius),
            _as_float_ptr(self.rest_lengths),
            ctypes.c_float(self.young_modulus),
            ctypes.c_float(self.torsion_modulus),
        )
        if not self.rod_ptr:
            raise RuntimeError("InitDirectElasticRod returned a null pointer.")
        self._destroyed = False
        atexit.register(self.destroy)

    def destroy(self):
        if self._destroyed:
            return
        if self.rod_ptr:
            self.lib.DestroyDirectElasticRod(self.rod_ptr)
            self.rod_ptr = None
        self._destroyed = True

    def set_gravity(self, gravity: np.ndarray):
        self.gravity[0, 0:3] = gravity.astype(np.float32)

    def set_bend_stiffness(self, bend_stiffness: float, twist_stiffness: float):
        self.bend_stiffness[:, 0] = bend_stiffness
        self.bend_stiffness[:, 1] = bend_stiffness
        self.bend_stiffness[:, 2] = twist_stiffness

    def set_rest_darboux(self, rest_bend_d1: float, rest_bend_d2: float, rest_twist: float):
        self.rest_darboux[:, 0] = rest_bend_d1
        self.rest_darboux[:, 1] = rest_bend_d2
        self.rest_darboux[:, 2] = rest_twist

    def set_solver_mode(self, use_banded: bool):
        self.use_banded = use_banded or not self.supports_non_banded

    def set_root_locked(self, locked: bool):
        self.root_locked = locked
        if locked:
            self.inv_masses[0] = 0.0
            self.quat_inv_masses[0] = 0.0
            self.velocities[0, 0:3] = 0.0
            self.angular_velocities[0, 0:3] = 0.0
        else:
            self.inv_masses[0] = self._root_inv_mass_unlocked
            self.quat_inv_masses[0] = self._root_quat_inv_mass_unlocked

    def toggle_root_lock(self):
        self.set_root_locked(not self.root_locked)

    def reset(self):
        self.positions[:] = self._initial_positions
        self.predicted_positions[:] = self._initial_positions
        self.velocities.fill(0.0)
        self.forces.fill(0.0)

        self.orientations[:] = self._initial_orientations
        self.predicted_orientations[:] = self._initial_orientations
        self.prev_orientations[:] = self._initial_orientations
        self.angular_velocities.fill(0.0)
        self.torques.fill(0.0)

    def predict_positions(self, dt: float, linear_damping: float):
        self.lib.PredictPositions_native(
            ctypes.c_float(dt),
            ctypes.c_float(linear_damping),
            ctypes.c_int(self.num_points),
            _as_ptr(self.positions, BtVector3),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.velocities, BtVector3),
            _as_ptr(self.forces, BtVector3),
            _as_float_ptr(self.inv_masses),
            _as_ptr(self.gravity, BtVector3),
        )

    def predict_rotations(self, dt: float, angular_damping: float):
        self.lib.PredictRotationsPBD(
            ctypes.c_float(dt),
            ctypes.c_float(angular_damping),
            ctypes.c_int(self.num_points),
            _as_ptr(self.orientations, BtQuaternion),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_ptr(self.angular_velocities, BtVector3),
            _as_ptr(self.torques, BtVector3),
            _as_float_ptr(self.quat_inv_masses),
        )

    def prepare_constraints(self, dt: float):
        self.lib.PrepareDirectElasticRodConstraints(
            self.rod_ptr,
            ctypes.c_int(self.num_edges),
            ctypes.c_float(dt),
            _as_ptr(self.bend_stiffness, BtVector3),
            _as_ptr(self.rest_darboux, BtVector3),
            _as_float_ptr(self.rest_lengths),
            ctypes.c_float(self.young_modulus),
            ctypes.c_float(self.torsion_modulus),
        )

    def update_constraints_banded(self):
        self.lib.UpdateConstraints_DirectElasticRodConstraintsBanded(
            self.rod_ptr,
            ctypes.c_int(self.num_points),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
        )

    def compute_jacobians_banded(self):
        self.lib.ComputeJacobians_DirectElasticRodConstraintsBanded(
            self.rod_ptr,
            ctypes.c_int(0),
            ctypes.c_int(self.num_edges),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
        )

    def assemble_jmjt_banded(self):
        self.lib.AssembleJMJT_DirectElasticRodConstraintsBanded(
            self.rod_ptr,
            ctypes.c_int(0),
            ctypes.c_int(self.num_edges),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
        )

    def project_jmjt_banded(self):
        self.lib.ProjectJMJT_DirectElasticRodConstraintsBanded(
            self.rod_ptr,
            ctypes.c_int(self.num_points),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
            _as_ptr(self.pos_corrections, BtVector3),
            _as_ptr(self.rot_corrections, BtQuaternion),
        )

    def project_direct(self):
        self.lib.ProjectDirectElasticRodConstraints(
            self.rod_ptr,
            ctypes.c_int(self.num_points),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
            _as_ptr(self.pos_corrections, BtVector3),
            _as_ptr(self.rot_corrections, BtQuaternion),
        )

    def integrate_positions(self, dt: float):
        self.lib.Integrate_native(
            ctypes.c_float(dt),
            ctypes.c_int(self.num_points),
            _as_ptr(self.positions, BtVector3),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.velocities, BtVector3),
            _as_float_ptr(self.inv_masses),
        )

    def integrate_rotations(self, dt: float):
        self.lib.IntegrateRotationsPBD(
            ctypes.c_float(dt),
            ctypes.c_int(self.num_points),
            _as_ptr(self.orientations, BtQuaternion),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_ptr(self.prev_orientations, BtQuaternion),
            _as_ptr(self.angular_velocities, BtVector3),
            _as_float_ptr(self.quat_inv_masses),
        )

    def apply_floor_collisions(self, floor_z: float, restitution: float = 0.0):
        if self.num_points == 0:
            return
        min_z = np.float32(floor_z + self.rod_radius)
        for i in range(self.num_points):
            z = self.positions[i, 2]
            if z < min_z:
                self.positions[i, 2] = min_z
                self.predicted_positions[i, 2] = min_z
                if self.velocities[i, 2] < 0.0:
                    self.velocities[i, 2] = -np.float32(restitution) * self.velocities[i, 2]

    def step(
        self,
        dt: float,
        linear_damping: float,
        angular_damping: float,
    ):
        self.predict_positions(dt, linear_damping)
        self.predict_rotations(dt, angular_damping)
        self.prepare_constraints(dt)

        if self.use_banded or not self.supports_non_banded:
            self.update_constraints_banded()
            self.compute_jacobians_banded()
            self.assemble_jmjt_banded()
            self.project_jmjt_banded()
        else:
            self.project_direct()

        self.integrate_positions(dt)
        self.integrate_rotations(dt)


class NumpyDirectRodState(DefKitDirectRodState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numpy_available = {
            "predict_positions": True,
            "integrate_positions": True,
            "predict_rotations": True,
            "integrate_rotations": True,
            "prepare_constraints": True,
            "update_constraints_banded": True,
            "compute_jacobians_banded": True,
            "assemble_jmjt_banded": True,
            "project_jmjt_banded": True,
            "project_direct": True,
        }
        self.numpy_enabled = {
            "predict_positions": True,
            "integrate_positions": True,
            "predict_rotations": True,
            "integrate_rotations": True,
            "prepare_constraints": True,
            "update_constraints_banded": True,
            "compute_jacobians_banded": True,
            "assemble_jmjt_banded": True,
            "project_jmjt_banded": True,
            "project_direct": True,
        }
        self.warp_device = wp.get_device()
        warp_default = self.warp_device.is_cuda
        self.warp_available = {step_name: False for step_name in self.numpy_available}
        self.warp_enabled = {step_name: False for step_name in self.numpy_available}
        for step_name in (
            "predict_positions",
            "integrate_positions",
            "predict_rotations",
            "integrate_rotations",
            "prepare_constraints",
            "project_direct",
        ):
            self.warp_available[step_name] = True
            self.warp_enabled[step_name] = warp_default
        self.direct_solve_backend = (
            DIRECT_SOLVE_WARP_BLOCK_THOMAS if warp_default else DIRECT_SOLVE_CPU_NUMPY
        )
        self.lambdas = np.zeros((self.num_edges, 6), dtype=np.float32)
        self.compliance = np.zeros((self.num_edges, 6), dtype=np.float32)
        self.constraint_values = np.zeros((self.num_edges, 6), dtype=np.float32)
        self.lambda_sum = np.zeros((self.num_edges, 6), dtype=np.float32)
        self.current_rest_lengths = self.rest_lengths.copy()
        self.current_rest_darboux = np.zeros((self.num_edges, 3), dtype=np.float32)
        self._update_cross_section_properties()
        self.last_constraint_max = 0.0
        self.last_delta_lambda_max = 0.0
        self.last_correction_max = 0.0

    def set_numpy_override(self, step_name: str, enabled: bool):
        if step_name not in self.numpy_available:
            raise ValueError(f"Unknown step: {step_name}")
        if not self.numpy_available[step_name]:
            self.numpy_enabled[step_name] = False
            return False
        self.numpy_enabled[step_name] = enabled
        return True

    def set_warp_override(self, step_name: str, enabled: bool):
        if step_name not in self.warp_available:
            raise ValueError(f"Unknown step: {step_name}")
        if not self.warp_available[step_name]:
            self.warp_enabled[step_name] = False
            return False
        self.warp_enabled[step_name] = enabled
        return True

    def set_direct_solve_backend(self, backend: str):
        if backend not in DIRECT_SOLVE_BACKENDS:
            raise ValueError(f"Unknown direct solve backend: {backend}")
        self.direct_solve_backend = backend

    def _use_warp_step(self, step_name: str) -> bool:
        return self.warp_available.get(step_name, False) and self.warp_enabled.get(step_name, False)

    def predict_positions(self, dt: float, linear_damping: float):
        if self._use_warp_step("predict_positions"):
            self._warp_predict_positions(dt, linear_damping)
        elif self.numpy_enabled["predict_positions"]:
            self._numpy_predict_positions(dt, linear_damping)
        else:
            super().predict_positions(dt, linear_damping)

    def integrate_positions(self, dt: float):
        if self._use_warp_step("integrate_positions"):
            self._warp_integrate_positions(dt)
        elif self.numpy_enabled["integrate_positions"]:
            self._numpy_integrate_positions(dt)
        else:
            super().integrate_positions(dt)

    def predict_rotations(self, dt: float, angular_damping: float):
        if self._use_warp_step("predict_rotations"):
            self._warp_predict_rotations(dt, angular_damping)
        elif self.numpy_enabled["predict_rotations"]:
            self._numpy_predict_rotations(dt, angular_damping)
        else:
            super().predict_rotations(dt, angular_damping)

    def integrate_rotations(self, dt: float):
        if self._use_warp_step("integrate_rotations"):
            self._warp_integrate_rotations(dt)
        elif self.numpy_enabled["integrate_rotations"]:
            self._numpy_integrate_rotations(dt)
        else:
            super().integrate_rotations(dt)

    def prepare_constraints(self, dt: float):
        if self._use_warp_step("prepare_constraints"):
            self._warp_prepare_constraints(dt)
            if self._requires_native_constraint_pipeline():
                super().prepare_constraints(dt)
        elif self.numpy_enabled["prepare_constraints"]:
            self._numpy_prepare_constraints(dt)
            if self._requires_native_constraint_pipeline():
                super().prepare_constraints(dt)
        else:
            super().prepare_constraints(dt)

    def update_constraints_banded(self):
        if self.numpy_enabled["update_constraints_banded"]:
            self._numpy_update_constraints_banded()
            if self._requires_native_constraint_pipeline():
                super().update_constraints_banded()
        else:
            super().update_constraints_banded()

    def compute_jacobians_banded(self):
        if self.numpy_enabled["compute_jacobians_banded"]:
            self._numpy_compute_jacobians_direct()
            if self._requires_native_constraint_pipeline():
                super().compute_jacobians_banded()
        else:
            super().compute_jacobians_banded()

    def assemble_jmjt_banded(self):
        if self.numpy_enabled["assemble_jmjt_banded"]:
            self._numpy_assemble_jmjt_banded()
            if self._requires_native_constraint_pipeline():
                super().assemble_jmjt_banded()
        else:
            super().assemble_jmjt_banded()

    def project_jmjt_banded(self):
        if self.numpy_enabled["project_jmjt_banded"]:
            self._numpy_project_jmjt_banded()
        else:
            super().project_jmjt_banded()

    def project_direct(self):
        if self._use_warp_step("project_direct"):
            if self.direct_solve_backend == DIRECT_SOLVE_WARP_BLOCK_THOMAS:
                self._warp_project_direct()
            elif self.direct_solve_backend == DIRECT_SOLVE_WARP_BANDED_CHOLESKY:
                self._warp_project_direct(use_banded_cholesky=True)
            else:
                self._numpy_project_direct()
        elif self.numpy_enabled["project_direct"]:
            self._numpy_project_direct()
        else:
            super().project_direct()

    def _numpy_predict_positions(self, dt: float, linear_damping: float):
        dt = np.float32(dt)
        damping = np.float32(linear_damping)
        damp = np.float32(1.0) - damping

        inv_mass = self.inv_masses[:, None]
        positions = self.positions[:, 0:3]
        velocities = self.velocities[:, 0:3]
        forces = self.forces[:, 0:3]
        gravity = self.gravity[0, 0:3]

        velocities[:] = velocities + (forces * inv_mass + gravity) * dt
        velocities[:] = velocities * damp

        predicted = positions + velocities * dt

        static_mask = self.inv_masses == 0.0
        velocities[static_mask] = 0.0
        predicted[static_mask] = positions[static_mask]

        self.predicted_positions[:, 0:3] = predicted
        self.predicted_positions[:, 3] = 0.0
        self.velocities[:, 3] = 0.0

    def _numpy_integrate_positions(self, dt: float):
        dt_inv = np.float32(1.0) / np.float32(dt)
        positions = self.positions[:, 0:3]
        predicted = self.predicted_positions[:, 0:3]
        velocities = self.velocities[:, 0:3]

        dynamic_mask = self.inv_masses != 0.0
        velocities[dynamic_mask] = (predicted[dynamic_mask] - positions[dynamic_mask]) * dt_inv
        positions[dynamic_mask] = predicted[dynamic_mask]

        self.positions[:, 3] = 0.0
        self.velocities[:, 3] = 0.0

    def _numpy_predict_rotations(self, dt: float, angular_damping: float):
        dt = np.float32(dt)
        half_dt = np.float32(0.5) * dt
        damp = np.float32(1.0) - np.float32(angular_damping)

        inv_mass = self.quat_inv_masses[:, None]
        ang_vel = self.angular_velocities[:, 0:3]
        torques = self.torques[:, 0:3]
        orientations = self.orientations

        dynamic_mask = self.quat_inv_masses != 0.0
        ang_vel[dynamic_mask] = (ang_vel[dynamic_mask] + torques[dynamic_mask] * inv_mass[dynamic_mask] * dt) * damp
        ang_vel[~dynamic_mask] = 0.0

        ang_vel_q = np.zeros_like(orientations)
        ang_vel_q[:, 0:3] = ang_vel

        qdot = self._numpy_quat_mul(ang_vel_q, orientations)
        predicted = orientations + qdot * half_dt
        predicted = self._numpy_quat_normalize(predicted, dynamic_mask)
        predicted[~dynamic_mask] = orientations[~dynamic_mask]

        self.predicted_orientations = predicted.astype(np.float32)
        self.angular_velocities[:, 3] = 0.0

    def _numpy_integrate_rotations(self, dt: float):
        dt_inv2 = np.float32(2.0) / np.float32(dt)
        dynamic_mask = self.quat_inv_masses != 0.0

        predicted = self.predicted_orientations
        orientations = self.orientations

        conj = orientations.copy()
        conj[:, 0:3] *= -1.0

        rel = self._numpy_quat_mul(predicted, conj)
        self.angular_velocities[dynamic_mask, 0:3] = rel[dynamic_mask, 0:3] * dt_inv2

        self.prev_orientations[dynamic_mask] = orientations[dynamic_mask]
        self.orientations[dynamic_mask] = predicted[dynamic_mask]
        self.angular_velocities[:, 3] = 0.0

    def _warp_predict_positions(self, dt: float, linear_damping: float):
        if self.num_points == 0:
            return
        device = self.warp_device
        positions = np.ascontiguousarray(self.positions[:, 0:3])
        velocities = np.ascontiguousarray(self.velocities[:, 0:3])
        forces = np.ascontiguousarray(self.forces[:, 0:3])
        inv_masses = np.ascontiguousarray(self.inv_masses)

        positions_wp = wp.array(positions, dtype=wp.vec3, device=device)
        velocities_wp = wp.array(velocities, dtype=wp.vec3, device=device)
        forces_wp = wp.array(forces, dtype=wp.vec3, device=device)
        inv_masses_wp = wp.array(inv_masses, dtype=wp.float32, device=device)
        predicted_wp = wp.empty(self.num_points, dtype=wp.vec3, device=device)

        gravity = wp.vec3(
            float(self.gravity[0, 0]),
            float(self.gravity[0, 1]),
            float(self.gravity[0, 2]),
        )

        wp.launch(
            _warp_predict_positions,
            dim=self.num_points,
            inputs=[
                positions_wp,
                velocities_wp,
                forces_wp,
                inv_masses_wp,
                gravity,
                float(dt),
                float(linear_damping),
                predicted_wp,
            ],
            device=device,
        )

        self.predicted_positions[:, 0:3] = predicted_wp.numpy()
        self.predicted_positions[:, 3] = 0.0
        self.velocities[:, 0:3] = velocities_wp.numpy()
        self.velocities[:, 3] = 0.0

    def _warp_integrate_positions(self, dt: float):
        if self.num_points == 0:
            return
        device = self.warp_device
        positions = np.ascontiguousarray(self.positions[:, 0:3])
        predicted = np.ascontiguousarray(self.predicted_positions[:, 0:3])
        velocities = np.ascontiguousarray(self.velocities[:, 0:3])
        inv_masses = np.ascontiguousarray(self.inv_masses)

        positions_wp = wp.array(positions, dtype=wp.vec3, device=device)
        predicted_wp = wp.array(predicted, dtype=wp.vec3, device=device)
        velocities_wp = wp.array(velocities, dtype=wp.vec3, device=device)
        inv_masses_wp = wp.array(inv_masses, dtype=wp.float32, device=device)

        wp.launch(
            _warp_integrate_positions,
            dim=self.num_points,
            inputs=[positions_wp, predicted_wp, velocities_wp, inv_masses_wp, float(dt)],
            device=device,
        )

        self.positions[:, 0:3] = positions_wp.numpy()
        self.positions[:, 3] = 0.0
        self.velocities[:, 0:3] = velocities_wp.numpy()
        self.velocities[:, 3] = 0.0

    def _warp_predict_rotations(self, dt: float, angular_damping: float):
        if self.num_points == 0:
            return
        device = self.warp_device
        orientations = np.ascontiguousarray(self.orientations)
        angular_vel = np.ascontiguousarray(self.angular_velocities[:, 0:3])
        torques = np.ascontiguousarray(self.torques[:, 0:3])
        quat_inv_masses = np.ascontiguousarray(self.quat_inv_masses)

        orientations_wp = wp.array(orientations, dtype=wp.quat, device=device)
        angular_vel_wp = wp.array(angular_vel, dtype=wp.vec3, device=device)
        torques_wp = wp.array(torques, dtype=wp.vec3, device=device)
        quat_inv_masses_wp = wp.array(quat_inv_masses, dtype=wp.float32, device=device)
        predicted_wp = wp.empty(self.num_points, dtype=wp.quat, device=device)

        wp.launch(
            _warp_predict_rotations,
            dim=self.num_points,
            inputs=[
                orientations_wp,
                angular_vel_wp,
                torques_wp,
                quat_inv_masses_wp,
                float(dt),
                float(angular_damping),
                predicted_wp,
            ],
            device=device,
        )

        self.predicted_orientations[:, :] = predicted_wp.numpy()
        self.angular_velocities[:, 0:3] = angular_vel_wp.numpy()
        self.angular_velocities[:, 3] = 0.0

    def _warp_integrate_rotations(self, dt: float):
        if self.num_points == 0:
            return
        device = self.warp_device
        orientations = np.ascontiguousarray(self.orientations)
        predicted = np.ascontiguousarray(self.predicted_orientations)
        prev_orientations = np.ascontiguousarray(self.prev_orientations)
        angular_vel = np.ascontiguousarray(self.angular_velocities[:, 0:3])
        quat_inv_masses = np.ascontiguousarray(self.quat_inv_masses)

        orientations_wp = wp.array(orientations, dtype=wp.quat, device=device)
        predicted_wp = wp.array(predicted, dtype=wp.quat, device=device)
        prev_orientations_wp = wp.array(prev_orientations, dtype=wp.quat, device=device)
        angular_vel_wp = wp.array(angular_vel, dtype=wp.vec3, device=device)
        quat_inv_masses_wp = wp.array(quat_inv_masses, dtype=wp.float32, device=device)

        wp.launch(
            _warp_integrate_rotations,
            dim=self.num_points,
            inputs=[
                orientations_wp,
                predicted_wp,
                prev_orientations_wp,
                angular_vel_wp,
                quat_inv_masses_wp,
                float(dt),
            ],
            device=device,
        )

        self.orientations[:, :] = orientations_wp.numpy()
        self.prev_orientations[:, :] = prev_orientations_wp.numpy()
        self.angular_velocities[:, 0:3] = angular_vel_wp.numpy()
        self.angular_velocities[:, 3] = 0.0

    def _warp_prepare_constraints(self, dt: float):
        self.lambdas.fill(0.0)
        self.lambda_sum.fill(0.0)

        self.current_rest_lengths = self.rest_lengths.copy()
        self.current_rest_darboux[:, 0] = self.rest_darboux[:, 0]
        self.current_rest_darboux[:, 1] = self.rest_darboux[:, 1]
        self.current_rest_darboux[:, 2] = self.rest_darboux[:, 2]

        if self.num_edges == 0:
            return

        device = self.warp_device
        rest_lengths_wp = wp.array(self.current_rest_lengths, dtype=wp.float32, device=device)
        bend_stiffness = np.ascontiguousarray(self.bend_stiffness[:, 0:3])
        bend_stiffness_wp = wp.array(bend_stiffness, dtype=wp.vec3, device=device)
        compliance_flat = np.ascontiguousarray(self.compliance.reshape(-1))
        compliance_wp = wp.array(compliance_flat, dtype=wp.float32, device=device)

        wp.launch(
            _warp_prepare_compliance,
            dim=self.num_edges,
            inputs=[
                rest_lengths_wp,
                bend_stiffness_wp,
                float(self.young_modulus),
                float(self.torsion_modulus),
                float(dt),
                compliance_wp,
            ],
            device=device,
        )

        self.compliance[:, :] = compliance_wp.numpy().reshape(self.num_edges, 6)

    def _warp_project_direct(self, use_banded_cholesky: bool = False):
        n_edges = self.num_edges
        if n_edges == 0:
            return

        device = self.warp_device
        n_dofs = 6 * n_edges

        positions = np.ascontiguousarray(self.predicted_positions[:, 0:3])
        orientations = np.ascontiguousarray(self.predicted_orientations)
        rest_lengths = np.ascontiguousarray(self.current_rest_lengths)
        rest_darboux = np.ascontiguousarray(self.current_rest_darboux)

        positions_wp = wp.array(positions, dtype=wp.vec3, device=device)
        orientations_wp = wp.array(orientations, dtype=wp.quat, device=device)
        rest_lengths_wp = wp.array(rest_lengths, dtype=wp.float32, device=device)
        rest_darboux_wp = wp.array(rest_darboux, dtype=wp.vec3, device=device)

        constraint_values_wp = wp.empty(n_edges * 6, dtype=wp.float32, device=device)
        jacobian_pos_wp = wp.array(np.zeros(n_edges * 36, dtype=np.float32), dtype=wp.float32, device=device)
        jacobian_rot_wp = wp.array(np.zeros(n_edges * 36, dtype=np.float32), dtype=wp.float32, device=device)

        wp.launch(
            _warp_update_constraints_direct,
            dim=n_edges,
            inputs=[
                positions_wp,
                orientations_wp,
                rest_lengths_wp,
                rest_darboux_wp,
                constraint_values_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_compute_jacobians_direct,
            dim=n_edges,
            inputs=[
                orientations_wp,
                rest_lengths_wp,
                jacobian_pos_wp,
                jacobian_rot_wp,
            ],
            device=device,
        )

        compliance_flat = np.ascontiguousarray(self.compliance.reshape(-1))
        compliance_wp = wp.array(compliance_flat, dtype=wp.float32, device=device)
        lambda_sum_flat = np.ascontiguousarray(self.lambda_sum.reshape(-1))
        lambda_sum_wp = wp.array(lambda_sum_flat, dtype=wp.float32, device=device)

        constraint_values = constraint_values_wp.numpy().reshape(n_edges, 6)
        self.constraint_values[:, :] = constraint_values
        if constraint_values.size > 0:
            self.last_constraint_max = float(np.max(np.linalg.norm(constraint_values, axis=1)))
        else:
            self.last_constraint_max = 0.0

        self.jacobian_pos = jacobian_pos_wp.numpy().reshape(n_edges, 6, 6)
        self.jacobian_rot = jacobian_rot_wp.numpy().reshape(n_edges, 6, 6)

        if use_banded_cholesky:
            ab_wp = wp.zeros((BAND_LDAB, n_dofs), dtype=wp.float32, device=device)
            wp.launch(
                _warp_assemble_jmjt_banded,
                dim=n_edges,
                inputs=[jacobian_pos_wp, jacobian_rot_wp, compliance_wp, int(n_dofs), ab_wp],
                device=device,
            )
            rhs_wp = wp.zeros(n_dofs, dtype=wp.float32, device=device)
            wp.launch(
                _warp_build_rhs,
                dim=n_dofs,
                inputs=[constraint_values_wp, compliance_wp, lambda_sum_wp, int(n_dofs), rhs_wp],
                device=device,
            )
            wp.launch(
                _warp_spbsv_u11_1rhs,
                dim=1,
                inputs=[int(n_dofs), ab_wp, rhs_wp],
                device=device,
            )
            delta_lambda = rhs_wp.numpy()
        elif n_dofs <= TILE:
            A_wp = wp.zeros((TILE, TILE), dtype=wp.float32, device=device)
            wp.launch(
                _warp_assemble_jmjt_dense,
                dim=n_edges,
                inputs=[jacobian_pos_wp, jacobian_rot_wp, compliance_wp, int(n_dofs), A_wp],
                device=device,
            )
            rhs_wp = wp.zeros(TILE, dtype=wp.float32, device=device)
            wp.launch(
                _warp_build_rhs,
                dim=TILE,
                inputs=[constraint_values_wp, compliance_wp, lambda_sum_wp, int(n_dofs), rhs_wp],
                device=device,
            )
            if n_dofs < TILE:
                wp.launch(
                    _warp_pad_diagonal,
                    dim=TILE,
                    inputs=[A_wp, int(n_dofs), int(TILE)],
                    device=device,
                )
            delta_lambda_wp = wp.zeros(TILE, dtype=wp.float32, device=device)
            wp.launch_tiled(
                _warp_cholesky_solve_tile,
                dim=[1, 1],
                inputs=[A_wp, rhs_wp],
                outputs=[delta_lambda_wp],
                block_dim=BLOCK_DIM,
                device=device,
            )
            delta_lambda = delta_lambda_wp.numpy()[:n_dofs]
        else:
            diag_blocks_wp = wp.zeros(n_edges * 36, dtype=wp.float32, device=device)
            offdiag_blocks_wp = wp.zeros(n_edges * 36, dtype=wp.float32, device=device)
            wp.launch(
                _warp_assemble_jmjt_blocks,
                dim=n_edges,
                inputs=[
                    jacobian_pos_wp,
                    jacobian_rot_wp,
                    compliance_wp,
                    int(n_edges),
                    diag_blocks_wp,
                    offdiag_blocks_wp,
                ],
                device=device,
            )
            rhs_wp = wp.zeros(n_dofs, dtype=wp.float32, device=device)
            wp.launch(
                _warp_build_rhs,
                dim=n_dofs,
                inputs=[constraint_values_wp, compliance_wp, lambda_sum_wp, int(n_dofs), rhs_wp],
                device=device,
            )
            c_blocks_wp = wp.zeros(n_edges * 36, dtype=wp.float32, device=device)
            d_prime_wp = wp.zeros(n_edges * 6, dtype=wp.float32, device=device)
            delta_lambda_wp = wp.zeros(n_edges * 6, dtype=wp.float32, device=device)
            wp.launch(
                _warp_block_thomas_solve,
                dim=1,
                inputs=[
                    diag_blocks_wp,
                    offdiag_blocks_wp,
                    rhs_wp,
                    int(n_edges),
                    c_blocks_wp,
                    d_prime_wp,
                    delta_lambda_wp,
                ],
                device=device,
            )
            delta_lambda = delta_lambda_wp.numpy()

        self.last_delta_lambda_max = float(np.max(np.abs(delta_lambda))) if delta_lambda.size > 0 else 0.0
        self.lambda_sum += delta_lambda.reshape(n_edges, 6)

        inv_masses = self.inv_masses
        corr_max = 0.0
        for i in range(n_edges):
            dl = delta_lambda[6 * i : 6 * i + 6]
            J_pos = self.jacobian_pos[i]
            J_rot = self.jacobian_rot[i]
            J_p0 = J_pos[:, 0:3]
            J_p1 = J_pos[:, 3:6]
            J_t0 = J_rot[:, 0:3]
            J_t1 = J_rot[:, 3:6]

            inv_m0 = inv_masses[i]
            inv_m1 = inv_masses[i + 1]

            if inv_m0 > 0.0:
                dp0 = inv_m0 * (J_p0.T @ dl)
                self.predicted_positions[i, 0:3] += dp0
                corr_max = max(corr_max, float(np.linalg.norm(dp0)))
            if inv_m1 > 0.0:
                dp1 = inv_m1 * (J_p1.T @ dl)
                self.predicted_positions[i + 1, 0:3] += dp1
                corr_max = max(corr_max, float(np.linalg.norm(dp1)))

            if self.quat_inv_masses[i] > 0.0:
                dtheta0 = 1.0 * (J_t0.T @ dl)
                self._apply_quaternion_correction_g(self.predicted_orientations, i, dtheta0)
                corr_max = max(corr_max, float(np.linalg.norm(dtheta0)))
            if self.quat_inv_masses[i + 1] > 0.0:
                dtheta1 = 1.0 * (J_t1.T @ dl)
                self._apply_quaternion_correction_g(self.predicted_orientations, i + 1, dtheta1)
                corr_max = max(corr_max, float(np.linalg.norm(dtheta1)))

        self.last_correction_max = corr_max

    def _update_cross_section_properties(self):
        radius = np.float32(self.rod_radius)
        self.cross_section_area = np.float32(np.pi) * radius * radius
        self.second_moment_area = np.float32(np.pi) * radius**4 / np.float32(4.0)
        self.polar_moment = np.float32(np.pi) * radius**4 / np.float32(2.0)

    def _step_enabled(self, step_name: str) -> bool:
        return self._use_warp_step(step_name) or self.numpy_enabled.get(step_name, False)

    def _requires_native_constraint_pipeline(self) -> bool:
        if not self.use_banded:
            return not self._step_enabled("project_direct")
        return not (
            self._step_enabled("update_constraints_banded")
            and self._step_enabled("compute_jacobians_banded")
            and self._step_enabled("assemble_jmjt_banded")
            and self._step_enabled("project_jmjt_banded")
        )

    def _numpy_prepare_constraints(self, dt: float):
        self.lambdas.fill(0.0)
        self.lambda_sum.fill(0.0)

        self.current_rest_lengths = self.rest_lengths.copy()
        self.current_rest_darboux[:, 0] = self.rest_darboux[:, 0]
        self.current_rest_darboux[:, 1] = self.rest_darboux[:, 1]
        self.current_rest_darboux[:, 2] = self.rest_darboux[:, 2]

        dt2 = np.float32(dt * dt)
        eps = np.float32(1.0e-10)

        E = np.float32(self.young_modulus)
        G = np.float32(self.torsion_modulus)
        A = self.cross_section_area
        # I = self.second_moment_area  # Unused in C++ reference Prepare logic
        # J = self.polar_moment        # Unused in C++ reference Prepare logic

        L = self.current_rest_lengths.astype(np.float32)
        inv_L = np.where(L > 0.0, np.float32(1.0) / L, np.float32(0.0))

        # C++ Reference "DirectElasticRod" logic:
        # 1. PrepareDirectElasticRodConstraints overwrites stiffness K with (Modulus * Slider).
        #    It ignores geometric factors (I, J, 1/L) which are physically required.
        # 2. Compliance is calculated as: alpha = 1 / (K * dt^2 * L).
        #    This scales with 1/L instead of L (as expected for alpha = L/EI).
        # We match this behavior to reproduce the reference simulation.

        k_bend1_ref = E * self.bend_stiffness[:, 0]
        k_bend2_ref = E * self.bend_stiffness[:, 1]
        k_twist_ref = G * self.bend_stiffness[:, 2]

        # Effective stiffness for compliance = 1 / (K_eff * dt^2)
        # 1 / (K_ref * dt^2 * L) = 1 / ( (K_ref * L) * dt^2 )
        k_bend1_eff = k_bend1_ref * L
        k_bend2_eff = k_bend2_ref * L
        k_twist_eff = k_twist_ref * L

        # C++ Reference uses hardcoded regularization for stretch compliance (~1e-12 * inv_dt^2).
        # It does NOT use physical stiffness (EA/L) for stretch.
        # This makes the stretch constraint effectively rigid.
        # We assume dt approx 1/60s -> inv_dt^2 ~ 3600. 1e-12 * 3600 = 3.6e-9.
        # Using a fixed small epsilon matches this "rigid" behavior better than physical stiffness.
        stretch_compliance_val = np.float32(1.0e-10)

        self.compliance[:, 0] = stretch_compliance_val
        self.compliance[:, 1] = stretch_compliance_val
        self.compliance[:, 2] = stretch_compliance_val
        self.compliance[:, 3] = np.float32(1.0) / (k_bend1_eff * dt2 + eps)
        self.compliance[:, 4] = np.float32(1.0) / (k_bend2_eff * dt2 + eps)
        self.compliance[:, 5] = np.float32(1.0) / (k_twist_eff * dt2 + eps)

    def _numpy_update_constraints_banded(self):
        positions = self.predicted_positions[:, 0:3]
        orientations = self.predicted_orientations
        rest_lengths = self.current_rest_lengths
        rest_darboux = self.current_rest_darboux

        max_constraint = 0.0
        for i in range(self.num_edges):
            p0 = positions[i]
            p1 = positions[i + 1]
            q0 = orientations[i]
            q1 = orientations[i + 1]

            L = rest_lengths[i]
            half_L = np.float32(0.5) * L

            # Local offsets to midpoint (assuming rod aligns with local Z)
            r0_local = np.array([0.0, 0.0, half_L], dtype=np.float32)
            r1_local = np.array([0.0, 0.0, -half_L], dtype=np.float32)

            r0_world = self._numpy_quat_rotate_vector(q0, r0_local)
            r1_world = self._numpy_quat_rotate_vector(q1, r1_local)

            c0 = p0 + r0_world
            c1 = p1 + r1_world

            # Stretch violation in World Space (p0 + r0) - (p1 + r1)
            # C++ uses: connector0 - connector1
            stretch_error = c0 - c1

            self.constraint_values[i, 0:3] = stretch_error

            q_rel = self._numpy_quat_mul_single(self._numpy_quat_conjugate(q0), q1)
            omega = q_rel[:3]
            darboux_error = omega - rest_darboux[i]
            self.constraint_values[i, 3:6] = darboux_error
            max_constraint = max(max_constraint, float(np.linalg.norm(self.constraint_values[i])))

        self.last_constraint_max = max_constraint

    def _numpy_project_direct(self):
        n_edges = self.num_edges
        if n_edges == 0:
            return

        self._numpy_update_constraints_banded()
        self._numpy_compute_jacobians_direct()

        n_dofs = 6 * n_edges
        A = np.zeros((n_dofs, n_dofs), dtype=np.float32)
        rhs = (-self.constraint_values).reshape(n_dofs)
        # C++ Non-Banded solver uses lambda sum (XPBD)
        rhs -= (self.compliance * self.lambda_sum).reshape(n_dofs)

        inv_masses = self.inv_masses
        
        # C++ Banded Solver Implicit Assumptions:
        # The JMJT assembly (J * JT) implicitly assumes Unit Mass (1.0) and Unit Inertia (1.0)
        # for the system matrix construction.
        # We replicate this by using 1.0 for masses/inertia in the LHS assembly loop.
        # The actual masses are applied only during the final correction step.
        inv_masses_lhs = np.ones_like(inv_masses)
        # However, static points (inv_mass == 0) should remain static in LHS too to avoid singular matrix?
        # Actually C++ uses J*JT and subtracts compliance from diagonal.
        # If mass is infinite (static), PBD usually handles it by 0 invMass.
        # But C++ Banded Code does NOT check for static mass in AssembleJMJT!
        # It iterates all constraints and adds J*JT.
        # So effectively it treats even static points as dynamic with mass=1 in the system solve,
        # but then multiplies by 0 correction at the end.
        inv_masses_lhs[:] = 1.0
        
        # Similar for inertia: C++ uses skew matrices which implies I_inv = Identity (1.0).
        inv_I_lhs = np.ones_like(self.quat_inv_masses)

        # Actual inverse inertia for correction step (matches C++ hardcoded 0.1 inertia -> inv=10.0)
        inv_I_correction = self.quat_inv_masses * np.float32(10.0)

        for i in range(n_edges):
            J_pos = self.jacobian_pos[i]
            J_rot = self.jacobian_rot[i]
            J_p0 = J_pos[:, 0:3]
            J_p1 = J_pos[:, 3:6]
            J_t0 = J_rot[:, 0:3]
            J_t1 = J_rot[:, 3:6]

            inv_m0 = inv_masses_lhs[i]
            inv_m1 = inv_masses_lhs[i + 1]
            inv_I0 = inv_I_lhs[i]
            inv_I1 = inv_I_lhs[i + 1]

            JMJT = (
                inv_m0 * (J_p0 @ J_p0.T)
                + inv_m1 * (J_p1 @ J_p1.T)
                + inv_I0 * (J_t0 @ J_t0.T)
                + inv_I1 * (J_t1 @ J_t1.T)
            )
            JMJT += np.diag(self.compliance[i])

            block = slice(6 * i, 6 * i + 6)
            A[block, block] += JMJT

            if i > 0:
                J_pos_prev = self.jacobian_pos[i - 1]
                J_rot_prev = self.jacobian_rot[i - 1]
                J_p1_prev = J_pos_prev[:, 3:6]
                J_t1_prev = J_rot_prev[:, 3:6]

                coupling = inv_m0 * (J_p1_prev @ J_p0.T) + inv_I0 * (J_t1_prev @ J_t0.T)
                prev_block = slice(6 * (i - 1), 6 * (i - 1) + 6)
                A[prev_block, block] += coupling
                A[block, prev_block] += coupling.T

        try:
            delta_lambda = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            delta_lambda = np.linalg.lstsq(A, rhs, rcond=None)[0]

        self.last_delta_lambda_max = float(np.max(np.abs(delta_lambda))) if delta_lambda.size > 0 else 0.0
        self.lambda_sum += delta_lambda.reshape(n_edges, 6)

        corr_max = 0.0
        for i in range(n_edges):
            dl = delta_lambda[6 * i : 6 * i + 6]
            J_pos = self.jacobian_pos[i]
            J_rot = self.jacobian_rot[i]
            J_p0 = J_pos[:, 0:3]
            J_p1 = J_pos[:, 3:6]
            J_t0 = J_rot[:, 0:3]
            J_t1 = J_rot[:, 3:6]

            # Use ACTUAL masses for correction
            inv_m0 = inv_masses[i]
            inv_m1 = inv_masses[i + 1]
            
            # C++ Correction: "corr_q[i].coeffs() = G * deltaLambdaBendingAndTorsion"
            # This implies correction is J_rot^T * lambda * inv_I_effective
            # where inv_I_effective seems to be 1.0 (Identity) because G maps R^3 to R^4 directly without inertia scaling?
            # Wait, computeMatrixG is just kinematic map.
            # If C++ does `corr_q = G * torque`, it implies angular velocity `w = torque`.
            # This means `inv_I = 1`.
            # BUT `m_inertiaTensor` was 0.1...
            # In `solveJMJT...`: `corr_q[i] = deltaQSoln`.
            # It seems the correction ignores `rot_inv_mass_scale`?
            # Let's try using inv_I = 1.0 for correction as well to match "G * dL" directly.
            inv_I0 = 1.0 # inv_I_correction[i] 
            inv_I1 = 1.0 # inv_I_correction[i + 1]
            
            # Actually, `inv_masses` are passed to C++ Project function.
            # But inertia is NOT passed. It is internal.
            # And Banded Solver `solveJMJT` does NOT use `m_Segments[i].m_inertiaTensor`.
            # So it effectively uses Identity inertia.
            
            if inv_m0 > 0.0:
                dp0 = inv_m0 * (J_p0.T @ dl)
                self.predicted_positions[i, 0:3] += dp0
                corr_max = max(corr_max, float(np.linalg.norm(dp0)))
            if inv_m1 > 0.0:
                dp1 = inv_m1 * (J_p1.T @ dl)
                self.predicted_positions[i + 1, 0:3] += dp1
                corr_max = max(corr_max, float(np.linalg.norm(dp1)))

            # Static points have inv_mass=0 so we don't update them.
            # But what about rotation? `quat_inv_masses` handles static root.
            # If root is static (inv_mass=0), we should check that.
            
            # Use self.quat_inv_masses as mask (0 or 1), multiplied by 1.0 effective inertia
            if self.quat_inv_masses[i] > 0.0:
                dtheta0 = 1.0 * (J_t0.T @ dl)
                self._apply_quaternion_correction_g(self.predicted_orientations, i, dtheta0)
                corr_max = max(corr_max, float(np.linalg.norm(dtheta0)))
            
            if self.quat_inv_masses[i+1] > 0.0:
                dtheta1 = 1.0 * (J_t1.T @ dl)
                self._apply_quaternion_correction_g(self.predicted_orientations, i + 1, dtheta1)
                corr_max = max(corr_max, float(np.linalg.norm(dtheta1)))

        self.last_correction_max = corr_max

    def _numpy_compute_jacobians_direct(self):
        n_edges = self.num_edges
        if n_edges == 0:
            return

        if not hasattr(self, "jacobian_pos") or self.jacobian_pos.shape[0] != n_edges:
            self.jacobian_pos = np.zeros((n_edges, 6, 6), dtype=np.float32)
            self.jacobian_rot = np.zeros((n_edges, 6, 6), dtype=np.float32)

        orientations = self.predicted_orientations
        rest_lengths = self.current_rest_lengths

        for i in range(n_edges):
            q0 = orientations[i]
            q1 = orientations[i + 1]
            L = rest_lengths[i]
            half_L = np.float32(0.5) * L

            # Local offsets
            r0_local = np.array([0.0, 0.0, half_L], dtype=np.float32)
            r1_local = np.array([0.0, 0.0, -half_L], dtype=np.float32)

            r0_world = self._numpy_quat_rotate_vector(q0, r0_local)
            r1_world = self._numpy_quat_rotate_vector(q1, r1_local)

            # Stretch constraint C = (p0 + r0) - (p1 + r1)
            # J_p0 = I, J_p1 = -I
            # J_q0 = -skew(r0), J_q1 = skew(r1)  (Note signs: r1 is in negative term but q1 adds to it? Wait)
            # C = p0 + R0 r0 - p1 - R1 r1
            # dC/dp0 = I
            # dC/dp1 = -I
            # dC/dtheta0 = -skew(R0 r0)
            # dC/dtheta1 = -(-skew(R1 r1)) = skew(R1 r1)

            I3 = np.eye(3, dtype=np.float32)
            
            self.jacobian_pos[i, 0:3, 0:3] = I3
            self.jacobian_pos[i, 0:3, 3:6] = -I3

            r0_skew = self._numpy_skew_symmetric(r0_world)
            r1_skew = self._numpy_skew_symmetric(r1_world)

            self.jacobian_rot[i, 0:3, 0:3] = -r0_skew
            self.jacobian_rot[i, 0:3, 3:6] = r1_skew

            jomega0, jomega1 = self._numpy_compute_bending_torsion_jacobians(q0, q1)
            g0 = self._numpy_compute_matrix_g(q0)
            g1 = self._numpy_compute_matrix_g(q1)
            self.jacobian_rot[i, 3:6, 0:3] = (jomega0 @ g0).astype(np.float32)
            self.jacobian_rot[i, 3:6, 3:6] = (jomega1 @ g1).astype(np.float32)

            self.jacobian_pos[i, 3:6, 0:3] = 0.0
            self.jacobian_pos[i, 3:6, 3:6] = 0.0

    def _numpy_assemble_jmjt_banded(self):
        n_edges = self.num_edges
        if n_edges == 0:
            return
        n_dofs = 6 * n_edges
        bandwidth = 6
        self.bandwidth = bandwidth

        if not hasattr(self, "A_banded") or self.A_banded.shape[1] != n_dofs:
            self.A_banded = np.zeros((2 * bandwidth + 1, n_dofs), dtype=np.float32)
            self.rhs = np.zeros(n_dofs, dtype=np.float32)

        self.A_banded.fill(0.0)

        # C++ Banded Solver Implicit Assumptions:
        # The JMJT assembly (J * JT) implicitly assumes Unit Mass (1.0) and Unit Inertia (1.0)
        # for the system matrix construction.
        inv_masses = np.ones(self.num_points, dtype=np.float32)
        inv_I = np.ones(self.num_points, dtype=np.float32)

        for i in range(n_edges):
            J_pos = self.jacobian_pos[i]
            J_rot = self.jacobian_rot[i]
            J_p0 = J_pos[:, 0:3]
            J_p1 = J_pos[:, 3:6]
            J_t0 = J_rot[:, 0:3]
            J_t1 = J_rot[:, 3:6]

            inv_m0 = inv_masses[i]
            inv_m1 = inv_masses[i + 1]
            inv_I0 = inv_I[i]
            inv_I1 = inv_I[i + 1]

            JMJT = (
                inv_m0 * (J_p0 @ J_p0.T)
                + inv_m1 * (J_p1 @ J_p1.T)
                + inv_I0 * (J_t0 @ J_t0.T)
                + inv_I1 * (J_t1 @ J_t1.T)
            )
            JMJT += np.diag(self.compliance[i])

            block_start = 6 * i
            for row in range(6):
                for col in range(6):
                    global_row = block_start + row
                    global_col = block_start + col
                    band_row = bandwidth + global_row - global_col
                    if 0 <= band_row < 2 * bandwidth + 1:
                        self.A_banded[band_row, global_col] += JMJT[row, col]

            if i > 0:
                J_pos_prev = self.jacobian_pos[i - 1]
                J_rot_prev = self.jacobian_rot[i - 1]
                J_p1_prev = J_pos_prev[:, 3:6]
                J_t1_prev = J_rot_prev[:, 3:6]
                coupling = inv_m0 * (J_p1_prev @ J_p0.T) + inv_I0 * (J_t1_prev @ J_t0.T)

                prev_block = 6 * (i - 1)
                for row in range(6):
                    for col in range(6):
                        global_row = prev_block + row
                        global_col = block_start + col
                        band_row = bandwidth + global_row - global_col
                        if 0 <= band_row < 2 * bandwidth + 1:
                            self.A_banded[band_row, global_col] += coupling[row, col]

                        global_row = block_start + col
                        global_col = prev_block + row
                        band_row = bandwidth + global_row - global_col
                        if 0 <= band_row < 2 * bandwidth + 1:
                            self.A_banded[band_row, global_col] += coupling[row, col]

    def _numpy_project_jmjt_banded(self):
        n_edges = self.num_edges
        if n_edges == 0:
            return

        n_dofs = 6 * n_edges
        self.rhs[:n_dofs] = (-self.constraint_values).reshape(n_dofs)

        # C++ Banded Solver Regularization:
        # The C++ spbsv_u11_1rhs solver explicitly regularizes the Cholesky factor diagonal
        # if it drops below 1e-6. This prevents instability at high stiffness (singular A).
        # We mimic this by adding a small epsilon to the diagonal of A_banded before solving.
        # This is critical for stability when compliance -> 0.
        regularization = np.float32(1.0e-6)
        self.A_banded[self.bandwidth, :n_dofs] += regularization

        try:
            from scipy.linalg import solve_banded  # noqa: PLC0415

            delta_lambda = solve_banded(
                (self.bandwidth, self.bandwidth),
                self.A_banded,
                self.rhs[:n_dofs],
                overwrite_ab=False,
                overwrite_b=False,
            )
        except Exception:
            A_dense = np.zeros((n_dofs, n_dofs), dtype=np.float32)
            for col in range(n_dofs):
                for band_row in range(2 * self.bandwidth + 1):
                    row = col + band_row - self.bandwidth
                    if 0 <= row < n_dofs:
                        A_dense[row, col] = self.A_banded[band_row, col]
            delta_lambda = np.linalg.solve(A_dense, self.rhs[:n_dofs])

        inv_masses = self.inv_masses
        # inv_I removed (unused)

        self.last_delta_lambda_max = float(np.max(np.abs(delta_lambda))) if delta_lambda.size > 0 else 0.0
        corr_max = 0.0
        for i in range(n_edges):
            dl = delta_lambda[6 * i : 6 * i + 6]
            J_pos = self.jacobian_pos[i]
            J_rot = self.jacobian_rot[i]
            J_p0 = J_pos[:, 0:3]
            J_p1 = J_pos[:, 3:6]
            J_t0 = J_rot[:, 0:3]
            J_t1 = J_rot[:, 3:6]

            # Use ACTUAL masses for position correction, but UNIT inertia for rotation (matching C++)
            inv_m0 = inv_masses[i]
            inv_m1 = inv_masses[i + 1]
            # inv_I0 = inv_I[i] # Unused, we use 1.0 masked by quat_inv_masses
            # inv_I1 = inv_I[i + 1]

            if inv_m0 > 0.0:
                dp0 = inv_m0 * (J_p0.T @ dl)
                self.predicted_positions[i, 0:3] += dp0
                corr_max = max(corr_max, float(np.linalg.norm(dp0)))
            if inv_m1 > 0.0:
                dp1 = inv_m1 * (J_p1.T @ dl)
                self.predicted_positions[i + 1, 0:3] += dp1
                corr_max = max(corr_max, float(np.linalg.norm(dp1)))
            
            # C++ applies rotation correction without inertia scaling (effectively I=1)
            # We must respect static mask (quat_inv_masses > 0)
            if self.quat_inv_masses[i] > 0.0:
                dtheta0 = 1.0 * (J_t0.T @ dl)
                self._apply_quaternion_correction_g(self.predicted_orientations, i, dtheta0)
                corr_max = max(corr_max, float(np.linalg.norm(dtheta0)))
            if self.quat_inv_masses[i+1] > 0.0:
                dtheta1 = 1.0 * (J_t1.T @ dl)
                self._apply_quaternion_correction_g(self.predicted_orientations, i + 1, dtheta1)
                corr_max = max(corr_max, float(np.linalg.norm(dtheta1)))

        self.last_correction_max = corr_max

    @staticmethod
    def _numpy_skew_symmetric(v: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _numpy_compute_bending_torsion_jacobians(q0: np.ndarray, q1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x0, y0, z0, w0 = q0
        x1, y1, z1, w1 = q1
        jomega0 = np.array(
            [
                [-w1, -z1, y1, x1],
                [z1, -w1, -x1, y1],
                [-y1, x1, -w1, z1],
            ],
            dtype=np.float32,
        )
        jomega1 = np.array(
            [
                [w0, z0, -y0, -x0],
                [-z0, w0, x0, -y0],
                [y0, -x0, w0, -z0],
            ],
            dtype=np.float32,
        )
        return jomega0, jomega1

    @staticmethod
    def _numpy_compute_matrix_g(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        return np.array(
            [
                [0.5 * w, 0.5 * z, -0.5 * y],
                [-0.5 * z, 0.5 * w, 0.5 * x],
                [0.5 * y, -0.5 * x, 0.5 * w],
                [-0.5 * x, -0.5 * y, -0.5 * z],
            ],
            dtype=np.float32,
        )

    def _apply_quaternion_correction(self, orientations: np.ndarray, idx: int, dtheta: np.ndarray):
        if np.linalg.norm(dtheta) < 1.0e-10:
            return
        q = orientations[idx]
        dq = np.array([0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2], 0.0], dtype=np.float32)
        q_new = q + self._numpy_quat_mul_single(dq, q)
        q_new /= np.linalg.norm(q_new)
        orientations[idx] = q_new

    @staticmethod
    def _apply_quaternion_correction_g(orientations: np.ndarray, idx: int, dtheta: np.ndarray):
        if np.linalg.norm(dtheta) < 1.0e-10:
            return
        q = orientations[idx]
        g = NumpyDirectRodState._numpy_compute_matrix_g(q)
        corr_q = (g @ dtheta).astype(np.float32)
        q_new = q + corr_q
        q_new /= np.linalg.norm(q_new)
        orientations[idx] = q_new

    @staticmethod
    def _numpy_quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        vx, vy, vz = v

        tx = np.float32(2.0) * (y * vz - z * vy)
        ty = np.float32(2.0) * (z * vx - x * vz)
        tz = np.float32(2.0) * (x * vy - y * vx)

        return np.array(
            [
                vx + w * tx + y * tz - z * ty,
                vy + w * ty + z * tx - x * tz,
                vz + w * tz + x * ty - y * tx,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _numpy_quat_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

    @staticmethod
    def _numpy_quat_mul_single(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _numpy_quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        return np.stack(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            axis=1,
        ).astype(np.float32)

    @staticmethod
    def _numpy_quat_normalize(quats: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        norms = np.where(norms < 1.0e-8, 1.0, norms)
        normalized = quats / norms
        if mask is None:
            return normalized
        result = quats.copy()
        result[mask] = normalized[mask]
        return result

class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.substeps = args.substeps
        self.linear_damping = args.linear_damping
        self.angular_damping = args.angular_damping
        self.bend_stiffness = args.bend_stiffness
        self.twist_stiffness = args.twist_stiffness
        self.rest_bend_d1 = args.rest_bend_d1
        self.rest_bend_d2 = args.rest_bend_d2
        self.rest_twist = args.rest_twist
        self.young_modulus_scale = args.young_modulus / 1.0e6
        self.torsion_modulus_scale = args.torsion_modulus / 1.0e6
        self.use_banded = args.use_banded
        self.compare_offset = args.compare_offset
        half_offset = 0.5 * self.compare_offset
        self.ref_offset = np.array([0.0, -half_offset, 0.0], dtype=np.float32)
        self.numpy_offset = np.array([0.0, half_offset, 0.0], dtype=np.float32)

        self.base_gravity = np.array(args.gravity, dtype=np.float32)
        self.gravity_enabled = True
        self.gravity_scale = 1.0
        self.floor_collision_enabled = True
        self.floor_height = 0.0
        self.floor_restitution = 0.0

        self.show_segments = True
        self.show_directors = False
        self.director_scale = 0.1

        self.root_move_speed = 1.0
        self.root_rotate_speed = 1.0
        self.root_rotation = 0.0

        self._gravity_key_was_down = False
        self._reset_key_was_down = False
        self._banded_key_was_down = False
        self._lock_key_was_down = False

        self.lib = DefKitDirectLibrary(args.dll_path, args.calling_convention)
        self.supports_non_banded = self.lib.ProjectDirectElasticRodConstraints is not None
        if not self.supports_non_banded:
            self.use_banded = True

        rod_radius = args.rod_radius if args.rod_radius is not None else args.particle_radius
        self.ref_rod = DefKitDirectRodState(
            lib=self.lib,
            num_points=args.num_points,
            segment_length=args.segment_length,
            mass=args.particle_mass,
            particle_height=args.particle_height,
            rod_radius=rod_radius,
            bend_stiffness=self.bend_stiffness,
            twist_stiffness=self.twist_stiffness,
            rest_bend_d1=self.rest_bend_d1,
            rest_bend_d2=self.rest_bend_d2,
            rest_twist=self.rest_twist,
            young_modulus=args.young_modulus,
            torsion_modulus=args.torsion_modulus,
            gravity=self.base_gravity,
            lock_root_rotation=args.lock_root_rotation,
            use_banded=self.use_banded,
        )
        self.numpy_rod = NumpyDirectRodState(
            lib=self.lib,
            num_points=args.num_points,
            segment_length=args.segment_length,
            mass=args.particle_mass,
            particle_height=args.particle_height,
            rod_radius=rod_radius,
            bend_stiffness=self.bend_stiffness,
            twist_stiffness=self.twist_stiffness,
            rest_bend_d1=self.rest_bend_d1,
            rest_bend_d2=self.rest_bend_d2,
            rest_twist=self.rest_twist,
            young_modulus=args.young_modulus,
            torsion_modulus=args.torsion_modulus,
            gravity=self.base_gravity,
            lock_root_rotation=args.lock_root_rotation,
            use_banded=self.use_banded,
        )

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        for i in range(args.num_points):
            mass = 0.0 if i == 0 else args.particle_mass
            ref_pos = tuple(self.ref_rod.positions[i, 0:3] + self.ref_offset)
            builder.add_particle(pos=ref_pos, vel=(0.0, 0.0, 0.0), mass=mass, radius=args.particle_radius)
        for i in range(args.num_points):
            mass = 0.0 if i == 0 else args.particle_mass
            numpy_pos = tuple(self.numpy_rod.positions[i, 0:3] + self.numpy_offset)
            builder.add_particle(pos=numpy_pos, vel=(0.0, 0.0, 0.0), mass=mass, radius=args.particle_radius)

        self.model = builder.finalize()
        self.state = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        self._ref_segment_colors = np.tile(np.array([0.2, 0.6, 1.0], dtype=np.float32), (args.num_points - 1, 1))
        self._numpy_segment_colors = np.tile(np.array([1.0, 0.6, 0.2], dtype=np.float32), (args.num_points - 1, 1))
        self._numpy_step_labels = {
            "predict_positions": "Predict Positions",
            "integrate_positions": "Integrate Positions",
            "predict_rotations": "Predict Rotations",
            "integrate_rotations": "Integrate Rotations",
            "prepare_constraints": "Prepare Constraints",
            "update_constraints_banded": "Update Constraints (banded)",
            "compute_jacobians_banded": "Compute Jacobians (banded)",
            "assemble_jmjt_banded": "Assemble JMJT (banded)",
            "project_jmjt_banded": "Project JMJT (banded)",
            "project_direct": "Project Direct (non-banded)",
        }
        self._warp_step_labels = dict(self._numpy_step_labels)

        self._sync_state_from_rods()
        self._update_gravity()
        self._ref_root_base_orientation = self.ref_rod.orientations[0].copy()
        self._numpy_root_base_orientation = self.numpy_rod.orientations[0].copy()

    def __del__(self):
        if hasattr(self, "ref_rod"):
            self.ref_rod.destroy()
        if hasattr(self, "numpy_rod"):
            self.numpy_rod.destroy()

    def _update_gravity(self):
        if self.gravity_enabled:
            gravity = self.base_gravity * self.gravity_scale
        else:
            gravity = np.zeros(3, dtype=np.float32)
        self.ref_rod.set_gravity(gravity)
        self.numpy_rod.set_gravity(gravity)

    def _sync_state_from_rods(self):
        ref_positions = self.ref_rod.positions[:, 0:3].astype(np.float32) + self.ref_offset
        numpy_positions = self.numpy_rod.positions[:, 0:3].astype(np.float32) + self.numpy_offset
        ref_velocities = self.ref_rod.velocities[:, 0:3].astype(np.float32)
        numpy_velocities = self.numpy_rod.velocities[:, 0:3].astype(np.float32)

        positions = np.vstack([ref_positions, numpy_positions])
        velocities = np.vstack([ref_velocities, numpy_velocities])

        self.state.particle_q.assign(wp.array(positions, dtype=wp.vec3, device=self.model.device))
        self.state.particle_qd.assign(wp.array(velocities, dtype=wp.vec3, device=self.model.device))

    def _handle_keyboard_input(self):
        if not hasattr(self.viewer, "is_key_down"):
            return

        try:
            import pyglet.window.key as key
        except ImportError:
            return

        g_down = self.viewer.is_key_down(key.G)
        if g_down and not self._gravity_key_was_down:
            self.gravity_enabled = not self.gravity_enabled
            self._update_gravity()
        self._gravity_key_was_down = g_down

        b_down = self.viewer.is_key_down(key.B)
        if b_down and not self._banded_key_was_down:
            if self.supports_non_banded:
                self.use_banded = not self.use_banded
                self.ref_rod.set_solver_mode(self.use_banded)
                self.numpy_rod.set_solver_mode(self.use_banded)
                self.use_banded = self.ref_rod.use_banded
        self._banded_key_was_down = b_down

        l_down = self.viewer.is_key_down(key.L)
        if l_down and not self._lock_key_was_down:
            self.ref_rod.toggle_root_lock()
            self.numpy_rod.toggle_root_lock()
        self._lock_key_was_down = l_down

        r_down = self.viewer.is_key_down(key.R)
        if r_down and not self._reset_key_was_down:
            self.ref_rod.reset()
            self.numpy_rod.reset()
            self.root_rotation = 0.0
            self._apply_root_rotation()
            self.sim_time = 0.0
            self._sync_state_from_rods()
        self._reset_key_was_down = r_down

        dx = 0.0
        dy = 0.0
        dz = 0.0

        if self.viewer.is_key_down(key.NUM_6):
            dx += self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_4):
            dx -= self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_8):
            dy += self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_2):
            dy -= self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_9):
            dz += self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_3):
            dz -= self.root_move_speed * self.frame_dt

        rotation_changed = False
        if self.viewer.is_key_down(key.NUM_7):
            self.root_rotation += self.root_rotate_speed * self.frame_dt
            rotation_changed = True
        if self.viewer.is_key_down(key.NUM_1):
            self.root_rotation -= self.root_rotate_speed * self.frame_dt
            rotation_changed = True

        if dx != 0.0 or dy != 0.0 or dz != 0.0:
            self._apply_root_translation(dx, dy, dz)
        if rotation_changed:
            self._apply_root_rotation()

    def step(self):
        self._handle_keyboard_input()

        sub_dt = self.frame_dt / max(self.substeps, 1)
        for _ in range(self.substeps):
            self.ref_rod.step(sub_dt, self.linear_damping, self.angular_damping)
            self.numpy_rod.step(sub_dt, self.linear_damping, self.angular_damping)
            if self.floor_collision_enabled:
                self.ref_rod.apply_floor_collisions(self.floor_height, self.floor_restitution)
                self.numpy_rod.apply_floor_collisions(self.floor_height, self.floor_restitution)

        self._sync_state_from_rods()
        self.sim_time += self.frame_dt

    def _apply_root_translation(self, dx: float, dy: float, dz: float):
        delta = np.array([dx, dy, dz], dtype=np.float32)
        for rod in (self.ref_rod, self.numpy_rod):
            pos = rod.positions[0, 0:3]
            new_pos = pos + delta
            rod.positions[0, 0:3] = new_pos
            rod.predicted_positions[0, 0:3] = new_pos
            rod.velocities[0, 0:3] = 0.0

    def _apply_root_rotation(self):
        q_twist = _quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), self.root_rotation)
        q_ref = _quat_multiply(self._ref_root_base_orientation, q_twist)
        q_numpy = _quat_multiply(self._numpy_root_base_orientation, q_twist)
        self.ref_rod.orientations[0] = q_ref
        self.ref_rod.predicted_orientations[0] = q_ref
        self.ref_rod.prev_orientations[0] = q_ref
        self.numpy_rod.orientations[0] = q_numpy
        self.numpy_rod.predicted_orientations[0] = q_numpy
        self.numpy_rod.prev_orientations[0] = q_numpy

    def _rotate_vector_by_quaternion(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        vx, vy, vz = v

        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)

        return np.array(
            [
                vx + w * tx + y * tz - z * ty,
                vy + w * ty + z * tx - x * tz,
                vz + w * tz + x * ty - y * tx,
            ],
            dtype=np.float32,
        )

    def _build_director_lines(self, rod: DefKitDirectRodState, offset: np.ndarray):
        num_edges = rod.num_points - 1
        positions = rod.positions[:, 0:3] + offset
        orientations = rod.orientations

        starts = np.zeros((num_edges * 3, 3), dtype=np.float32)
        ends = np.zeros((num_edges * 3, 3), dtype=np.float32)
        colors = np.zeros((num_edges * 3, 3), dtype=np.float32)

        for i in range(num_edges):
            midpoint = 0.5 * (positions[i] + positions[i + 1])
            q = orientations[i]

            d1 = self._rotate_vector_by_quaternion(np.array([1.0, 0.0, 0.0], dtype=np.float32), q)
            d2 = self._rotate_vector_by_quaternion(np.array([0.0, 1.0, 0.0], dtype=np.float32), q)
            d3 = self._rotate_vector_by_quaternion(np.array([0.0, 0.0, 1.0], dtype=np.float32), q)

            base = i * 3
            starts[base] = midpoint
            ends[base] = midpoint + d1 * self.director_scale
            colors[base] = [1.0, 0.0, 0.0]

            starts[base + 1] = midpoint
            ends[base + 1] = midpoint + d2 * self.director_scale
            colors[base + 1] = [0.0, 1.0, 0.0]

            starts[base + 2] = midpoint
            ends[base + 2] = midpoint + d3 * self.director_scale
            colors[base + 2] = [0.0, 0.0, 1.0]

        return starts, ends, colors

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)

        if self.show_segments:
            ref_starts = (self.ref_rod.positions[:-1, 0:3] + self.ref_offset).astype(np.float32)
            ref_ends = (self.ref_rod.positions[1:, 0:3] + self.ref_offset).astype(np.float32)
            numpy_starts = (self.numpy_rod.positions[:-1, 0:3] + self.numpy_offset).astype(np.float32)
            numpy_ends = (self.numpy_rod.positions[1:, 0:3] + self.numpy_offset).astype(np.float32)
            self.viewer.log_lines(
                "/rod_reference",
                wp.array(ref_starts, dtype=wp.vec3, device=self.model.device),
                wp.array(ref_ends, dtype=wp.vec3, device=self.model.device),
                wp.array(self._ref_segment_colors, dtype=wp.vec3, device=self.model.device),
            )
            self.viewer.log_lines(
                "/rod_numpy",
                wp.array(numpy_starts, dtype=wp.vec3, device=self.model.device),
                wp.array(numpy_ends, dtype=wp.vec3, device=self.model.device),
                wp.array(self._numpy_segment_colors, dtype=wp.vec3, device=self.model.device),
            )
        else:
            self.viewer.log_lines("/rod_reference", None, None, None)
            self.viewer.log_lines("/rod_numpy", None, None, None)

        if self.show_directors:
            ref_starts, ref_ends, ref_colors = self._build_director_lines(self.ref_rod, self.ref_offset)
            numpy_starts, numpy_ends, numpy_colors = self._build_director_lines(self.numpy_rod, self.numpy_offset)
            self.viewer.log_lines(
                "/directors_reference",
                wp.array(ref_starts, dtype=wp.vec3, device=self.model.device),
                wp.array(ref_ends, dtype=wp.vec3, device=self.model.device),
                wp.array(ref_colors, dtype=wp.vec3, device=self.model.device),
            )
            self.viewer.log_lines(
                "/directors_numpy",
                wp.array(numpy_starts, dtype=wp.vec3, device=self.model.device),
                wp.array(numpy_ends, dtype=wp.vec3, device=self.model.device),
                wp.array(numpy_colors, dtype=wp.vec3, device=self.model.device),
            )
        else:
            self.viewer.log_lines("/directors_reference", None, None, None)
            self.viewer.log_lines("/directors_numpy", None, None, None)

        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Direct Cosserat Rod: Reference + NumPy/Warp")
        ui.text(f"Particles per rod: {self.ref_rod.num_points}")
        ui.text("Reference: blue, Candidate: orange")
        ui.separator()

        _changed, self.substeps = ui.slider_int("Substeps", self.substeps, 1, 16)
        _changed, self.linear_damping = ui.slider_float("Linear Damping", self.linear_damping, 0.0, 0.05)
        _changed, self.angular_damping = ui.slider_float("Angular Damping", self.angular_damping, 0.0, 0.05)

        ui.separator()
        offset_changed, self.compare_offset = ui.slider_float("Compare Offset", self.compare_offset, 0.1, 5.0)
        if offset_changed:
            half_offset = 0.5 * self.compare_offset
            self.ref_offset = np.array([0.0, -half_offset, 0.0], dtype=np.float32)
            self.numpy_offset = np.array([0.0, half_offset, 0.0], dtype=np.float32)
            self._sync_state_from_rods()

        ui.separator()
        changed_bend_k, self.bend_stiffness = ui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        changed_twist_k, self.twist_stiffness = ui.slider_float(
            "Twist Stiffness", self.twist_stiffness, 0.0, 1.0
        )
        if changed_bend_k or changed_twist_k:
            self.ref_rod.set_bend_stiffness(self.bend_stiffness, self.twist_stiffness)
            self.numpy_rod.set_bend_stiffness(self.bend_stiffness, self.twist_stiffness)

        ui.separator()
        ui.text("Material Moduli")
        changed_young, self.young_modulus_scale = ui.slider_float(
            "Young Modulus (x1e6)", self.young_modulus_scale, 0.01, 100.0
        )
        changed_torsion, self.torsion_modulus_scale = ui.slider_float(
            "Torsion Modulus (x1e6)", self.torsion_modulus_scale, 0.01, 100.0
        )
        if changed_young or changed_torsion:
            young_modulus = float(self.young_modulus_scale) * 1.0e6
            torsion_modulus = float(self.torsion_modulus_scale) * 1.0e6
            self.ref_rod.young_modulus = young_modulus
            self.ref_rod.torsion_modulus = torsion_modulus
            self.numpy_rod.young_modulus = young_modulus
            self.numpy_rod.torsion_modulus = torsion_modulus

        ui.separator()
        ui.text("Rest Shape (Darboux Vector)")
        changed_rest_d1, self.rest_bend_d1 = ui.slider_float("Rest Bend d1", self.rest_bend_d1, -0.5, 0.5)
        changed_rest_d2, self.rest_bend_d2 = ui.slider_float("Rest Bend d2", self.rest_bend_d2, -0.5, 0.5)
        changed_rest_twist, self.rest_twist = ui.slider_float("Rest Twist", self.rest_twist, -0.5, 0.5)
        if changed_rest_d1 or changed_rest_d2 or changed_rest_twist:
            self.ref_rod.set_rest_darboux(self.rest_bend_d1, self.rest_bend_d2, self.rest_twist)
            self.numpy_rod.set_rest_darboux(self.rest_bend_d1, self.rest_bend_d2, self.rest_twist)

        ui.separator()
        gravity_changed, self.gravity_enabled = ui.checkbox("Gravity (G)", self.gravity_enabled)
        scale_changed, self.gravity_scale = ui.slider_float("Gravity Scale", self.gravity_scale, 0.0, 2.0)
        if gravity_changed or scale_changed:
            self._update_gravity()

        ui.separator()
        ui.text("Floor Collision")
        _changed, self.floor_collision_enabled = ui.checkbox(
            "Enable Floor Collision", self.floor_collision_enabled
        )
        _changed, self.floor_height = ui.slider_float("Floor Height", self.floor_height, -1.0, 1.0)
        _changed, self.floor_restitution = ui.slider_float(
            "Floor Restitution", self.floor_restitution, 0.0, 1.0
        )

        ui.separator()
        if self.supports_non_banded:
            changed_banded, self.use_banded = ui.checkbox("Use Banded Solver", self.use_banded)
            if changed_banded:
                self.ref_rod.set_solver_mode(self.use_banded)
                self.numpy_rod.set_solver_mode(self.use_banded)
                self.use_banded = self.ref_rod.use_banded
        else:
            ui.text("Non-banded solver not available in this DLL build.")

        ui.separator()
        ui.text("Warp Overrides (candidate rod)")
        ui.text("Warp overrides take priority over NumPy for the same step.")
        available_warp_steps = []
        for step_name, label in self._warp_step_labels.items():
            if self.numpy_rod.warp_available.get(step_name, False):
                available_warp_steps.append((step_name, label))
        for step_name, label in available_warp_steps:
            current = self.numpy_rod.warp_enabled.get(step_name, False)
            changed, enabled = ui.checkbox(f"{label} (Warp)", current)
            if changed:
                self.numpy_rod.set_warp_override(step_name, enabled)
        pending_warp = len(self._warp_step_labels) - len(available_warp_steps)
        if pending_warp > 0:
            ui.text(f"Pending Warp steps: {pending_warp}")

        ui.separator()
        ui.text("Direct Solve Backend (non-banded)")
        if self.use_banded:
            ui.text("Disable banded solver to use direct backend.")
        else:
            backend_labels = [
                "Warp Block-Thomas",
                "Warp Banded Cholesky",
                "CPU NumPy",
            ]
            backend_values = [
                DIRECT_SOLVE_WARP_BLOCK_THOMAS,
                DIRECT_SOLVE_WARP_BANDED_CHOLESKY,
                DIRECT_SOLVE_CPU_NUMPY,
            ]
            current_backend = self.numpy_rod.direct_solve_backend
            try:
                current_idx = backend_values.index(current_backend)
            except ValueError:
                current_idx = 0
            changed, new_idx = ui.combo("Direct Solve Backend", current_idx, backend_labels)
            if changed:
                self.numpy_rod.set_direct_solve_backend(backend_values[new_idx])
            if self.numpy_rod.direct_solve_backend == DIRECT_SOLVE_CPU_NUMPY:
                ui.text("CPU NumPy direct solve will be used for Project Direct.")
            elif self.numpy_rod.direct_solve_backend == DIRECT_SOLVE_WARP_BANDED_CHOLESKY:
                ui.text("Warp banded Cholesky (spbsv_u11_1rhs-style).")
            else:
                ui.text("Warp block-tridiagonal Thomas solve.")

        ui.separator()
        ui.text("NumPy Overrides (candidate rod)")
        available_numpy_steps = []
        for step_name, label in self._numpy_step_labels.items():
            if self.numpy_rod.numpy_available.get(step_name, False):
                available_numpy_steps.append((step_name, label))
        for step_name, label in available_numpy_steps:
            current = self.numpy_rod.numpy_enabled.get(step_name, False)
            changed, enabled = ui.checkbox(f"{label} (NumPy)", current)
            if changed:
                self.numpy_rod.set_numpy_override(step_name, enabled)
        pending_numpy = len(self._numpy_step_labels) - len(available_numpy_steps)
        if pending_numpy > 0:
            ui.text(f"Pending NumPy steps: {pending_numpy} (see DEFKIT_TO_NUMPY.md)")

        ui.separator()
        ui.text("NumPy Direct Stabilization")
        ui.text(f"NumPy max |C|: {self.numpy_rod.last_constraint_max:.3e}")
        ui.text(f"NumPy max |Δλ|: {self.numpy_rod.last_delta_lambda_max:.3e}")
        ui.text(f"NumPy max correction: {self.numpy_rod.last_correction_max:.3e}")

        ui.separator()
        _changed, self.show_segments = ui.checkbox("Show Rod Segments", self.show_segments)
        _changed, self.show_directors = ui.checkbox("Show Directors", self.show_directors)
        _changed, self.director_scale = ui.slider_float("Director Scale", self.director_scale, 0.01, 0.3)

        ui.separator()
        ui.text("Root Control (Numpad, both rods)")
        _changed, self.root_move_speed = ui.slider_float("Move Speed", self.root_move_speed, 0.1, 5.0)
        _changed, self.root_rotate_speed = ui.slider_float("Rotate Speed", self.root_rotate_speed, 0.1, 3.0)
        ui.text(f"  Rotation: {self.root_rotation:.2f} rad")
        ui.text("  4/6: X-, X+  8/2: Y+, Y-  9/3: Z+, Z-")
        ui.text("  7/1: Rotate +Z/-Z")

        ui.separator()
        ui.text("Controls:")
        ui.text("  G: Toggle gravity")
        ui.text("  B: Toggle banded solver")
        ui.text("  L: Toggle root lock (position + rotation)")
        ui.text("  R: Reset")

    def test_final(self):
        ref_anchor = self.ref_rod.positions[0, 0:3]
        ref_initial = self.ref_rod._initial_positions[0, 0:3]
        ref_dist = float(np.linalg.norm(ref_anchor - ref_initial))
        assert ref_dist < 1.0e-3, f"Reference anchor moved too far: {ref_dist}"

        numpy_anchor = self.numpy_rod.positions[0, 0:3]
        numpy_initial = self.numpy_rod._initial_positions[0, 0:3]
        numpy_dist = float(np.linalg.norm(numpy_anchor - numpy_initial))
        assert numpy_dist < 1.0e-3, f"NumPy anchor moved too far: {numpy_dist}"

        if not np.all(np.isfinite(self.ref_rod.positions[:, 0:3])):
            raise AssertionError("Non-finite reference positions detected")
        if not np.all(np.isfinite(self.numpy_rod.positions[:, 0:3])):
            raise AssertionError("Non-finite NumPy positions detected")


def create_parser():
    import argparse  # noqa: PLC0415

    parser = newton.examples.create_parser()
    parser.add_argument(
        "--dll-path",
        type=str,
        default=None,
        help="Path to DefKitAdv.dll. If omitted, attempts to load from PATH.",
    )
    parser.add_argument(
        "--calling-convention",
        type=str,
        choices=["cdecl", "stdcall"],
        default="cdecl",
        help="Calling convention used by the DLL (cdecl or stdcall).",
    )
    parser.add_argument("--num-points", type=int, default=64, help="Number of rod points.")
    parser.add_argument("--segment-length", type=float, default=0.025, help="Rest length per segment.")
    parser.add_argument("--particle-mass", type=float, default=1.0, help="Mass per particle (root fixed).")
    parser.add_argument("--particle-radius", type=float, default=0.02, help="Particle visualization radius.")
    parser.add_argument("--particle-height", type=float, default=1.0, help="Initial rod height (z).")
    parser.add_argument(
        "--rod-radius",
        type=float,
        default=None,
        help="Physical rod radius for direct solver (defaults to particle-radius).",
    )
    parser.add_argument(
        "--compare-offset",
        type=float,
        default=0.0,
        help="Y-offset separating reference and NumPy rods.",
    )
    parser.add_argument("--substeps", type=int, default=4, help="Integration substeps per frame.")
    parser.add_argument("--bend-stiffness", type=float, default=1.0, help="Per-edge bend stiffness.")
    parser.add_argument("--twist-stiffness", type=float, default=1.0, help="Per-edge twist stiffness.")
    parser.add_argument("--rest-bend-d1", type=float, default=0.0, help="Rest bend around d1 axis (rad/segment).")
    parser.add_argument("--rest-bend-d2", type=float, default=0.0, help="Rest bend around d2 axis (rad/segment).")
    parser.add_argument("--rest-twist", type=float, default=0.0, help="Rest twist around d3 axis (rad/segment).")
    parser.add_argument("--young-modulus", type=float, default=1.0e6, help="Young's modulus multiplier.")
    parser.add_argument("--torsion-modulus", type=float, default=1.0e6, help="Torsion modulus multiplier.")
    parser.add_argument(
        "--use-banded",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use banded direct solver (disable to use non-banded if available).",
    )
    parser.add_argument("--linear-damping", type=float, default=0.001, help="Linear damping coefficient.")
    parser.add_argument("--angular-damping", type=float, default=0.001, help="Angular damping coefficient.")
    parser.add_argument(
        "--gravity",
        type=float,
        nargs=3,
        default=[0.0, 0.0, -9.81],
        help="Gravity vector (x y z).",
    )
    parser.add_argument(
        "--lock-root-rotation",
        action="store_true",
        default=False,
        help="Lock root rotation by zeroing quaternion inverse mass.",
    )
    return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(create_parser())

    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    example = Example(viewer, args)
    newton.examples.run(example, args)
