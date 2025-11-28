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

"""
KAMINO: Math Operations
"""

from __future__ import annotations

import numpy as np
import warp as wp
from warp._src.types import Any, Float

from .types import (
    float32,
    mat22f,
    mat33f,
    mat34f,
    mat44f,
    mat66f,
    quatf,
    vec3f,
    vec4f,
    vec6f,
)

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

FLOAT32_MIN = wp.constant(float32(np.finfo(np.float32).min))
"""The lowest 32-bit floating-point value."""

FLOAT32_MAX = wp.constant(float32(np.finfo(np.float32).max))
"""The highest 32-bit floating-point value."""

FLOAT32_EPS = wp.constant(float32(np.finfo(np.float32).eps))
"""The smallest 32-bit floating-point value that is not zero."""

UNIT_X = wp.constant(vec3f(1.0, 0.0, 0.0))
""" 3D unit vector for the X axis """

UNIT_Y = wp.constant(vec3f(0.0, 1.0, 0.0))
""" 3D unit vector for the Y axis """

UNIT_Z = wp.constant(vec3f(0.0, 0.0, 1.0))
""" 3D unit vector for the Z axis """

PI = wp.constant(3.141592653589793)
"""Convenience constant for PI"""

TWO_PI = wp.constant(6.283185307179586)
"""Convenience constant for 2 * PI"""

HALF_PI = wp.constant(1.5707963267948966)
"""Convenience constant for PI / 2"""

COS_PI_6 = wp.constant(0.8660254037844387)
"""Convenience constant for cos(PI / 6)"""

I_2 = wp.constant(mat22f(1, 0, 0, 1))
""" The 2x2 identity matrix."""

I_3 = wp.constant(mat33f(1, 0, 0, 0, 1, 0, 0, 0, 1))
""" The 3x3 identity matrix."""

I_4 = wp.constant(mat44f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))
""" The 4x4 identity matrix."""

I_6 = wp.constant(
    mat66f(1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1)
)
""" The 6x6 identity matrix."""


###
# General-purpose functions
###


@wp.func
def squared_norm(x: Any) -> Float:
    return wp.dot(x, x)


###
# Rotation matrices
###


@wp.func
def R_x(theta: float32) -> mat33f:
    """
    Computes the rotation matrix around the X axis.

    Args:
        theta (float32): The angle in radians.

    Returns:
        mat33f: The rotation matrix.
    """
    c = wp.cos(theta)
    s = wp.sin(theta)
    return mat33f(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)


@wp.func
def R_y(theta: float32) -> mat33f:
    """
    Computes the rotation matrix around the Y axis.

    Args:
        theta (float32): The angle in radians.

    Returns:
        mat33f: The rotation matrix.
    """
    c = wp.cos(theta)
    s = wp.sin(theta)
    return mat33f(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)


@wp.func
def R_z(theta: float32) -> mat33f:
    """
    Computes the rotation matrix around the Z axis.

    Args:
        theta (float32): The angle in radians.

    Returns:
        mat33f: The rotation matrix.
    """
    c = wp.cos(theta)
    s = wp.sin(theta)
    return mat33f(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)


@wp.func
def unskew(S: mat33f) -> vec3f:
    """
    Extracts the 3D vector from a 3x3 skew-symmetric matrix.

    Args:
        S (mat33f): The 3x3 skew-symmetric matrix.

    Returns:
        vec3f: The vector extracted from the skew-symmetric matrix.
    """
    return vec3f(S[2, 1], S[0, 2], S[1, 0])


###
# Quaternions
###


@wp.func
def G_of(q: quatf) -> mat34f:
    """
    Computes the G matrix from a quaternion.

    Args:
        q (quatf): The quaternion.

    Returns:
        mat34f: The G matrix.
    """
    G = mat34f(0.0)
    G[0, 0] = q.w
    G[0, 1] = -q.z
    G[0, 2] = q.y
    G[0, 3] = -q.x
    G[1, 0] = q.z
    G[1, 1] = q.w
    G[1, 2] = -q.x
    G[1, 3] = -q.y
    G[2, 0] = -q.y
    G[2, 1] = q.x
    G[2, 2] = q.w
    G[2, 3] = -q.z
    return G


@wp.func
def H_of(q: quatf) -> mat34f:
    """
    Computes the H matrix from a quaternion.

    Args:
        q (quatf): The quaternion.

    Returns:
        mat34f: The H matrix.
    """
    H = mat34f(0.0)
    H[0, 0] = q.w
    H[0, 1] = q.z
    H[0, 2] = -q.y
    H[0, 3] = -q.x
    H[1, 0] = -q.z
    H[1, 1] = q.w
    H[1, 2] = q.x
    H[1, 3] = -q.y
    H[2, 0] = q.y
    H[2, 1] = -q.x
    H[2, 2] = q.w
    H[2, 3] = -q.z
    return H


@wp.func
def quat_from_vec4(v: vec4f) -> quatf:
    """
    Convert a vec4f to a quaternion type.
    """
    return quatf(v[0], v[1], v[2], v[3])


@wp.func
def quat_to_vec4(q: quatf) -> vec4f:
    """
    Convert a quaternion type to a vec4f.
    """
    return vec4f(q.x, q.y, q.z, q.w)


@wp.func
def quat_conj(q: quatf) -> quatf:
    """
    Compute the conjugate of a quaternion.
    The conjugate of a quaternion q = (x, y, z, w) is defined as: q_conj = (x, y, z, -w)
    """
    return quatf(q.x, q.y, q.z, -q.w)


@wp.func
def quat_positive(q: quatf) -> quatf:
    """
    Compute the positive representation of a quaternion.
    The positive representation is defined as the quaternion with a non-negative scalar part.
    """
    if q.w < 0.0:
        s = -1.0
    else:
        s = 1.0
    return s * q


@wp.func
def quat_imaginary(q: quatf) -> vec3f:
    """
    Extract the imaginary part of a quaternion.
    The imaginary part is defined as the vector part of the quaternion (x, y, z).
    """
    return vec3f(q.x, q.y, q.z)


@wp.func
def quat_apply(q: quatf, v: vec3f) -> vec3f:
    """
    Apply a quaternion to a vector.
    The quaternion is applied to the vector using the formula:
    v' = s * v + q.w * uv + qv x uv, where s = q.w^2, uv = 2 * qv x v, and qv is the imaginary part of the quaternion.
    """
    qv = quat_imaginary(q)
    uv = 2.0 * wp.cross(qv, v)
    s = wp.dot(q, q)
    return s * v + q.w * uv + wp.cross(qv, uv)


@wp.func
def quat_derivative(q: quatf, omega: vec3f) -> quatf:
    """
    Computes the quaternion derivative from a quaternion and angular velocity.

    Args:
        q (quatf): The quaternion of the current pose of the body.
        omega (vec3f): The angular velocity of the body.

    Returns:
        quatf: The quaternion derivative.
    """
    vdq = 0.5 * wp.transpose(G_of(q)) * omega
    dq = wp.quaternion(vdq.x, vdq.y, vdq.z, vdq.w, dtype=float32)
    return dq


@wp.func
def quat_log(q: quatf) -> vec3f:
    """
    Computes the logarithm of a quaternion using the stable
    `4 * atan()` formulation to render a rotation vector.
    """
    p = quat_positive(q)
    pv = quat_imaginary(p)
    pv_norm_sq = wp.dot(pv, pv)
    pw_sq = p.w * p.w
    pv_norm = wp.sqrt(pv_norm_sq)

    # Check if the norm of the imaginary part is infinitesimal
    if pv_norm_sq > FLOAT32_EPS:
        # Regular solution for larger angles
        # Use more stable 4 * atan() formulation over the 2 * atan(pv_norm / pw)
        # TODO: angle = 4.0 * wp.atan2(pv_norm, (p.w + wp.sqrt(pw_sq + pv_norm_sq)))
        angle = 4.0 * wp.atan(pv_norm / (p.w + wp.sqrt(pw_sq + pv_norm_sq)))
        c = angle / pv_norm
    else:
        # Taylor expansion solution for small angles
        # For the alternative branch use the limit of angle / pv_norm for angle -> 0.0
        c = (2.0 - wp.static(2.0 / 3.0) * (pv_norm_sq / pw_sq)) / p.w

    # Return the scaled imaginary part of the quaternion
    return c * pv


@wp.func
def quat_log_decomposed(q: quatf) -> vec4f:
    """
    Computes the logarithm of a quaternion using the stable
    `4 * atan()` formulation to render an angle-axis vector.

    The output is a vec4f with the following format:
        - `a = [x, y, z, c]` is the angle-axis output
        - `[x, y, z]` is the axis of rotation
        - `c` is the angle.
    """
    p = quat_positive(q)
    pv = quat_imaginary(p)
    pv_norm_sq = wp.dot(pv, pv)
    pw_sq = p.w * p.w
    pv_norm = wp.sqrt(pv_norm_sq)

    # Check if the norm of the imaginary part is infinitesimal
    if pv_norm_sq > FLOAT32_EPS:
        # Regular solution for larger angles
        # Use more stable 4 * atan() formulation over the 2 * atan(pv_norm / pw)
        # TODO: angle = 4.0 * wp.atan2(pv_norm, (p.w + wp.sqrt(pw_sq + pv_norm_sq)))
        angle = 4.0 * wp.atan(pv_norm / (p.w + wp.sqrt(pw_sq + pv_norm_sq)))
        c = angle / pv_norm
    else:
        # Taylor expansion solution for small angles
        # For the alternative branch use the limit of angle / pv_norm for angle -> 0.0
        c = (2.0 - wp.static(2.0 / 3.0) * (pv_norm_sq / pw_sq)) / p.w

    # Return the scaled imaginary part of the quaternion
    return vec4f(pv.x, pv.y, pv.z, c)


@wp.func
def quat_exp(v: vec3f) -> quatf:
    """
    Computes the exponential map of a 3D vector as a quaternion.
    using Rodrigues' formula: R = I + sin(θ)*K (1-cos(θ)*K^2),
    were q = quat(R).

    Args:
        v (vec3f): The 3D rotation vector to be mapped to quaternion space.

    Returns:
        quatf: The quaternion resulting from the exponential map of the input rotation vector.
    """
    eps = FLOAT32_EPS
    q = wp.quat_identity(dtype=float32)
    vn = wp.length(v)
    if vn > eps:
        a = 0.5 * vn
        sina = wp.sin(a)
        cosa = wp.cos(a)
        vu = wp.normalize(v)
        q.x = sina * vu.x
        q.y = sina * vu.y
        q.z = sina * vu.z
        q.w = cosa
    else:
        q.x = 0.5 * v.x
        q.y = 0.5 * v.y
        q.z = 0.5 * v.z
        q.w = 1.0
    return q


@wp.func
def quat_product(q1: quatf, q2: quatf) -> quatf:
    """
    Computes the quaternion product of two quaternions.

    Args:
        q1 (quatf): The first quaternion.
        q2 (quatf): The second quaternion.

    Returns:
        quatf: The result of the quaternion product.
    """
    q3 = wp.quat_identity(dtype=float32)
    q3.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
    q3.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
    q3.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
    q3.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    return q3


@wp.func
def quat_box_plus(q: quatf, v: vec3f) -> quatf:
    """
    Computes the box-plus operation for a quaternion and a vector:
        R(q) [+] v == exp(v) * R(q), where R(q) is the rotation matrix of the quaternion q.

    Args:
        q (vec3f): The quaternion.
        v (vec3f): The vector.

    Returns:
        quatf: The result of the box-plus operation.
    """
    return quat_product(quat_exp(v), q)


@wp.func
def quat_from_x_rot(angle_rad: float32) -> quatf:
    """
    Computes a unit quaternion corresponding to rotation by given angle about the x axis
    """
    return wp.quatf(wp.sin(0.5 * angle_rad), 0.0, 0.0, wp.cos(0.5 * angle_rad))


@wp.func
def quat_from_y_rot(angle_rad: float32) -> quatf:
    """
    Computes a unit quaternion corresponding to rotation by given angle about the y axis
    """
    return wp.quatf(0.0, wp.sin(0.5 * angle_rad), 0.0, wp.cos(0.5 * angle_rad))


@wp.func
def quat_from_z_rot(angle_rad: float32) -> quatf:
    """
    Computes a unit quaternion corresponding to rotation by given angle about the z axis
    """
    return wp.quatf(0.0, 0.0, wp.sin(0.5 * angle_rad), wp.cos(0.5 * angle_rad))


@wp.func
def quat_to_euler_xyz(q: quatf) -> vec3f:
    """
    Converts a unit quaternion to XYZ Euler angles (also known as Cardan angles).
    """
    rpy = vec3f(0.0)
    R_20 = -2.0 * (q.x * q.z - q.w * q.y)
    if wp.abs(R_20) < 1.0:
        rpy[1] = wp.asin(-R_20)
        rpy[0] = wp.atan2(2.0 * (q.y * q.z + q.w * q.x), q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z)
        rpy[2] = wp.atan2(2.0 * (q.x * q.y + q.w * q.z), q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z)
    else:  # Gimbal lock
        rpy[0] = wp.atan2(-2.0 * (q.x * q.y - q.w * q.z), q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z)
        rpy[1] = wp.half_pi if R_20 <= -1.0 else -wp.half_pi
        rpy[2] = 0.0
    return rpy


@wp.func
def quat_left_jacobian_inverse(q: quatf) -> mat33f:
    """
    Computes the left-Jacobian inverse of the quaternion log map
    """
    p = quat_positive(q)
    pv = quat_imaginary(p)
    pv_norm_sq = wp.dot(pv, pv)
    pw_sq = p.w * p.w
    pv_norm = wp.sqrt(pv_norm_sq)

    # Check if the norm of the imaginary part is infinitesimal
    if pv_norm_sq > FLOAT32_EPS:
        # Regular solution for larger angles
        c0 = 2.0 * wp.atan(pv_norm / (p.w + wp.sqrt(pw_sq + pv_norm_sq))) / pv_norm
        c1 = (1.0 - c0 * p.w) / pv_norm_sq
    else:
        # Taylor expansion solution for small angles
        c1 = wp.static(1.0 / 3.0) / pw_sq
        c0 = (1.0 - c1 * pv_norm_sq) / p.w

    return wp.identity(3, dtype=float32) - wp.skew(c0 * pv) + wp.skew(c1 * pv) * wp.skew(pv)


@wp.func
def quat_normalized_apply(q: quatf, v: vec3f) -> vec3f:
    """
    Combines quaternion normalization and applying a unit quaternion to a vector
    """
    qv = quat_imaginary(q)
    s = wp.dot(q, q)
    uv_s = (2.0 / s) * wp.cross(qv, v)
    return v + q[3] * uv_s + wp.cross(qv, uv_s)


@wp.func
def quat_conj_normalized_apply(q: quatf, v: vec3f) -> vec3f:
    """
    Combines quaternion conjugation, normalization and applying a unit quaternion to a vector
    """
    qv = quat_imaginary(q)
    s = wp.dot(q, q)
    uv_s = (2.0 / s) * wp.cross(qv, v)
    return v - q[3] * uv_s + wp.cross(qv, uv_s)


###
# Unit Quaternions
###


@wp.func
def unit_quat_apply(q: quatf, v: vec3f) -> vec3f:
    """
    Applies a unit quaternion to a vector (making use of the unit norm assumption to simplify the result)
    """
    qv = quat_imaginary(q)
    uv = 2.0 * wp.cross(qv, v)
    return v + q.w * uv + wp.cross(qv, uv)


@wp.func
def unit_quat_conj_apply(q: quatf, v: vec3f) -> vec3f:
    """
    Applies the conjugate of a unit quaternion to a vector (making use of the unit norm assumption to simplify
    the result)
    """
    qv = quat_imaginary(q)
    uv = 2.0 * wp.cross(qv, v)
    return v - q.w * uv + wp.cross(qv, uv)


@wp.func
def unit_quat_to_rotation_matrix(q: quatf) -> mat33f:
    """
    Converts a unit quaternion to a rotation matrix (making use of the unit norm assumption to simplify the result)
    """
    xx = 2.0 * q.x * q.x
    xy = 2.0 * q.x * q.y
    xz = 2.0 * q.x * q.z
    wx = 2.0 * q.w * q.x
    yy = 2.0 * q.y * q.y
    yz = 2.0 * q.y * q.z
    wy = 2.0 * q.w * q.y
    zz = 2.0 * q.z * q.z
    wz = 2.0 * q.w * q.z
    return mat33f(1.0 - yy - zz, xy - wz, xz + wy, xy + wz, 1.0 - xx - zz, yz - wx, xz - wy, yz + wx, 1.0 - xx - yy)


@wp.func
def unit_quat_conj_to_rotation_matrix(q: quatf) -> mat33f:
    """
    Converts the conjugate of a unit quaternion to a rotation matrix (making use of the unit norm assumption
    to simplify the result); this is simply the transpose of unit_quat_to_rotation_matrix(q)
    """
    xx = 2.0 * q.x * q.x
    xy = 2.0 * q.x * q.y
    xz = 2.0 * q.x * q.z
    wx = 2.0 * q.w * q.x
    yy = 2.0 * q.y * q.y
    yz = 2.0 * q.y * q.z
    wy = 2.0 * q.w * q.y
    zz = 2.0 * q.z * q.z
    wz = 2.0 * q.w * q.z
    return mat33f(1.0 - yy - zz, xy + wz, xz - wy, xy - wz, 1.0 - xx - zz, yz + wx, xz + wy, yz - wx, 1.0 - xx - yy)


@wp.func
def unit_quat_apply_jacobian(q: quatf, v: vec3f) -> mat34f:
    """
    Returns the Jacobian of unit_quat_apply(q, v) with respect to q
    """
    xX = 2.0 * q.x * v[0]
    xY = 2.0 * q.x * v[1]
    xZ = 2.0 * q.x * v[2]
    yX = 2.0 * q.y * v[0]
    yY = 2.0 * q.y * v[1]
    yZ = 2.0 * q.y * v[2]
    zX = 2.0 * q.z * v[0]
    zY = 2.0 * q.z * v[1]
    zZ = 2.0 * q.z * v[2]
    wX = 2.0 * q.w * v[0]
    wY = 2.0 * q.w * v[1]
    wZ = 2.0 * q.w * v[2]
    return mat34f(
        yY + zZ,
        -2.0 * yX + xY + wZ,
        -2.0 * zX + xZ - wY,
        yZ - zY,
        -2.0 * xY + yX - wZ,
        xX + zZ,
        -2.0 * zY + yZ + wX,
        zX - xZ,
        -2.0 * xZ + zX + wY,
        -2.0 * yZ + zY - wX,
        xX + yY,
        xY - yX,
    )


@wp.func
def unit_quat_conj_apply_jacobian(q: quatf, v: vec3f) -> mat34f:
    """
    Returns the Jacobian of unit_quat_conj_apply(q, v) with respect to q
    """
    xX = 2.0 * q.x * v[0]
    xY = 2.0 * q.x * v[1]
    xZ = 2.0 * q.x * v[2]
    yX = 2.0 * q.y * v[0]
    yY = 2.0 * q.y * v[1]
    yZ = 2.0 * q.y * v[2]
    zX = 2.0 * q.z * v[0]
    zY = 2.0 * q.z * v[1]
    zZ = 2.0 * q.z * v[2]
    wX = 2.0 * q.w * v[0]
    wY = 2.0 * q.w * v[1]
    wZ = 2.0 * q.w * v[2]
    return mat34f(
        yY + zZ,
        -2.0 * yX + xY - wZ,
        -2.0 * zX + xZ + wY,
        zY - yZ,
        -2.0 * xY + yX + wZ,
        xX + zZ,
        -2.0 * zY + yZ - wX,
        xZ - zX,
        -2.0 * xZ + zX - wY,
        -2.0 * yZ + zY + wX,
        xX + yY,
        yX - xY,
    )


###
# Screws
###


@wp.func
def screw(linear: vec3f, angular: vec3f) -> vec6f:
    """
    Constructs a 6D screw (as `vec6f`) from 3D linear and angular components.

    Args:
        linear (vec3f): The linear component of the screw.
        angular (vec3f): The angular component of the screw.

    Returns:
        vec6f: The resulting screw represented as a 6D vector.
    """
    return vec6f(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])


@wp.func
def screw_linear(s: vec6f) -> vec3f:
    """
    Extracts the linear component from a 6D screw vector.

    Args:
        s (vec6f): The 6D screw vector.

    Returns:
        vec3f: The linear component of the screw.
    """
    return vec3f(s[0], s[1], s[2])


@wp.func
def screw_angular(s: vec6f) -> vec3f:
    """
    Extracts the angular component from a 6D screw vector.

    Args:
        s (vec6f): The 6D screw vector.

    Returns:
        vec3f: The angular component of the screw.
    """
    return vec3f(s[3], s[4], s[5])
