# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: Math Operations
"""

from __future__ import annotations

import numpy as np
import warp as wp
from warp._src.types import Any

from .....core.types import Axis, AxisType
from .types import (
    mat34f,
    mat63f,
)

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

FLOAT32_MIN = wp.constant(wp.float32(np.finfo(np.float32).min))
"""The lowest 32-bit floating-point value."""

FLOAT32_MAX = wp.constant(wp.float32(np.finfo(np.float32).max))
"""The highest 32-bit floating-point value."""

FLOAT32_EPS = wp.constant(wp.float32(np.finfo(np.float32).eps))
"""Machine epsilon for 32-bit float: the smallest value such that 1.0 + eps != 1.0."""

I_3 = wp.constant(wp.mat33f(1, 0, 0, 0, 1, 0, 0, 0, 1))
""" The 3x3 identity matrix."""


###
# Rotation matrices
###


def axis_to_mat33(axis: AxisType) -> wp.mat33f:
    """Return a 3x3 frame matrix whose first column is the unit vector for ``axis``.

    The remaining two columns are the standard basis vectors cycled so the matrix
    is a permutation of the identity (i.e. a proper rotation).

    Args:
        axis: Axis identifier (:class:`Axis`, string, or int) to use as the first
            column of the output matrix.
    """
    a = Axis.from_any(axis)
    if a == Axis.X:
        return wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if a == Axis.Y:
        return wp.mat33f(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    return wp.mat33f(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)


###
# Quaternions
###


@wp.func
def G_of(q: wp.quatf) -> mat34f:
    """
    Computes the G matrix from a quaternion.

    Args:
        q: The quaternion.

    Returns:
        The G matrix.
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
def H_of(q: wp.quatf) -> mat34f:
    """
    Computes the H matrix from a quaternion.

    Args:
        q: The quaternion.

    Returns:
        The H matrix.
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
def quat_positive(q: wp.quatf) -> wp.quatf:
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
def quat_imaginary(q: wp.quatf) -> wp.vec3f:
    """
    Extract the imaginary part of a quaternion.
    The imaginary part is defined as the vector part of the quaternion (x, y, z).
    """
    return wp.vec3f(q.x, q.y, q.z)


@wp.func
def quat_log(q: wp.quatf) -> wp.vec3f:
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
def quat_exp(v: wp.vec3f) -> wp.quatf:
    """
    Computes the exponential map of a 3D vector as a quaternion.
    using Rodrigues' formula: R = I + sin(θ)*K (1-cos(θ)*K^2),
    were q = quat(R).

    Args:
        v: The 3D rotation vector to be mapped to quaternion space.

    Returns:
        The quaternion resulting from the exponential map of the input rotation vector.
    """
    eps = FLOAT32_EPS
    q = wp.quat_identity(dtype=wp.float32)
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
def quat_left_jacobian_inverse(q: wp.quatf) -> wp.mat33f:
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

    return wp.identity(3, dtype=wp.float32) - wp.skew(c0 * pv) + wp.skew(c1 * pv) * wp.skew(pv)


@wp.func
def quat_twist_angle(q: wp.quatf, axis: wp.vec3f) -> wp.float32:
    """
    Computes the twist angle of a quaternion around a specific axis.

    This function isolates the rotation component of ``q`` that occurs purely
    around the provided ``axis`` (Twist-Swing decomposition) and returns
    its angle in [-pi, pi].
    """
    # positive quaternion guarantees angle is in [-pi, pi]
    p = quat_positive(q)
    pv = quat_imaginary(p)
    angle = 2.0 * wp.atan2(wp.dot(pv, axis), p.w)
    return angle


###
# Unit Quaternions
###


@wp.func
def unit_quat_apply(q: wp.quatf, v: wp.vec3f) -> wp.vec3f:
    """
    Applies a unit quaternion to a vector (making use of the unit norm assumption to simplify the result)
    """
    qv = quat_imaginary(q)
    uv = 2.0 * wp.cross(qv, v)
    return v + q.w * uv + wp.cross(qv, uv)


@wp.func
def unit_quat_conj_apply(q: wp.quatf, v: wp.vec3f) -> wp.vec3f:
    """
    Applies the conjugate of a unit quaternion to a vector (making use of the unit norm assumption to simplify
    the result)
    """
    qv = quat_imaginary(q)
    uv = 2.0 * wp.cross(qv, v)
    return v - q.w * uv + wp.cross(qv, uv)


@wp.func
def unit_quat_to_rotation_matrix(q: wp.quatf) -> wp.mat33f:
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
    return wp.mat33f(1.0 - yy - zz, xy - wz, xz + wy, xy + wz, 1.0 - xx - zz, yz - wx, xz - wy, yz + wx, 1.0 - xx - yy)


@wp.func
def unit_quat_conj_to_rotation_matrix(q: wp.quatf) -> wp.mat33f:
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
    return wp.mat33f(1.0 - yy - zz, xy + wz, xz - wy, xy - wz, 1.0 - xx - zz, yz + wx, xz + wy, yz - wx, 1.0 - xx - yy)


@wp.func
def unit_quat_apply_jacobian(q: wp.quatf, v: wp.vec3f) -> mat34f:
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
def unit_quat_conj_apply_jacobian(q: wp.quatf, v: wp.vec3f) -> mat34f:
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
def screw_transform_matrix_from_points(r_A: wp.vec3f, r_B: wp.vec3f) -> wp.spatial_matrixf:
    """
    Generates a 6x6 screw transformation matrix given the starting (`r_A`)
    and ending (`r_B`) positions defining the line-of-action of the screw.

    Both positions are assumed to be expressed in world coordinates,
    and the line-of-action can be thought of as an effective lever-arm
    from point `A` to point `B`.

    This function thus renders the matrix screw transformation from point `A` to point `B` as:

    `W_BA := [[I_3  , 0_3],[S_BA , I_3]]`,

    where `S_BA` is the skew-symmetric matrix of the vector `r_BA = r_A - r_B`.

    Args:
        r_A: The starting position of the line-of-action in world coordinates.
        r_B: The ending position of the line-of-action in world coordinates.

    Returns:
        The 6x6 screw transformation matrix.
    """
    # Initialize the wrench matrix
    W_BA = wp.identity(n=6, dtype=wp.float32)

    # Fill the lower left block with the skew-symmetric matrix
    S_BA = wp.skew(r_A - r_B)
    for i in range(3):
        for j in range(3):
            W_BA[3 + i, j] = S_BA[i, j]

    # Return the wrench matrix
    return W_BA


###
# Wrenches
###


W_C_I = wp.constant(mat63f(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0))
"""Identity-like matrix to initialize contact wrench matrices."""


@wp.func
def contact_wrench_matrix_from_points(r_k: wp.vec3f, r_i: wp.vec3f) -> mat63f:
    """
    Generates a 6x3 screw transformation matrix given the contact (`r_k`)
    and body CoM (`r_i`) positions defining the line-of-action of the force.

    Both positions are assumed to be expressed in world coordinates,
    and the line-of-action can be thought of as an effective lever-arm
    from point `k` to point `i`.

    This function thus renders the matrix screw transformation from point `k` to point `i` as:

    `W_ki := [[I_3],[S_ki]]`,

    where `S_ki` is the skew-symmetric matrix of the vector `r_ki = r_k - r_i`.

    Args:
        r_k: The position of the contact point in world coordinates.
        r_i: The position of the body CoM in world coordinates.

    Returns:
        The 6x6 screw transformation matrix.
    """
    # Initialize the wrench matrix
    W_ki = W_C_I

    # Fill the lower left block with the skew-symmetric matrix
    S_ki = wp.skew(r_k - r_i)
    for i in range(3):
        for j in range(3):
            W_ki[3 + i, j] = S_ki[i, j]

    # Return the wrench matrix
    return W_ki


@wp.func
def expand6d(X: wp.mat33f) -> wp.spatial_matrixf:
    """
    Expands a 3x3 rotation matrix to a 6x6 matrix operator by filling
    the upper left and lower right blocks with the input matrix.

    Args:
        X: The 3x3 matrix to be expanded.

    Returns:
        The expanded 6x6 matrix.
    """
    # Initialize the 6D matrix
    X_6d = wp.spatial_matrixf(0.0)

    # Fill the upper left 3x3 block with the input matrix
    for i in range(3):
        for j in range(3):
            X_6d[i, j] = X[i, j]
            X_6d[3 + i, 3 + j] = X[i, j]

    # Return the expanded matrix
    return X_6d


@wp.func
def concat6d(X1: wp.mat33f, X2: wp.mat33f) -> wp.spatial_matrixf:
    """
    Concatenates two 3x3 rotation matrix as diagonal blocks of a 6x6 matrix.

    Args:
        X1: The 3x3 top-left matrix.
        X2: The 3x3 bottom-right matrix.

    Returns:
        The 6x6 matrix concatenating X1 and X2 along the diagonal.
    """
    # Initialize the 6D matrix
    X_6d = wp.spatial_matrixf(0.0)

    # Fill the upper left 3x3 block with the input matrix
    for i in range(3):
        for j in range(3):
            X_6d[i, j] = X1[i, j]
            X_6d[3 + i, 3 + j] = X2[i, j]

    # Return the expanded matrix
    return X_6d


###
# Dynamics
###


@wp.func
def compute_body_twist_update_with_eom(
    dt: wp.float32,
    g: wp.vec3f,
    inv_m_i: wp.float32,
    I_i: wp.mat33f,
    inv_I_i: wp.mat33f,
    u_i: wp.spatial_vectorf,
    w_i: wp.spatial_vectorf,
) -> tuple[wp.vec3f, wp.vec3f]:
    # Extract linear and angular parts
    v_i = wp.spatial_top(u_i)
    omega_i = wp.spatial_bottom(u_i)
    S_i = wp.skew(omega_i)
    f_i = wp.spatial_top(w_i)
    tau_i = wp.spatial_bottom(w_i)

    # Compute velocity update equations
    v_i_n = v_i + dt * (g + inv_m_i * f_i)
    omega_i_n = omega_i + dt * inv_I_i @ (-S_i @ (I_i @ omega_i) + tau_i)

    # Return the updated velocities
    return v_i_n, omega_i_n


@wp.func
def compute_body_pose_update_with_logmap(
    dt: wp.float32,
    p_i: wp.transformf,
    v_i: wp.vec3f,
    omega_i: wp.vec3f,
) -> wp.transformf:
    # Extract linear and angular parts
    r_i = wp.transform_get_translation(p_i)
    q_i = wp.transform_get_rotation(p_i)

    # Compute configuration update equations
    r_i_n = r_i + dt * v_i
    q_i_n = quat_exp(dt * omega_i) * q_i
    p_i_n = wp.transformf(r_i_n, q_i_n)

    # Return the new pose and twist
    return p_i_n


###
# Indexing
###


@wp.func
def tril_index(row: Any, col: Any) -> Any:
    """
    Computes the index in a flattened lower-triangular matrix.

    Args:
        row: The row index.
        col: The column index.

    Returns:
        The index in the flattened lower-triangular matrix.
    """
    return (row * (row + 1)) // 2 + col
