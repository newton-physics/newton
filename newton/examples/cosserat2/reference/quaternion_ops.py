# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Quaternion math utilities for NumPy-based Cosserat rod simulation.

Quaternion convention: [x, y, z, w] (scalar last, matching Warp/Eigen conventions).

Reference: "Position And Orientation Based Cosserat Rods" paper
https://animation.rwth-aachen.de/publication/0550/
"""

import numpy as np
from numpy.typing import NDArray


def quat_multiply(q1: NDArray, q2: NDArray) -> NDArray:
    """Multiply two quaternions: q1 * q2.

    Args:
        q1: First quaternion [x, y, z, w].
        q2: Second quaternion [x, y, z, w].

    Returns:
        Product quaternion [x, y, z, w].
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quat_conjugate(q: NDArray) -> NDArray:
    """Compute quaternion conjugate.

    Args:
        q: Quaternion [x, y, z, w].

    Returns:
        Conjugate quaternion [-x, -y, -z, w].
    """
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_normalize(q: NDArray) -> NDArray:
    """Normalize a quaternion to unit length.

    Args:
        q: Quaternion [x, y, z, w].

    Returns:
        Normalized quaternion.
    """
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / norm


def quat_to_rotation_matrix(q: NDArray) -> NDArray:
    """Convert quaternion to 3x3 rotation matrix.

    Args:
        q: Quaternion [x, y, z, w].

    Returns:
        3x3 rotation matrix.
    """
    x, y, z, w = q

    # Precompute products
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )


def quat_rotate_vector(q: NDArray, v: NDArray) -> NDArray:
    """Rotate a vector by a quaternion: q * v * conjugate(q).

    Args:
        q: Quaternion [x, y, z, w].
        v: Vector [x, y, z].

    Returns:
        Rotated vector [x, y, z].
    """
    # Convert vector to quaternion with w=0
    v_quat = np.array([v[0], v[1], v[2], 0.0])

    # q * v * conjugate(q)
    q_conj = quat_conjugate(q)
    result = quat_multiply(quat_multiply(q, v_quat), q_conj)

    return result[:3]


def quat_rotate_vector_inv(q: NDArray, v: NDArray) -> NDArray:
    """Rotate a vector by the inverse of a quaternion: conjugate(q) * v * q.

    Args:
        q: Quaternion [x, y, z, w].
        v: Vector [x, y, z].

    Returns:
        Rotated vector [x, y, z].
    """
    q_conj = quat_conjugate(q)
    return quat_rotate_vector(q_conj, v)


def quat_rotate_e3(q: NDArray) -> NDArray:
    """Rotate the e3 unit vector [0, 0, 1] by quaternion q.

    This is an optimized version of quat_rotate_vector(q, [0, 0, 1]).
    Computes the third column of the rotation matrix (the d3 director).

    From pbd_rods C++ code:
        d3[0] = 2.0 * (q.x * q.z + q.w * q.y)
        d3[1] = 2.0 * (q.y * q.z - q.w * q.x)
        d3[2] = q.w^2 - q.x^2 - q.y^2 + q.z^2

    Args:
        q: Quaternion [x, y, z, w].

    Returns:
        Third director d3 = R(q) * e3.
    """
    x, y, z, w = q

    return np.array(
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            w * w - x * x - y * y + z * z,
        ]
    )


def quat_e3_bar(q: NDArray) -> NDArray:
    """Compute q * e3_conjugate for quaternion correction.

    This is used in the stretch/shear constraint correction formula.
    Computes q * [0, 0, -1, 0] where [0, 0, -1, 0] is the conjugate
    of the pure quaternion representing e3 = [0, 0, 1, 0].

    From pbd_rods C++ code (Eigen uses w,x,y,z constructor order):
        q_e_3_bar = Quaternion(q.z, -q.y, q.x, -q.w)
    Which in [x,y,z,w] format is: [-y, x, -w, z]

    Args:
        q: Quaternion [x, y, z, w].

    Returns:
        Quaternion q * e3_bar = [-y, x, -w, z].
    """
    x, y, z, w = q
    return np.array([-y, x, -w, z])


def compute_darboux_vector(q0: NDArray, q1: NDArray, rest_darboux: NDArray) -> NDArray:
    """Compute the Darboux vector (curvature) between two quaternions.

    The Darboux vector omega = conjugate(q0) * q1 represents the relative
    rotation between adjacent edge frames.

    This function also handles the quaternion double-cover by choosing
    the shorter rotation path using full quaternion norm comparison.

    Args:
        q0: First edge quaternion [x, y, z, w].
        q1: Second edge quaternion [x, y, z, w].
        rest_darboux: Rest Darboux vector as quaternion [x, y, z, w].

    Returns:
        Darboux vector deviation [x, y, z] (imaginary part of omega - rest_darboux).
    """
    # Compute Darboux vector: omega = conjugate(q0) * q1
    q0_conj = quat_conjugate(q0)
    omega = quat_multiply(q0_conj, q1)

    # Handle quaternion double-cover: choose shorter path
    # Compare full quaternion norms |omega + rest|^2 vs |omega - rest|^2
    omega_plus = omega + rest_darboux
    omega_minus = omega - rest_darboux

    # Use full quaternion squared norm (all 4 components)
    if np.dot(omega_plus, omega_plus) < np.dot(omega_minus, omega_minus):
        return omega_plus[:3]
    else:
        return omega_minus[:3]
