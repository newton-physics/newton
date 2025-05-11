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

import warp as wp


@wp.func
def velocity_at_point(qd: wp.spatial_vector, r: wp.vec3):
    """
    Returns the velocity of a point relative to the frame with the given spatial velocity.

    Args:
        qd (spatial_vector): The spatial velocity of the frame.
        r (vec3): The position of the point relative to the frame.

    Returns:
        vec3: The velocity of the point.
    """
    return wp.cross(wp.spatial_top(qd), r) + wp.spatial_bottom(qd)


@wp.func
def quat_twist(axis: wp.vec3, q: wp.quat):
    """
    Returns the twist around an axis.
    """

    # project imaginary part onto axis
    a = wp.vec3(q[0], q[1], q[2])
    proj = wp.dot(a, axis)
    a = proj * axis
    # if proj < 0.0:
    #     # ensure twist points in same direction as axis
    #     a = -a
    return wp.normalize(wp.quat(a[0], a[1], a[2], q[3]))


@wp.func
def quat_twist_angle(axis: wp.vec3, q: wp.quat):
    """
    Returns the angle of the twist around an axis.
    """
    return 2.0 * wp.acos(quat_twist(axis, q)[3])


@wp.func
def quat_decompose(q: wp.quat):
    """
    Decompose a quaternion into a sequence of 3 rotations around x,y',z' respectively, i.e.: q = q_z''q_y'q_x.
    """

    R = wp.matrix_from_cols(
        wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0)),
        wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)),
        wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)),
    )

    # https://www.sedris.org/wg8home/Documents/WG80485.pdf
    phi = wp.atan2(R[1, 2], R[2, 2])
    sinp = -R[0, 2]
    if wp.abs(sinp) >= 1.0:
        theta = wp.HALF_PI * wp.sign(sinp)
    else:
        theta = wp.asin(-R[0, 2])
    psi = wp.atan2(R[0, 1], R[0, 0])

    return -wp.vec3(phi, theta, psi)


@wp.func
def quat_to_rpy(q: wp.quat):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = wp.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = wp.clamp(t2, -1.0, 1.0)
    pitch_y = wp.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = wp.atan2(t3, t4)

    return wp.vec3(roll_x, pitch_y, yaw_z)


@wp.func
def quat_to_euler(q: wp.quat, i: int, j: int, k: int) -> wp.vec3:
    """
    Convert a quaternion into Euler angles.

    :math:`i, j, k` are the indices in :math:`[0, 1, 2]` of the axes to use
    (:math:`i \\neq j, j \\neq k`).

    Reference: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0276302

    Args:
        q (quat): The quaternion to convert
        i (int): The index of the first axis
        j (int): The index of the second axis
        k (int): The index of the third axis

    Returns:
        vec3: The Euler angles (in radians)
    """
    # i, j, k are actually assumed to follow 1-based indexing but
    # we want to be compatible with quat_from_euler
    i += 1
    j += 1
    k += 1
    not_proper = True
    if i == k:
        not_proper = False
        k = 6 - i - j  # because i + j + k = 1 + 2 + 3 = 6
    e = float((i - j) * (j - k) * (k - i)) / 2.0  # Levi-Civita symbol
    a = q[0]
    b = q[i]
    c = q[j]
    d = q[k] * e
    if not_proper:
        a -= q[j]
        b += q[k] * e
        c += q[0]
        d -= q[i]
    t2 = wp.acos(2.0 * (a * a + b * b) / (a * a + b * b + c * c + d * d) - 1.0)
    tp = wp.atan2(b, a)
    tm = wp.atan2(d, c)
    t1 = 0.0
    t3 = 0.0
    if wp.abs(t2) < 1e-6:
        t3 = 2.0 * tp - t1
    elif wp.abs(t2 - wp.HALF_PI) < 1e-6:
        t3 = 2.0 * tm + t1
    else:
        t1 = tp - tm
        t3 = tp + tm
    if not_proper:
        t2 -= wp.HALF_PI
        t3 *= e
    return wp.vec3(t1, t2, t3)


@wp.func
def quat_from_euler(e: wp.vec3, i: int, j: int, k: int) -> wp.quat:
    """
    Convert Euler angles to a quaternion.

    :math:`i, j, k` are the indices in :math:`[0, 1, 2]` of the axes in which the Euler angles are provided
    (:math:`i \\neq j, j \\neq k`), e.g. (0, 1, 2) for Euler sequence XYZ.

    Args:
        e (vec3): The Euler angles (in radians)
        i (int): The index of the first axis
        j (int): The index of the second axis
        k (int): The index of the third axis

    Returns:
        quat: The quaternion
    """
    # Half angles
    half_e = e / 2.0

    # Precompute sines and cosines of half angles
    cr = wp.cos(half_e[i])
    sr = wp.sin(half_e[i])
    cp = wp.cos(half_e[j])
    sp = wp.sin(half_e[j])
    cy = wp.cos(half_e[k])
    sy = wp.sin(half_e[k])

    # Components of the quaternion based on the rotation sequence
    return wp.quat(
        (cy * sr * cp - sy * cr * sp),
        (cy * cr * sp + sy * sr * cp),
        (sy * cr * cp - cy * sr * sp),
        (cy * cr * cp + sy * sr * sp),
    )


@wp.func
def transform_twist(t: wp.transform, x: wp.spatial_vector):
    # Frank & Park definition 3.20, pg 100

    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    w = wp.quat_rotate(q, w)
    v = wp.quat_rotate(q, v) + wp.cross(p, w)

    return wp.spatial_vector(w, v)


@wp.func
def transform_wrench(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    v = wp.quat_rotate(q, v)
    w = wp.quat_rotate(q, w) + wp.cross(p, v)

    return wp.spatial_vector(w, v)


__all__ = [
    "quat_decompose",
    "quat_from_euler",
    "quat_to_euler",
    "quat_to_rpy",
    "quat_twist",
    "quat_twist_angle",
    "transform_twist",
    "transform_wrench",
    "velocity_at_point",
]