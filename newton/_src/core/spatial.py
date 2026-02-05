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

from .types import Axis, AxisType


@wp.func
def velocity_at_point(qd: wp.spatial_vector, r: wp.vec3):
    """
    Return the velocity of a point relative to the frame that owns the
    provided spatial velocity.

    Args:
        qd (spatial_vector): The spatial velocity of the frame.
        r (vec3): The position of the point relative to the frame.

    Returns:
        vec3: The velocity of the point.
    """
    return wp.spatial_top(qd) + wp.cross(wp.spatial_bottom(qd), r)


@wp.func
def quat_twist(axis: wp.vec3, q: wp.quat):
    """Return the twist around an axis."""

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
    """Return the angle of the twist around an axis."""
    return 2.0 * wp.acos(quat_twist(axis, q)[3])


@wp.func
def quat_velocity(q_now: wp.quat, q_prev: wp.quat, dt: float) -> wp.vec3:
    """Approximate angular velocity from successive world quaternions (world frame).

    Uses right-trivialized mapping via dq = q_now * conj(q_prev).

    Args:
        q_now: Current orientation in world frame.
        q_prev: Previous orientation in world frame.
        dt: Time step [s].

    Returns:
        wp.vec3: Angular velocity omega in world frame [rad/s].
    """
    # Normalize inputs
    q1 = wp.normalize(q_now)
    q0 = wp.normalize(q_prev)

    # Enforce shortest-arc by aligning quaternion hemisphere
    if wp.dot(q1, q0) < 0.0:
        q0 = wp.quat(-q0[0], -q0[1], -q0[2], -q0[3])

    # dq = q1 * conj(q0)
    dq = wp.normalize(wp.mul(q1, wp.quat_inverse(q0)))

    axis, angle = wp.quat_to_axis_angle(dq)
    return axis * (angle / dt)


@wp.func
def quat_decompose(q: wp.quat):
    """Decompose a quaternion into extrinsic Euler angles.

    Calculates Euler angles for a sequence of rotations around fixed world axes
    in the order of X, then Y, then Z.

    The corresponding matrix multiplication for a column vector :math:`v` is:

    .. math::

       v_{\\text{rotated}} = R_z(\\text{angle}_z) R_y(\\text{angle}_y) R_x(\\text{angle}_x) v

    Args:
        q (wp.quat): The input quaternion to decompose.

    Returns:
        wp.vec3: The Euler angles :math:`(\\text{angle}_x, \\text{angle}_y, \\text{angle}_z)` in radians.
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
    """Convert a quaternion into Euler angles (roll, pitch, yaw).

    The returned angles represent a sequence of extrinsic rotations following the
    Z-Y-X convention (Tait-Bryan angles).

    - **yaw**: Rotation about the *z*-axis.
    - **pitch**: Rotation about the *y*-axis.
    - **roll**: Rotation about the *x*-axis.

    All angles are in radians and are applied counter-clockwise. Note that Warp's
    quaternion components are stored in `(x, y, z, w)` order.

    Args:
        q (wp.quat): The input quaternion to convert.

    Returns:
        wp.vec3: The Euler angles `(roll, pitch, yaw)` in radians.
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
    """Convert a quaternion into Euler angles.

    :math:`i, j, k` are the indices in :math:`[0, 1, 2]` of the axes to use
    (:math:`i \\neq j, j \\neq k`).

    Reference: https://doi.org/10.1371/journal.pone.0276302

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
    """Convert Euler angles to a quaternion.

    The integers ``i, j, k`` select axes in the set ``{0, 1, 2}`` that
    determine the Euler-sequence used.  They must satisfy ``i ≠ j`` and
    ``j ≠ k``.  For example, the XYZ sequence corresponds to ``(0, 1, 2)``.

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
    """Transform a spatial twist between coordinate frames.

    This routine applies the rigid-body twist transformation defined in
    *Frank & Park, Modern Robotics* (Definition 3.20, p. 100).

    Given a spatial twist ``x = (v, ω)`` expressed in the *source* frame and a
    homogeneous transform ``t`` (source → destination), the returned twist
    ``x' = (v', ω')`` represents the same motion expressed in the destination
    frame:

    .. math::

       x' = \\begin{bmatrix} R & [p]_{\\times} R \\\\ 0 & R \\end{bmatrix} x

    where *R* and *p* are the rotation and translation components of ``t`` and
    ``[p]_x`` is the skew-symmetric matrix of *p*.

    Args:
        t (transform): The transform from the **source** frame to the
            **destination** frame.
        x (spatial_vector): The spatial twist expressed in the source frame.

    Returns:
        spatial_vector: The twist expressed in the destination frame.
    """

    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    v = wp.spatial_top(x)
    w = wp.spatial_bottom(x)

    w = wp.quat_rotate(q, w)
    v = wp.quat_rotate(q, v) + wp.cross(p, w)

    return wp.spatial_vector(v, w)


@wp.func
def transform_wrench(t: wp.transform, x: wp.spatial_vector):
    """Transform a spatial wrench between coordinate frames.

    A spatial wrench is the dual vector to a spatial twist and consists of a
    force-torque pair ``x = (f, τ)``.

    Given a wrench expressed in the *source* frame and a transform ``t``
    (source → destination), this function returns the equivalent wrench in the
    destination frame:

    .. math::

       x' = \\begin{bmatrix} R & 0 \\\\ [p]_{\\times} R & R \\end{bmatrix} x

    Args:
        t (transform): The transform from the **source** frame to the
            **destination** frame.
        x (spatial_vector): The spatial wrench expressed in the source frame.

    Returns:
        spatial_vector: The wrench expressed in the destination frame.
    """

    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    f = wp.spatial_top(x)
    tau = wp.spatial_bottom(x)

    f = wp.quat_rotate(q, f)
    tau = wp.quat_rotate(q, tau) + wp.cross(p, f)

    return wp.spatial_vector(f, tau)


__axis_rotations = {}


def quat_between_axes(*axes: AxisType) -> wp.quat:
    """Compute the rotation between a sequence of axes.

    This function returns a quaternion that represents the cumulative rotation
    through a sequence of axes. For example, for axes (a, b, c), it computes
    the rotation from a to c by composing the rotation from a to b and b to c.

    Args:
        axes (AxisType): A sequence of axes, e.g., ('x', 'y', 'z').

    Returns:
        wp.quat: The total rotation quaternion.
    """
    q = wp.quat_identity()
    for i in range(len(axes) - 1):
        src = Axis.from_any(axes[i])
        dst = Axis.from_any(axes[i + 1])
        if (src.value, dst.value) in __axis_rotations:
            dq = __axis_rotations[(src.value, dst.value)]
        else:
            dq = wp.quat_between_vectors(src.to_vec3(), dst.to_vec3())
            __axis_rotations[(src.value, dst.value)] = dq
        q *= dq
    return q


@wp.func
def angvel_between_quats(q1: wp.quat, q2: wp.quat, dt: float):
    """
    Compute angular velocity vector omega (rad/s) taking orientation q1 to q2 over time dt.

    Args:
        q1, q2 : wp.quat
            Unit quaternions representing orientations.
        dt : float
            Time step in seconds (> 0).

    Returns:
        omega : ndarray shape (3,)
            Angular velocity vector in rad/s.

    Notes:
        - Uses omega = (2/dt) * log( q_rel ), where q_rel = q2 * conj(q1) with shortest-arc fix.
        - The returned omega corresponds to the rotation that maps q1 to q2.
        - q1 and q2 are normalized internally; dt must be positive.
    """

    if wp.dot(q1, q2) < 0.0:
        q2 = -q2

    q_rel = q2 * wp.quat_inverse(q1)
    q_rel = wp.normalize(q_rel)

    v = wp.vec3(q_rel.x, q_rel.y, q_rel.z)
    w = wp.clamp(q_rel.w, -1.0, 1.0)

    v_norm = wp.length(v)

    # Stable handling for small angles
    eps = 1e-12
    if v_norm < eps:
        omega = (2.0 / dt) * v
    else:
        phi = wp.atan2(v_norm, w)
        n = v / v_norm
        theta = 2.0 * phi
        omega = (theta / dt) * n

    return omega


__all__ = [
    "angvel_between_quats",
    "quat_between_axes",
    "quat_decompose",
    "quat_from_euler",
    "quat_to_euler",
    "quat_to_rpy",
    "quat_twist",
    "quat_twist_angle",
    "quat_velocity",
    "transform_twist",
    "transform_wrench",
    "velocity_at_point",
]
