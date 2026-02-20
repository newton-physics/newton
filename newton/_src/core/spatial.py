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
def quat_between_vectors_robust(from_vec: wp.vec3, to_vec: wp.vec3, eps: float = 1.0e-8) -> wp.quat:
    """Robustly compute the quaternion that rotates ``from_vec`` to ``to_vec``.

    This is a safer version of :func:`warp.quat_between_vectors` that handles the
    anti-parallel (180-degree) singularity by selecting a deterministic axis
    orthogonal to ``from_vec``.

    Args:
        from_vec: Source vector (assumed normalized).
        to_vec: Target vector (assumed normalized).
        eps: Tolerance for parallel/anti-parallel checks.

    Returns:
        wp.quat: Rotation quaternion q such that q * from_vec = to_vec.
    """
    d = wp.dot(from_vec, to_vec)

    if d >= 1.0 - eps:
        return wp.quat_identity()

    if d <= -1.0 + eps:
        # Deterministic axis orthogonal to from_vec.
        # Prefer cross with X, fallback to Y if nearly parallel.
        helper = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(from_vec[0]) >= 0.9:
            helper = wp.vec3(0.0, 1.0, 0.0)

        axis = wp.cross(from_vec, helper)
        axis_len = wp.length(axis)
        if axis_len <= eps:
            axis = wp.cross(from_vec, wp.vec3(0.0, 0.0, 1.0))
            axis_len = wp.length(axis)

        # Final fallback: if axis is still degenerate, pick an arbitrary axis.
        if axis_len <= eps:
            return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)

        axis = axis / axis_len
        return wp.quat_from_axis_angle(axis, wp.pi)

    return wp.quat_between_vectors(from_vec, to_vec)


@wp.func
def quat_velocity(q_now: wp.quat, q_prev: wp.quat, dt: float) -> wp.vec3:
    """Approximate angular velocity from successive world quaternions (world frame).

    Uses right-trivialized mapping via dq = q_now * conj(q_prev).

    Args:
        q_now: Current orientation in world frame.
        q_prev: Previous orientation in world frame.
        dt: Time step [s].

    Returns:
        Angular velocity omega in world frame [rad/s].
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


__axis_rotations = {}


def quat_between_axes(*axes: AxisType) -> wp.quat:
    """Compute the rotation between a sequence of axes.

    This function returns a quaternion that represents the cumulative rotation
    through a sequence of axes. For example, for axes (a, b, c), it computes
    the rotation from a to c by composing the rotation from a to b and b to c.

    Args:
        axes: A sequence of axes, e.g., ('x', 'y', 'z').

    Returns:
        The total rotation quaternion.
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


__all__ = [
    "quat_between_axes",
    "quat_between_vectors_robust",
    "quat_velocity",
]
