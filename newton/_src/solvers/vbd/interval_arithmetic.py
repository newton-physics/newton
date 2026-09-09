# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused interval arithmetic for VBD rigid point-plane trajectory tests.

This module intentionally implements only the operations needed to enclose the
fixed-axis rigid trajectory used by Planar-DAT. SolverVBD uses it only through
an explicitly experimental Stage 2 option.

Algebraic operations expand a correctly rounded float32 result by one ULP.
Trigonometric endpoints receive a wider provisional expansion because Warp's
CPU/CUDA sine and cosine error bounds are not part of its public contract.
Consequently, this module is conservative in the tested configurations but
must not yet be treated as a formal machine-level interval implementation.
"""

import warp as wp

wp.set_module_options({"enable_backward": False})

_PI = wp.constant(3.141592653589793)
_TRIG_ULP_PADDING = wp.constant(4)


@wp.struct
class FloatInterval:
    """Closed float32 interval with lower and upper endpoints."""

    lower: float
    upper: float


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
return nextafterf(value, INFINITY);
#else
return __builtin_nextafterf(value, INFINITY);
#endif
"""
)
def next_float_up(value: float) -> float:
    """Return the next representable float32 toward positive infinity."""

    ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
return nextafterf(value, -INFINITY);
#else
return __builtin_nextafterf(value, -INFINITY);
#endif
"""
)
def next_float_down(value: float) -> float:
    """Return the next representable float32 toward negative infinity."""

    ...


@wp.func
def _expand_up(value: float, ulps: int) -> float:
    result = value
    for _i in range(ulps):
        result = next_float_up(result)
    return result


@wp.func
def _expand_down(value: float, ulps: int) -> float:
    result = value
    for _i in range(ulps):
        result = next_float_down(result)
    return result


@wp.func
def interval(lower: float, upper: float) -> FloatInterval:
    """Construct an interval from ordered endpoints."""

    result = FloatInterval()
    result.lower = lower
    result.upper = upper
    return result


@wp.func
def point_interval(value: float) -> FloatInterval:
    """Represent one float32 input value exactly as a singleton interval."""

    return interval(value, value)


@wp.func
def interval_add(a: FloatInterval, b: FloatInterval) -> FloatInterval:
    """Outward-rounded interval addition."""

    return interval(next_float_down(a.lower + b.lower), next_float_up(a.upper + b.upper))


@wp.func
def interval_mul(a: FloatInterval, b: FloatInterval) -> FloatInterval:
    """Outward-rounded multiplication of finite intervals."""

    p00 = a.lower * b.lower
    p01 = a.lower * b.upper
    p10 = a.upper * b.lower
    p11 = a.upper * b.upper
    lower = wp.min(wp.min(p00, p01), wp.min(p10, p11))
    upper = wp.max(wp.max(p00, p01), wp.max(p10, p11))
    return interval(next_float_down(lower), next_float_up(upper))


@wp.func
def _linear_sinusoid_interval(angle: FloatInterval, cos_coeff: float, sin_coeff: float) -> FloatInterval:
    """Enclose ``cos_coeff*cos(u) + sin_coeff*sin(u)`` without losing correlation."""

    value_lower = cos_coeff * wp.cos(angle.lower) + sin_coeff * wp.sin(angle.lower)
    value_upper = cos_coeff * wp.cos(angle.upper) + sin_coeff * wp.sin(angle.upper)
    lower = _expand_down(wp.min(value_lower, value_upper), _TRIG_ULP_PADDING)
    upper = _expand_up(wp.max(value_lower, value_upper), _TRIG_ULP_PADDING)

    # Writing the scalar sinusoid as amplitude*cos(u-phase) gives its only
    # possible interior maximum and minimum on the shortest-arc domain.
    amplitude = wp.sqrt(cos_coeff * cos_coeff + sin_coeff * sin_coeff)
    phase = wp.atan2(sin_coeff, cos_coeff)
    if phase >= angle.lower and phase <= angle.upper:
        upper = wp.max(upper, _expand_up(amplitude, _TRIG_ULP_PADDING))
    minimum_angle = phase + _PI
    if minimum_angle >= angle.lower and minimum_angle <= angle.upper:
        lower = wp.min(lower, _expand_down(-amplitude, _TRIG_ULP_PADDING))
    return interval(lower, upper)


@wp.func
def rigid_point_plane_signed_distance_interval(
    t_lower: float,
    t_upper: float,
    n: wp.vec3,
    d: wp.vec3,
    c0: wp.vec3,
    dx: wp.vec3,
    axis: wp.vec3,
    angle: float,
    offset0: wp.vec3,
) -> FloatInterval:
    """Enclose the rigid point's signed plane distance on a time interval.

    Preconditions:
        0 <= t_lower <= t_upper <= 1, 0 <= angle <= pi, and axis is the unit
        world-space rotation axis used by the rigid trajectory whenever angle
        is nonzero. The zero-angle identity trajectory also permits a zero axis.
    """

    time = interval(t_lower, t_upper)
    angle_interval = interval_mul(time, point_interval(angle))
    # The mathematical shortest-arc domain is known even if outward arithmetic
    # expands a zero endpoint by one subnormal.
    angle_interval.lower = wp.max(angle_interval.lower, 0.0)
    parallel = axis * wp.dot(axis, offset0)
    perpendicular = offset0 - parallel
    cross = wp.cross(axis, offset0)

    constant = wp.dot(n, c0 - d + parallel)
    linear = wp.dot(n, dx)
    cos_coeff = wp.dot(n, perpendicular)
    sin_coeff = wp.dot(n, cross)

    result = interval_add(point_interval(constant), interval_mul(time, point_interval(linear)))
    return interval_add(result, _linear_sinusoid_interval(angle_interval, cos_coeff, sin_coeff))


@wp.func
def rigid_point_plane_signed_distance_derivative_interval(
    t_lower: float,
    t_upper: float,
    n: wp.vec3,
    dx: wp.vec3,
    axis: wp.vec3,
    angle: float,
    offset0: wp.vec3,
) -> FloatInterval:
    """Enclose the signed plane-distance derivative on a time interval.

    This uses the derivative of the same fixed-axis Rodrigues trajectory as
    :func:`rigid_point_plane_signed_distance_interval`. Its preconditions on
    time, angle, and the unit rotation axis are identical.
    """

    time = interval(t_lower, t_upper)
    angle_interval = interval_mul(time, point_interval(angle))
    angle_interval.lower = wp.max(angle_interval.lower, 0.0)
    parallel = axis * wp.dot(axis, offset0)
    perpendicular = offset0 - parallel
    cross = wp.cross(axis, offset0)

    linear = wp.dot(n, dx)
    # d/dt [C cos(angle*t) + D sin(angle*t)]
    #   = angle [D cos(angle*t) - C sin(angle*t)].
    cos_coeff = angle * wp.dot(n, cross)
    sin_coeff = -angle * wp.dot(n, perpendicular)
    return interval_add(point_interval(linear), _linear_sinusoid_interval(angle_interval, cos_coeff, sin_coeff))
